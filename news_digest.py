"""
News Digest Pipeline — Clean Rewrite
======================================
Two-stage funnel: cheap vector similarity → expensive LLM.
Only articles semantically close to your use-cases reach the LLM.

Embedding strategy — two fixed windows per article (no chunking loop):
  • Lede    (words 0–150)   : front-loaded signal for news/events/results
  • Extended (words 100–500): reaches past boilerplate intros in tech/research articles
  At most 2 embeddings per article. Best distance across both windows wins.

Debugging flags:
  --dry-run       Vector search only, no LLM. Shows distances to calibrate threshold.
  --show-rejects  Include LLM-rejected articles in the HTML report with rejection reasons.
                  Every rejected card has a direct link to the original article.

Requirements:
    pip install feedparser httpx chromadb openai pydantic-settings jinja2 trafilatura

Environment variables (or .env file):
    OPENROUTER_API_KEY=sk-or-...
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import json
import logging
import re
import signal
import sqlite3
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import feedparser
import httpx
import chromadb
import trafilatura
from jinja2 import Environment, BaseLoader
from openai import AsyncOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("news_digest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config(BaseSettings):
    """All tuneable knobs. Override any via environment variable or .env file."""

    openrouter_api_key: str

    db_path: Path = Path("./seen_news.db")
    chroma_path: Path = Path("./chroma_db")
    usecases_path: Path = Path("my_usecases.json")
    reports_dir: Path = Path("./reports")

    # Vector search — lower = more similar. Tune with --dry-run first.
    distance_threshold: float = 1.2

    # Models
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"

    # RSS
    rss_articles_per_query: int = 3

    # Article representation strategy (two fixed windows, no chunking loop):
    #
    #   lede_words         — title + first N words. Catches local news, events,
    #                        results where signal is always front-loaded.
    #
    #   extended_start/end — title + words [start:end]. Catches tech/research
    #                        articles whose intro is generic boilerplate and
    #                        real detail starts after word ~100.
    #                        Set extended_end_word=0 to disable this window.
    #
    # Both windows are embedded in a single API call (2 embeddings per article
    # max). Best distance across both windows wins.
    lede_words: int = 150
    extended_start_word: int = 100
    extended_end_word: int = 500

    # LLM context limit (words sent to LLM for evaluation)
    max_content_words: int = 3000

    # HTTP
    http_timeout: float = 15.0
    max_concurrent_fetches: int = 8

    # Retry
    max_retries: int = 3
    retry_base_delay: float = 1.0

    # Modes
    dry_run: bool = False
    show_rejects: bool = False  # include LLM-rejected articles in the HTML report

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UseCase:
    category: str
    search_string: str         # specific narrative prose — used for ChromaDB embeddings only
    match_criteria: str        # intent-based description — used as the LLM's acceptance criteria
    search_queries: list[str]  # Google News search terms


@dataclass
class RawArticle:
    title: str
    summary: str
    link: str


@dataclass
class EvaluatedArticle:
    """An article that passed both the vector gate and the LLM gate."""
    title: str
    url: str
    category: str
    summary: str
    distance: float


@dataclass
class RejectedArticle:
    """
    An article that passed the vector gate but was rejected by the LLM.
    Captured so you can understand why the LLM said no and click through
    to the original article to judge for yourself.
    """
    title: str
    url: str
    category: str          # use-case it matched semantically
    distance: float        # semantic distance (it passed the threshold)
    rejection_reason: str  # the LLM's own explanation


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def retry_async(coro_fn, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
    """
    Calls `await coro_fn(*args, **kwargs)` with exponential backoff.
    max_retries and base_delay are consumed here — NOT forwarded to coro_fn.
    """
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                break
            delay = base_delay * (2 ** attempt)
            log.warning(
                "Retry %d/%d for %s in %.1fs — %s",
                attempt + 1, max_retries,
                getattr(coro_fn, "__name__", repr(coro_fn)),
                delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# SQLite — single persistent connection per pipeline run
# ---------------------------------------------------------------------------

class SeenDB:
    """Thread-safe SQLite wrapper that stays open for the pipeline lifetime."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_articles (
                url          TEXT PRIMARY KEY,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
        log.debug("SQLite ready at %s", db_path)

    def is_seen(self, url: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_articles WHERE url = ?", (url,)
        ).fetchone()
        return row is not None

    def mark_seen(self, url: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_articles (url) VALUES (?)", (url,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# OpenRouter embeddings
# ---------------------------------------------------------------------------

async def get_embeddings(texts: list[str], cfg: Config) -> list[list[float]]:
    """
    Fetches embeddings from OpenRouter. Returns vectors in the same order
    as the input texts.
    """
    if not texts:
        return []

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {cfg.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.embedding_model,
                "input": texts,
                "encoding_format": "float",
            },
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Embedding API {resp.status_code}: {resp.text[:300]}")

    body = resp.json()

    if "error" in body:
        err = body["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"Embedding API error: {msg}")

    if "data" not in body:
        raise RuntimeError(f"Unexpected embedding response keys: {list(body.keys())}")

    sorted_items = sorted(body["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_items]


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def build_chroma_collection(cfg: Config) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(cfg.chroma_path),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    # No embedding_function — we supply embeddings manually
    return client.get_or_create_collection(name="news_use_cases")


async def sync_usecases(cfg: Config, collection: chromadb.Collection) -> list[UseCase]:
    """Load use-cases from JSON, embed search_strings, upsert into ChromaDB."""
    raw = json.loads(cfg.usecases_path.read_text(encoding="utf-8"))

    use_cases: list[UseCase] = []
    ids, docs, metas = [], [], []

    for i, item in enumerate(raw):
        uc = UseCase(
            category=item["category"],
            search_string=item["search_string"],
            match_criteria=item["match_criteria"],
            search_queries=item["search_queries"],
        )
        use_cases.append(uc)
        ids.append(f"uc_{i}")
        docs.append(uc.search_string)
        metas.append({
            "category": uc.category,
            "match_criteria": uc.match_criteria,
            "search_queries": json.dumps(uc.search_queries),
        })

    log.info("Embedding %d use-cases…", len(use_cases))
    embeddings = await retry_async(
        get_embeddings, docs, cfg,
        max_retries=cfg.max_retries,
        base_delay=cfg.retry_base_delay,
    )

    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    log.info("Synced %d use-cases into ChromaDB", len(use_cases))
    return use_cases


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------

def fetch_rss_articles(use_cases: list[UseCase], max_per_query: int) -> list[RawArticle]:
    """Pull headlines from Google News RSS. Deduplicates by URL."""
    seen_urls: set[str] = set()
    articles: list[RawArticle] = []

    log.info("Fetching RSS feeds…")
    for uc in use_cases:
        for query in uc.search_queries:
            time_query = f"{query} when:1d"
            encoded = urllib.parse.quote(time_query)
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
            )

            try:
                feed = feedparser.parse(rss_url)
            except Exception as exc:
                log.warning("RSS parse error for '%s': %s", query, exc)
                continue

            if not feed.entries:
                log.debug("No entries for query: %s", query)
                continue

            for entry in feed.entries[:max_per_query]:
                url = getattr(entry, "link", None)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                articles.append(RawArticle(
                    title=entry.get("title", "No title"),
                    summary=entry.get("summary", ""),
                    link=url,
                ))

    log.info("Collected %d unique articles from RSS", len(articles))
    return articles


# ---------------------------------------------------------------------------
# Full-text fetching
# ---------------------------------------------------------------------------

async def fetch_full_text(client: httpx.AsyncClient, article: RawArticle, cfg: Config) -> str:
    """
    Downloads the article page and extracts body text via trafilatura.
    Falls back to the RSS summary on any error or thin extraction.
    """
    fallback = re.sub(r"<[^>]+>", " ", article.summary).strip()

    try:
        resp = await client.get(
            article.link,
            follow_redirects=True,
            timeout=cfg.http_timeout,
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as exc:
        log.debug("HTTP fetch failed for %s: %s", article.link, exc)
        return fallback

    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    except Exception as exc:
        log.debug("trafilatura failed for %s: %s", article.link, exc)
        extracted = None

    if extracted and len(extracted.split()) > 50:
        return extracted

    log.debug("Extraction thin, using RSS fallback for %s", article.link)
    return fallback


# ---------------------------------------------------------------------------
# Article representations (replaces generic chunking)
# ---------------------------------------------------------------------------

def build_representations(title: str, content: str, cfg: Config) -> list[tuple[str, str]]:
    """
    Returns 1–2 (label, text) pairs to embed and query against ChromaDB.
    Best distance across both windows wins — so we catch signal regardless
    of where it sits in the article.

    Window 1 — lede (words 0 → lede_words)
        News is written inverted-pyramid style: the most important info is
        always at the top. Handles local news, events, scores, market moves.

    Window 2 — extended (words extended_start → extended_end)
        Tech release notes and research articles often open with generic
        boilerplate before the real detail. This window reaches past it.
        Skipped when the article is too short or extended_end_word=0.
    """
    words = content.split()
    representations: list[tuple[str, str]] = []

    # Window 1: lede
    lede_body = " ".join(words[: cfg.lede_words])
    representations.append(("lede", f"Title: {title}\nContent: {lede_body}"))

    # Window 2: extended
    if cfg.extended_end_word > 0 and len(words) > cfg.extended_start_word:
        ext_body = " ".join(words[cfg.extended_start_word : cfg.extended_end_word])
        if len(ext_body.split()) >= 50:  # skip if it adds no meaningful new text
            representations.append(("extended", f"Title: {title}\nContent: {ext_body}"))

    return representations


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

async def vector_search(
    collection: chromadb.Collection,
    title: str,
    content: str,
    cfg: Config,
) -> tuple[float, str, str]:
    """
    Embeds article representations in one batch call, queries ChromaDB,
    returns (best_distance, matched_category, matched_criteria).
    """
    representations = build_representations(title, content, cfg)
    if not representations:
        log.debug("No content to embed for: %s", title)
        return float("inf"), "", ""

    labels = [label for label, _ in representations]
    texts  = [text  for _, text  in representations]

    try:
        embeddings = await retry_async(
            get_embeddings, texts, cfg,
            max_retries=cfg.max_retries,
            base_delay=cfg.retry_base_delay,
        )
    except Exception as exc:
        log.warning("Embedding failed for '%s': %s", title[:60], exc)
        return float("inf"), "", ""

    try:
        results = collection.query(
            query_embeddings=embeddings,
            n_results=1,
            include=["distances", "metadatas", "documents"],
        )
    except Exception as exc:
        log.warning("ChromaDB query failed for '%s': %s", title[:60], exc)
        return float("inf"), "", ""

    best_distance    = float("inf")
    matched_category = ""
    matched_criteria = ""

    for i, distances in enumerate(results["distances"]):
        dist = distances[0]
        cat  = results["metadatas"][i][0]["category"]
        log.debug("  [%s] dist=%.4f  category=%s", labels[i], dist, cat)
        if dist < best_distance:
            best_distance    = dist
            matched_category = cat
            matched_criteria = results["metadatas"][i][0].get("match_criteria", "")

    return best_distance, matched_category, matched_criteria


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------

async def evaluate_article(
    llm: AsyncOpenAI,
    collection: chromadb.Collection,
    article: RawArticle,
    content: str,
    cfg: Config,
) -> tuple[Optional[EvaluatedArticle], Optional[RejectedArticle]]:
    """
    Stage 1 — vector gate (cheap).
    Stage 2 — LLM gate (expensive, only if stage 1 passes).

    Returns:
      (EvaluatedArticle, None)  — full match
      (None, RejectedArticle)   — passed vector gate, LLM said no (with reason)
      (None, None)              — vector gate rejected, or an error occurred
    """
    distance, category, criteria = await vector_search(
        collection, article.title, content, cfg
    )

    if distance > cfg.distance_threshold:
        log.info(
            "  ❌ VECTOR REJECT  dist=%.4f (threshold=%.2f)  %s",
            distance, cfg.distance_threshold, article.title[:70],
        )
        return None, None

    log.info(
        "  🔍 VECTOR PASS    dist=%.4f  category=%s  → LLM  %s",
        distance, category, article.title[:70],
    )

    if cfg.dry_run:
        # Show every vector-pass as a match so you can see what would reach the LLM
        return EvaluatedArticle(
            title=article.title,
            url=article.link,
            category=f"{category} [DRY RUN]",
            summary="Dry-run mode — LLM evaluation skipped.",
            distance=distance,
        ), None

    # Truncate to protect context window
    safe_content = " ".join(content.split()[: cfg.max_content_words])

    system_prompt = (
        "You are a news relevance filter. Your job is to decide if an article belongs "
        "to a category based on the acceptance criteria below.\n\n"
        f"CATEGORY: {category}\n\n"
        f"ACCEPTANCE CRITERIA:\n{criteria}\n\n"
        "RULES:\n"
        "1. Judge the article against the ACCEPTANCE CRITERIA above — not against any "
        "specific example you might expect. A valid article does not need to match a "
        "specific player, team, event, or phrasing.\n"
        "2. Reject generic news, clickbait, or articles that only mention the topic in passing.\n"
        "3. Respond ONLY with valid JSON:\n"
        '   {"is_match": <bool>, "summary": "<one sentence or null if rejected>", '
        '"rejection_reason": "<concise explanation of why rejected, or null if accepted>"}'
    )
    user_msg = f"Title: {article.title}\n\nContent:\n{safe_content}"

    try:
        response = await retry_async(
            llm.chat.completions.create,
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_retries=cfg.max_retries,
            base_delay=cfg.retry_base_delay,
        )
        payload = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        log.warning("LLM returned invalid JSON for '%s': %s", article.title[:60], exc)
        return None, None
    except Exception as exc:
        log.warning("LLM call failed for '%s': %s", article.title[:60], exc)
        return None, None

    if not payload.get("is_match"):
        reason = payload.get("rejection_reason") or "No reason provided."
        log.info(
            "  ❌ LLM REJECT  [%s] dist=%.4f  reason: %s  |  %s",
            category, distance, reason, article.title[:70],
        )
        return None, RejectedArticle(
            title=article.title,
            url=article.link,
            category=category,
            distance=distance,
            rejection_reason=reason,
        )

    log.info("  ✅ MATCH [%s] dist=%.4f  %s", category, distance, article.title[:70])
    return EvaluatedArticle(
        title=article.title,
        url=article.link,
        category=category,
        summary=payload.get("summary") or "No summary provided.",
        distance=distance,
    ), None


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

async def process_articles(
    articles: list[RawArticle],
    llm: AsyncOpenAI,
    collection: chromadb.Collection,
    db: SeenDB,
    cfg: Config,
) -> tuple[list[EvaluatedArticle], list[RejectedArticle]]:
    """
    Fetch full text and evaluate all articles concurrently.
    Returns (matches, rejects). rejects is only populated when the LLM ran.
    """
    semaphore = asyncio.Semaphore(cfg.max_concurrent_fetches)
    matches: list[EvaluatedArticle] = []
    rejects: list[RejectedArticle]  = []

    async def handle_one(article: RawArticle) -> None:
        if db.is_seen(article.link):
            log.debug("Already seen, skipping: %s", article.link)
            return

        async with semaphore:
            log.info("Processing: %s", article.title[:75])

            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (compatible; NewsDigest/3.0)"},
                follow_redirects=True,
            ) as http:
                content = await fetch_full_text(http, article, cfg)

            match, reject = await evaluate_article(llm, collection, article, content, cfg)

        # Mark seen regardless of outcome to avoid re-processing next run
        db.mark_seen(article.link)

        if match:
            matches.append(match)
        elif reject:
            rejects.append(reject)

    await asyncio.gather(*[handle_one(a) for a in articles])
    return matches, rejects


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>News Digest — {{ timestamp }}</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      line-height: 1.65; color: #1a1a2e; background: #f0f2f5;
      max-width: 900px; margin: 0 auto; padding: 2rem 1rem;
    }
    header { margin-bottom: 2rem; }
    header h1 {
      font-size: 1.8rem; color: #16213e;
      border-bottom: 3px solid #0f3460; padding-bottom: .5rem;
    }
    header p { color: #555; margin-top: .4rem; }

    /* ── Section headings ───────────────────────────────── */
    .section-heading {
      font-size: .9rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .08em; color: #666; margin: 2rem 0 1rem;
      display: flex; align-items: center; gap: .6rem;
    }
    .section-heading::after { content: ""; flex: 1; height: 1px; background: #ddd; }

    /* ── Match cards ────────────────────────────────────── */
    .card {
      background: #fff; border-radius: 10px; padding: 1.2rem 1.5rem;
      margin-bottom: 1.1rem; box-shadow: 0 2px 8px rgba(0,0,0,.07);
      border-left: 5px solid #0f3460;
    }
    .tag {
      display: inline-block; background: #e8eaf6; color: #3949ab;
      padding: 3px 10px; border-radius: 12px; font-size: .75rem;
      font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
      margin-bottom: .6rem;
    }
    .card h2 { font-size: 1.05rem; margin-bottom: .45rem; }
    .card h2 a { text-decoration: none; color: #16213e; }
    .card h2 a:hover { color: #0f3460; text-decoration: underline; }
    .summary { color: #444; font-size: .92rem; }
    .meta {
      font-size: .77rem; color: #999; margin-top: .85rem;
      border-top: 1px solid #f0f0f0; padding-top: .6rem;
      display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;
    }
    .meta a {
      color: #3949ab; font-weight: 600; text-decoration: none; font-size: .8rem;
    }
    .meta a:hover { text-decoration: underline; }

    /* ── Reject cards ───────────────────────────────────── */
    .reject-card {
      background: #fff8f8; border-radius: 10px; padding: 1.1rem 1.4rem;
      margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.05);
      border-left: 5px solid #e57373;
    }
    .reject-tag {
      display: inline-block; background: #fce4e4; color: #c62828;
      padding: 2px 9px; border-radius: 12px; font-size: .73rem;
      font-weight: 700; text-transform: uppercase; letter-spacing: .04em;
      margin-bottom: .5rem;
    }
    .reject-card h3 { font-size: .97rem; margin-bottom: .4rem; }
    .reject-card h3 a { text-decoration: none; color: #4a1010; }
    .reject-card h3 a:hover { text-decoration: underline; color: #c62828; }
    .reject-reason {
      font-size: .86rem; color: #7b3333;
      background: #fff0f0; border-radius: 6px;
      padding: .45rem .75rem; margin-top: .45rem;
    }
    .reject-reason strong { color: #c62828; }
    .reject-meta {
      font-size: .75rem; color: #bbb; margin-top: .7rem;
      border-top: 1px solid #f5e0e0; padding-top: .5rem;
      display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap;
    }
    .reject-meta a {
      color: #e57373; font-weight: 600; text-decoration: none; font-size: .77rem;
    }
    .reject-meta a:hover { text-decoration: underline; }
    .reject-hint {
      font-size: .82rem; color: #888; margin-bottom: 1rem;
      background: #fff; border-radius: 8px; padding: .75rem 1rem;
      border: 1px solid #f0d0d0;
    }
    .reject-hint code { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: .8rem; }

    .empty { text-align: center; padding: 3rem; color: #888; font-size: 1.05rem; }
    footer { text-align: center; color: #aaa; font-size: .8rem; margin-top: 2.5rem; }
  </style>
</head>
<body>
  <header>
    <h1>📰 Daily Intel Digest</h1>
    <p>Generated {{ timestamp }} &mdash;
      <strong>{{ articles | length }}</strong>
      match{{ 'es' if articles | length != 1 else '' }}
      {% if rejects %}
        &bull; <strong>{{ rejects | length }}</strong>
        LLM rejection{{ 's' if rejects | length != 1 else '' }}
      {% endif %}
    </p>
  </header>

  {# ── Matched articles ──────────────────────────────────── #}
  <div class="section-heading">✅ Matched articles</div>
  {% if articles %}
    {% for item in articles %}
    <div class="card">
      <span class="tag">{{ item.category }}</span>
      <h2>
        <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">
          {{ item.title }}
        </a>
      </h2>
      <p class="summary"><strong>Why it matters:</strong> {{ item.summary }}</p>
      <div class="meta">
        <span>Semantic distance: {{ "%.4f" | format(item.distance) }}</span>
        <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">↗ Open article</a>
      </div>
    </div>
    {% endfor %}
  {% else %}
    <div class="empty">No matching articles found today.</div>
  {% endif %}

  {# ── LLM-rejected articles (only rendered when --show-rejects is active) ── #}
  {% if rejects %}
    <div class="section-heading">🔍 Passed vector gate — rejected by LLM</div>
    <div class="reject-hint">
      These articles were semantically close to your use-cases but the LLM decided
      they weren't specific enough. Click any title or <strong>↗ Open article</strong>
      to read it yourself. If the LLM is wrong, consider refining your
      <code>search_string</code> or raising <code>--threshold</code>.
    </div>
    {% for item in rejects %}
    <div class="reject-card">
      <span class="reject-tag">{{ item.category }}</span>
      <h3>
        <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">
          {{ item.title }}
        </a>
      </h3>
      <div class="reject-reason">
        <strong>LLM reason:</strong> {{ item.rejection_reason }}
      </div>
      <div class="reject-meta">
        <span>Semantic distance: {{ "%.4f" | format(item.distance) }}</span>
        <a href="{{ item.url }}" target="_blank" rel="noopener noreferrer">↗ Open article</a>
      </div>
    </div>
    {% endfor %}
  {% endif %}

  <footer>News Digest Pipeline &bull; {{ timestamp }}</footer>
</body>
</html>"""


def generate_report(
    matches: list[EvaluatedArticle],
    rejects: list[RejectedArticle],
    reports_dir: Path,
    show_rejects: bool,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = reports_dir / f"Digest_{timestamp}.html"

    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(REPORT_TEMPLATE)
    rendered = template.render(
        timestamp=timestamp,
        articles=matches,
        rejects=rejects if show_rejects else [],
    )

    out_path.write_text(rendered, encoding="utf-8")
    log.info("Report saved → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _shutdown(signame: str) -> None:
        log.warning("Received %s — cancelling all tasks…", signame)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Not supported on Windows
            loop.add_signal_handler(sig, _shutdown, sig.name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def async_main(cfg: Config) -> None:
    log.info("=== NEWS DIGEST PIPELINE START ===")
    if cfg.dry_run:
        log.info("DRY-RUN MODE — LLM skipped, all vector-passes appear as matches")
    if cfg.show_rejects:
        log.info("SHOW-REJECTS MODE — LLM-rejected articles will appear in the report")

    if not cfg.usecases_path.exists():
        log.error("Use-cases file not found: %s", cfg.usecases_path)
        sys.exit(1)

    db = SeenDB(cfg.db_path)

    try:
        collection = build_chroma_collection(cfg)
        use_cases  = await sync_usecases(cfg, collection)

        raw_articles = fetch_rss_articles(use_cases, cfg.rss_articles_per_query)
        if not raw_articles:
            log.warning("No articles fetched. Check search queries and RSS connectivity.")
            return

        llm = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter_api_key,
        )

        matches, rejects = await process_articles(raw_articles, llm, collection, db, cfg)

        # Most relevant first within each section
        matches.sort(key=lambda a: a.distance)
        rejects.sort(key=lambda a: a.distance)

        report_path = generate_report(matches, rejects, cfg.reports_dir, cfg.show_rejects)

        vector_rejected = len(raw_articles) - len(matches) - len(rejects)
        log.info(
            "=== DONE — %d matched | %d LLM-rejected | %d vector-rejected | %d total ===",
            len(matches), len(rejects), vector_rejected, len(raw_articles),
        )
        log.info("Report → %s", report_path)

    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="News Digest Pipeline — semantic news filtering"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG logging (shows per-window distances, extraction details).",
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Override distance threshold (default 1.2). Lower = stricter vector gate.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Skip LLM calls. Every article that passes the vector gate appears in the "
            "report. Use this to calibrate --threshold before spending LLM credits."
        ),
    )
    parser.add_argument(
        "--show-rejects", action="store_true",
        help=(
            "Include LLM-rejected articles in the HTML report with the LLM's rejection "
            "reason and a direct link to the original article. Use this to debug why the "
            "LLM is filtering out articles you expected to match."
        ),
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = Config()

    overrides: dict = {}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.show_rejects:
        overrides["show_rejects"] = True
    if args.threshold is not None:
        overrides["distance_threshold"] = args.threshold
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    install_signal_handlers(loop)

    try:
        loop.run_until_complete(async_main(cfg))
    except asyncio.CancelledError:
        log.info("Pipeline cancelled by signal.")
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()