<![CDATA[# 📰 Newsy — AI-Powered News Digest Pipeline

A Python 3.11+ pipeline that curates a personalised daily news digest using a **two-stage funnel**: cheap vector-similarity search followed by expensive LLM evaluation. Only the articles that are semantically close to your interests ever reach the LLM, keeping costs low and signal high.

> Built with async Python, OpenRouter, ChromaDB, and Jinja2.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **Two-stage filtering** | Vector gate (ChromaDB) → LLM gate (OpenRouter) |
| **Async everywhere** | `httpx` + `asyncio` for concurrent RSS fetching, full-text extraction, embedding, and LLM calls |
| **Configurable use cases** | Define topics, search queries, and acceptance criteria in a single JSON file |
| **Smart article windowing** | Lede + extended windows ensure signal is captured regardless of article structure |
| **Deduplication** | SQLite-backed `SeenDB` prevents re-processing across runs |
| **HTML digest reports** | Styled, responsive reports saved to `reports/` with matched & rejected sections |
| **CLI debugging tools** | `--dry-run` to calibrate thresholds; `--show-rejects` to audit LLM decisions |

---

## 📂 Project Structure

```
newsScraper/
├── news_digest.py          # Main pipeline script (config, data classes, pipeline, report)
├── my_usecases.json        # Use-case definitions — topics, queries, criteria
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Template for environment variables
├── .env                    # Your actual secrets (git-ignored)
├── seen_news.db            # SQLite DB tracking processed article URLs (auto-created)
├── chroma_db/              # ChromaDB persistent storage for use-case embeddings
├── usecase_vectordb/       # Additional vector DB storage
├── reports/                # Generated HTML digest reports
└── .github/
    └── copilot-instructions.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- An **[OpenRouter](https://openrouter.ai/)** API key (provides access to embedding + LLM models)

### 1. Clone & create a virtual environment

```sh
git clone https://github.com/nikhildr22/newsy.git
cd newsy

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Configure environment

Copy the example file and fill in your API key:

```sh
cp .env.example .env
```

Open `.env` and set at minimum:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

All other settings have sensible defaults. See [Configuration Reference](#-configuration-reference) below.

### 4. Define your use cases

Edit `my_usecases.json` to specify the topics you care about. Each use case needs:

| Field | Purpose |
|---|---|
| `category` | Human-readable category name |
| `search_queries` | List of Google News RSS search terms |
| `search_string` | Narrative prose describing ideal articles — embedded into ChromaDB |
| `match_criteria` | Instructions the LLM uses to accept/reject articles |

### 5. Run the pipeline

```sh
python news_digest.py
```

The report is saved to `reports/Digest_YYYY-MM-DD_HH-MM.html`.

---

## 🛠 CLI Options

| Flag | Description |
|---|---|
| `--dry-run` | Skip LLM calls. All articles passing the vector gate appear as matches. Use to calibrate `--threshold`. |
| `--show-rejects` | Include LLM-rejected articles in the report with rejection reasons and direct links. |
| `--threshold <float>` | Override the vector distance threshold (default `1.2`). Lower = stricter. |
| `--debug` | Enable `DEBUG` logging — shows per-window distances, extraction details, etc. |

**Example — calibrate your threshold:**

```sh
python news_digest.py --dry-run --threshold 1.0
```

---

## ⚙ Configuration Reference

All settings live in the `Config` class and can be overridden via **environment variables** or a **`.env`** file.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | Your OpenRouter API key |
| `DB_PATH` | `./seen_news.db` | SQLite database for deduplication |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistent storage path |
| `USECASES_PATH` | `my_usecases.json` | Path to use-case definitions |
| `REPORTS_DIR` | `./reports` | Output directory for HTML reports |
| `DISTANCE_THRESHOLD` | `1.2` | Vector similarity threshold (lower = stricter) |
| `LLM_MODEL` | `openai/gpt-4o-mini` | LLM model for article evaluation |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `RSS_ARTICLES_PER_QUERY` | `3` | Max articles per RSS query |
| `LEDE_WORDS` | `150` | Word count for the lede embedding window |
| `EXTENDED_START_WORD` | `100` | Start of the extended embedding window |
| `EXTENDED_END_WORD` | `500` | End of the extended window (set `0` to disable) |
| `MAX_CONTENT_WORDS` | `3000` | Max words sent to the LLM |
| `HTTP_TIMEOUT` | `15.0` | HTTP request timeout (seconds) |
| `MAX_CONCURRENT_FETCHES` | `8` | Concurrency limit for article fetching |
| `MAX_RETRIES` | `3` | Retry count for API calls |
| `RETRY_BASE_DELAY` | `1.0` | Base delay for exponential backoff (seconds) |

---

## 📄 Output

HTML reports are written to `reports/` with timestamped filenames:

```
reports/Digest_2026-03-06_09-02.html
```

Each report contains:
- **Matched articles** — category tag, title (linked), LLM summary, and semantic distance
- **Rejected articles** *(when `--show-rejects` is used)* — rejection reason and a link to judge for yourself

---

## 💡 Tips & Pitfalls

1. **Start with `--dry-run`** — calibrate your `distance_threshold` before spending LLM credits.
2. **Write specific `search_string` values** — vague descriptions produce poor embeddings; use concrete, narrative prose.
3. **Keep `match_criteria` clear** — tell the LLM exactly what to accept *and* what to reject.
4. **Check your `.env`** — missing or incorrect API keys surface as cryptic HTTP errors.
5. **Monitor costs** — each non-dry run makes embedding + LLM calls via OpenRouter.

---

## 📖 Further Reading

See **[PROCESS.md](./PROCESS.md)** for a deep dive into how the pipeline works end-to-end.

---

## 📜 License

MIT License
]]>
