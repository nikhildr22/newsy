# Copilot Workspace Instructions for newsScraper

## Overview
This project is a Python 3.11+ news digest generator that uses LLMs and vector search to filter and summarize news articles. It is configured via environment variables or a `.env` file, and uses async libraries for HTTP and OpenAI API access.

## Build & Run
- **Install dependencies:**
  ```sh
  pip install -r requirements.txt
  ```
- **Run the main script:**
  ```sh
  python news_digest.py
  ```
- **Key CLI options:**
  - `--dry-run`: Vector search only, no LLM. Shows distances to calibrate threshold.
  - `--show-rejects`: Include LLM-rejected articles in the HTML report with rejection reasons.

## Project Structure
- `news_digest.py`: Main script, contains all core logic and configuration.
- `requirements.txt`: Python dependencies.
- `my_usecases.json`: Use case definitions for filtering news.
- `chroma_db/`, `usecase_vectordb/`: Vector database storage.
- `reports/`: Output HTML reports.

## Configuration
- Environment variables or `.env` file (auto-loaded by pydantic-settings).
- All tuneable knobs are in the config class in `news_digest.py`.
- Key paths (DB, chroma, usecases, reports) are configurable.

## Conventions & Pitfalls
- Use async/await for all I/O (httpx, OpenAI, etc.).
- Always calibrate vector threshold with `--dry-run` before running with LLM.
- Usecases must have clear `search_string` and `match_criteria` fields.
- If you see unexpected results, check for missing/incorrect environment variables.

## Example Prompts
- "Summarize the architecture of this project."
- "How do I add a new use case for a different news topic?"
- "What does the --dry-run option do?"

## Next Steps
- Consider adding agent customizations for usecase management or report generation automation.
- For complex workflows, create additional instructions files scoped to specific directories (e.g., `usecase_vectordb/`).
