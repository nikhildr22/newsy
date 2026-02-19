# NewsScraper

A Python 3.11+ news digest generator that uses LLMs and vector search to filter and summarize news articles. Easily configurable and designed for async performance.

## Features
- Async HTTP and OpenAI API access
- Vector search for article filtering
- LLM-based summarization and rejection
- Configurable use cases via JSON
- HTML report generation
- CLI options for dry-run and debugging

## Project Structure
- `news_digest.py`: Main script with all core logic and configuration
- `requirements.txt`: Python dependencies
- `my_usecases.json`: Use case definitions for filtering news
- `chroma_db/`, `usecase_vectordb/`: Vector database storage
- `reports/`: Output HTML reports

## Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure environment:**
   - Set environment variables or create a `.env` file (auto-loaded)
   - All configuration is in the config class in `news_digest.py`

3. **Run the main script:**
   ```sh
   python news_digest.py
   ```

## CLI Options
- `--dry-run`: Vector search only, no LLM. Shows distances to calibrate threshold.
- `--show-rejects`: Include LLM-rejected articles in the HTML report with rejection reasons.

## Adding Use Cases
- Edit `my_usecases.json` to add or modify use cases
- Each use case must have a `search_string` and `match_criteria`

## Tips & Pitfalls
- Use async/await for all I/O
- Calibrate vector threshold with `--dry-run` before running with LLM
- Check environment variables if you see unexpected results

## Output
- HTML reports are saved in the `reports/` directory

## Example Prompts
- "Summarize the architecture of this project."
- "How do I add a new use case for a different news topic?"
- "What does the --dry-run option do?"

## License
MIT License
