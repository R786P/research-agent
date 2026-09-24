# Research Agent

Flask app where an LLM agent researches a goal: it **plans** search queries, uses a **web search tool**, and **writes a short cited report** from the results.

## Flow
User Goal -> Plan -> Web Search Tool -> Summarize -> Sourced Report

## Setup
Environment variables:
- `GEMINI_API_KEY` (required) - your Google AI Studio key. Never commit it to GitHub.
- `GEMINI_MODEL` (optional) - defaults to `gemini-2.5-flash`. Set it to any current Gemini model your key supports.
- `RATE_LIMIT_PER_HOUR` (optional) - requests per IP per hour, default 10.
- `DEMO_MODE=1` (optional) - runs with fake data, no API key needed, for testing the UI.

## Run locally
```
pip install -r requirements.txt
GEMINI_API_KEY=your_key python app.py
```

## Deploy on Render (free)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app --timeout 120`
- Add `GEMINI_API_KEY` under Environment.

## Notes
- Search uses DuckDuckGo (`ddgs`) with a Wikipedia fallback. No search API key needed.
- Search results are treated as untrusted text. Reports can still contain mistakes, so always check the sources.
- Includes a simple per-IP rate limit to protect your API quota.
