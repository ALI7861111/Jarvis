#!/usr/bin/env python3
"""
web_search.py

Web search via the Tavily Search API (https://tavily.com).

Replaces a previous DuckDuckGo HTML scraper: html.duckduckgo.com started
serving an "anomaly detected" bot-check page to plain HTTP clients, which
silently produced zero results with no error. Tavily is a paid/free-tier
API built for this (agent/LLM search) use case and already returns
extracted page content per result, so no separate fetch-and-parse step
is needed.

Setup:
    Get a key at https://tavily.com and set TAVILY_API_KEY in .env.

Usage:
    python -m tools.web_search "how to make omelette" -n 5 -o results.json
"""

import argparse
import asyncio
import json

import aiohttp

import config

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Tavily returns one page of up to 20 results per call (no true pagination),
# so n_pages is translated into a result count instead of literal pages.
RESULTS_PER_PAGE = 5
MAX_RESULTS = 20


class WebSearchError(RuntimeError):
    """Raised when web search cannot run at all (e.g. missing API key)."""


async def web_search_async(query: str, n_pages: int = 1, max_chars: int = 5000,
                            max_concurrency: int = 8) -> dict:
    """Search the web via Tavily and return content for each result.

    Kept the same {"search": ..., "pages": {"page_1": [...]}} shape as the
    old scraper so callers (research_workflow.py, the researcher agent's
    tool schema) don't need to change. `n_pages` maps to Tavily's
    `max_results` (RESULTS_PER_PAGE per unit, capped at MAX_RESULTS);
    `max_concurrency` is accepted for backward compatibility but unused --
    Tavily is a single API call, not per-result fetches.

    Raises:
        WebSearchError: if TAVILY_API_KEY is not configured.
    """
    if not config.TAVILY_API_KEY:
        raise WebSearchError(
            "TAVILY_API_KEY is not set -- add it to .env to enable web search."
        )

    max_results = min(max(n_pages, 1) * RESULTS_PER_PAGE, MAX_RESULTS)
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_raw_content": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                TAVILY_SEARCH_URL, json=payload,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    # A single query failing (rate limit, transient 5xx) shouldn't
                    # abort a whole multi-query research run -- log and move on.
                    print(f"[web_search] Tavily API error {resp.status}: {body}")
                    return {"search": query, "pages": {}}
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"[web_search] Tavily request failed: {e}")
        return {"search": query, "pages": {}}

    items = [
        {
            "title": r.get("title", ""),
            "link": r.get("url", ""),
            "content": (r.get("content") or "")[:max_chars],
        }
        for r in body.get("results", [])
    ]

    return {"search": query, "pages": {"page_1": items} if items else {}}


def web_search(query: str, n_pages: int = 1, max_chars: int = 5000, max_concurrency: int = 8) -> dict:
    """Sync wrapper -- call this if you don't want to deal with asyncio yourself."""
    return asyncio.run(web_search_async(
        query, n_pages=n_pages, max_chars=max_chars, max_concurrency=max_concurrency,
    ))


def main():
    parser = argparse.ArgumentParser(description="Web search via the Tavily API.")
    parser.add_argument("query", default="How to make an omelette", help="Search query")
    parser.add_argument("-n", "--pages", type=int, default=1,
                         help=f"Result multiplier ({RESULTS_PER_PAGE} results per unit, capped at {MAX_RESULTS})")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file path")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max chars of content kept per result")
    args = parser.parse_args()

    data = web_search(args.query, n_pages=args.pages, max_chars=args.max_chars)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in data["pages"].values())
    print(f"Saved {total} result(s) to {args.output}")


if __name__ == "__main__":
    main()
