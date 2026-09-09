#!/usr/bin/env python3
"""
web_search.py

Async web search scraper: no Selenium, no browser process. Uses plain
async HTTP requests for both the search-results pages and the individual
result pages, fetched concurrently.

NOTE: this targets html.duckduckgo.com, not Google. As of 2025, Google
serves a JS-required interstitial (/httpservice/retry/enablejs) to any
non-browser client -- no header/cookie combination gets past it without
a real browser (Selenium/Playwright), which defeats the point of a fast,
low-RAM scraper. DuckDuckGo's HTML endpoint is server-rendered and does
not have this wall, so it's used here as the working equivalent.

Install:
    pip install aiohttp beautifulsoup4 lxml

Usage:
    python web_search.py "how to make omelette" -n 3 -o results.json
"""

import argparse
import asyncio
import json
import random
import re
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

SEARCH_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def clean_result_link(href: str) -> str:
    # DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded-url>&rut=...
    if "uddg=" in href:
        qs = parse_qs(urlparse(href, scheme="https").query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    return href


def parse_results(html: str):
    soup = BeautifulSoup(html, "lxml")
    results = []
    for block in soup.select("div.result"):
        title_tag = block.select_one("a.result__a")
        if not title_tag:
            continue
        href = clean_result_link(title_tag.get("href", ""))
        if not href.startswith("http"):
            continue
        results.append({"title": title_tag.get_text(strip=True), "link": href})
    return results


def extract_next_params(html: str) -> dict | None:
    # DuckDuckGo's "More results" form carries a vqd/dc continuation token
    # that must be POSTed back to get page 2+; a bare `s=` offset is ignored.
    soup = BeautifulSoup(html, "lxml")
    for form in soup.select("form"):
        if form.select_one("input[name=dc]"):
            return {i.get("name"): i.get("value", "") for i in form.select("input") if i.get("name")}
    return None


async def fetch_search_page(session: aiohttp.ClientSession, query: str,
                             next_params: dict | None = None, timeout: int = 10):
    """Fetch one page of results. Pass the previous call's returned
    next_params to advance to the following page; None means "first page".
    Returns (results, next_params_for_the_page_after_this_one)."""
    try:
        if next_params is None:
            request = session.get(
                "https://html.duckduckgo.com/html/", params={"q": query},
                headers=SEARCH_HEADERS, timeout=timeout,
            )
        else:
            request = session.post(
                "https://html.duckduckgo.com/html/", data=next_params,
                headers=SEARCH_HEADERS, timeout=timeout,
            )
        async with request as resp:
            if resp.status != 200:
                return [], None
            html = await resp.text()
            return parse_results(html), extract_next_params(html)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return [], None


async def fetch_content(session: aiohttp.ClientSession, item: dict, sem: asyncio.Semaphore,
                         max_chars: int = 5000, timeout: int = 10):
    async with sem:
        try:
            async with session.get(item["link"], headers={"User-Agent": USER_AGENT},
                                    timeout=timeout, allow_redirects=True) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    item["content"] = f"[Skipped non-HTML content: {content_type}]"
                    return item
                html = await resp.text(errors="ignore")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            item["content"] = f"[Error fetching content: {e}]"
            return item

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
        item["content"] = text[:max_chars]
        return item


async def web_search_async(query: str, n_pages: int = 1,
                            max_chars: int = 5000, max_concurrency: int = 8,
                            page_delay_range=(0.5, 1.5)) -> dict:
    output = {"search": query, "pages": {}}
    sem = asyncio.Semaphore(max_concurrency)

    connector = aiohttp.TCPConnector(limit=max_concurrency)
    next_params = None
    async with aiohttp.ClientSession(connector=connector) as session:
        for page_num in range(n_pages):
            page_key = f"page_{page_num + 1}"
            results, next_params = await fetch_search_page(session, query, next_params)

            if not results:
                # No more results, or blocked -- stop early rather than
                # writing empty/garbage pages.
                break

            tasks = [fetch_content(session, r, sem, max_chars=max_chars) for r in results]
            enriched = await asyncio.gather(*tasks)
            output["pages"][page_key] = list(enriched)

            if next_params is None:
                break

            # Small politeness delay between *search* page requests only
            # (content fetches are already concurrency-limited).
            await asyncio.sleep(random.uniform(*page_delay_range))

    return output


def web_search(query: str, n_pages: int = 1, max_chars: int = 5000, max_concurrency: int = 8) -> dict:
    """Sync wrapper -- call this if you don't want to deal with asyncio yourself."""
    return asyncio.run(web_search_async(
        query, n_pages=n_pages, max_chars=max_chars, max_concurrency=max_concurrency,
    ))


def main():
    parser = argparse.ArgumentParser(description="Fast async DuckDuckGo search scraper (no Selenium).")
    parser.add_argument("query", default="How to make an omelette", help="Search query")
    parser.add_argument("-n", "--pages", type=int, default=1, help="Number of result pages to scrape")
    parser.add_argument("-o", "--output", default="results.json", help="Output JSON file path")
    parser.add_argument("--max-chars", type=int, default=5000, help="Max chars of content kept per result")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent content fetches")
    args = parser.parse_args()

    data = web_search(
        args.query,
        n_pages=args.pages,
        max_chars=args.max_chars,
        max_concurrency=args.concurrency,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    total = sum(len(v) for v in data["pages"].values())
    print(f"Saved {total} result(s) across {len(data['pages'])} page(s) to {args.output}")


if __name__ == "__main__":
    main()
