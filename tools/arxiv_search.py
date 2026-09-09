#!/usr/bin/env python3
"""
arxiv_search.py

Search arXiv.org for papers and optionally download their PDFs. Uses
arXiv's public Atom API (export.arxiv.org/api/query) -- no API key needed,
no scraping. arXiv asks scripts not to hammer their PDF server, so
downloads are sequential with a delay between requests.

Install:
    pip install aiohttp

Usage:
    python arxiv_search.py "transformer attention" -n 5
    python arxiv_search.py "transformer attention" -n 5 --download -o papers/
"""

import argparse
import asyncio
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import aiohttp

ATOM_NS = "{http://www.w3.org/2005/Atom}"

API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "jarvis-arxiv-tool/1.0"


def _text(el, tag: str) -> str:
    node = el.find(f"{ATOM_NS}{tag}")
    return node.text.strip() if node is not None and node.text else ""


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id_url = _text(entry, "id")
        arxiv_id = arxiv_id_url.rsplit("/", 1)[-1] if arxiv_id_url else ""

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        authors = [_text(author, "name") for author in entry.findall(f"{ATOM_NS}author")]

        papers.append({
            "id": arxiv_id,
            "title": re.sub(r"\s+", " ", _text(entry, "title")).strip(),
            "summary": re.sub(r"\s+", " ", _text(entry, "summary")).strip(),
            "authors": authors,
            "published": _text(entry, "published"),
            "pdf_url": pdf_url,
            "abs_url": arxiv_id_url,
        })
    return papers


async def arxiv_search_async(query: str, max_results: int = 10, timeout: int = 15) -> list[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession() as session:
        async with session.get(API_URL, params=params, headers=headers, timeout=timeout) as resp:
            resp.raise_for_status()
            xml_text = await resp.text()
    return parse_feed(xml_text)


def arxiv_search(query: str, max_results: int = 10) -> list[dict]:
    """Sync wrapper -- call this if you don't want to deal with asyncio yourself."""
    return asyncio.run(arxiv_search_async(query, max_results=max_results))


def _safe_filename(title: str, arxiv_id: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", title).strip("_")[:80]
    return f"{arxiv_id}_{slug}.pdf" if slug else f"{arxiv_id}.pdf"


async def download_paper(session: aiohttp.ClientSession, paper: dict, out_dir: Path, timeout: int = 30) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / _safe_filename(paper["title"], paper["id"])
    async with session.get(paper["pdf_url"], headers={"User-Agent": USER_AGENT}, timeout=timeout) as resp:
        resp.raise_for_status()
        data = await resp.read()
    dest.write_bytes(data)
    return str(dest)


async def download_papers_async(papers: list[dict], out_dir: str = "papers", delay: float = 3.0) -> list[str]:
    out_path = Path(out_dir)
    paths = []
    async with aiohttp.ClientSession() as session:
        for i, paper in enumerate(papers):
            if not paper.get("pdf_url"):
                continue
            paths.append(await download_paper(session, paper, out_path))
            if i < len(papers) - 1:
                await asyncio.sleep(delay)
    return paths


def download_papers(papers: list[dict], out_dir: str = "papers", delay: float = 3.0) -> list[str]:
    """Sync wrapper -- call this if you don't want to deal with asyncio yourself."""
    return asyncio.run(download_papers_async(papers, out_dir=out_dir, delay=delay))


async def arxiv_search_and_download_async(query: str, max_results: int = 3, out_dir: str = "papers",
                                            delay: float = 3.0) -> list[dict]:
    """Search arXiv and download the matching PDFs in one step, so callers
    (including tool-calling LLMs) never need to hand a paper dict back in --
    just a query string. Each returned paper dict gains a "local_path" key
    (None if that paper had no pdf_url)."""
    papers = await arxiv_search_async(query, max_results=max_results)
    out_path = Path(out_dir)
    async with aiohttp.ClientSession() as session:
        for i, paper in enumerate(papers):
            if not paper.get("pdf_url"):
                paper["local_path"] = None
                continue
            paper["local_path"] = await download_paper(session, paper, out_path)
            if i < len(papers) - 1:
                await asyncio.sleep(delay)
    return papers


def arxiv_search_and_download(query: str, max_results: int = 3, out_dir: str = "papers", delay: float = 3.0) -> list[dict]:
    """Sync wrapper -- searches arXiv and downloads matching PDFs in one call."""
    return asyncio.run(arxiv_search_and_download_async(query, max_results=max_results, out_dir=out_dir, delay=delay))


def main():
    parser = argparse.ArgumentParser(description="Search arXiv and optionally download paper PDFs.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--max-results", type=int, default=5, help="Number of papers to fetch")
    parser.add_argument("--download", action="store_true", help="Download PDFs of the found papers")
    parser.add_argument("-o", "--output-dir", default="papers", help="Directory to save downloaded PDFs")
    args = parser.parse_args()

    papers = arxiv_search(args.query, max_results=args.max_results)

    for p in papers:
        print(f"[{p['id']}] {p['title']}")
        print(f"    authors: {', '.join(p['authors'])}")
        print(f"    pdf: {p['pdf_url']}")

    if args.download:
        paths = download_papers(papers, out_dir=args.output_dir)
        print(f"\nDownloaded {len(paths)} paper(s) to {args.output_dir}/")


if __name__ == "__main__":
    main()
