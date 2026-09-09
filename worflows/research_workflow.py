#!/usr/bin/env python3
"""
research_workflow.py

Deterministic research pipeline:

    Research Workflow
         |
         v
    Understand Task
         |
         v
    Generate Search Queries
         |
         v
    Researcher Agent  -- web_search() per query, reads webpages
         |
         v
    Evaluate Sources
         |
         v
    Summarizer Agent
         |
         v
    Final Research Report

Unlike agents/researcher.py (a free-form AssistantAgent that decides for
itself whether/how to search), this module runs research as a fixed
sequence of steps so each stage's output can be inspected and the process
reliably ends in a written report.

Usage:
    python -m worflows.research_workflow "impact of tariffs on semiconductor supply chains"
"""

import asyncio
import json
from dataclasses import dataclass

from autogen_agentchat.agents import AssistantAgent

from model_client import get_model_client
from tools.web_search import web_search_async


@dataclass
class Source:
    title: str
    link: str
    content: str
    query: str
    relevance: int = 0
    reason: str = ""


@dataclass
class ResearchResult:
    task: str
    understanding: str
    queries: list[str]
    sources: list[Source]
    report: str


def _agent(name: str, system_message: str, model_client) -> AssistantAgent:
    return AssistantAgent(name=name, model_client=model_client, system_message=system_message)


async def _ask(agent: AssistantAgent, prompt: str) -> str:
    result = await agent.run(task=prompt)
    return str(result.messages[-1].content).strip()


async def understand_task(task: str, model_client) -> str:
    """Step 1: restate the task as a concrete research brief -- the core
    question(s) to answer, key sub-topics, and any constraints."""
    agent = _agent(
        "task_analyst",
        "You turn a user's research request into a short, concrete research "
        "brief: the core question(s) to answer, key sub-topics, and any "
        "constraints (timeframe, geography, source type). 3-6 sentences, "
        "no preamble, no questions back to the user.",
        model_client,
    )
    return await _ask(agent, task)


async def generate_search_queries(understanding: str, model_client, n: int = 4) -> list[str]:
    """Step 2: turn the brief into concrete, diverse search-engine queries."""
    agent = _agent(
        "query_writer",
        f"You write {n} diverse, concrete web search queries that together "
        "cover a research brief. Vary angles (definitions, recent news, "
        "data/statistics, expert opinion, counterarguments) rather than "
        "repeating the same phrasing. Reply with ONLY a JSON array of "
        f"{n} strings -- no prose, no markdown fences.",
        model_client,
    )
    raw = await _ask(agent, understanding)
    try:
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:n]
    except json.JSONDecodeError:
        pass
    # Fallback if the model didn't return JSON: one query per non-empty line.
    lines = [line.strip("-* \t") for line in raw.splitlines() if line.strip()]
    return lines[:n] if lines else [understanding]


async def run_researcher(queries: list[str], max_chars: int = 3000) -> list[Source]:
    """Step 3: the Researcher Agent -- web_search() for each query, reading
    the fetched webpages, collecting every result as a Source."""
    sources: list[Source] = []
    for query in queries:
        result = await web_search_async(query, n_pages=1, max_chars=max_chars)
        for page in result["pages"].values():
            for item in page:
                sources.append(Source(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    content=item.get("content", ""),
                    query=query,
                ))
    return sources


async def evaluate_sources(understanding: str, sources: list[Source], model_client,
                            keep_top: int = 8) -> list[Source]:
    """Step 4: score each source's relevance to the brief and drop the
    rest -- keeps irrelevant or empty-content pages out of the report."""
    if not sources:
        return []

    agent = _agent(
        "source_evaluator",
        "You judge how relevant each numbered source is to a research "
        "brief, on a 0-10 scale (0 = irrelevant/empty, 10 = directly "
        "answers it). Reply with ONLY a JSON array of objects "
        '[{"index": <int>, "relevance": <int>, "reason": "<short phrase>"}], '
        "one entry per source, no prose.",
        model_client,
    )

    listing = "\n\n".join(
        f"[{i}] {s.title} ({s.link})\n{s.content[:800]}"
        for i, s in enumerate(sources)
    )
    raw = await _ask(agent, f"Research brief:\n{understanding}\n\nSources:\n{listing}")

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = []

    by_index = {s["index"]: s for s in scores if isinstance(s, dict) and "index" in s}
    for i, source in enumerate(sources):
        info = by_index.get(i)
        if info:
            source.relevance = int(info.get("relevance", 0))
            source.reason = str(info.get("reason", ""))

    ranked = sorted(sources, key=lambda s: s.relevance, reverse=True)
    kept = [s for s in ranked if s.relevance > 0][:keep_top]
    # If scoring failed entirely (e.g. bad JSON), fall back to keeping
    # whatever was fetched rather than producing an empty report.
    return kept or ranked[:keep_top]


async def summarize_report(task: str, understanding: str, sources: list[Source], model_client) -> str:
    """Step 5: the Summarizer Agent -- turn the surviving sources into a
    structured final report, citing which source backs each claim."""
    agent = _agent(
        "summarizer",
        "You write a clear, well-structured research report from the "
        "provided sources. Use markdown headings, synthesize across "
        "sources rather than listing them one by one, and cite sources "
        "inline as [n] matching the numbering given. End with a 'Sources' "
        "section listing each [n] as a markdown link. Do not invent facts "
        "not present in the sources -- if the sources don't cover part of "
        "the brief, say so explicitly.",
        model_client,
    )

    listing = "\n\n".join(
        f"[{i + 1}] {s.title} ({s.link})\n{s.content}"
        for i, s in enumerate(sources)
    )
    prompt = (
        f"Original task: {task}\n\nResearch brief: {understanding}\n\n"
        f"Sources:\n{listing or '(no relevant sources found)'}"
    )
    return await _ask(agent, prompt)


async def run_research_workflow(task: str, n_queries: int = 4, keep_top: int = 8) -> ResearchResult:
    """Run the full pipeline: Understand Task -> Generate Search Queries ->
    Researcher Agent -> Evaluate Sources -> Summarizer Agent -> Report."""
    model_client = get_model_client()

    understanding = await understand_task(task, model_client)
    queries = await generate_search_queries(understanding, model_client, n=n_queries)
    sources = await run_researcher(queries)
    kept = await evaluate_sources(understanding, sources, model_client, keep_top=keep_top)
    report = await summarize_report(task, understanding, kept, model_client)

    return ResearchResult(
        task=task, understanding=understanding, queries=queries,
        sources=kept, report=report,
    )


async def research_workflow(task: str, n_queries: int = 4, keep_top: int = 8) -> str:
    """Run the full research pipeline on a topic or question and return the
    final written report (markdown, with a Sources section).

    Pipeline: understand the task -> generate diverse search queries ->
    web-search each one and read the pages -> evaluate/filter sources by
    relevance -> synthesize a cited report. Use this for a broad research
    request instead of calling web_search yourself when the task needs
    multiple angles covered and a single coherent write-up, rather than a
    quick one-off lookup.

    Args:
        task: The research question or topic to investigate.
        n_queries: How many distinct search queries to generate.
        keep_top: Max number of sources to keep after relevance evaluation.
    """
    result = await run_research_workflow(task, n_queries=n_queries, keep_top=keep_top)
    return result.report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the research workflow end-to-end.")
    parser.add_argument("task", help="What to research")
    parser.add_argument("-n", "--n-queries", type=int, default=4, help="Number of search queries to generate")
    parser.add_argument("--keep-top", type=int, default=8, help="Max sources to keep after evaluation")
    args = parser.parse_args()

    result = asyncio.run(run_research_workflow(args.task, n_queries=args.n_queries, keep_top=args.keep_top))

    print(f"\n=== Understanding ===\n{result.understanding}")
    print("\n=== Search Queries ===\n" + "\n".join(f"- {q}" for q in result.queries))
    print(f"\n=== Sources kept ({len(result.sources)}) ===")
    for i, s in enumerate(result.sources):
        print(f"[{i + 1}] {s.title} ({s.link}) -- relevance {s.relevance}: {s.reason}")
    print(f"\n=== Final Research Report ===\n{result.report}")


if __name__ == "__main__":
    main()
