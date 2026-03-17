"""
Web Search — unified interface over multiple search backends.

Two backends are supported:
  duckduckgo  — Zero configuration, uses duckduckgo-search library.
                Reasonable for personal use; no API key needed.
  searxng     — Self-hosted meta-search via Docker. More private.
                docker run -d -p 8080:8080 searxng/searxng

Both return the same format: list of {"title", "url", "snippet"} dicts.

Rate limiting is the caller's responsibility (ReactLoop handles it).
This module is intentionally stateless — just search and return.
"""

from typing import Optional
import config


def search(query: str, max_results: int = None) -> list[dict]:
    """
    Search the web using the configured backend.

    Returns list of {"title": str, "url": str, "snippet": str} dicts.
    Returns empty list on any network or import error — callers should
    handle the empty case gracefully (offline fallback).
    """
    max_results = max_results or config.SEARCH_MAX_RESULTS
    backend = config.SEARCH_BACKEND.lower()

    if backend == "duckduckgo":
        return _duckduckgo(query, max_results)
    elif backend == "searxng":
        return _searxng(query, max_results)
    else:
        raise ValueError(f"Unknown SEARCH_BACKEND: {backend!r}. Use 'duckduckgo' or 'searxng'.")


def _duckduckgo(query: str, max_results: int) -> list[dict]:
    # Package was renamed from duckduckgo-search to ddgs
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError(
                "ddgs not installed.\n"
                "Run: pip install ddgs"
            )

    try:
        results = list(DDGS().text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except Exception:
        # Network errors, rate limits, etc. — fail silently so offline fallback works
        return []


def _searxng(query: str, max_results: int) -> list[dict]:
    try:
        import requests
        resp = requests.get(
            f"{config.SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": "general"},
            timeout=config.WEB_READER_TIMEOUT_SECS,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:max_results]
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in results
        ]
    except Exception:
        return []


def format_results_for_prompt(results: list[dict], query: str) -> str:
    """Format search results as a readable block for the model."""
    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results):
        lines.append(f"[{i + 1}] {r['title']}")
        lines.append(f"    {r['url']}")
        if r["snippet"]:
            lines.append(f"    {r['snippet']}")
        lines.append("")

    return "\n".join(lines).strip()
