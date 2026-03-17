"""
Wikipedia tool — structured knowledge lookup.

Uses the Wikipedia REST API (no key needed). More reliable than web search
for established factual queries.
"""

import requests

_API_URL = "https://en.wikipedia.org/api/rest_v1"
_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 8
_MAX_CHARS = 4000


def search_wikipedia(query: str, max_results: int = 3) -> dict:
    """
    Search Wikipedia and return article summaries.

    Returns:
        {
            "query": str,
            "results": [{"title": str, "summary": str, "url": str}],
            "error": str | None,
        }
    """
    try:
        resp = requests.get(
            _SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"query": query, "results": [], "error": f"Search failed: {e}"}

    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return {"query": query, "results": [], "error": None}

    results = []
    for item in search_results:
        title = item["title"]
        summary = _get_summary(title)
        if summary:
            results.append({
                "title": title,
                "summary": summary,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            })

    return {"query": query, "results": results, "error": None}


def _get_summary(title: str) -> str | None:
    """Fetch the summary extract for a Wikipedia article."""
    try:
        resp = requests.get(
            f"{_API_URL}/page/summary/{title.replace(' ', '_')}",
            timeout=_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = data.get("extract", "")
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + "..."
        return text
    except Exception:
        return None
