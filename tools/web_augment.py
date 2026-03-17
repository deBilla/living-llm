"""
Web search augmentation — works alongside limbiq.

When limbiq returns low-confidence or empty results,
trigger a web search and store findings through limbiq's API.
"""

import time

from limbiq import Limbiq
from llm_backend import LLMBackend
import config


# Signals that a query might benefit from web search
_SEARCHABLE_SIGNALS = [
    "what is the current", "latest", "today", "price of",
    "who is", "when did", "how many", "what happened",
    "news about", "update on", "status of", "weather",
    "where is", "how to", "what are the", "recent",
]

# Signals that a query is NOT worth searching
_NOT_SEARCHABLE = [
    "how are you", "how do you feel", "what do you think",
    "write me", "create a", "help me with", "can you",
    "tell me a joke", "hello", "hi ", "thanks", "thank you",
    "good morning", "good night", "bye",
]


class WebAugmenter:
    """Bridges limbiq's memory system and web search."""

    def __init__(self, lq: Limbiq, llm: LLMBackend):
        self.lq = lq
        self.llm = llm

    def maybe_augment(self, query: str, limbiq_result) -> dict | None:
        """
        Check if limbiq's result needs web augmentation.
        Returns a dict with context and search_log, or None.
        """
        # If limbiq found priority memories, no need to search
        if limbiq_result.priority_count > 0:
            return None

        # If limbiq found enough relevant memories, skip
        if limbiq_result.memories_retrieved >= 3:
            return None

        # Check if this is a searchable query
        if not self._is_searchable(query):
            return None

        # Search the web
        try:
            from tools.web_search import search, format_results_for_prompt
            results = search(query, max_results=3)
        except Exception as e:
            print(f"  [Web augment] Search failed: {e}")
            return None

        if not results:
            return None

        search_log = [{"type": "search", "query": query, "results": results}]
        snippets = format_results_for_prompt(results, query)

        return {
            "context": snippets,
            "search_log": search_log,
        }

    def store_web_facts(self, query: str, search_log: list[dict], response: str):
        """
        After a web search, extract durable facts and store through limbiq.
        Uses dopamine to tag web-sourced facts as priority with source metadata.
        """
        # Collect source URLs
        source_urls = []
        for entry in search_log:
            if entry["type"] == "search":
                source_urls.extend(r.get("url", "") for r in entry.get("results", []))
            elif entry["type"] == "read" and entry.get("success"):
                source_urls.append(entry["url"])

        # Extract facts via LLM
        facts = self._extract_facts(query, response)
        if not facts:
            return

        for fact in facts:
            tagged = f"[Web] {fact}"
            self.lq.dopamine(tagged)
            print(f"  [Web fact stored] {fact[:60]}...")

    def _extract_facts(self, query: str, response: str) -> list[str]:
        """Use the LLM to extract durable facts from a search interaction."""
        prompt = (
            "You just answered a question using web search results.\n\n"
            f"User's question: {query}\n"
            f"Your response: {response[:800]}\n\n"
            "Extract 1-3 key facts from this interaction that are worth storing.\n"
            "Each fact must be self-contained and factual.\n"
            "If nothing useful, respond NONE.\n"
            "One fact per line, no bullets."
        )

        raw = self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )

        if "NONE" in raw.upper()[:20]:
            return []

        return [f.strip() for f in raw.strip().split("\n") if len(f.strip()) > 10]

    def _is_searchable(self, query: str) -> bool:
        """Determine if a query is worth searching the web for."""
        query_lower = query.lower().strip()

        for signal in _NOT_SEARCHABLE:
            if signal in query_lower:
                return False

        for signal in _SEARCHABLE_SIGNALS:
            if signal in query_lower:
                return True

        # Ambiguous — let the LLM decide (cheap call)
        decision = self.llm.chat(
            [{"role": "user", "content": (
                "Would searching the internet help answer this question? "
                f'Just reply YES or NO.\n\nQuestion: "{query}"'
            )}],
            max_tokens=5,
            temperature=0.0,
        ).strip().upper()

        return "YES" in decision
