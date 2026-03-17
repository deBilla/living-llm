"""
Web Knowledge — extracts durable facts from search interactions and stores them.

After the model performs a web search and responds, this module uses the LLM to
distill 1-3 key facts worth keeping. Facts are stored in the WEB memory tier with:
  - Source URL attribution
  - Initial confidence (0.7 — lower than user-stated facts)
  - Retrieval timestamp (for time-based decay)

Over time, web knowledge decays: news-type facts expire after 7 days, general
facts after 30 days. Facts that survive long enough at very low confidence are
deleted — the model accepts it doesn't know the current state anymore.

Why extract facts separately rather than storing the full response?
The full response contains conversational filler. The extracted facts are
dense, self-contained sentences optimised for future memory injection. They're
what the compressor would produce anyway — we just do it at insert time for
web knowledge instead of waiting for the nightly compression cycle.
"""

import time
from typing import Optional

import config
from memory.store import MemoryStore, MemoryTier, Memory
from llm_backend import LLMBackend

# Signals that suggest a fact is time-sensitive / news-type
_NEWS_SIGNALS = frozenset({
    "today", "yesterday", "this week", "this month", "just announced",
    "breaking", "latest", "recently", "as of", "currently", "right now",
    "price", "stock", "market", "score", "weather", "forecast",
})


class WebKnowledgeExtractor:
    """
    Bridges the web search pipeline and the memory store.

    Lifecycle:
      1. extract_and_store()  — called after each search turn
      2. decay_old_knowledge() — called periodically (consolidation cycle)
    """

    def __init__(self, store: MemoryStore, llm: LLMBackend):
        self.store = store
        self.llm = llm

    def extract_and_store(
        self,
        query: str,
        search_log: list[dict],
        model_response: str,
    ) -> list[Memory]:
        """
        After a web search turn, extract durable facts and persist them.

        Uses a short LLM call to distill the model's response into 1-3
        self-contained facts. If the interaction didn't produce anything
        worth keeping (e.g. the search failed), returns an empty list.
        """
        # Collect which URLs were actually consulted this turn
        source_urls = []
        for entry in search_log:
            if entry["type"] == "search":
                source_urls.extend(r.get("url", "") for r in entry.get("results", []))
            elif entry["type"] == "read" and entry.get("success"):
                source_urls.append(entry["url"])

        extraction_prompt = (
            "You just answered a question using web search results.\n\n"
            f"User's question: {query}\n"
            f"Your response: {model_response[:800]}\n\n"
            "Extract 1-3 key facts from this interaction that are worth storing for future conversations.\n"
            "Each fact must be:\n"
            "  - Self-contained (understandable without original context)\n"
            "  - Factual, not conversational filler\n"
            "  - Include a date or recency marker if the information is time-sensitive\n"
            "  - One sentence\n\n"
            "If nothing durable is worth storing, respond with exactly: NONE\n\n"
            "Output one fact per line, no bullets, no numbering."
        )

        raw = self.llm.chat(
            [{"role": "user", "content": extraction_prompt}],
            max_tokens=200,
            temperature=0.2,
        )

        if "NONE" in raw.upper()[:20]:
            return []

        stored: list[Memory] = []
        for line in raw.strip().split("\n"):
            fact = line.strip().lstrip("-•*0123456789. ")
            if len(fact) < 15:
                continue

            memory = self.store.store_memory(
                content=fact,
                tier=MemoryTier.WEB,
                metadata={
                    "type": "web_knowledge",
                    "source_query": query,
                    "source_urls": source_urls[:3],  # Top 3 sources
                    "retrieved_at": time.time(),
                    "confidence": config.WEB_CONFIDENCE_INITIAL,
                    "is_news": self._is_news(fact),
                },
            )
            stored.append(memory)

        return stored

    def decay_old_knowledge(self) -> dict:
        """
        Apply time-based confidence decay to web knowledge.

        News-type facts (TTL: 7 days) decay faster than general facts (30 days).
        Facts that fall below confidence 0.1 are deleted — they're stale enough
        to be more misleading than helpful.

        Returns a summary dict for display.
        """
        web_memories = self.store.get_memories_by_tier(MemoryTier.WEB)
        now = time.time()

        decayed = 0
        deleted = 0

        for mem in web_memories:
            meta = dict(mem.metadata)  # Copy
            age_days = (now - meta.get("retrieved_at", mem.created_at)) / 86400
            ttl = config.WEB_NEWS_TTL_DAYS if meta.get("is_news") else config.WEB_KNOWLEDGE_TTL_DAYS

            if age_days <= ttl:
                continue  # Still fresh

            # Days past TTL determine how much confidence to remove
            past_ttl = age_days - ttl
            current_conf = meta.get("confidence", config.WEB_CONFIDENCE_INITIAL)
            new_conf = max(0.0, current_conf - past_ttl * config.WEB_CONFIDENCE_DECAY_RATE)

            if new_conf <= 0.1:
                self.store.delete_memory(mem.id)
                deleted += 1
            else:
                meta["confidence"] = round(new_conf, 3)
                self.store.update_memory_metadata(mem.id, meta)
                decayed += 1

        return {"decayed": decayed, "deleted": deleted}

    @staticmethod
    def _is_news(content: str) -> bool:
        """
        Heuristic: is this a time-sensitive / news-type fact?
        News facts get a shorter TTL (7 days vs 30 days).
        """
        lower = content.lower()
        return any(signal in lower for signal in _NEWS_SIGNALS)
