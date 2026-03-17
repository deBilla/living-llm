"""
Augmented Recall — the bridge between blurry memory and web search.

Flow:
1. User asks a question
2. Retrieve memories
3. Classify: CLEAR / BLURRY / ABSENT
4. If CLEAR  -> respond from memory (no search needed)
5. If BLURRY -> use memory as seed, search web to fill gaps, merge results
6. If ABSENT -> decide if searchable, search or say "I don't know"
7. After responding, update memory with enriched information

This is the "tip of the tongue" resolver — when you vaguely remember
something, you look it up to make the memory sharp again.
"""

import time

from memory.store import MemoryStore, MemoryTier
from memory.confidence import MemoryConfidenceClassifier, MemoryClarity, MemoryAssessment
from memory.retriever import MemoryRetriever
from tools.web_search import search
from tools.web_reader import read_page
from llm_backend import LLMBackend
import config


class AugmentedRecall:
    def __init__(self, store: MemoryStore, llm: LLMBackend, retriever: MemoryRetriever):
        self.store = store
        self.llm = llm
        self.retriever = retriever
        self.classifier = MemoryConfidenceClassifier(llm)
        self.last_assessment: MemoryAssessment | None = None
        self.last_search_results: list[dict] | None = None
        self.last_urls_read: list[str] = []

    def recall_and_augment(self, query: str) -> dict:
        """
        Full augmented recall pipeline.

        Returns:
            {
                "clarity":          "clear" | "blurry" | "absent",
                "memory_context":   str,                  # formatted for prompt injection
                "web_context":      str | None,            # web results if used
                "search_performed": bool,
                "memory_updated":   bool,                 # whether we sharpened a blurry memory
                "assessment":       MemoryAssessment,
                "search_log":       list[dict],           # for web knowledge extraction
            }
        """
        self.last_urls_read = []
        self.last_search_results = None

        # Step 1: Retrieve memories
        memories = self.retriever.recall(query)

        # Step 2: Classify clarity
        assessment = self.classifier.assess(query, memories)
        self.last_assessment = assessment

        result = {
            "clarity": assessment.clarity.value,
            "memory_context": self.retriever.format_context(assessment.memories),
            "web_context": None,
            "search_performed": False,
            "memory_updated": False,
            "assessment": assessment,
            "search_log": [],
        }

        # Step 3: Route based on clarity
        if assessment.clarity == MemoryClarity.CLEAR:
            return result

        if assessment.clarity == MemoryClarity.BLURRY:
            web_result = self._augment_blurry_memory(query, assessment)
            if web_result:
                result["web_context"] = web_result["context"]
                result["search_performed"] = True
                result["memory_updated"] = web_result["memory_updated"]
                result["search_log"] = web_result.get("search_log", [])
            return result

        if assessment.clarity == MemoryClarity.ABSENT:
            if self._is_searchable(query):
                web_result = self._search_fresh(query)
                if web_result:
                    result["web_context"] = web_result["context"]
                    result["search_performed"] = True
                    result["search_log"] = web_result.get("search_log", [])
            return result

        return result

    # ── Blurry memory augmentation ────────────────────────────

    def _augment_blurry_memory(self, query: str, assessment: MemoryAssessment) -> dict | None:
        """
        Use blurry memory as a search seed. Instead of searching the user's
        raw question, search using memory fragments + what's missing.
        """
        search_query = assessment.suggested_search
        if not search_query:
            return None

        print(f"  [Augmented Recall] Blurry memory detected. Searching: '{search_query}'")
        results = search(search_query)
        self.last_search_results = results

        if not results:
            return None

        search_log = [{"type": "search", "query": search_query, "results": results}]

        # Read top 2 results for detail
        context_parts = []
        for r in results[:2]:
            page = read_page(r["url"], max_chars=2000)
            if page.get("content"):
                context_parts.append(f"Source: {r['url']}\n{page['content']}")
                self.last_urls_read.append(r["url"])
                search_log.append({"type": "read", "url": r["url"], "content": page["content"][:500]})

        if not context_parts:
            context_parts = [
                f"[{r['title']}] ({r['url']}): {r['snippet']}"
                for r in results[:3]
            ]

        web_context = "\n\n".join(context_parts)

        # Merge blurry memory + web results
        memory_text = "\n".join(f"- {m.content}" for m in assessment.memories)

        merge_prompt = (
            f"I had these blurry memories about a topic:\n{memory_text}\n\n"
            f"I searched the web and found this additional information:\n{web_context}\n\n"
            f'Based on both sources, extract the key facts that answer this question: "{query}"\n\n'
            "Rules:\n"
            "- If the web results CONTRADICT the memory, trust the web results (they're more current)\n"
            "- If the web results CONFIRM the memory, note the confirmation\n"
            "- If the web results ADD new details, include them\n"
            "- Output each fact on its own line starting with a tag\n\n"
            "Format:\n"
            "[memory] fact from memory that was confirmed\n"
            "[web] new fact from web search\n"
            "[corrected] fact where web corrected a blurry memory"
        )

        merged_facts = self.llm.chat(
            [{"role": "user", "content": merge_prompt}],
            max_tokens=300,
            temperature=0.2,
        )

        # Store enriched knowledge back into memory (sharpening the blur)
        memory_updated = False
        if merged_facts and ("[web]" in merged_facts.lower() or "[corrected]" in merged_facts.lower()):
            self.store.store_memory(
                content=merged_facts,
                tier=MemoryTier.WEB,
                metadata={
                    "type": "augmented_recall",
                    "original_query": query,
                    "source_urls": self.last_urls_read,
                    "blurry_memory_ids": [m.id for m in assessment.memories],
                    "retrieved_at": time.time(),
                    "confidence": 0.8,
                },
            )
            memory_updated = True

        return {
            "context": f"[Augmented recall — blurry memory + web search]\n{merged_facts}",
            "memory_updated": memory_updated,
            "search_log": search_log,
        }

    # ── Fresh search (no memory) ──────────────────────────────

    def _search_fresh(self, query: str) -> dict | None:
        """Search from scratch when no memory exists."""
        print(f"  [Augmented Recall] No memory found. Searching: '{query}'")
        results = search(query)
        self.last_search_results = results

        if not results:
            return None

        search_log = [{"type": "search", "query": query, "results": results}]

        snippets = "\n".join(
            f"[{r['title']}] ({r['url']}): {r['snippet']}"
            for r in results[:3]
        )

        return {
            "context": f"[Web search results]\n{snippets}",
            "memory_updated": False,
            "search_log": search_log,
        }

    # ── Searchability heuristic ───────────────────────────────

    def _is_searchable(self, query: str) -> bool:
        """
        Determine if a query is worth searching the web for.

        Fast keyword check first; LLM fallback for ambiguous queries.
        """
        query_lower = query.lower().strip()

        not_searchable = [
            "how are you", "how do you feel", "what do you think",
            "write me", "create a", "help me with", "can you",
            "tell me a joke", "hello", "hi ", "thanks", "thank you",
            "good morning", "good night", "bye",
        ]
        for signal in not_searchable:
            if signal in query_lower:
                return False

        searchable = [
            "what is the current", "latest", "today", "price of",
            "who is", "when did", "how many", "what happened",
            "news about", "update on", "status of", "weather",
            "where is", "how to", "what are the", "recent",
        ]
        for signal in searchable:
            if signal in query_lower:
                return True

        # Ambiguous — let the LLM decide (cheap call, 5 tokens max)
        decision = self.llm.chat(
            [{"role": "user", "content": (
                "Would searching the internet help answer this question? "
                f'Just reply YES or NO.\n\nQuestion: "{query}"'
            )}],
            max_tokens=5,
            temperature=0.0,
        ).strip().upper()

        return "YES" in decision

    # ── User feedback loop ────────────────────────────────────

    def check_user_feedback(self, user_response: str):
        """
        If the last response used augmented recall, check if the user
        confirmed or denied the information, and adjust confidence.
        """
        if not self.last_assessment or self.last_assessment.clarity == MemoryClarity.CLEAR:
            return

        response_lower = user_response.lower()

        confirmation = any(s in response_lower for s in [
            "yes", "exactly", "correct", "right", "that's it", "yeah", "yep",
        ])
        denial = any(s in response_lower for s in [
            "no", "wrong", "incorrect", "not right", "that's not", "nah",
        ])

        if not confirmation and not denial:
            return

        for mem in self.last_assessment.memories:
            meta = dict(mem.metadata)
            conf = meta.get("confidence", config.WEB_CONFIDENCE_INITIAL)
            if confirmation:
                meta["confidence"] = min(conf + 0.15, 1.0)
            elif denial:
                meta["confidence"] = max(conf - 0.25, 0.05)
            self.store.update_memory_metadata(mem.id, meta)

        tag = "confirmed" if confirmation else "denied"
        print(f"  [Feedback] User {tag} — memory confidence adjusted")
