"""
Memory Retriever — associative recall across all memory tiers.

This mimics how human memory works: you don't exhaustively search everything
you've ever experienced. Instead, the current context triggers associated 
memories by semantic similarity. Relevant things surface; irrelevant things don't.

Memory is injected into the prompt as context, giving the LLM the illusion
of persistent memory without any weight updates.
"""

from memory.store import MemoryStore, MemoryTier, Memory
import config


class MemoryRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def recall(self, query: str, top_k: int = None) -> list[Memory]:
        """
        Retrieve the most relevant memories for the current context.

        Priority tiers (highest to lowest):
          LONG  — always included (survived compression → highest value)
          MID   — always included (survived compression)
          WEB   — semantic search, top 2, confidence-weighted score
          SHORT — semantic search, fills remaining slots

        Why explicit fetching for LONG/MID instead of pure semantic search?
        "User said: who am i?" echoes score ~0.59 similarity to "who am i"
        while the actual long-term facts about the user score ~0.14 — the echoes
        completely bury the useful knowledge. See the fix commit for details.
        """
        top_k = top_k or config.TOP_K_MEMORIES

        # LONG + MID: always include, assign floors above any SHORT/WEB noise
        long_memories = self.store.get_memories_by_tier(MemoryTier.LONG)
        mid_memories = self.store.get_memories_by_tier(MemoryTier.MID)

        permanent = long_memories + mid_memories
        permanent_ids = {m.id for m in permanent}

        for mem in long_memories:
            mem.relevance_score = max(mem.relevance_score, 0.8) * 1.3
        for mem in mid_memories:
            mem.relevance_score = max(mem.relevance_score, 0.6) * 1.1

        # WEB: semantic search, up to 2 slots, weighted by stored confidence
        web_candidates = self.store.search_memories(query, top_k=6, tier=MemoryTier.WEB)
        web_candidates = [m for m in web_candidates if m.id not in permanent_ids]
        for mem in web_candidates:
            confidence = mem.metadata.get("confidence", config.WEB_CONFIDENCE_INITIAL)
            mem.relevance_score = max(mem.relevance_score, 0.1) * confidence
        web_candidates = sorted(web_candidates, key=lambda m: m.relevance_score, reverse=True)

        web_slots = min(2, len(web_candidates))

        # SHORT: fill remaining slots
        used_ids = permanent_ids | {m.id for m in web_candidates[:web_slots]}
        short_quota = max(top_k - len(permanent) - web_slots, 1)
        short_candidates = self.store.search_memories(
            query, top_k=short_quota * 3, tier=MemoryTier.SHORT
        )
        short_candidates = [m for m in short_candidates if m.id not in used_ids]
        for mem in short_candidates:
            if mem.access_count > 5:
                mem.relevance_score *= 1.1

        all_memories = (
            permanent
            + web_candidates[:web_slots]
            + short_candidates[:short_quota]
        )
        all_memories.sort(key=lambda m: m.relevance_score, reverse=True)
        return all_memories[:top_k]

    def format_context(self, memories: list[Memory]) -> str:
        """
        Format retrieved memories into a context block for prompt injection.
        
        Structures by tier so the LLM understands the fidelity level:
        - Long-term facts are presented as established knowledge
        - Mid-term gists as recent impressions  
        - Short-term as recent conversation fragments
        """
        if not memories:
            return ""

        sections = []

        long_term = [m for m in memories if m.tier == MemoryTier.LONG]
        mid_term = [m for m in memories if m.tier == MemoryTier.MID]
        short_term = [m for m in memories if m.tier == MemoryTier.SHORT]

        web_knowledge = [m for m in memories if m.tier == MemoryTier.WEB]

        if long_term:
            facts = "\n".join(f"  - {m.content}" for m in long_term)
            sections.append(f"[KNOWN FACTS about this user — use these confidently when asked]\n{facts}")

        if mid_term:
            gists = "\n".join(f"  - {m.content}" for m in mid_term)
            sections.append(f"[STORED FACTS from previous conversations — use these directly, do not say you don't know]\n{gists}")

        if web_knowledge:
            # Show confidence and source so the model treats web facts as provisional
            web_lines = []
            for m in web_knowledge:
                conf = m.metadata.get("confidence", config.WEB_CONFIDENCE_INITIAL)
                urls = m.metadata.get("source_urls", [])
                src = urls[0] if urls else "web search"
                web_lines.append(f"  [{conf:.0%} confidence | source: {src}] {m.content}")
            sections.append(f"[Knowledge from web searches]\n" + "\n".join(web_lines))

        if short_term:
            recent = "\n".join(f"  {m.content}" for m in short_term)
            sections.append(f"[Recent exchanges]\n{recent}")

        return "\n\n".join(sections)

    def build_memory_prompt(self, query: str) -> str:
        """
        Full pipeline: query → recall → format → ready for injection.
        Returns empty string if no relevant memories found.
        """
        memories = self.recall(query)
        if not memories:
            return ""

        context = self.format_context(memories)
        return f"""<memory_context>
IMPORTANT: The following is real stored information from previous conversations with this person.
When the user asks about anything covered here, USE THIS INFORMATION directly — do NOT say you don't know or don't have that information.

{context}
</memory_context>"""
