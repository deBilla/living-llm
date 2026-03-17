"""
Memory Confidence Classifier — decides if memories are clear enough to use.

Classifies retrieved memories into three states:
  CLEAR  — high confidence, detailed, recent. Use directly.
  BLURRY — partial match, old, compressed, or low confidence. Augment with web search.
  ABSENT — nothing found. Search from scratch or say "I don't know."

The heuristic pass avoids an LLM call when the answer is obvious. The LLM
evaluation only fires for borderline cases where the heuristic can't decide.
"""

from enum import Enum
from dataclasses import dataclass, field

from memory.store import Memory, MemoryTier
from llm_backend import LLMBackend


class MemoryClarity(Enum):
    CLEAR = "clear"
    BLURRY = "blurry"
    ABSENT = "absent"


@dataclass
class MemoryAssessment:
    clarity: MemoryClarity
    memories: list[Memory]
    confidence: float               # Aggregate confidence 0.0-1.0
    missing_details: list[str]      # What's unclear or incomplete
    suggested_search: str | None    # Auto-generated search query if blurry


class MemoryConfidenceClassifier:
    """
    Analyzes retrieved memories and determines if they're clear enough
    to use directly, or if web augmentation is needed.
    """

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def assess(self, query: str, memories: list[Memory]) -> MemoryAssessment:
        """
        Given a user query and retrieved memories, determine clarity.

        Fast heuristic pass first; LLM evaluation only for borderline cases.
        """
        if not memories:
            return MemoryAssessment(
                clarity=MemoryClarity.ABSENT,
                memories=[],
                confidence=0.0,
                missing_details=["No relevant memories found"],
                suggested_search=self._generate_search_query(query, []),
            )

        # Aggregate scores
        avg_score = sum(m.relevance_score for m in memories) / len(memories)
        max_score = max(m.relevance_score for m in memories)
        avg_age = sum(m.session_count for m in memories) / len(memories)
        compressed_ratio = sum(
            1 for m in memories if m.tier in (MemoryTier.MID, MemoryTier.LONG)
        ) / len(memories)

        # Fast path: clearly sufficient.
        # Note: compressed memories (MID/LONG) have boosted scores from the
        # retriever (LONG >= 1.04, MID >= 0.66). High scores on compressed
        # memories mean they survived lossy compression — they're the most
        # valuable, not "blurry". Only check age for SHORT memories.
        if max_score > 0.6:
            return MemoryAssessment(
                clarity=MemoryClarity.CLEAR,
                memories=memories,
                confidence=avg_score,
                missing_details=[],
                suggested_search=None,
            )

        # Fast path: very weak match
        if max_score < 0.2 and len(memories) <= 1:
            return MemoryAssessment(
                clarity=MemoryClarity.ABSENT,
                memories=memories,
                confidence=avg_score,
                missing_details=["Weak/irrelevant matches only"],
                suggested_search=self._generate_search_query(query, memories),
            )

        # Borderline — ask the LLM
        return self._llm_assess(query, memories, avg_score)

    # ── Internal ──────────────────────────────────────────────

    def _llm_assess(self, query: str, memories: list[Memory], avg_score: float) -> MemoryAssessment:
        memory_text = "\n".join(f"- [{m.tier.value}] {m.content}" for m in memories)

        eval_prompt = (
            f'A user asked: "{query}"\n\n'
            f"Here are the memories I have related to this:\n{memory_text}\n\n"
            "Evaluate these memories:\n"
            "1. Can I answer the user's question fully and confidently from these memories alone? (YES/NO)\n"
            "2. If NO, what specific details are missing or unclear? (list them)\n"
            "3. What web search query would help fill the gaps? (one short query)\n\n"
            "Respond in this exact format:\n"
            "SUFFICIENT: YES or NO\n"
            'MISSING: [comma-separated list of missing details, or "none"]\n'
            'SEARCH: [suggested search query, or "none"]'
        )

        response = self.llm.chat(
            [{"role": "user", "content": eval_prompt}],
            max_tokens=150,
            temperature=0.1,
        )

        sufficient = "SUFFICIENT: YES" in response.upper()

        missing = []
        if "MISSING:" in response:
            missing_line = response.split("MISSING:")[1].split("\n")[0].strip()
            if missing_line.lower() not in ("none", "[none]", ""):
                missing = [m.strip().strip("[]") for m in missing_line.split(",") if m.strip()]

        search_query = None
        if "SEARCH:" in response:
            search_line = response.split("SEARCH:")[1].split("\n")[0].strip()
            if search_line.lower() not in ("none", "[none]", ""):
                search_query = search_line.strip('"').strip("[]")

        if sufficient:
            return MemoryAssessment(
                clarity=MemoryClarity.CLEAR,
                memories=memories,
                confidence=avg_score,
                missing_details=[],
                suggested_search=None,
            )

        return MemoryAssessment(
            clarity=MemoryClarity.BLURRY,
            memories=memories,
            confidence=avg_score,
            missing_details=missing,
            suggested_search=search_query or self._generate_search_query(query, memories),
        )

    def _generate_search_query(self, user_query: str, memories: list[Memory]) -> str:
        """Generate a search query from user question + any partial memories."""
        if not memories:
            return user_query

        memory_hints = " ".join(m.content[:50] for m in memories[:2])

        prompt = (
            f'The user asked: "{user_query}"\n'
            f"I have these partial/blurry memories: {memory_hints}\n\n"
            "Generate a short web search query (3-8 words) that would help me find the "
            "specific information I'm missing. Just output the query, nothing else."
        )

        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.2,
        ).strip().strip('"')
