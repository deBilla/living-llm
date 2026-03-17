"""
Memory Compressor — the lossy compression pipeline.

This is the heart of the "forgetting is a feature" insight.
Full conversations compress into gists. Gists compress into facts.
Details fade, meaning persists.

            ┌──────────────┐
            │ Full convo    │  HIGH FIDELITY
            │ (every word)  │
            └──────┬───────┘
                   │  LLM extracts gist
                   ▼
            ┌──────────────┐
            │ Gist          │  MEDIUM FIDELITY
            │ (2-4 sentences)│
            └──────┬───────┘
                   │  LLM extracts abstract facts
                   ▼
            ┌──────────────┐
            │ Facts         │  LOW FIDELITY, HIGH DURABILITY
            │ (bullet points)│  "User is a strong developer
            │               │   interested in AI/ML"
            └──────────────┘
"""

from memory.store import MemoryStore, MemoryTier
from llm_backend import LLMBackend


class MemoryCompressor:
    def __init__(self, store: MemoryStore, llm: LLMBackend):
        self.store = store
        self.llm = llm

    def compress_conversations(self) -> int:
        """
        Compress unprocessed conversations into mid-term gists.
        This is like the brain processing the day's experiences.
        
        Returns the number of conversations compressed.
        """
        uncompressed = self.store.get_uncompressed_conversations()
        count = 0

        for conv_id, messages in uncompressed:
            # Format conversation as readable text
            conv_text = self._format_conversation(messages)

            if len(conv_text.strip()) < 50:
                # Too short to be worth compressing
                self.store.mark_conversation_compressed(conv_id)
                continue

            # Extract atomic facts — each stored separately so embedding search
            # can match individual facts rather than compound gist sentences.
            facts = self.llm.extract_atomic_facts(conv_text)

            if facts:
                for fact in facts:
                    self.store.store_memory(
                        content=fact,
                        tier=MemoryTier.MID,
                        metadata={"source_conversation": conv_id, "type": "fact"},
                    )
                count += 1

            self.store.mark_conversation_compressed(conv_id)

        return count

    def consolidate_to_long_term(self) -> bool:
        """
        Compress accumulated mid-term gists into long-term abstract facts.
        This is the "sleep consolidation" — reorganizing understanding.
        
        Only runs when there are enough gists to synthesize from.
        Returns True if consolidation happened.
        """
        mid_memories = self.store.get_memories_by_tier(MemoryTier.MID)

        # Need at least 3 gists to synthesize meaningful long-term facts
        if len(mid_memories) < 3:
            return False

        gists = [m.content for m in mid_memories]

        # Use the LLM to extract durable facts from accumulated gists
        facts = self.llm.extract_facts(gists)

        if facts:
            self.store.store_memory(
                content=facts,
                tier=MemoryTier.LONG,
                metadata={
                    "type": "consolidated_facts",
                    "source_gist_count": len(gists),
                },
            )
            return True

        return False

    def run_full_cycle(self) -> dict:
        """
        Run the complete compression pipeline:
        1. Compress new conversations → gists
        2. Expire old short-term memories
        3. Consolidate gists → long-term facts (if enough accumulated)
        4. Expire old mid-term memories
        
        Returns stats about what happened.
        """
        results = {
            "conversations_compressed": 0,
            "memories_expired": 0,
            "long_term_consolidated": False,
        }

        # Step 1: Compress conversations to gists
        results["conversations_compressed"] = self.compress_conversations()

        # Step 2: Expire old short-term memories
        results["memories_expired"] = self.store.expire_old_memories()

        # Step 3: Consolidate to long-term (only if enough gists)
        results["long_term_consolidated"] = self.consolidate_to_long_term()

        return results

    @staticmethod
    def _format_conversation(messages: list[dict]) -> str:
        """Format messages into readable text for the compressor."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n\n".join(lines)
