"""
Conversation Engine — the orchestrator.

Ties together the LLM, memory system, LoRA adapter, and web search.

For each user message:
1. Search memory for relevant context (LONG, MID, WEB, SHORT tiers)
2. Build system prompt with memory + optional tool-use instructions
3. Run the ReAct loop (intercepts tool calls, executes web searches)
4. Store the exchange in short-term memory
5. If web search happened, extract and store learned facts
6. On conversation end, trigger compression and optionally LoRA training

LoRA integration:
When a trained adapter is available, inference switches to MLXBackend.
The user can toggle this with /adapter on|off.

Web search integration:
The ReAct loop lets the model call web_search() and read_page() tools.
Tool calls are intercepted, executed, and results fed back until the
model produces a final answer. Learned facts are stored in the WEB tier.
"""

import re

import config
from llm_backend import LLMBackend, MLXBackend
from memory.store import MemoryStore, MemoryTier
from memory.compressor import MemoryCompressor
from memory.retriever import MemoryRetriever
from memory.training_data import prepare_training_data, count_new_conversations, mark_training_complete
from memory.web_knowledge import WebKnowledgeExtractor
from training.adapter_manager import AdapterManager
from training.lora_trainer import LoRATrainer
from training.eval import AdapterEvaluator
from tools.react_loop import ReactLoop
from tools.augmented_recall import AugmentedRecall
from memory.confidence import MemoryClarity

_TOOL_TAG_RE = re.compile(r"<tool_(?:call|result)>.*?</tool_(?:call|result)>", re.DOTALL)


class ConversationEngine:
    def __init__(self):
        self.llm = LLMBackend()           # llama-cpp, always available (base model)
        self.store = MemoryStore()
        self.compressor = MemoryCompressor(self.store, self.llm)
        self.retriever = MemoryRetriever(self.store)
        self.web_knowledge = WebKnowledgeExtractor(self.store, self.llm)
        self.messages: list[dict] = []
        self._turn_count = 0

        # Augmented recall — bridges blurry memory and web search
        self.augmented_recall = AugmentedRecall(self.store, self.llm, self.retriever)

        # ReAct loop — session-scoped rate limiting persists across turns
        self.react_loop = ReactLoop(
            max_iterations=3,
            max_calls_per_iteration=2,
        )

        # LoRA adapter state
        self.adapter_manager = AdapterManager()
        self.trainer = LoRATrainer()
        self.evaluator = AdapterEvaluator()
        self._mlx: MLXBackend | None = None
        self._adapter_enabled = True
        self._pending_adapter: str | None = None
        self._training_active = False

        if config.LORA_ENABLED:
            self._try_load_active_adapter(quiet=True)

    # ── Session ────────────────────────────────────────────────

    def start_session(self):
        """Begin a new conversation session and age existing memories."""
        self.store.new_session()
        self.messages = []
        self._turn_count = 0
        self.react_loop.reset_session()

        stats = self.store.get_stats()
        parts = [
            f"{stats['short']} short",
            f"{stats['mid']} mid",
            f"{stats['long']} long-term",
        ]
        if stats.get("web", 0):
            parts.append(f"{stats['web']} web")
        adapter_note = " | adapter: on" if (self._mlx and self._adapter_enabled) else ""
        print(f"  Memory: {' / '.join(parts)}{adapter_note}")

    def respond(self, user_input: str) -> str:
        """
        Process a user message and return the assistant's response.

        Pipeline:
        1. Augmented recall — classify memory as CLEAR / BLURRY / ABSENT
        2. If BLURRY, use memory as seed to search the web and merge results
        3. If ABSENT and searchable, search fresh
        4. Build prompt with memory + web context
        5. Run the ReAct loop for any additional tool calls
        6. Store exchange; extract web knowledge if searches happened
        7. Check user feedback on previous augmented recall
        """
        self._turn_count += 1

        if self._pending_adapter:
            self._activate_pending_adapter()

        # Check if previous turn's augmented recall got user feedback
        if config.WEB_SEARCH_ENABLED:
            self.augmented_recall.check_user_feedback(user_input)

        # Step 1: Augmented recall — memory retrieval + optional web search
        if config.WEB_SEARCH_ENABLED:
            recall_result = self.augmented_recall.recall_and_augment(user_input)
        else:
            recall_result = {
                "clarity": "clear",
                "memory_context": self.retriever.build_memory_prompt(user_input),
                "web_context": None,
                "search_performed": False,
                "memory_updated": False,
                "search_log": [],
            }

        # Step 2: Build system prompt
        system_content = config.SYSTEM_PROMPT
        if config.WEB_SEARCH_ENABLED:
            system_content += "\n\n" + config.TOOL_USE_PROMPT

        if recall_result["memory_context"]:
            system_content += "\n\n" + recall_result["memory_context"]
            print(f"  [Context] Memory context injected ({len(recall_result['memory_context'])} chars)")

        if recall_result["web_context"]:
            system_content += (
                "\n\n<web_search_results>\n"
                + recall_result["web_context"]
                + "\n</web_search_results>"
            )
            print(f"  [Context] Web context injected ({len(recall_result['web_context'])} chars)")

        # Add clarity hint so the model knows how to frame its response
        if recall_result["clarity"] == "blurry":
            system_content += (
                "\n\nNOTE: Some of the context above came from augmenting blurry memories "
                "with web search. When referencing this information, clearly distinguish "
                "what you remembered vs what you just looked up. Example: 'I remembered "
                "you mentioned something about X, and I looked it up to get the details...'"
            )
        elif recall_result["clarity"] == "absent" and recall_result["search_performed"]:
            system_content += (
                "\n\nNOTE: You had no memory of this topic, so the context above is from "
                "a fresh web search. Be transparent: 'I don't have any previous knowledge "
                "about this, but I searched and found...'"
            )

        # Step 3: Build messages
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.messages[-(6 * 2):])  # Last 6 turns

        # Inject memory summary directly into the user message so the 8B model
        # can't miss it. Small models pay far more attention to user turns than
        # system content buried early in the context window.
        user_msg = user_input
        has_context = recall_result["memory_context"] or recall_result["web_context"]
        if has_context:
            context_parts = []
            if recall_result["memory_context"]:
                context_parts.append(recall_result["memory_context"])
            if recall_result["web_context"]:
                context_parts.append(
                    "<web_search_results>\n"
                    + recall_result["web_context"]
                    + "\n</web_search_results>"
                )
            injected = "\n\n".join(context_parts)
            user_msg = (
                f"Here is what you remember about me from past conversations "
                f"(use this naturally, do NOT mention section names or tags):\n\n"
                f"{injected}\n\n"
                f"Now answer this: {user_input}"
            )

        messages.append({"role": "user", "content": user_msg})

        backend = self._active_backend()

        # Step 4: Run ReAct loop (may do additional searches beyond augmented recall)
        search_log = list(recall_result.get("search_log", []))

        if config.WEB_SEARCH_ENABLED:
            response, react_log = self.react_loop.run(backend, messages)
            search_log.extend(react_log)

            bad_urls = self.react_loop.verify_citations(response, search_log)
            if bad_urls:
                print(f"  [Warning: response mentions unverified URLs: {bad_urls}]")
        else:
            response = backend.chat(messages)

        # Clean up any stray tool tags that slipped through
        response = _TOOL_TAG_RE.sub("", response).strip()

        # Step 5: Store exchange
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})

        self.store.store_memory(
            content=f"User said: {user_input}",
            tier=MemoryTier.SHORT,
            metadata={"turn": self._turn_count, "type": "user_message"},
        )

        # Step 6: Extract web knowledge from any searches
        if search_log:
            self.web_knowledge.extract_and_store(user_input, search_log, response)

        # Step 7: Log recall outcome
        if recall_result["search_performed"]:
            updated = "updated" if recall_result["memory_updated"] else "unchanged"
            print(f"  [Recall] {recall_result['clarity']} -> searched web -> memory {updated}")

        return response

    def end_session(self):
        """End conversation: store it, compress memories, optionally train."""
        if not self.messages:
            return {}

        self.store.store_conversation(self.messages)
        results = {"turns": self._turn_count}

        if config.CONSOLIDATE_ON_EXIT:
            print("\n  Consolidating memories...")
            compression_results = self.compressor.run_full_cycle()
            results.update(compression_results)

            if compression_results["conversations_compressed"] > 0:
                print(f"  Compressed {compression_results['conversations_compressed']} conversation(s) into gists")
            if compression_results["long_term_consolidated"]:
                print("  Consolidated gists into long-term facts")
            if compression_results["memories_expired"] > 0:
                print(f"  Expired {compression_results['memories_expired']} old memories")

            # Web knowledge decay (calendar-based, not session-based)
            decay = self.web_knowledge.decay_old_knowledge()
            if decay["deleted"] > 0:
                print(f"  Expired {decay['deleted']} stale web knowledge entries")
            if decay["decayed"] > 0:
                print(f"  Decayed confidence on {decay['decayed']} web knowledge entries")

            # Auto-train if enough fresh conversations
            if config.LORA_ENABLED and not self._training_active:
                new_convos = count_new_conversations(self.store)
                if self.adapter_manager.should_auto_train(new_convos):
                    print(f"  {new_convos} new conversation(s) ready — triggering LoRA training in background...")
                    self._start_training_background()

        stats = self.store.get_stats()
        web_note = f" / {stats['web']} web" if stats.get("web", 0) else ""
        print(f"  Memory now: {stats['short']} short / {stats['mid']} mid / {stats['long']} long-term{web_note}")

        return results

    # ── Web search commands ────────────────────────────────────

    def forced_search(self, query: str) -> str:
        """
        Directly execute a web search and generate a response.

        Used by the /search command — bypasses the ReAct loop for reliability
        (the model is explicitly given the results rather than deciding to search).
        """
        try:
            from tools.web_search import search as do_search, format_results_for_prompt
            results = do_search(query)
        except Exception as e:
            return f"Search failed: {e}"

        if not results:
            return f"Searched for '{query}' but found no results. Network may be unavailable."

        results_text = format_results_for_prompt(results, query)
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"I searched the web for: {query}\n\n"
                    f"Here are the results:\n\n{results_text}\n\n"
                    "Please summarise what you found and answer the question. Cite your sources."
                ),
            },
        ]

        response = self._active_backend().chat(messages)
        search_log = [{"type": "search", "query": query, "results": results}]
        self.web_knowledge.extract_and_store(query, search_log, response)
        return response

    def get_web_knowledge(self) -> list:
        """Return all stored web knowledge entries."""
        return self.store.get_memories_by_tier(MemoryTier.WEB)

    def clear_web_knowledge(self):
        """Delete all WEB tier memories."""
        memories = self.store.get_memories_by_tier(MemoryTier.WEB)
        for m in memories:
            self.store.delete_memory(m.id)
        print(f"  Cleared {len(memories)} web knowledge entries.")

    def decay_web_knowledge(self):
        """Manually trigger confidence decay on web knowledge."""
        result = self.web_knowledge.decay_old_knowledge()
        print(f"  Decayed: {result['decayed']}  |  Deleted (expired): {result['deleted']}")

    # ── LoRA adapter commands ──────────────────────────────────

    def train_now(self) -> bool:
        """Manually trigger LoRA training in background."""
        if not config.LORA_ENABLED:
            print("  LoRA is disabled (set LORA_ENABLED = True in config.py)")
            return False
        if self._training_active:
            print("  Training already in progress — wait for it to complete")
            return False
        if not self.trainer.is_available():
            print("  mlx_lm not found. Install with: pip install mlx-lm")
            return False
        new_convos = count_new_conversations(self.store)
        if new_convos == 0:
            print("  No new conversations to train on.")
            print("  (Need compressed conversations — have a few chats and run /quit first)")
            return False
        if new_convos < config.LORA_MIN_CONVERSATIONS:
            print(f"  Need at least {config.LORA_MIN_CONVERSATIONS} conversations, have {new_convos}.")
            print("  Have more chats, /quit to compress, then try again.")
            return False
        print(f"  Starting training on {new_convos} conversation(s)...")
        return self._start_training_background()

    def adapter_on(self) -> bool:
        """Re-enable the LoRA adapter."""
        self._adapter_enabled = True
        if self._mlx is None:
            self._try_load_active_adapter(quiet=False)
        if self._mlx:
            print("  Adapter enabled.")
            return True
        print("  No adapter found. Run /train first.")
        return False

    def adapter_off(self):
        """Disable the adapter (fall back to base model)."""
        self._adapter_enabled = False
        print("  Adapter disabled — using base model.")

    def compare_responses(self, prompt: str) -> dict | None:
        """Run the same prompt through base and adapted model."""
        if self._mlx is None or not self._mlx.is_loaded():
            print("  No adapter loaded. Run /train first.")
            return None

        print("  Running base model...")
        base_resp = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.01)
        print("  Running adapted model...")
        adapted_resp = self._mlx.chat([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.01)

        comparison = self.evaluator.compare_responses(prompt, self.llm, self._mlx)
        self.evaluator.log_comparison(comparison, adapter_path=self.adapter_manager.get_active_adapter() or "")

        return {"prompt": prompt, "base": base_resp, "adapted": adapted_resp}

    def get_adapter_status(self) -> dict:
        status = self.adapter_manager.get_status()
        status["adapter_active"] = self._adapter_enabled and self._mlx is not None and self._mlx.is_loaded()
        status["training_running"] = self._training_active
        status["mlx_available"] = MLXBackend.is_available()
        status.update(self.evaluator.get_summary())
        return status

    # ── Augmented recall commands ─────────────────────────────

    def debug_recall(self, query: str) -> dict:
        """Run augmented recall and return the full assessment (for /recall)."""
        recall_result = self.augmented_recall.recall_and_augment(query)
        assessment = recall_result.get("assessment")
        return {
            "clarity": recall_result["clarity"],
            "confidence": assessment.confidence if assessment else 0.0,
            "missing_details": assessment.missing_details if assessment else [],
            "suggested_search": assessment.suggested_search if assessment else None,
            "memories_found": len(assessment.memories) if assessment else 0,
            "memory_texts": [m.content for m in (assessment.memories if assessment else [])],
            "search_performed": recall_result["search_performed"],
            "memory_updated": recall_result["memory_updated"],
            "web_context": recall_result.get("web_context"),
        }

    def sharpen_memories(self) -> int:
        """
        Proactively enrich blurry memories with web search.
        Like reviewing and updating old notes.
        """
        all_memories = (
            self.store.get_memories_by_tier(MemoryTier.MID)
            + self.store.get_memories_by_tier(MemoryTier.LONG)
        )

        sharpened = 0
        for memory in all_memories:
            if memory.session_count < 3 and memory.relevance_score > 0.7:
                continue

            assessment = self.augmented_recall.classifier.assess(
                memory.content, [memory]
            )

            if assessment.clarity == MemoryClarity.BLURRY and assessment.suggested_search:
                result = self.augmented_recall._augment_blurry_memory(
                    memory.content, assessment
                )
                if result and result["memory_updated"]:
                    sharpened += 1
                    print(f"  Sharpened: {memory.content[:60]}...")

        return sharpened

    # ── Memory debug ───────────────────────────────────────────

    def get_memory_debug(self) -> dict:
        """Return current memory state for debugging."""
        return {
            "stats": self.store.get_stats(),
            "session_id": self.store.session_id,
            "turn_count": self._turn_count,
            "long_term": [m.content for m in self.store.get_memories_by_tier(MemoryTier.LONG)],
            "mid_term": [m.content for m in self.store.get_memories_by_tier(MemoryTier.MID)],
            "web_knowledge": self.store.get_memories_by_tier(MemoryTier.WEB),
        }

    # ── Internal ───────────────────────────────────────────────

    def _active_backend(self):
        if self._adapter_enabled and self._mlx is not None and self._mlx.is_loaded():
            return self._mlx
        return self.llm

    def _try_load_active_adapter(self, quiet: bool = False):
        adapter_path = self.adapter_manager.get_active_adapter()
        if not adapter_path:
            return
        if not MLXBackend.is_available():
            if not quiet:
                print("  mlx_lm not installed — adapter not loaded.")
            return
        try:
            self._mlx = MLXBackend(adapter_path=adapter_path)
            self._mlx.load()
        except Exception as e:
            if not quiet:
                print(f"  Failed to load adapter: {e}")
            self._mlx = None

    def _start_training_background(self) -> bool:
        training_dir = prepare_training_data(self.store)
        if training_dir is None:
            new_convos = count_new_conversations(self.store)
            print(f"  Not enough quality data yet ({new_convos} new, need {config.LORA_MIN_CONVERSATIONS})")
            return False

        self._training_active = True

        def _on_done(adapter_path: str | None):
            self._training_active = False
            if adapter_path:
                # Only now mark conversations as used — failed training shouldn't consume them
                mark_training_complete(training_dir)
                self.adapter_manager.on_training_complete(adapter_path)
                self._pending_adapter = adapter_path
                self.evaluator.log_training_event(
                    adapter_path=adapter_path,
                    num_conversations=count_new_conversations(self.store),
                    training_iters=self.trainer._compute_iters(count_new_conversations(self.store)),
                )
                print(f"\n  [Training complete — adapter ready, will load on next message]")
            else:
                print("\n  [Training failed — check output above]")

        new_convos = count_new_conversations(self.store)
        self.trainer.train_background(training_dir, num_conversations=new_convos, callback=_on_done)
        return True

    def _activate_pending_adapter(self):
        path = self._pending_adapter
        self._pending_adapter = None
        if not path:
            return
        try:
            if self._mlx:
                self._mlx.unload()
            self._mlx = MLXBackend(adapter_path=path)
            self._mlx.load()
            print("  [New adapter loaded — adapted model now active]")
        except Exception as e:
            print(f"  [Adapter load failed: {e}]")
            self._mlx = None
