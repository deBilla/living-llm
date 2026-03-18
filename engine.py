"""
Conversation Engine — now powered by limbiq.

The engine is a thin orchestrator:
1. Ask limbiq for memory context (lq.process)
2. Build prompt with that context
3. Send to LLM (with optional ReAct tool loop)
4. Tell limbiq what happened (lq.observe)
5. On session end, limbiq consolidates (lq.end_session)

All memory, learning, and adaptation logic lives in limbiq.
LoRA training and web search remain in living-llm (limbiq is memory-only).
"""

import re
import json
import threading

import config
from limbiq import Limbiq
from limbiq.steering import enable_steering
from llm_backend import MLXBackend
from tools.react_loop import ReactLoop
from tools.web_augment import WebAugmenter
from training.adapter_manager import AdapterManager
from training.lora_trainer import LoRATrainer
from training.eval import AdapterEvaluator
from memory.training_data import prepare_training_data, count_new_conversations, mark_training_complete

_TOOL_TAG_RE = re.compile(r"<tool_(?:call|result)>.*?</tool_(?:call|result)>", re.DOTALL)


class ConversationEngine:
    def __init__(self):
        # LoRA adapter state — checked before MLX load
        self.adapter_manager = AdapterManager()
        self.trainer = LoRATrainer()
        self.evaluator = AdapterEvaluator()
        self._adapter_enabled = True
        self._pending_adapter: str | None = None
        self._training_active = False

        # ONE model instance — MLX handles everything:
        # primary generation, LoRA adapters, and steering vectors
        adapter_path = self.adapter_manager.get_active_adapter() if config.LORA_ENABLED else None
        self._mlx = MLXBackend(adapter_path=adapter_path)
        self._mlx.load()

        # Initialize limbiq — uses the same MLX model for compression
        self.lq = Limbiq(
            store_path=config.LIMBIQ_STORE_PATH,
            user_id=config.USER_ID,
            embedding_model=config.EMBEDDING_MODEL,
            llm_fn=self._llm_compress_fn,
        )

        # Steering uses the SAME MLX model — no second load
        enable_steering(
            self.lq,
            model=self._mlx._model,
            tokenizer=self._mlx._tokenizer,
        )

        self.messages: list[dict] = []
        self._turn_count = 0

        # Web search augmentation — bridges limbiq and web search
        self.web_augmenter = WebAugmenter(self.lq, self._mlx)

        # ReAct loop — session-scoped rate limiting persists across turns
        self.react_loop = ReactLoop(
            max_iterations=3,
            max_calls_per_iteration=2,
        )

    def _llm_compress_fn(self, prompt: str) -> str:
        """Adapter: limbiq needs a simple fn(str) -> str for compression."""
        return self._mlx.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )

    # ── Session ────────────────────────────────────────────────

    def start_session(self):
        """Begin a new conversation session."""
        self.lq.start_session()
        self.messages = []
        self._turn_count = 0
        self.react_loop.reset_session()

        stats = self.lq.get_stats()
        adapter_note = " | adapter: on" if (self._adapter_enabled and self._mlx.adapter_path) else ""
        print(f"  Limbiq memory: {stats}{adapter_note}")

    def respond(self, user_input: str) -> str:
        """
        Process a user message and return the assistant's response.

        Pipeline:
        1. Ask limbiq for enriched memory context
        2. Optionally augment with web search
        3. Build prompt with memory + web context
        4. Run the ReAct loop for tool calls
        5. Store exchange; tell limbiq what happened
        """
        self._turn_count += 1

        if self._pending_adapter:
            self._activate_pending_adapter()

        # Step 1: Ask limbiq for enriched context
        result = self.lq.process(
            message=user_input,
            conversation_history=self.messages[-(6 * 2):],
        )

        # Log signals for debugging
        if result.signals_fired:
            for sig in result.signals_fired:
                print(f"  [{sig.signal_type}] {sig.trigger}")

        # Step 2: Optionally augment with web search
        web_context = None
        search_log = []
        if config.WEB_SEARCH_ENABLED:
            web_result = self.web_augmenter.maybe_augment(user_input, result)
            if web_result:
                web_context = web_result["context"]
                search_log = web_result.get("search_log", [])

        # Step 3: Build system prompt
        system_content = config.SYSTEM_PROMPT
        if config.WEB_SEARCH_ENABLED:
            system_content += "\n\n" + config.TOOL_USE_PROMPT

        if result.context:
            system_content += "\n\n" + result.context
            print(f"  [Context] Memory context injected ({len(result.context)} chars)")

        if web_context:
            system_content += (
                "\n\n<web_search_results>\n"
                + web_context
                + "\n</web_search_results>"
            )
            print(f"  [Context] Web context injected ({len(web_context)} chars)")

        # Step 4: Build messages
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.messages[-(6 * 2):])

        # Inject memory summary directly into the user message so the 8B model
        # can't miss it. Small models pay far more attention to user turns than
        # system content buried early in the context window.
        user_msg = user_input
        if result.context and not web_context:
            # Memory-only: strong instruction to use memory, no web distraction
            user_msg = (
                f"Here is what you remember about me from past conversations "
                f"(use this naturally, do NOT mention section names or tags). "
                f"Answer using ONLY this memory — do NOT search the web:\n\n"
                f"{result.context}\n\n"
                f"Now answer this: {user_input}"
            )
        elif result.context and web_context:
            # Both memory and web: prioritize memory
            user_msg = (
                f"Here is what you remember about me (PRIORITY — use this first):\n\n"
                f"{result.context}\n\n"
                f"Here are web search results (use ONLY if memory doesn't answer the question):\n\n"
                f"<web_search_results>\n{web_context}\n</web_search_results>\n\n"
                f"Now answer this: {user_input}"
            )
        elif web_context:
            # Web-only: no memory available
            user_msg = (
                f"<web_search_results>\n{web_context}\n</web_search_results>\n\n"
                f"Now answer this: {user_input}"
            )

        messages.append({"role": "user", "content": user_msg})

        # Step 5: Generate response
        # If memory already has good context, skip the ReAct tool loop entirely —
        # prevents the model from issuing redundant web searches for things it
        # already knows from memory.
        memory_sufficient = result.priority_count > 0 or result.memories_retrieved >= 3
        if config.WEB_SEARCH_ENABLED and not memory_sufficient:
            response, react_log = self.react_loop.run(self._mlx, messages)
            search_log.extend(react_log)

            bad_urls = self.react_loop.verify_citations(response, search_log)
            if bad_urls:
                print(f"  [Warning: response mentions unverified URLs: {bad_urls}]")
        else:
            response = self._mlx.chat(messages)

        # Clean up any stray tool tags that slipped through
        response = _TOOL_TAG_RE.sub("", response).strip()

        # Step 6: Store exchange and tell limbiq in the BACKGROUND
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response})

        threading.Thread(
            target=self._observe_async,
            args=(user_input, response, list(search_log)),
            daemon=True,
        ).start()

        return response

    def _observe_async(self, message: str, response: str, search_log: list):
        """Run limbiq observation and web fact storage in background."""
        try:
            self.lq.observe(message=message, response=response)
        except Exception as e:
            print(f"  [Limbiq observe error: {e}]")

        if search_log:
            try:
                self.web_augmenter.store_web_facts(message, search_log, response)
            except Exception as e:
                print(f"  [Web fact storage error: {e}]")

    def end_session(self):
        """End session — limbiq handles all consolidation."""
        if not self.messages:
            return {}

        print("\n  Limbiq consolidating...")
        results = self.lq.end_session()
        print(f"  Compressed: {results.get('compressed', 0)} facts")
        print(f"  Suppressed: {results.get('suppressed', 0)} stale memories")
        print(f"  Deleted: {results.get('deleted', 0)} old suppressed")

        # Auto-train if enough fresh conversations
        if config.LORA_ENABLED and not self._training_active:
            new_convos = count_new_conversations(self.lq._core.store)
            if self.adapter_manager.should_auto_train(new_convos):
                print(f"  {new_convos} new conversation(s) ready — triggering LoRA training...")
                self._start_training_background()

        stats = self.lq.get_stats()
        print(f"  Memory now: {stats}")

        return results

    # ── Feedback ───────────────────────────────────────────────

    def handle_feedback(self, feedback_type: str, detail: str = None):
        """Handle explicit user feedback signals."""
        if feedback_type == "positive":
            if self.messages:
                last_user_msg = None
                for m in reversed(self.messages):
                    if m["role"] == "user":
                        last_user_msg = m["content"]
                        break
                if last_user_msg:
                    self.lq.dopamine(f"User positively received response about: {last_user_msg[:200]}")

        elif feedback_type == "correction" and detail:
            self.lq.correct(detail)

    # ── Web search commands ────────────────────────────────────

    def forced_search(self, query: str) -> str:
        """Directly execute a web search and generate a response."""
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

        response = self._mlx.chat(messages)

        # Store web facts through limbiq
        search_log = [{"type": "search", "query": query, "results": results}]
        self.web_augmenter.store_web_facts(query, search_log, response)

        return response

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
        new_convos = count_new_conversations(self.lq._core.store)
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
        adapter_path = self.adapter_manager.get_active_adapter()
        if not adapter_path:
            print("  No adapter found. Run /train first.")
            return False
        self._adapter_enabled = True
        # Reload MLX with the adapter
        self._mlx.unload()
        self._mlx = MLXBackend(adapter_path=adapter_path)
        self._mlx.load()
        print("  Adapter enabled.")
        return True

    def adapter_off(self):
        """Disable the adapter (reload base model without adapter)."""
        self._adapter_enabled = False
        self._mlx.unload()
        self._mlx = MLXBackend()
        self._mlx.load()
        print("  Adapter disabled — using base model.")

    def compare_responses(self, prompt: str) -> dict | None:
        """Run the same prompt through base and adapted model."""
        adapter_path = self.adapter_manager.get_active_adapter()
        if not adapter_path:
            print("  No adapter loaded. Run /train first.")
            return None

        # Temporarily load base model (no adapter) for comparison
        print("  Running base model...")
        base_mlx = MLXBackend()
        base_mlx.load()
        base_resp = base_mlx.chat([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.01)
        base_mlx.unload()

        print("  Running adapted model...")
        adapted_resp = self._mlx.chat([{"role": "user", "content": prompt}], max_tokens=256, temperature=0.01)

        comparison = self.evaluator.compare_responses(prompt, base_mlx, self._mlx)
        self.evaluator.log_comparison(comparison, adapter_path=adapter_path)

        return {"prompt": prompt, "base": base_resp, "adapted": adapted_resp}

    def get_adapter_status(self) -> dict:
        status = self.adapter_manager.get_status()
        status["adapter_active"] = self._adapter_enabled and self._mlx.adapter_path is not None
        status["training_running"] = self._training_active
        status["mlx_available"] = MLXBackend.is_available()
        status.update(self.evaluator.get_summary())
        return status

    # ── Memory debug ───────────────────────────────────────────

    def get_memory_debug(self) -> dict:
        """Return current memory state for debugging / UI display."""
        return {
            "stats": self.lq.get_stats(),
            "graph_stats": self.lq.get_graph_stats(),
            "world_summary": self.lq.get_world_summary(),
            "priority": [m.content for m in self.lq.get_priority_memories()],
            "suppressed_count": len(self.lq.get_suppressed()),
            "recent_signals": [
                {"type": s.signal_type if isinstance(s.signal_type, str) else s.signal_type.value,
                 "trigger": s.trigger}
                for s in self.lq.get_signal_log(limit=10)
            ],
            "turn_count": self._turn_count,
        }

    # ── Internal ───────────────────────────────────────────────

    def _start_training_background(self) -> bool:
        training_dir = prepare_training_data(self.lq._core.store)
        if training_dir is None:
            new_convos = count_new_conversations(self.lq._core.store)
            print(f"  Not enough quality data yet ({new_convos} new, need {config.LORA_MIN_CONVERSATIONS})")
            return False

        self._training_active = True

        def _on_done(adapter_path: str | None):
            self._training_active = False
            if adapter_path:
                mark_training_complete(training_dir)
                self.adapter_manager.on_training_complete(adapter_path)
                self._pending_adapter = adapter_path
                self.evaluator.log_training_event(
                    adapter_path=adapter_path,
                    num_conversations=count_new_conversations(self.lq._core.store),
                    training_iters=self.trainer._compute_iters(count_new_conversations(self.lq._core.store)),
                )
                print(f"\n  [Training complete — adapter ready, will load on next message]")
            else:
                print("\n  [Training failed — check output above]")

        new_convos = count_new_conversations(self.lq._core.store)
        self.trainer.train_background(training_dir, num_conversations=new_convos, callback=_on_done)
        return True

    def _activate_pending_adapter(self):
        path = self._pending_adapter
        self._pending_adapter = None
        if not path:
            return
        try:
            self._mlx.unload()
            self._mlx = MLXBackend(adapter_path=path)
            self._mlx.load()
            print("  [New adapter loaded — adapted model now active]")
        except Exception as e:
            print(f"  [Adapter load failed: {e}]")
            # Fallback: reload base model without adapter
            self._mlx = MLXBackend()
            self._mlx.load()
