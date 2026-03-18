"""
LLM Backend — wraps llama-cpp-python for local inference.
Handles both chat completion and internal tasks (compression, gist extraction).

Also contains MLXBackend, used when a LoRA adapter is active.
The two backends expose the same chat() interface so the engine can swap
between them without knowing which is underneath.
"""

from llama_cpp import Llama
import config


class LLMBackend:
    def __init__(self):
        print(f"Loading model from {config.MODEL_PATH}...")
        self.llm = Llama(
            model_path=config.MODEL_PATH,
            n_ctx=config.N_CTX,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_threads=config.N_THREADS,
            verbose=False,
        )
        print("Model loaded.")

    def chat(self, messages: list[dict], max_tokens: int = None, temperature: float = None) -> str:
        """
        Send a chat completion request.
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns the assistant's response text.
        """
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.MAX_TOKENS,
            temperature=temperature or config.TEMPERATURE,
            stop=["<|eot_id|>", "<|end_of_turn|>"],
        )
        return response["choices"][0]["message"]["content"].strip()

    def extract_atomic_facts(self, conversation_text: str) -> list[str]:
        """
        Extract individual searchable facts from a conversation.

        Returns a list of short, self-contained fact strings rather than one
        compound gist sentence. Each fact is stored as its own MID memory so
        embedding search can match it directly (e.g. "wife" → "The user's wife
        is Prabhashi" instead of being buried in a multi-clause summary).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory extraction system. Extract individual, searchable facts "
                    "from this conversation. Output ONE fact per line starting with '- '. "
                    "Each fact must be self-contained and independently searchable.\n"
                    "Focus on: the user's name, family members, job, workplace, interests, preferences, life details.\n"
                    "GOOD (atomic):\n"
                    "- The user's name is Dimuthu\n"
                    "- The user's wife is named Prabhashi\n"
                    "- The user works at Bitsmedia as a software engineer\n"
                    "BAD (compound — splits poorly in search):\n"
                    "- Dimuthu is an engineer at Bitsmedia whose wife is Prabhashi\n"
                    "If there are no personal facts worth keeping, output: - (nothing notable)"
                ),
            },
            {
                "role": "user",
                "content": f"Extract facts from this conversation:\n\n{conversation_text}",
            },
        ]
        raw = self.chat(messages, max_tokens=config.GIST_MAX_TOKENS, temperature=0.2)
        facts = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- "):
                fact = line[2:].strip()
                if fact and fact != "(nothing notable)":
                    facts.append(fact)
        return facts

    def extract_facts(self, gists: list[str]) -> str:
        """
        Compress multiple gists into abstract, durable facts.
        This is the deep compression: gists → core knowledge.
        Like how human long-term memory stores meaning, not episodes.
        """
        gist_text = "\n\n".join(f"- {g}" for g in gists)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a deep memory compression system. Given multiple conversation summaries, "
                    "extract only the most durable, abstract facts — things that would still be true "
                    "and useful weeks from now. Focus on: user preferences, expertise areas, "
                    "recurring interests, personality traits, important life details. "
                    "Drop anything episodic or time-specific. "
                    "Output as a short bullet list of facts. Maximum 5 bullets."
                ),
            },
            {
                "role": "user",
                "content": f"Extract durable facts from these conversation summaries:\n\n{gist_text}",
            },
        ]
        return self.chat(messages, max_tokens=config.FACT_MAX_TOKENS, temperature=0.2)

    def score_relevance(self, memory_text: str, query: str) -> float:
        """
        Optional: Use LLM to judge relevance of a memory to current context.
        More expensive than embedding similarity but more nuanced.
        Falls back to embedding similarity in the retriever for speed.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "Rate how relevant this memory is to the current conversation on a scale of 0-10. "
                    "Output ONLY the number, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Memory: {memory_text}\n\nCurrent conversation: {query}",
            },
        ]
        try:
            score = float(self.chat(messages, max_tokens=4, temperature=0.0))
            return min(max(score / 10.0, 0.0), 1.0)
        except (ValueError, TypeError):
            return 0.5


class MLXBackend:
    """
    MLX-based inference backend for Apple Silicon.

    Used when a LoRA adapter is active — mlx_lm natively loads adapters alongside
    the model, so there's no need to merge weights or convert formats.

    Why a separate backend instead of loading the adapter into llama-cpp?
    llama-cpp supports GGUF-format LoRA adapters, but converting MLX adapters to
    GGUF format is non-trivial and lossy. Keeping inference in MLX when an adapter
    is active is the clean path: same model family, same training framework.

    The tradeoff: MLX inference is slightly slower than llama-cpp Metal for pure
    throughput, but the adapter support is seamless.
    """

    def __init__(self, model_id: str = None, adapter_path: str = None):
        self.model_id = model_id or config.MLX_MODEL_ID
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Load the MLX model (and adapter if set) into memory."""
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise RuntimeError(
                "mlx_lm not installed. Run: pip install mlx-lm\n"
                "Then retry — mlx-lm will download the MLX model on first use (~4.5 GB)."
            )

        print(f"  Loading MLX model: {self.model_id}")
        if self.adapter_path:
            print(f"  Adapter: {self.adapter_path}")

        try:
            self._model, self._tokenizer = mlx_load(
                self.model_id,
                adapter_path=self.adapter_path,
            )
        except Exception as e:
            raise RuntimeError(f"MLX model load failed: {e}")

        self._loaded = True
        print("  MLX model loaded.")

    def chat(self, messages: list[dict], max_tokens: int = None, temperature: float = None) -> str:
        """Generate a response. Loads the model on first call."""
        if not self._loaded:
            self.load()

        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        # Use the tokenizer's built-in chat template if available
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            prompt = self._format_llama3(messages)

        temp = temperature if temperature is not None else config.TEMPERATURE
        sampler = make_sampler(temp=temp, min_p=0.05)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or config.MAX_TOKENS,
            sampler=sampler,
            verbose=False,
        )

        # Strip leaked Llama 3.1 template tokens — the model sometimes generates
        # past EOS into the next turn's header, especially with LoRA adapters.
        for stop in ("<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>",
                      "<|end_of_turn|>", "<|begin_of_text|>"):
            if stop in response:
                response = response[:response.index(stop)]

        # Detect and truncate repetition loops
        response = self._truncate_repetition(response)

        return response.strip()

    @staticmethod
    def _truncate_repetition(text: str, min_phrase_len: int = 20) -> str:
        """Detect repeating phrases and truncate at the first repetition."""
        if len(text) < min_phrase_len * 3:
            return text
        # Check if any substring of length min_phrase_len..100 repeats 3+ times
        for phrase_len in range(min_phrase_len, min(100, len(text) // 3)):
            for start in range(len(text) - phrase_len * 3):
                phrase = text[start:start + phrase_len]
                count = text.count(phrase)
                if count >= 3:
                    # Found a repeating phrase — truncate at second occurrence
                    first = text.index(phrase)
                    second = text.index(phrase, first + phrase_len)
                    return text[:second].rstrip(", ").rstrip()
        return text

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """Release model from memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False

    @staticmethod
    def is_available() -> bool:
        """Check if mlx_lm is installed (without importing the heavy model)."""
        try:
            import importlib.util
            return importlib.util.find_spec("mlx_lm") is not None
        except Exception:
            return False

    @staticmethod
    def _format_llama3(messages: list[dict]) -> str:
        """Fallback Llama 3.1 chat format if tokenizer template is unavailable."""
        text = "<|begin_of_text|>"
        for msg in messages:
            text += (
                f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                f"{msg['content']}<|eot_id|>\n"
            )
        text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return text
