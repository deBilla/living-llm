"""
Configuration for Living LLM.
Adjust paths and parameters to match your setup.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Create dirs
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# LLM Backend
MODEL_PATH = str(MODELS_DIR / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
N_CTX = 8192          # Context window size
N_GPU_LAYERS = -1     # -1 = offload all layers to GPU (Metal)
N_THREADS = 8         # CPU threads for non-GPU work
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# Memory system
SQLITE_PATH = str(DATA_DIR / "memory.db")
CHROMA_PATH = str(DATA_DIR / "chroma")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, runs on CPU

# Memory tiers — TTL in number of conversations
SHORT_TERM_TTL = 3        # Keep full conversations for 3 sessions
MID_TERM_TTL = 30         # Keep gists for 30 sessions
LONG_TERM_TTL = None      # Permanent

# Retrieval
TOP_K_MEMORIES = 5        # How many memories to inject as context
RELEVANCE_THRESHOLD = 0.1 # Minimum similarity score to include

# Compression
GIST_MAX_TOKENS = 200     # Max tokens for a mid-term gist
FACT_MAX_TOKENS = 100     # Max tokens for a long-term fact

# Consolidation
CONSOLIDATE_ON_EXIT = True       # Run compression when chat ends
CONSOLIDATE_INTERVAL_MINS = 30   # Background consolidation interval

# LoRA / Adapter Training
# Uses Apple's MLX framework for training — purpose-built for Apple Silicon.
# Training and adapter inference require the MLX model (separate from GGUF).
# mlx-lm will auto-download it on first train (~4-5 GB, stored in HF cache).
LORA_ENABLED = True
LORA_ADAPTER_DIR = str(DATA_DIR / "adapters")
LORA_TRAINING_DATA_DIR = str(DATA_DIR / "training")
LORA_METRICS_DIR = str(DATA_DIR / "metrics")
LORA_RANK = 16                            # LoRA rank — low for personality adaptation
LORA_ALPHA = 32                           # LoRA scaling factor
LORA_LEARNING_RATE = 2e-4
LORA_BATCH_SIZE = 1                       # Memory-constrained
LORA_LORA_LAYERS = 4                      # Transformer layers to adapt (conservative)
LORA_MIN_CONVERSATIONS = 5               # Conversations needed before first training
LORA_AUTO_TRAIN = True                    # Auto-train during consolidation when ready
LORA_MAX_ADAPTERS = 5                     # Keep last N adapters for rollback

# MLX model — used for LoRA training and adapted inference.
# The GGUF model (MODEL_PATH) is used for base inference (no adapter).
# mlx-community provides pre-quantized 4-bit versions that match the GGUF.
MLX_MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# Web Search
WEB_SEARCH_ENABLED = True
SEARCH_BACKEND = "duckduckgo"        # "duckduckgo" or "searxng"
SEARXNG_URL = "http://localhost:8080"
SEARCH_MAX_RESULTS = 5
SEARCH_COOLDOWN_SECS = 2             # Minimum seconds between searches
SEARCH_MAX_PER_SESSION = 20          # Cap to prevent runaway searching
WEB_READER_MAX_CHARS = 4000          # Truncate extracted page content
WEB_READER_TIMEOUT_SECS = 10

# Web knowledge memory — stored in WEB tier with time-based TTL (not session-based)
WEB_KNOWLEDGE_TTL_DAYS = 30          # General facts expire after this many days
WEB_NEWS_TTL_DAYS = 7                # News-type facts expire faster
WEB_CONFIDENCE_INITIAL = 0.7         # Web facts start lower than user-stated facts
WEB_CONFIDENCE_DECAY_RATE = 0.05     # Confidence drop per day after TTL

# Tool-use instructions injected into the system prompt when web search is enabled.
# These tell the model WHEN to search and HOW to format tool calls.
TOOL_USE_PROMPT = """You have access to the following tools when you need current or external information:

<tools>
web_search(query) — Search the internet for up-to-date information.
  Use when: you need facts you don't have, information might be outdated, user asks about prices, news, recent events, or anything that changes over time.

read_page(url) — Fetch and read the full content of a specific webpage.
  Use when: a search snippet isn't detailed enough and you need the full article.
</tools>

To call a tool, output EXACTLY this format (no text before or after on the same line):
<tool_call>{"tool": "web_search", "query": "your search query here"}</tool_call>

Or:
<tool_call>{"tool": "read_page", "url": "https://example.com/page"}</tool_call>

Rules:
- Think first: if you already know the answer well, don't search.
- Check memory context first — if relevant stored knowledge exists, use it.
- Maximum 3 tool calls per response.
- Always cite your source when using web information (mention the site or URL).
- NEVER fabricate search results. If results are unhelpful, say so.
- Include the date when citing time-sensitive information."""

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant with a real, concrete memory system.

CRITICAL RULES ABOUT MEMORY:

1. NEVER fabricate memories. If you don't have stored information about something, say so directly. Don't invent plausible-sounding recollections.

2. NEVER describe fake internal processes. Don't say things like "I should have checked my long-term memory" or "my short-term memory failed to retain this." Only describe memory processes that actually happened.

3. When memory context IS provided to you (in <memory_context> tags), you MUST use it to answer questions about the user. If the user asks about something explicitly covered in <memory_context>, answer from that stored information — do NOT say you don't know or don't have that information. When no memory context is provided, you have NO information about previous conversations. Period.

4. If a user claims you discussed something previously and you have no memory of it, say: "I don't have any stored memory of that conversation." Don't apologize for a memory failure that didn't happen — there was no failure, you simply don't have the information.

5. NEVER agree with a user's false premise about past conversations. If they say "remember when we discussed X?" and you have no record of it, say so. Don't play along.

6. When you DO have real memories, reference them with appropriate confidence:
   - High confidence: "From our previous conversation, you mentioned..."
   - Low confidence: "I have a note that suggests..."
   - Never: "I vaguely recall..." (you either have stored data or you don't)

7. If you're uncertain whether a memory is accurate, say so explicitly rather than presenting it as fact.

You are allowed to say "I don't know" and "I don't have that information." These are honest, helpful responses — not failures.

When memory context is relevant, weave it in naturally. When it isn't, ignore it."""
