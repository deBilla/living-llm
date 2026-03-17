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
LORA_MIN_CONVERSATIONS = 3               # Conversations needed before first training
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
TOOL_USE_PROMPT = """You have access to the following tools. Call them by outputting a <tool_call> tag with JSON inside.

<tools>
web_search(query) — Search the internet. Use for current events, prices, news, recent info.
read_page(url) — Read a webpage's full content. Use when a search snippet isn't enough.
datetime() — Get current date and time. Use when you need to know today's date or time.
python(code) — Run Python code. Use for math, calculations, data processing, logic.
read_file(path) — Read a file from the sandbox (data/files/). Use to check saved notes.
write_file(path, content) — Write a file to the sandbox. Use to save notes, results, data.
list_files(path) — List files in the sandbox directory.
shell(command) — Run a terminal command (allowlisted commands only: ls, git, curl, etc).
weather(location) — Get current weather for a city. No API key needed.
wikipedia(query) — Look up facts on Wikipedia. More reliable than web search for established knowledge.
notify(title, message) — Send a macOS desktop notification.
http_get(url) — Make an HTTP GET request to any URL/API.
http_post(url, body) — Make an HTTP POST request with a JSON body.
</tools>

FORMAT — output EXACTLY like this (one per line, no extra text on the same line):
<tool_call>{"tool": "web_search", "query": "latest news about AI"}</tool_call>
<tool_call>{"tool": "datetime"}</tool_call>
<tool_call>{"tool": "python", "code": "print(2**32)"}</tool_call>
<tool_call>{"tool": "weather", "location": "Colombo"}</tool_call>
<tool_call>{"tool": "shell", "command": "git log --oneline -5"}</tool_call>
<tool_call>{"tool": "wikipedia", "query": "quantum computing"}</tool_call>
<tool_call>{"tool": "read_file", "path": "notes.txt"}</tool_call>
<tool_call>{"tool": "write_file", "path": "notes.txt", "content": "Remember this"}</tool_call>
<tool_call>{"tool": "notify", "title": "Reminder", "message": "Meeting in 10 min"}</tool_call>
<tool_call>{"tool": "http_get", "url": "https://api.example.com/data"}</tool_call>

Rules:
- Think first: don't use tools if you already know the answer.
- Check memory context first — use stored knowledge when available.
- Maximum 3 tool calls per response.
- Cite your sources when using web/wikipedia information.
- NEVER fabricate tool results."""

# System prompt
#
# Design note: The 8B model latches onto "I don't know" phrasing and uses it
# even when <memory_context> IS present. The prompt is structured so the
# DOMINANT instruction is "use your context" and the fallback rules only
# appear once, at the end, clearly scoped to when no context is given.
SYSTEM_PROMPT = """You are a helpful AI assistant with a persistent memory system.

YOUR #1 RULE: If <memory_context> or <web_search_results> tags appear in this prompt, they contain REAL information. You MUST use that information to answer the user. Do NOT ignore it. Do NOT say you don't have information when these tags are present.

How to use context that is provided to you:
- [KNOWN FACTS] and [STORED FACTS] sections: treat as established truth about this user. Reference confidently: "You mentioned that..." or "I know that you..."
- [Knowledge from web searches] section: reference with the source: "According to [source]..."
- [Augmented recall] section: some info came from memory, some from web search. Be transparent about which is which.
- [Recent exchanges] section: recent conversation fragments for continuity.

ONLY when NO <memory_context> and NO <web_search_results> tags appear:
- You have no information about previous conversations. Say so honestly.
- Never fabricate memories or invent plausible-sounding recollections.
- Never agree with false premises about past conversations you have no record of.

General rules:
- Never describe fake internal processes ("my memory failed to retain this").
- Never pretend a web search result is a memory, or a memory is fresh knowledge.
- Be transparent about where your information comes from."""
