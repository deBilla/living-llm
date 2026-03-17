# Living LLM

A locally-running language model with persistent memory, lossy compression, web-augmented recall, LoRA neuroplasticity, and a full agent tool suite. Runs entirely on Apple Silicon.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Chat Interface (Terminal / Gradio)           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Conversation Engine                      │
│                                                          │
│  Augmented Recall ─→ Classify (CLEAR/BLURRY/ABSENT)      │
│       │                    │                │             │
│    Use memory         Search + merge     Fresh search     │
│    directly           with memory        or say IDK       │
│       └────────────────────┴────────────────┘             │
│                            │                              │
│                     ReAct Tool Loop                       │
│              (13 tools, max 3 calls/turn)                 │
└────┬──────────────────┬──────────────────┬───────────────┘
     │                  │                  │
┌────▼────────┐  ┌──────▼───────┐  ┌──────▼───────────────┐
│ LLM Backend │  │ Memory System│  │ LoRA Adapter (MLX)   │
│ llama.cpp   │  │              │  │                      │
│ Metal GPU   │  │ SHORT → MID  │  │ Trains on compressed │
│             │  │  → LONG → WEB│  │ conversations        │
│ Llama 3.1   │  │              │  │ Auto-loads on next   │
│ 8B Q4_K_M   │  │ SQLite +     │  │ message              │
│             │  │ ChromaDB     │  │                      │
└─────────────┘  └──────────────┘  └──────────────────────┘
```

## Memory Tiers

| Tier | What it stores | TTL | Compression |
|------|---------------|-----|-------------|
| **SHORT** | Full conversation turns | 3 sessions | None |
| **MID** | Atomic facts extracted by LLM | 30 sessions | Lossy — one fact per entry |
| **LONG** | Abstract knowledge synthesized from 3+ gists | Permanent | Deep lossy |
| **WEB** | Facts learned from web searches | 7–30 days (calendar) | Confidence decay |

Each tier compresses the one above it. Details fade, meaning persists — like human memory.

## Agent Tools

The model has access to 13 tools via a ReAct (Reason-Act) loop:

| Tool | Description |
|------|-------------|
| `web_search` | DuckDuckGo/SearXNG internet search |
| `read_page` | Extract content from a URL (trafilatura) |
| `datetime` | Current date, time, timezone |
| `python` | Sandboxed Python REPL (10s timeout, restricted builtins) |
| `read_file` | Read files from sandbox (`data/files/`) |
| `write_file` | Write files to sandbox |
| `list_files` | List sandbox directory |
| `shell` | Run allowlisted terminal commands (git, ls, curl, etc.) |
| `weather` | Current weather via Open-Meteo (free, no API key) |
| `wikipedia` | Wikipedia article search and summaries |
| `notify` | macOS desktop notifications |
| `http_get` | HTTP GET to any URL/API |
| `http_post` | HTTP POST with JSON body |

Tools are called by the model using `<tool_call>` XML tags. The ReAct loop intercepts, executes, and feeds results back until the model produces a final answer.

## Augmented Recall

The bridge between blurry memory and web search:

1. User asks a question → retrieve memories
2. **CLEAR** — high confidence, use directly
3. **BLURRY** — partial match → use memory as seed for targeted web search → merge results → sharpen the memory
4. **ABSENT** — nothing found → fresh web search if the query is searchable, otherwise say "I don't know"

User feedback (confirmation/denial) adjusts memory confidence scores.

## LoRA Neuroplasticity

- After ≥3 compressed conversations, LoRA fine-tuning runs via `mlx_lm` on Apple Silicon
- Trains on the MLX-format model (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`)
- Base GGUF model stays untouched; adapted inference uses `MLXBackend`
- Training runs in a background thread; adapter loads on next message
- Old adapters are cleaned up (keeps last 5)

## Setup

### Prerequisites

- macOS with Apple Silicon (M4 Pro recommended, 24GB RAM)
- Python 3.11+
- ~10GB disk space (model + MLX model for training)

### Install

```bash
cd living-llm
python3 -m venv venv
source venv/bin/activate

# Core (Metal GPU acceleration)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install -r requirements.txt
```

### Download model

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/
```

The MLX model for LoRA training downloads automatically on first `/train` (~4.5 GB).

### Run

```bash
# Terminal mode
python main.py

# Gradio web UI
python main.py --ui
```

## Commands

| Command | Description |
|---------|-------------|
| `/memory` | Inspect memory state (all tiers) |
| `/search <query>` | Force a web search |
| `/recall <query>` | Debug augmented recall assessment |
| `/sharpen` | Enrich blurry memories with web search |
| `/knowledge` | Show stored web knowledge |
| `/knowledge clear` | Clear all web knowledge |
| `/knowledge decay` | Run confidence decay |
| `/train` | Trigger LoRA training |
| `/adapter` | Show adapter status |
| `/adapter compare` | Compare base vs adapted model |
| `/adapter off\|on` | Toggle LoRA adapter |
| `/new` | Start a new session |
| `/quit` | End session (triggers compression) |

## Project Structure

```
living-llm/
├── main.py                  # Entry point — terminal chat + Gradio UI
├── engine.py                # Orchestrator — memory, tools, LLM, LoRA
├── llm_backend.py           # LLM backends (llama-cpp + MLX)
├── config.py                # All configuration
├── consolidate.py           # Background memory consolidation
├── eval_confabulation.py    # Confabulation test suite
├── memory/
│   ├── store.py             # SQLite + ChromaDB dual store
│   ├── compressor.py        # Lossy compression pipeline
│   ├── retriever.py         # Semantic retrieval with tier boosting
│   ├── confidence.py        # Memory clarity classifier (CLEAR/BLURRY/ABSENT)
│   ├── web_knowledge.py     # Web fact extraction and decay
│   └── training_data.py     # Conversation → JSONL for LoRA
├── tools/
│   ├── react_loop.py        # ReAct tool execution loop
│   ├── augmented_recall.py  # Blurry memory → web search bridge
│   ├── web_search.py        # DuckDuckGo / SearXNG
│   ├── web_reader.py        # URL → clean text (trafilatura)
│   ├── datetime_tool.py     # Current date/time
│   ├── python_exec.py       # Sandboxed Python REPL
│   ├── file_tools.py        # Read/write/list files (sandboxed)
│   ├── shell_exec.py        # Allowlisted shell commands
│   ├── weather.py           # Open-Meteo weather API
│   ├── wikipedia.py         # Wikipedia search + summaries
│   ├── notify.py            # macOS desktop notifications
│   └── http_request.py      # Generic HTTP GET/POST
├── training/
│   ├── lora_trainer.py      # mlx_lm.lora subprocess wrapper
│   ├── adapter_manager.py   # Adapter lifecycle and rollback
│   └── eval.py              # Base vs adapted response comparison
├── models/                  # GGUF models (gitignored)
└── data/                    # All persistent state (gitignored)
```

## How It Works

1. **You chat** → conversation stored in short-term memory at full fidelity
2. **Augmented recall** → memories retrieved and classified (clear/blurry/absent); blurry memories seed web searches
3. **ReAct tools** → model can call 13 tools (search, code, weather, files, etc.) during response generation
4. **Memory injection** → context injected directly into the user message so the 8B model reliably uses it
5. **Compression** → on `/quit`, conversations compress into atomic facts (mid-term) and abstract knowledge (long-term)
6. **LoRA training** → when enough conversations accumulate, a LoRA adapter trains in the background
7. **Next conversation** → the model remembers you, uses its tools, and gets better over time
