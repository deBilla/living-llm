# Living LLM

A locally-running language model with persistent memory, neurotransmitter-inspired learning, web-augmented recall, LoRA neuroplasticity, and a full agent tool suite. Powered by [limbiq](https://github.com/deBilla/limbiq). Runs entirely on Apple Silicon.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Chat Interface (Terminal / Gradio)           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Conversation Engine                      │
│                                                          │
│  lq.process(msg) ──→ Build prompt ──→ LLM ──→ lq.observe│
│       │                    │                │            │
│  Memory context      Web augment if    ReAct tool loop   │
│  from limbiq         few memories      (13 tools)        │
│       └────────────────────┴────────────────┘            │
│                            │                             │
│                   lq.end_session()                        │
│              (compress + suppress stale)                  │
└────┬──────────────────┬──────────────────┬──────────────┘
     │                  │                  │
┌────▼────────┐  ┌──────▼───────┐  ┌──────▼───────────────┐
│ LLM Backend │  │   Limbiq     │  │ LoRA Adapter (MLX)   │
│ llama.cpp   │  │              │  │                      │
│ Metal GPU   │  │ Dopamine     │  │ Trains on compressed │
│             │  │ (priority)   │  │ conversations        │
│ Llama 3.1   │  │ GABA         │  │ Auto-loads on next   │
│ 8B Q4_K_M   │  │ (suppress)   │  │ message              │
│             │  │ SQLite +     │  │                      │
│             │  │ Embeddings   │  │                      │
└─────────────┘  └──────────────┘  └──────────────────────┘
```

## Limbiq Signals

Living LLM delegates all memory management to **limbiq**, a neurotransmitter-inspired adaptive learning library.

| Signal | Meaning | When it fires |
|--------|---------|---------------|
| **Dopamine** | "This matters, remember it" | Personal info shared, corrections, positive feedback |
| **GABA** | "Suppress this, let it fade" | Denials, contradictions, stale memories |

- **Dopamine-tagged** memories are always included in context (priority)
- **GABA-suppressed** memories are excluded from retrieval but can be restored
- **Corrections** combine both: dopamine on new info + GABA on old

## Memory Tiers

| Tier | What it stores | Lifecycle |
|------|---------------|-----------|
| **SHORT** | Raw conversation turns | Aged each session, suppressed when stale |
| **MID** | Atomic facts compressed from conversations | Created at session end |
| **PRIORITY** | Dopamine-tagged high-importance facts | Always included in context |

Limbiq handles compression via `end_session()` — conversations are distilled into atomic facts, stale memories are suppressed, and old suppressed memories are deleted.

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

## Web Search Augmentation

When limbiq returns few or no memories for a query, the web augmenter kicks in:

1. Check if the query is searchable (keyword heuristic + LLM fallback)
2. Search the web via DuckDuckGo/SearXNG
3. Inject results as additional context
4. Extract durable facts and store them via `lq.dopamine()` with `[Web]` prefix

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
| `/memory` | Inspect limbiq memory state |
| `/signals` | Show recent signal history (dopamine/GABA events) |
| `/priority` | Show all dopamine-tagged priority memories |
| `/suppress` | Show all GABA-suppressed memories |
| `/dopamine <fact>` | Manually tag a fact as high-priority |
| `/gaba <id>` | Manually suppress a memory by ID |
| `/correct <info>` | Correct a wrong memory (dopamine new + GABA old) |
| `/good` | Mark last response as positive (fires dopamine) |
| `/bad` | Mark last response as negative |
| `/restore <id>` | Restore a GABA-suppressed memory |
| `/search <query>` | Force a web search |
| `/export` | Export full limbiq state as JSON |
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
├── engine.py                # Thin orchestrator — limbiq + LLM + tools
├── llm_backend.py           # LLM backends (llama-cpp + MLX)
├── config.py                # All configuration
├── consolidate.py           # Standalone consolidation + training trigger
├── eval_confabulation.py    # Confabulation test suite
├── migrate_to_limbiq.py     # One-time migration from old memory system
├── memory/
│   └── training_data.py     # Conversation → JSONL for LoRA
├── tools/
│   ├── react_loop.py        # ReAct tool execution loop
│   ├── web_augment.py       # Bridge between limbiq and web search
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
    ├── limbiq/              # Limbiq's memory store
    ├── training/            # LoRA training data
    ├── adapters/            # LoRA adapter checkpoints
    └── metrics/             # Evaluation logs
```

## How It Works

1. **You chat** → limbiq processes the message and returns enriched memory context
2. **Context injection** → memory injected directly into the user message so the 8B model reliably uses it
3. **Web augmentation** → if limbiq has few memories, web search fills the gap
4. **ReAct tools** → model can call 13 tools (search, code, weather, files, etc.) during response generation
5. **Observe** → limbiq observes the exchange, fires dopamine/GABA signals as appropriate
6. **Compression** → on `/quit`, `lq.end_session()` compresses conversations into atomic facts and suppresses stale memories
7. **LoRA training** → when enough conversations accumulate, a LoRA adapter trains in the background
8. **Next conversation** → the model remembers you, uses its tools, and gets better over time

## Migration

If upgrading from the pre-limbiq version (with `data/memory.db` and `data/chroma/`):

```bash
python migrate_to_limbiq.py
```

This migrates long-term memories as priority, mid-term as observed, and web knowledge with `[Web]` tags. After migration you can delete the old `data/memory.db` and `data/chroma/` directories.
