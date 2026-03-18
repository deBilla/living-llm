# Living LLM

A locally-running language model with persistent memory, neurotransmitter-inspired learning, knowledge graph reasoning, web-augmented recall, LoRA neuroplasticity, and a full agent tool suite. Powered by [limbiq](https://github.com/deBilla/limbiq). Runs entirely on Apple Silicon.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Chat Interface (Terminal / Gradio)           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  Conversation Engine                      │
│                                                          │
│  lq.process(msg) ──→ Build prompt ──→ MLX ──→ lq.observe│
│       │                    │              │   (async)    │
│  Graph query first    Web augment if   ReAct tool loop   │
│  + world summary      few memories     (13 tools)        │
│  + memory context                                        │
│       └────────────────────┴────────────────┘            │
│                            │                             │
│                   lq.end_session()                        │
│         (compress + graph inference + suppress stale)     │
└────┬───────────────────────┬───────────────────┬────────┘
     │                       │                   │
┌────▼────────────┐  ┌──────▼───────┐  ┌────────▼─────────┐
│ MLX Backend     │  │   Limbiq     │  │ Activation       │
│ (single model)  │  │              │  │ Steering         │
│                 │  │ 5 Signals    │  │                  │
│ Llama 3.1 8B   │  │ Knowledge    │  │ Shares the same  │
│ 4-bit quantized │  │ Graph        │  │ MLX model — no   │
│                 │  │ SQLite +     │  │ second load      │
│ + LoRA adapter  │  │ Embeddings   │  │                  │
└─────────────────┘  └──────────────┘  └──────────────────┘
```

**Single model architecture:** One MLX model instance serves all purposes — primary generation, LoRA adapter inference, limbiq compression, and activation steering. No more llama.cpp + MLX duplication.

## Limbiq Signals

Living LLM delegates all memory management to **limbiq**, a neurotransmitter-inspired adaptive learning library.

| Signal | Meaning | When it fires |
|--------|---------|---------------|
| **Dopamine** | "This matters, remember it" | Personal info shared, corrections, positive feedback |
| **GABA** | "Suppress this, let it fade" | Denials, contradictions, stale memories |
| **Serotonin** | "This is a behavioral pattern" | Repeated user preferences → crystallized rules |
| **Acetylcholine** | "Focus on this domain" | Sustained topic discussion → knowledge clusters |
| **Norepinephrine** | "Topic shifted, be careful" | Abrupt topic changes → widened retrieval + caution |

- **Dopamine-tagged** memories are always included in context (priority)
- **GABA-suppressed** memories are excluded from retrieval but can be restored
- **Corrections** combine both: dopamine on new info + GABA on old
- **Serotonin rules** shape response style (concise, casual, technical, etc.)
- **Acetylcholine clusters** group domain knowledge for deep topic recall
- **Norepinephrine** widens retrieval and adds caution flags on topic shifts

## Knowledge Graph

Limbiq builds a personal knowledge graph from conversations — entities (people, places, companies) and relationships (father, wife, works_at). A deterministic inference engine computes implied relationships without using the LLM:

```
User tells limbiq:
  "My father is Upananda"     →  Dimuthu --[father]--> Upananda
  "My wife is Prabhashi"      →  Dimuthu --[wife]--> Prabhashi

Limbiq infers (no LLM needed):
  Upananda --[father_in_law_of]--> Prabhashi
```

**Token efficiency:** Instead of injecting 5 raw memory strings (~200 tokens), the graph produces a compact world summary (~40 tokens):

> "Your father is Upananda (Prabhashi's father-in-law). Your wife is Prabhashi. You work at Bitsmedia."

Graph queries ("Who is Upananda to my wife?") are answered deterministically — zero LLM cost, ~15 tokens injected.

## Memory Tiers

| Tier | What it stores | Lifecycle |
|------|---------------|-----------|
| **SHORT** | Raw conversation turns | Aged each session, suppressed when stale |
| **MID** | Atomic facts compressed from conversations | Created at session end |
| **PRIORITY** | Dopamine-tagged high-importance facts | Always included in context |

Limbiq handles compression via `end_session()` — conversations are distilled into atomic facts, entities are extracted into the knowledge graph, inference runs, stale memories are suppressed, and old suppressed memories are deleted.

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

Tools are called by the model using `<tool_call>` XML tags. The ReAct loop intercepts, executes, and feeds results back until the model produces a final answer. When memory already answers the question, the ReAct loop is skipped entirely to avoid redundant web searches.

## Web Search Augmentation

When limbiq returns few or no memories for a query, the web augmenter kicks in:

1. Check if the query is searchable (keyword heuristic — no LLM fallback)
2. Search the web via DuckDuckGo/SearXNG
3. Inject results as additional context
4. Extract durable facts and store them via `lq.dopamine()` with `[Web]` prefix

Memory-first priority: if limbiq has good context (priority memories or 3+ relevant memories), web search is skipped entirely.

## LoRA Neuroplasticity

- After ≥3 compressed conversations, LoRA fine-tuning runs via `mlx_lm` on Apple Silicon
- Trains on the MLX-format model (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`)
- Same MLX model serves both base and adapted inference (single instance)
- Training runs in a background thread; adapter loads on next message
- Old adapters are cleaned up (keeps last 5)

## Activation Steering

Limbiq maps neurotransmitter signals to steering vectors injected at specific transformer layers:

| Signal | Steering Effect |
|--------|----------------|
| Dopamine | Amplify attention to memory context + confidence |
| GABA | Increase honesty + decrease confidence |
| Serotonin | Persona vectors based on crystallized rules |
| Norepinephrine | Heightened caution + memory attention |
| Acetylcholine | Technical depth + helpfulness |

Steering vectors are injected during the forward pass at target layers only — non-steered layers have zero overhead. The steering model shares the same MLX model instance as primary generation.

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

# For development with local limbiq:
pip install -e ../limbiq
```

### Download model

```bash
pip install huggingface-hub
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/
```

The MLX model for LoRA training and inference downloads automatically on first run (~4.5 GB).

### Run

```bash
# Terminal mode
python main.py

# Gradio web UI
python main.py --ui

# Standalone consolidation
python consolidate.py          # Single cycle
python consolidate.py --stats  # Memory stats
python consolidate.py --train  # Consolidate + trigger LoRA training
```

## Commands

| Command | Description |
|---------|-------------|
| `/memory` | Inspect limbiq memory state + graph stats |
| `/signals` | Show recent signal history (all 5 neurotransmitters) |
| `/priority` | Show all dopamine-tagged priority memories |
| `/suppress` | Show all GABA-suppressed memories |
| `/graph` | Show knowledge graph — entities, relations, inferences |
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
| `/quit` | End session (triggers compression + graph inference) |

## Project Structure

```
living-llm/
├── main.py                  # Entry point — terminal chat + Gradio UI
├── engine.py                # Thin orchestrator — limbiq + MLX + tools
├── llm_backend.py           # MLX backend (single model for everything)
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
    ├── limbiq/              # Limbiq's memory store + knowledge graph
    ├── training/            # LoRA training data
    ├── adapters/            # LoRA adapter checkpoints
    └── metrics/             # Evaluation logs
```

## How It Works

1. **You chat** → limbiq queries the knowledge graph first, then retrieves memories
2. **Context injection** → graph answer + compact world summary + ungraphed memories injected into user message
3. **Memory-first routing** → if memory answers the question, web search and ReAct loop are skipped
4. **Web augmentation** → if limbiq has few memories, web search fills the gap
5. **ReAct tools** → model can call 13 tools during response generation (only when memory is insufficient)
6. **Async observe** → limbiq observes the exchange in a background thread (doesn't block response)
7. **Entity extraction** → limbiq extracts entities and relationships from every exchange into the graph
8. **Compression** → on `/quit`, conversations are compressed, graph inference runs, stale memories are suppressed
9. **LoRA training** → when enough conversations accumulate, a LoRA adapter trains in the background
10. **Next conversation** → the model remembers you, reasons about relationships, and gets better over time

## Migration

If upgrading from the pre-limbiq version (with `data/memory.db` and `data/chroma/`):

```bash
python migrate_to_limbiq.py
```

This migrates long-term memories as priority, mid-term as observed, and web knowledge with `[Web]` tags. After migration you can delete the old `data/memory.db` and `data/chroma/` directories.
