# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Base dependencies (macOS/Apple Silicon)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install limbiq rich gradio huggingface-hub

# LoRA training (Apple Silicon — pulls mlx as dependency)
pip install mlx-lm

# Web search, page extraction, tools
pip install ddgs trafilatura requests
```

Models (GGUF format) must be placed in `models/`. The active model is set via `MODEL_PATH` in `config.py`.

On first `/train`, mlx-lm downloads `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (~4.5 GB) to the HuggingFace cache.

## Running

```bash
python main.py          # Terminal chat
python main.py --ui     # Gradio web UI
python consolidate.py   # Single consolidation cycle
python consolidate.py --stats  # Memory stats
python consolidate.py --train  # Consolidate + trigger LoRA training
```

**In-chat commands:** `/memory`, `/signals`, `/priority`, `/suppress`, `/dopamine <fact>`, `/gaba <id>`, `/correct <info>`, `/good`, `/bad`, `/restore <id>`, `/search <query>`, `/export`, `/new`, `/quit`, `/train`, `/adapter`, `/adapter compare`, `/adapter off`, `/adapter on`

## Architecture

Living LLM is a **thin demo application** for [limbiq](https://github.com/deBilla/limbiq). All memory, learning, and adaptation logic lives in the limbiq library. The app provides:
- A chat interface (Gradio / terminal)
- An LLM backend (llama.cpp + optional MLX for LoRA)
- A web search module
- Wiring that connects them through limbiq's 3-method API: `process -> LLM -> observe`

**Core loop (`engine.py` -> `respond()`):**
1. `lq.process(message)` -> limbiq returns enriched memory context
2. Optionally augment with web search if limbiq found few memories
3. Build prompt: system + memory context + web context + user message
4. Memory context injected into **user message** (not just system prompt) — 8B models ignore system prompt content but attend to user turns
5. ReAct loop runs tool calls (web_search, python, weather, etc.)
6. `lq.observe(message, response)` -> limbiq fires signals and stores exchange
7. On session end: `lq.end_session()` -> compression + stale memory suppression

**Limbiq signals:**
- **Dopamine** — "This matters, remember it." Fires on personal info, corrections, positive feedback. Tagged memories always surface in context.
- **GABA** — "Suppress this, let it fade." Fires on denials, contradictions, stale memories. Suppression is soft and reversible.

**Key design decision — memory injection location:** Context is injected into the user message as `"Here is what you remember about me... Now answer this: {user_input}"`. This forces the 8B model to use stored memories.

**Web search integration (`tools/web_augment.py`):**
- When limbiq returns low-confidence results, `WebAugmenter` triggers a web search
- Facts extracted from search results are stored via `lq.dopamine()` with `[Web]` prefix
- ReAct loop (`tools/react_loop.py`) handles in-conversation tool calls (13 tools)

**ReAct tool loop (`tools/react_loop.py`):**
- 13 tools: `web_search`, `read_page`, `datetime`, `python`, `read_file`, `write_file`, `list_files`, `shell`, `weather`, `wikipedia`, `notify`, `http_get`, `http_post`
- Tools dispatched via `_execute()`, results wrapped in `<tool_result>` tags
- Session-scoped rate limiting (max 20 searches/session, 2s cooldown)

**LoRA neuroplasticity (`training/`):**
- After >=3 compressed conversations, LoRA adapter trains via `mlx_lm lora` (note: NOT `mlx_lm.lora`)
- Training config passed via JSON file with `-c` flag: `lora_parameters: {rank, scale, dropout}`
- `MLXBackend` used for inference when adapter active; uses `make_sampler(temp=...)` for generation
- MLX backend strips leaked Llama 3.1 template tokens (`<|eot_id|>`, `<|start_header_id|>`, etc.) post-generation
- Training data lifecycle: `prepare_training_data()` writes `pending_ids.json`; `mark_training_complete()` marks as used only on success (prevents failed training from consuming data)
- Training data module accesses limbiq's store via `lq._core.store.db` for conversation data

## Configuration

All tunable parameters are in `config.py`:
- `MODEL_PATH`, `N_CTX` (8192), `N_GPU_LAYERS`, `TEMPERATURE`, `MAX_TOKENS`
- `LIMBIQ_STORE_PATH`, `USER_ID`, `EMBEDDING_MODEL`
- `LORA_ENABLED`, `LORA_MIN_CONVERSATIONS` (3), `LORA_AUTO_TRAIN`, `LORA_MAX_ADAPTERS`
- `LORA_RANK`, `LORA_LORA_LAYERS`, `LORA_LEARNING_RATE`, `MLX_MODEL_ID`
- Web search: `WEB_SEARCH_ENABLED`, `SEARCH_BACKEND`, `SEARCH_MAX_RESULTS`

## Confabulation Test Suite

```bash
python eval_confabulation.py --phase baseline  # Raw model, no memory
python eval_confabulation.py --phase memory     # Memory system active
python eval_confabulation.py --phase lora       # Memory + LoRA adapter
python eval_confabulation.py --phase compare    # Side-by-side comparison
python eval_confabulation.py --phase show --show-phase memory  # Inspect responses
```

Results saved to `data/eval_results/`. Test categories: `false_memory` (1.x), `confab_theater` (3.x), `false_premise` (5.x).

## Data Persistence

- `data/limbiq/` — Limbiq's persistent storage (SQLite + embeddings, auto-managed)
- `data/training/` — JSONL training files, `pending_ids.json`, `used_ids.json`
- `data/adapters/` — LoRA adapters; `active_adapter.json` tracks current
- `data/metrics/` — evaluation log (`metrics.jsonl`)
- `data/eval_results/` — confabulation test results (JSON)

All directories auto-created. Deleting `data/` resets all state.

## Migration from Old Memory System

If you have existing data from the pre-limbiq version (data/memory.db + data/chroma/):
```bash
python migrate_to_limbiq.py   # Migrates old data to limbiq
```

## Project Structure

```
living-llm/
├── main.py                 # Chat interface (terminal + Gradio)
├── engine.py               # Thin orchestrator using limbiq
├── llm_backend.py          # LLM wrapper (llama-cpp + MLX)
├── config.py               # All configuration
├── consolidate.py          # Standalone consolidation + training trigger
├── migrate_to_limbiq.py    # One-time migration script
├── tools/
│   ├── react_loop.py       # ReAct tool dispatch (13 tools)
│   ├── web_augment.py      # Bridge between limbiq and web search
│   ├── web_search.py       # DuckDuckGo/SearXNG search
│   ├── web_reader.py       # Page content extraction
│   └── ... (datetime, python, file, shell, weather, wikipedia, notify, http)
├── memory/
│   └── training_data.py    # Conversation -> JSONL for LoRA
├── training/
│   ├── adapter_manager.py  # Adapter lifecycle
│   ├── lora_trainer.py     # MLX LoRA training
│   └── eval.py             # Base vs adapted comparison
└── data/
    ├── limbiq/             # Limbiq's persistent storage
    ├── training/           # LoRA training data
    ├── adapters/           # LoRA adapter checkpoints
    └── metrics/            # Evaluation logs
```
