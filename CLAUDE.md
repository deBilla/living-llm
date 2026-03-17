# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Base dependencies (macOS/Apple Silicon)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install sentence-transformers chromadb rich gradio huggingface-hub

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
python consolidate.py --watch  # Background consolidation loop
python consolidate.py --stats  # Memory stats
python consolidate.py --train  # Consolidate + trigger LoRA training
```

**In-chat commands:** `/memory`, `/search <query>`, `/recall <query>`, `/sharpen`, `/knowledge`, `/knowledge clear`, `/knowledge decay`, `/new`, `/quit`, `/train`, `/adapter`, `/adapter compare`, `/adapter off`, `/adapter on`

## Architecture

**Inference pipeline (`engine.py` → `respond()`):**
1. User input → augmented recall classifies memory as CLEAR / BLURRY / ABSENT
2. BLURRY: memory seeds a targeted web search, LLM merges memory + web results
3. ABSENT + searchable: fresh web search
4. Memory context injected into **user message** (not system prompt) — 8B models ignore system prompt content but attend to user turns
5. ReAct loop runs: model can call tools via `<tool_call>` XML tags, loop intercepts and executes, feeds results back (max 3 iterations)
6. Exchange stored in short-term memory; web knowledge extracted if searches happened

**Key design decision — memory injection location:** Context is injected into the user message as `"Here is what you remember about me... Now answer this: {user_input}"`. This forces the 8B model to use stored memories. Section headers use natural language (not tags like `[KNOWN FACTS]`) to prevent the model from leaking formatting into responses.

**Augmented recall (`tools/augmented_recall.py` + `memory/confidence.py`):**
- `MemoryConfidenceClassifier` uses a fast heuristic path (score > 0.6 → CLEAR, score < 0.2 → ABSENT) with LLM evaluation for borderline cases
- BLURRY memories are used as search seeds for targeted web queries, then LLM merges memory + web results with `[memory]`/`[web]`/`[corrected]` tags
- User feedback (confirmation/denial) adjusts memory confidence scores

**Consolidation pipeline (`consolidate.py` / `memory/compressor.py`):**
- Full conversations → **atomic facts** via `extract_atomic_facts()` (one fact per MID memory entry for better embedding search)
- 3+ facts → abstract **long-term facts** (permanent)
- Details fade, meaning persists — lossy like human memory

**Memory tiers (`memory/store.py`):**
- `SHORT` — raw conversation turns, TTL: 3 sessions
- `MID` — atomic facts (one per entry), TTL: 30 sessions
- `LONG` — abstract facts, permanent; boosted x1.3 in retrieval
- `WEB` — web-learned facts, calendar TTL (7-30 days), confidence decay

**SQLite thread safety:** Gradio runs handlers in worker threads. `MemoryStore` uses `threading.local()` for per-thread SQLite connections to avoid `InterfaceError`.

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

## Configuration

All tunable parameters are in `config.py`:
- `MODEL_PATH`, `N_CTX` (8192), `N_GPU_LAYERS`, `TEMPERATURE`, `MAX_TOKENS`
- `SHORT_TERM_TTL`, `MID_TERM_TTL`, `TOP_K_MEMORIES`, `RELEVANCE_THRESHOLD` (0.1)
- `LORA_ENABLED`, `LORA_MIN_CONVERSATIONS` (3), `LORA_AUTO_TRAIN`, `LORA_MAX_ADAPTERS`
- `LORA_RANK`, `LORA_LORA_LAYERS`, `LORA_LEARNING_RATE`, `MLX_MODEL_ID`

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

- `data/memory.db` — SQLite (memory records + raw conversations)
- `data/chroma/` — ChromaDB vector store (embeddings)
- `data/training/` — JSONL training files, `pending_ids.json`, `used_ids.json`
- `data/adapters/` — LoRA adapters; `active_adapter.json` tracks current
- `data/metrics/` — evaluation log (`metrics.jsonl`)
- `data/eval_results/` — confabulation test results (JSON)

All directories auto-created. Deleting `data/` resets all state.
