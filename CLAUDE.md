# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Base dependencies (macOS/Apple Silicon)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install sentence-transformers chromadb rich gradio huggingface-hub

# LoRA training (Apple Silicon — pulls mlx as dependency)
pip install mlx-lm

# Web search and page extraction
pip install ddgs trafilatura requests
```

Models (GGUF format) must be placed in `models/`. The active model is set via `MODEL_PATH` in `config.py`.

On first `/train`, mlx-lm downloads `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (~4.5 GB) to the HuggingFace cache.

## Running

```bash
# Terminal chat interface
python main.py

# Gradio web UI
python main.py --ui

# Memory consolidation (single cycle)
python consolidate.py

# Memory consolidation (watch mode, runs every N minutes)
python consolidate.py --watch

# Show memory stats
python consolidate.py --stats
```

**In-chat commands:** `/memory`, `/search <query>`, `/knowledge`, `/knowledge clear`, `/knowledge decay`, `/new`, `/quit`, `/train`, `/adapter`, `/adapter compare`, `/adapter off`, `/adapter on`

```bash
# Consolidate + trigger LoRA training in one step
python consolidate.py --train
```

## Architecture

This is a proof-of-concept for persistent LLM memory without retraining — all memory is injected into the context window.

**Inference loop (`engine.py`):**
1. User input → semantic search in ChromaDB for relevant memories
2. Retrieved memories injected as `<memory_context>` block in system prompt
3. LLM (llama-cpp-python + Metal) generates response using last 6 turns
4. Exchange stored as short-term memory in SQLite + ChromaDB

**Consolidation pipeline (`consolidate.py` / `memory/compressor.py`):**
- Runs on exit or on a background interval
- Full conversations → 2–4 sentence **gists** (mid-term, expire after 30 sessions)
- 3+ gists → abstract **facts** (long-term, permanent)
- Mimics lossy human memory: details fade, meaning persists

**Memory tiers (`memory/store.py`):**
- `SHORT` — raw conversation turns, TTL: 3 sessions
- `MID` — LLM-compressed gists, TTL: 30 sessions
- `LONG` — abstract facts, permanent; boosted ×1.3 in retrieval scoring

**Retrieval scoring (`memory/retriever.py`):**
- Cosine similarity via ChromaDB, with tier boosts (long-term ×1.3, mid-term ×1.1) and access-frequency boost (×1.1)

**Web search + knowledge layer (`tools/` + `memory/web_knowledge.py`):**
- When `WEB_SEARCH_ENABLED = True`, the system prompt is extended with `TOOL_USE_PROMPT` listing two tools: `web_search()` and `read_page()`
- `respond()` runs through `ReactLoop` instead of calling the LLM directly; the loop intercepts `<tool_call>` tags, executes them, and feeds results back until the model produces a final answer (max 3 iterations)
- After any turn where searches happened, `WebKnowledgeExtractor` uses a short LLM call to distill 1-3 durable facts and stores them in the new `WEB` memory tier (SQLite + ChromaDB)
- WEB tier uses calendar-based TTL (not session-count): news facts expire in 7 days (`WEB_NEWS_TTL_DAYS`), general facts in 30 days (`WEB_KNOWLEDGE_TTL_DAYS`). Confidence decays toward zero and the fact is deleted at ≤0.1
- In retrieval, WEB memories appear between MID and SHORT: scored by `relevance × confidence`, up to 2 slots, displayed with source URL and confidence percentage so the model treats them as provisional
- `/search <query>` bypasses the ReAct loop and directly injects search results (more reliable for forced searches)
- `duckduckgo-search` was renamed to `ddgs` — use `pip install ddgs`

**LoRA neuroplasticity layer (`training/`):**
- After ≥5 conversations are compressed, training data is prepared and a LoRA adapter is trained via `mlx_lm.lora`
- Training uses a separate MLX-format model (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`) — the GGUF is untouched
- When an adapter is active, the engine uses `MLXBackend` (in `llm_backend.py`) instead of `LLMBackend`; both expose the same `chat()` interface
- Training runs in a background thread; the new adapter is picked up on the next user message
- `training/adapter_manager.py` tracks which adapter is active (persists across sessions) and enforces rollback retention (`LORA_MAX_ADAPTERS`)
- `training/eval.py` logs response comparisons to `data/metrics/metrics.jsonl` for drift detection

## Configuration

All tunable parameters are in `config.py`:
- `MODEL_PATH`, `N_CTX`, `N_GPU_LAYERS`, `TEMPERATURE`, `MAX_TOKENS`
- `SHORT_TERM_TTL`, `MID_TERM_TTL`, `TOP_K_MEMORIES`, `RELEVANCE_THRESHOLD`
- `GIST_MAX_TOKENS`, `FACT_MAX_TOKENS`, `CONSOLIDATE_INTERVAL_MINS`
- `LORA_ENABLED`, `LORA_MIN_CONVERSATIONS`, `LORA_AUTO_TRAIN`, `LORA_MAX_ADAPTERS`
- `LORA_RANK`, `LORA_LORA_LAYERS`, `LORA_LEARNING_RATE`, `MLX_MODEL_ID`

## Confabulation Test Suite

`eval_confabulation.py` validates that the system doesn't fabricate memories or perform fake self-reflection. Run it across three phases to isolate what each layer contributes:

```bash
# Phase 0: raw model, no memory context
python eval_confabulation.py --phase baseline

# Phase 1: memory system active, no adapter
python eval_confabulation.py --phase memory

# Phase 2: memory + LoRA adapter
python eval_confabulation.py --phase lora

# Side-by-side comparison of all phases
python eval_confabulation.py --phase compare

# Inspect full responses for a phase
python eval_confabulation.py --phase show --show-phase memory
python eval_confabulation.py --phase show --show-phase memory --test-id 1.1
```

Results are saved to `data/eval_results/` as JSON. Expected progression: baseline ~20–40% → memory-only ~50–70% → LoRA ~70–90%. The memory→LoRA delta measures what weight adaptation contributed beyond prompt engineering.

**Test categories** (`ALL_TESTS` in `eval_confabulation.py`):
- `false_memory` (1.x) — prompts that tempt the model to invent past conversations
- `confab_theater` (3.x) — prompts that catch fake self-reflection about fake memory processes
- `false_premise` (5.x) — prompts that assert things that never happened and check if the model agrees

**Scoring**: PASS = good signal found, no bad signal. FAIL = bad signal found, no good. MIXED = both. UNCLEAR = neither. Signals are case-insensitive substring matches defined per-test.

**System prompt guardrails** (`config.py → SYSTEM_PROMPT`): Seven explicit rules injected at every turn — no fabricating memories, no describing fake processes, no agreeing with false premises, honest "I don't have that" responses.

## Data Persistence

- `data/memory.db` — SQLite (memory records + raw conversations)
- `data/chroma/` — ChromaDB vector store (embeddings for semantic search)
- `data/training/` — JSONL training files (`train.jsonl`, `valid.jsonl`, batch metadata, `used_ids.json`)
- `data/adapters/` — saved LoRA adapters; `active_adapter.json` tracks which is loaded
- `data/metrics/` — evaluation log (`metrics.jsonl`)

- `data/eval_results/` — confabulation test results per phase (JSON)

All directories are created automatically. Deleting `data/` resets all state including adapters.
