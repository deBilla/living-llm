# Living LLM — Proof of Concept

A proof-of-concept implementation of the "Next-Level LLM" architecture: a locally-running language model with persistent memory, lossy compression, and the ability to learn from every conversation.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Chat Interface                 │
│               (Terminal / Gradio UI)             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│              Conversation Engine                  │
│  Manages turns, injects relevant memory context  │
└──────────┬───────────────────────┬───────────────┘
           │                       │
┌──────────▼──────────┐ ┌─────────▼───────────────┐
│   LLM Backend       │ │   Memory System          │
│   (llama.cpp via    │ │                          │
│    llama-cpp-python) │ │  ┌───────────────────┐  │
│                      │ │  │ Short-term buffer  │  │
│  Quantized Llama 3.1 │ │  │ (full fidelity)   │  │
│  8B Q4_K_M           │ │  ├───────────────────┤  │
│                      │ │  │ Mid-term memory    │  │
│                      │ │  │ (compressed gists) │  │
│                      │ │  │ via LLM extraction │  │
│                      │ │  ├───────────────────┤  │
│                      │ │  │ Long-term memory   │  │
│                      │ │  │ (abstract facts &  │  │
│                      │ │  │  core knowledge)   │  │
│                      │ │  └───────────────────┘  │
│                      │ │                          │
│                      │ │  Relevance scoring       │
│                      │ │  (embedding similarity)  │
└──────────────────────┘ └─────────────────────────┘
                                    │
                       ┌────────────▼─────────────┐
                       │   Consolidation Daemon    │
                       │   (runs periodically)     │
                       │                           │
                       │   Short → Mid → Long      │
                       │   compression pipeline    │
                       └───────────────────────────┘
```

## Memory Tiers (Your "Lossy Compression" Insight)

| Tier | Fidelity | Example | TTL |
|------|----------|---------|-----|
| **Short-term** | Full conversation turns | "User asked about TCP, I explained the 3-way handshake" | Last 3 conversations |
| **Mid-term** | Compressed gists | "User is interested in networking protocols, has intermediate knowledge" | 30 days |
| **Long-term** | Abstract facts | "User: strong developer, interested in AI/ML and systems design" | Permanent |

Each tier compresses the one above it. Details fade, meaning persists — exactly like human memory.

## Setup

### Prerequisites

- macOS with Apple Silicon (M4 Pro recommended)
- Python 3.11+
- ~6GB disk space for model

### Step 1: Create environment

```bash
mkdir living-llm && cd living-llm
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install sentence-transformers chromadb rich gradio
```

The `metal` wheel enables GPU acceleration on Apple Silicon — this is critical for performance.

### Step 3: Download model

```bash
# Using Hugging Face CLI
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir models/

# Or for Llama 3.1 8B (better quality, recommended if available)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --local-dir models/
```

### Step 4: Run

```bash
# Terminal mode
python main.py

# With Gradio web UI
python main.py --ui
```

## Project Structure

```
living-llm/
├── main.py                 # Entry point — chat loop & UI
├── engine.py               # Conversation engine — orchestrates LLM + memory
├── llm_backend.py          # LLM wrapper (llama-cpp-python)
├── memory/
│   ├── __init__.py
│   ├── store.py            # Memory store — SQLite + ChromaDB
│   ├── compressor.py       # Compression pipeline (short → mid → long)
│   └── retriever.py        # Relevance-based memory retrieval
├── consolidate.py          # Consolidation daemon (the "sleep" cycle)
├── config.py               # All configuration in one place
├── models/                 # Downloaded GGUF models go here
└── data/                   # SQLite DB + ChromaDB persist here
```

## How It Works

1. **You chat** → conversation is stored in short-term memory at full fidelity
2. **Memory retrieval** → before each response, the system searches all memory tiers for relevant context and injects it into the prompt
3. **Compression** → after a conversation ends, the LLM itself extracts gists (mid-term) and abstract facts (long-term) from the full conversation
4. **Consolidation** → periodically, old short-term memories compress into mid-term, old mid-term compress into long-term. Details fade, meaning persists.
5. **Next conversation** → the model "remembers" you — not perfectly, but meaningfully. Like a human would.

## What This Proves

- An LLM can accumulate knowledge across conversations without retraining
- Lossy compression of memory is not just possible but beneficial — gists are more useful than raw transcripts for context injection
- The base model (frozen weights) + dynamic memory system is a viable architecture for continuous learning
- Memory retrieval by relevance mimics how human recall is triggered by association, not exhaustive search
# living-llm
