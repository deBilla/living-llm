"""
LoRA Training Pipeline — fine-tunes a LoRA adapter on accumulated conversation data.

Uses Apple's MLX framework via mlx-lm, which is purpose-built for Apple Silicon.
On an M4 Pro with 24GB RAM, training 50-100 examples takes roughly 3-8 minutes.

Why MLX over HuggingFace/PEFT?
- Native Apple Silicon GPU via Metal (not PyTorch's MPS, which has missing ops)
- 2-4x faster than HuggingFace for LoRA on M-series chips
- Tighter unified-memory management — no CPU/GPU copies
- mlx_lm.lora is a mature, well-tested CLI for exactly this workflow

Architecture note:
Training uses a separate MLX-quantized model (mlx-community/Meta-Llama-3.1-8B-Instruct-4bit),
downloaded automatically by mlx-lm on first use (~4.5 GB in HuggingFace cache).
The GGUF model (used for base inference) stays untouched.
The resulting adapter is in MLX safetensors format. When loaded, the engine
switches to MLXBackend for inference — slightly different inference stack but
same model weights + your adaptation layer on top.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import config


class LoRATrainer:
    """
    Manages the LoRA fine-tuning process using mlx-lm.

    Training flow:
      1. Receive path to training data directory (train.jsonl + valid.jsonl)
      2. Invoke mlx_lm.lora as a subprocess with configured hyperparameters
      3. Save adapter weights to data/adapters/{name}/ with metadata JSON
      4. Return adapter path for AdapterManager to activate
    """

    def __init__(self):
        self.adapter_dir = Path(config.LORA_ADAPTER_DIR)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def train(self, training_data_dir: str, num_conversations: int = 0) -> Optional[str]:
        """
        Run LoRA training synchronously. Blocks until complete (2-10 minutes).

        Returns the adapter directory path on success, None on failure.
        Training output (loss curves, etc.) streams directly to the terminal
        so the user can see progress.
        """
        adapter_name = f"adapter_{int(time.time())}"
        adapter_path = self.adapter_dir / adapter_name
        adapter_path.mkdir(parents=True, exist_ok=True)

        iters = self._compute_iters(num_conversations)

        print(f"  LoRA training: {num_conversations} conversation(s), {iters} iters")
        print(f"  Model: {config.MLX_MODEL_ID}")
        print(f"  Adapter → {adapter_path}")
        print("  (First run will download the MLX model ~4.5 GB if not cached)\n")

        # Write a LoRA config file — newer mlx_lm versions removed --rank
        # from CLI args and expect it in a config file instead.
        # Keys must match CONFIG_DEFAULTS in mlx_lm.lora exactly.
        lora_config = {
            "lora_parameters": {
                "rank": config.LORA_RANK,
                "scale": config.LORA_ALPHA / config.LORA_RANK,
                "dropout": 0.0,
            },
        }
        config_path = adapter_path / "lora_config.json"
        config_path.write_text(json.dumps(lora_config, indent=2))

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", config.MLX_MODEL_ID,
            "--data", training_data_dir,
            "--train",
            "--iters", str(iters),
            "--learning-rate", str(config.LORA_LEARNING_RATE),
            "--adapter-path", str(adapter_path),
            "--batch-size", str(config.LORA_BATCH_SIZE),
            "--grad-checkpoint",  # Trade speed for memory — important on 24GB
            "-c", str(config_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                timeout=1800,  # 30-minute hard cap
            )
        except subprocess.TimeoutExpired:
            print("  Training timed out (>30 min) — reduce LORA_LORA_LAYERS or data size")
            return None
        except FileNotFoundError:
            print("  mlx_lm not found. Install with: pip install mlx-lm")
            return None

        if result.returncode != 0:
            print(f"  Training failed (exit code {result.returncode})")
            return None

        # Confirm adapter files were written
        adapter_files = (
            list(adapter_path.glob("*.safetensors"))
            + list(adapter_path.glob("adapters.npz"))
        )
        if not adapter_files:
            print("  Training finished but no adapter weights found — something went wrong")
            return None

        self._save_metadata(adapter_path, num_conversations, iters)
        print(f"\n  Adapter saved: {adapter_path.name}")
        return str(adapter_path)

    def train_background(self, training_data_dir: str, num_conversations: int = 0, callback=None):
        """
        Start training in a background thread. Returns immediately.

        callback(adapter_path_or_None) is called on the background thread when done.
        The engine uses this to set a pending-adapter flag that gets picked up on
        the next user message (avoids cross-thread model swapping).
        """
        import threading

        def _run():
            path = self.train(training_data_dir, num_conversations)
            if callback:
                callback(path)

        thread = threading.Thread(target=_run, daemon=True, name="lora-trainer")
        thread.start()
        return thread

    def is_available(self) -> bool:
        """Check if mlx_lm is installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import mlx_lm"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _compute_iters(self, num_conversations: int) -> int:
        """
        Scale iterations with data size. More data = more iters, capped to avoid
        overfitting small datasets. Rule of thumb: ~15 iters per conversation.
        """
        return min(max(num_conversations * 15, 50), 600)

    def _save_metadata(self, adapter_path: Path, num_conversations: int, iters: int):
        """Persist training provenance alongside the adapter weights."""
        meta = {
            "created_at": time.time(),
            "created_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.MLX_MODEL_ID,
            "num_conversations": num_conversations,
            "training_iters": iters,
            "lora_rank": config.LORA_RANK,
            "lora_layers": config.LORA_LORA_LAYERS,
            "learning_rate": config.LORA_LEARNING_RATE,
        }
        (adapter_path / "metadata.json").write_text(json.dumps(meta, indent=2))
