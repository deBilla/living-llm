"""
Adapter Manager — lifecycle management for LoRA adapters.

Tracks which adapters exist, which is active, and when to trigger new training.
Think of this as version control for the model's accumulated learning.

The rollback system is important: LoRA adaptation can degrade quality if trained
on too-short or off-topic conversations. Keeping the last N adapters means we can
quickly revert if the adapted model behaves worse than the base.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import config

_ACTIVE_FILE = "active_adapter.json"


class AdapterManager:
    """
    Manages the collection of trained LoRA adapters.

    Each adapter is a directory under data/adapters/ containing:
      - adapters.safetensors (or adapters.npz) — the learned weights
      - adapter_config.json                    — LoRA configuration
      - metadata.json                          — training provenance (our addition)
    """

    def __init__(self):
        self.adapter_dir = Path(config.LORA_ADAPTER_DIR)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    def list_adapters(self) -> list[dict]:
        """List all available adapters, most recent first."""
        adapters = []
        for path in sorted(self.adapter_dir.glob("adapter_*/"), reverse=True):
            meta_file = path / "metadata.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text())
                meta["path"] = str(path)
                meta["name"] = path.name
                adapters.append(meta)
            except Exception:
                pass
        return adapters

    def get_latest_adapter(self) -> Optional[str]:
        """Return the path to the most recently trained adapter."""
        adapters = self.list_adapters()
        return adapters[0]["path"] if adapters else None

    def get_active_adapter(self) -> Optional[str]:
        """
        Return the currently active adapter path, or None if disabled/absent.

        Reads from active_adapter.json so the choice persists across sessions.
        Returns None if the active adapter's files have been deleted.
        """
        active_file = self.adapter_dir / _ACTIVE_FILE
        if not active_file.exists():
            return None
        try:
            data = json.loads(active_file.read_text())
            path = data.get("path")
            if path and Path(path).exists():
                return path
        except Exception:
            pass
        return None

    def set_active_adapter(self, adapter_path: Optional[str]):
        """Activate a specific adapter (or clear the active adapter if None)."""
        active_file = self.adapter_dir / _ACTIVE_FILE
        active_file.write_text(json.dumps({
            "path": adapter_path,
            "set_at": time.time(),
        }))

    def get_active_metadata(self) -> Optional[dict]:
        """Return metadata dict for the currently active adapter."""
        path = self.get_active_adapter()
        if not path:
            return None
        meta_file = Path(path) / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta["path"] = path
                meta["name"] = Path(path).name
                return meta
            except Exception:
                pass
        return {"path": path, "name": Path(path).name}

    def on_training_complete(self, adapter_path: str):
        """
        Called after a successful training run.
        Activates the new adapter and prunes old ones.
        """
        self.set_active_adapter(adapter_path)
        self._cleanup_old_adapters()

    def should_auto_train(self, num_new_conversations: int) -> bool:
        """True if auto-train is enabled and we have enough fresh data."""
        if not config.LORA_AUTO_TRAIN or not config.LORA_ENABLED:
            return False
        return num_new_conversations >= config.LORA_MIN_CONVERSATIONS

    def get_status(self) -> dict:
        """Return a display-ready summary of adapter state."""
        active = self.get_active_adapter()
        meta = self.get_active_metadata()
        all_adapters = self.list_adapters()

        status = {
            "lora_enabled": config.LORA_ENABLED,
            "adapter_loaded": active is not None,
            "num_adapters": len(all_adapters),
        }
        if meta:
            status["adapter_name"] = meta.get("name", "?")
            status["trained_at"] = meta.get("created_at_human", "?")
            status["trained_on_conversations"] = meta.get("num_conversations", "?")
            status["training_iters"] = meta.get("training_iters", "?")

        if all_adapters and len(all_adapters) > 1:
            status["previous_adapters"] = len(all_adapters) - 1

        return status

    def _cleanup_old_adapters(self):
        """Remove adapters beyond LORA_MAX_ADAPTERS, keeping the newest."""
        adapters = self.list_adapters()
        active = self.get_active_adapter()
        to_delete = adapters[config.LORA_MAX_ADAPTERS:]
        for adapter in to_delete:
            # Never delete the currently active adapter
            if adapter["path"] == active:
                continue
            try:
                shutil.rmtree(adapter["path"])
            except Exception:
                pass
