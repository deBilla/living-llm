"""
Evaluation & Monitoring — tracks whether LoRA adaptation is actually helping.

The core question: does the adapted model respond differently in ways that fit
this user's patterns better than the base model?

Metrics tracked:
- Response similarity (Jaccard on tokens): how much does adaptation change responses?
  Too similar = not working. Too different = potential drift.
- Drift detection: flag if base and adapted responses diverge sharply (similarity < 0.1)
- Training history: log each training run for longitudinal analysis

Intentionally lightweight — this is a POC, not a benchmark harness.
All metrics are appended to data/metrics/metrics.jsonl as newline-delimited JSON.
"""

import json
import time
from pathlib import Path
from typing import Optional

import config


class AdapterEvaluator:
    def __init__(self):
        self.metrics_dir = Path(config.LORA_METRICS_DIR)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def compare_responses(
        self,
        prompt: str,
        base_backend,
        adapted_backend,
        system_prompt: str = "",
    ) -> dict:
        """
        Run the same prompt through base and adapted model at temperature=0
        (deterministic), then compute similarity.

        temperature=0 removes randomness so differences reflect the adaptation,
        not sampling noise.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        base_response = base_backend.chat(messages, max_tokens=256, temperature=0.01)
        adapted_response = adapted_backend.chat(messages, max_tokens=256, temperature=0.01)

        similarity = self._token_overlap(base_response, adapted_response)
        drift_detected = similarity < 0.1

        return {
            "prompt": prompt,
            "base_response": base_response,
            "adapted_response": adapted_response,
            "similarity_score": round(similarity, 3),
            "drift_detected": drift_detected,
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def log_comparison(self, comparison: dict, adapter_path: str = ""):
        """Persist a comparison result."""
        self._append({
            "type": "comparison",
            "adapter": adapter_path,
            **comparison,
        })

    def log_training_event(self, adapter_path: str, num_conversations: int, training_iters: int):
        """Record that a training run occurred."""
        self._append({
            "type": "training",
            "adapter_path": adapter_path,
            "num_conversations": num_conversations,
            "training_iters": training_iters,
            "timestamp": time.time(),
            "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    def load_metrics(self, limit: int = 50) -> list[dict]:
        """Load logged metrics, most recent first."""
        metrics_file = self.metrics_dir / "metrics.jsonl"
        if not metrics_file.exists():
            return []
        entries = []
        for line in metrics_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return list(reversed(entries))[:limit]

    def get_summary(self) -> dict:
        """Brief summary of evaluation history for display."""
        metrics = self.load_metrics(limit=200)
        trainings = [m for m in metrics if m.get("type") == "training"]
        comparisons = [m for m in metrics if m.get("type") == "comparison"]

        summary: dict = {
            "total_training_runs": len(trainings),
            "total_comparisons": len(comparisons),
        }
        if comparisons:
            avg_sim = sum(c.get("similarity_score", 0) for c in comparisons) / len(comparisons)
            summary["avg_response_similarity"] = round(avg_sim, 3)
        if trainings:
            summary["last_training"] = trainings[0].get("timestamp_human", "?")
            summary["total_conversations_trained"] = sum(
                t.get("num_conversations", 0) for t in trainings
            )
        return summary

    def _token_overlap(self, a: str, b: str) -> float:
        """
        Jaccard similarity on whitespace-tokenized text.
        Fast proxy for semantic similarity — good enough for drift detection.
        """
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _append(self, entry: dict):
        metrics_file = self.metrics_dir / "metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
