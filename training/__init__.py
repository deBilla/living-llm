"""
LoRA training subsystem for Living LLM.

Components:
  LoRATrainer      — runs mlx_lm.lora training on accumulated conversation data
  AdapterManager   — lifecycle management: list, activate, rollback adapters
  AdapterEvaluator — lightweight metrics: response similarity, drift detection
"""

from training.lora_trainer import LoRATrainer
from training.adapter_manager import AdapterManager
from training.eval import AdapterEvaluator

__all__ = ["LoRATrainer", "AdapterManager", "AdapterEvaluator"]
