"""
Consolidation — simplified for limbiq.

Limbiq handles memory compression and cleanup via lq.end_session().
This script wraps that for standalone/scheduled usage, and optionally
triggers LoRA training.

Usage:
    python consolidate.py              # Run once
    python consolidate.py --stats      # Show memory stats only
    python consolidate.py --train      # Consolidate + trigger LoRA training
"""

import time
import argparse

from limbiq import Limbiq
from llm_backend import LLMBackend
from memory.training_data import prepare_training_data, count_new_conversations
from training.adapter_manager import AdapterManager
from training.lora_trainer import LoRATrainer
import config


def _make_llm_fn(llm: LLMBackend):
    """Create the compression adapter function for limbiq."""
    def compress_fn(prompt: str) -> str:
        return llm.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
    return compress_fn


def run_consolidation(verbose: bool = True, train: bool = False):
    """
    Run a single consolidation cycle via limbiq.

    If train=True, kick off LoRA training after consolidation.
    """
    if verbose:
        print("Initializing consolidation...")

    llm = LLMBackend()
    lq = Limbiq(
        store_path=config.LIMBIQ_STORE_PATH,
        user_id=config.USER_ID,
        embedding_model=config.EMBEDDING_MODEL,
        llm_fn=_make_llm_fn(llm),
    )

    if verbose:
        stats = lq.get_stats()
        print(f"Before: {stats}")

    results = lq.end_session()

    if verbose:
        print(f"\nResults:")
        print(f"  Compressed: {results.get('compressed', 0)} facts")
        print(f"  Suppressed: {results.get('suppressed', 0)} stale memories")
        print(f"  Deleted: {results.get('deleted', 0)} old suppressed")

        stats = lq.get_stats()
        print(f"\nAfter: {stats}")

    # LoRA training step
    should_train = train or config.LORA_AUTO_TRAIN
    if should_train and config.LORA_ENABLED:
        new_convos = count_new_conversations(lq._core.store)
        adapter_manager = AdapterManager()
        if adapter_manager.should_auto_train(new_convos):
            if verbose:
                print(f"\n  {new_convos} new conversation(s) ready for LoRA training...")
            training_dir = prepare_training_data(lq._core.store)
            if training_dir:
                trainer = LoRATrainer()
                if not trainer.is_available():
                    if verbose:
                        print("  mlx_lm not installed — skipping training.")
                else:
                    adapter_path = trainer.train(training_dir, num_conversations=new_convos)
                    if adapter_path:
                        adapter_manager.on_training_complete(adapter_path)
                        results["adapter_trained"] = adapter_path
                        if verbose:
                            print(f"  Adapter saved: {adapter_path}")
            elif verbose:
                print(f"  Not enough quality data yet (need {config.LORA_MIN_CONVERSATIONS} conversations)")
        elif verbose and config.LORA_AUTO_TRAIN:
            remaining = config.LORA_MIN_CONVERSATIONS - new_convos
            print(f"\n  LoRA: {new_convos}/{config.LORA_MIN_CONVERSATIONS} conversations — need {remaining} more")

    return results


def show_stats():
    """Display current memory statistics."""
    lq = Limbiq(
        store_path=config.LIMBIQ_STORE_PATH,
        user_id=config.USER_ID,
    )

    stats = lq.get_stats()
    print(f"\n{'=' * 40}")
    print(f"  Living LLM — Memory Statistics (Limbiq)")
    print(f"{'=' * 40}")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print(f"{'=' * 40}")

    priority = lq.get_priority_memories()
    if priority:
        print(f"\n  Priority memories (Dopamine-tagged):")
        for m in priority:
            print(f"    {m.content[:200]}")

    suppressed = lq.get_suppressed()
    if suppressed:
        print(f"\n  Suppressed memories: {len(suppressed)}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory consolidation (limbiq)")
    parser.add_argument("--stats", action="store_true", help="Show memory stats only")
    parser.add_argument("--train", action="store_true", help="Consolidate + LoRA training")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    else:
        run_consolidation(train=args.train)
