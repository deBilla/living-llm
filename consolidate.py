"""
Consolidation Daemon — the "sleep cycle".

Can be run standalone to process accumulated memories in the background.
This is separate from the main chat loop so it can run on a schedule.

Usage:
    python consolidate.py              # Run once
    python consolidate.py --watch      # Run continuously every N minutes
    python consolidate.py --stats      # Just show memory stats
    python consolidate.py --train      # Run consolidation then trigger LoRA training
"""

import time
import argparse

from llm_backend import LLMBackend
from memory.store import MemoryStore
from memory.compressor import MemoryCompressor
from memory.training_data import prepare_training_data, count_new_conversations
from training.adapter_manager import AdapterManager
from training.lora_trainer import LoRATrainer
import config


def run_consolidation(verbose: bool = True, train: bool = False):
    """
    Run a single consolidation cycle.

    If train=True (or LORA_AUTO_TRAIN is set and enough data exists),
    kick off LoRA training after compression. Training runs synchronously
    when called from the CLI — the daemon waits for it to complete.
    """
    if verbose:
        print("Initializing consolidation...")

    store = MemoryStore()
    llm = LLMBackend()
    compressor = MemoryCompressor(store, llm)

    if verbose:
        stats = store.get_stats()
        print(f"Before: {stats['short']} short / {stats['mid']} mid / {stats['long']} long-term")

    results = compressor.run_full_cycle()

    if verbose:
        print(f"\nResults:")
        print(f"  Conversations compressed: {results['conversations_compressed']}")
        print(f"  Memories expired: {results['memories_expired']}")
        print(f"  Long-term consolidated: {results['long_term_consolidated']}")

        stats = store.get_stats()
        print(f"\nAfter: {stats['short']} short / {stats['mid']} mid / {stats['long']} long-term")

    # LoRA training step
    should_train = train or config.LORA_AUTO_TRAIN
    if should_train and config.LORA_ENABLED:
        new_convos = count_new_conversations(store)
        adapter_manager = AdapterManager()
        if adapter_manager.should_auto_train(new_convos):
            if verbose:
                print(f"\n  {new_convos} new conversation(s) ready for LoRA training...")
            training_dir = prepare_training_data(store)
            if training_dir:
                trainer = LoRATrainer()
                if not trainer.is_available():
                    if verbose:
                        print("  mlx_lm not installed — skipping training. Run: pip install mlx-lm")
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
            print(f"\n  LoRA: {new_convos}/{config.LORA_MIN_CONVERSATIONS} conversations — need {remaining} more to train")

    return results


def show_stats():
    """Display current memory statistics."""
    store = MemoryStore()
    stats = store.get_stats()

    print(f"\n{'=' * 40}")
    print(f"  Living LLM — Memory Statistics")
    print(f"{'=' * 40}")
    print(f"  Short-term memories:  {stats['short']}")
    print(f"  Mid-term gists:       {stats['mid']}")
    print(f"  Long-term facts:      {stats['long']}")
    print(f"  Stored conversations: {stats['conversations']}")
    print(f"{'=' * 40}")

    # Show long-term knowledge
    from memory.store import MemoryTier
    long_memories = store.get_memories_by_tier(MemoryTier.LONG)
    if long_memories:
        print(f"\n  Long-term knowledge:")
        for m in long_memories:
            print(f"    {m.content[:200]}")

    mid_memories = store.get_memories_by_tier(MemoryTier.MID)
    if mid_memories:
        print(f"\n  Recent gists ({len(mid_memories)}):")
        for m in mid_memories[:5]:
            print(f"    {m.content[:200]}")
    print()


def watch_mode():
    """Run consolidation on a loop."""
    interval = config.CONSOLIDATE_INTERVAL_MINS * 60
    print(f"Watching for new memories every {config.CONSOLIDATE_INTERVAL_MINS} minutes...")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            run_consolidation(verbose=True)
            print(f"\nSleeping {config.CONSOLIDATE_INTERVAL_MINS} minutes...\n")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory consolidation daemon")
    parser.add_argument("--watch", action="store_true", help="Run continuously")
    parser.add_argument("--stats", action="store_true", help="Show memory stats only")
    parser.add_argument("--train", action="store_true", help="Run consolidation then LoRA training")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.watch:
        watch_mode()
    else:
        run_consolidation(train=args.train)
