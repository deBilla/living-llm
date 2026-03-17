"""
Training Data Preparation — converts stored conversations to LoRA training format.

This is the bridge between the memory system and the learning system.

Why only compressed conversations?
The compression pipeline has already "voted" on these conversations as worth
remembering. If an exchange was interesting enough to survive into a gist, the
interaction pattern is worth reinforcing via fine-tuning. This acts as a natural
quality filter — we only learn from what mattered.

Training format: Llama 3.1 chat template with "text" field in JSONL.
mlx_lm.lora expects train.jsonl + valid.jsonl in the data directory.
"""

import json
import time
from pathlib import Path
from typing import Optional

import config

# Llama 3.1 special tokens
_BOS = "<|begin_of_text|>"
_START_HEADER = "<|start_header_id|>"
_END_HEADER = "<|end_header_id|>"
_EOT = "<|eot_id|>"


def _format_llama3_chat(messages: list[dict]) -> str:
    """
    Format a message list into Llama 3.1 chat template format.

    This is the exact format the model was instruction-tuned on. Adapting using
    the same template keeps us aligned with base pretraining — the adapter learns
    to modulate content, not fight the tokenizer's learned structure.
    """
    text = _BOS
    for msg in messages:
        text += f"{_START_HEADER}{msg['role']}{_END_HEADER}\n\n{msg['content']}{_EOT}\n"
    # Signal that the model should now generate the assistant turn
    text += f"{_START_HEADER}assistant{_END_HEADER}\n\n"
    return text


def _is_quality_exchange(messages: list[dict]) -> bool:
    """
    Filter out exchanges not worth learning from.

    Quality signals:
    - At least one full user/assistant exchange
    - Non-trivial total content (rules out test messages)
    - No suspiciously short assistant responses (likely errors)
    """
    user_msgs = [m for m in messages if m["role"] == "user"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    if not user_msgs or not assistant_msgs:
        return False

    total_content = sum(len(m["content"]) for m in messages if m["role"] != "system")
    if total_content < 100:
        return False

    # Any near-empty assistant response suggests something went wrong
    if any(len(m["content"]) < 20 for m in assistant_msgs):
        return False

    return True


def _extract_turn_pairs(messages: list[dict]) -> list[tuple[str, str, dict | None]]:
    """
    Extract (user, assistant, system) turn tuples from a conversation.

    Each turn pair becomes one training example. We include the system message
    (which contains any memory context that was active at the time) so the adapter
    learns to USE memory context naturally — not just respond, but respond with memory.
    """
    system_msg = next((m for m in messages if m["role"] == "system"), None)
    pairs = []
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            pairs.append((messages[i]["content"], messages[i + 1]["content"], system_msg))
    return pairs


def _get_used_conversation_ids() -> set:
    """Load the set of conversation IDs already used for training."""
    used_file = Path(config.LORA_TRAINING_DATA_DIR) / "used_ids.json"
    if used_file.exists():
        try:
            return set(json.loads(used_file.read_text()))
        except Exception:
            pass
    return set()


def _mark_conversations_used(conv_ids: list[str]):
    """Record that these conversation IDs have been incorporated into training data."""
    used_file = Path(config.LORA_TRAINING_DATA_DIR) / "used_ids.json"
    existing = _get_used_conversation_ids()
    existing.update(conv_ids)
    used_file.write_text(json.dumps(list(existing)))


def count_new_conversations(store) -> int:
    """Return how many compressed conversations haven't been used for training yet."""
    used_ids = _get_used_conversation_ids()
    rows = store.conn.execute(
        "SELECT id FROM conversations WHERE compressed = 1"
    ).fetchall()
    return sum(1 for (row_id,) in rows if row_id not in used_ids)


def prepare_training_data(store) -> Optional[str]:
    """
    Pull compressed conversations from the store and write them as JSONL training files.

    Returns the path to the training data directory (containing train.jsonl + valid.jsonl),
    or None if there isn't enough new data.

    mlx_lm.lora requires:
      - <data_dir>/train.jsonl  — 80% of examples
      - <data_dir>/valid.jsonl  — 20% of examples (or at least 1 example)
    Each line: {"text": "<full formatted prompt>"}
    """
    training_dir = Path(config.LORA_TRAINING_DATA_DIR)
    training_dir.mkdir(parents=True, exist_ok=True)

    used_ids = _get_used_conversation_ids()

    rows = store.conn.execute(
        "SELECT id, messages FROM conversations WHERE compressed = 1 ORDER BY created_at ASC"
    ).fetchall()

    new_conversations = [
        (row[0], json.loads(row[1]))
        for row in rows
        if row[0] not in used_ids
    ]

    if len(new_conversations) < config.LORA_MIN_CONVERSATIONS:
        return None

    examples = []
    used_conv_ids = []

    for conv_id, messages in new_conversations:
        if not _is_quality_exchange(messages):
            continue

        pairs = _extract_turn_pairs(messages)

        for user_content, assistant_content, system_msg in pairs:
            turn_messages = []
            if system_msg:
                turn_messages.append(system_msg)
            turn_messages.append({"role": "user", "content": user_content})
            turn_messages.append({"role": "assistant", "content": assistant_content})

            formatted = _format_llama3_chat(turn_messages)
            # Append EOS after the assistant response so the model learns where to stop
            formatted += assistant_content + _EOT
            examples.append({"text": formatted})

        used_conv_ids.append(conv_id)

    if not examples:
        return None

    # 80/20 split — valid needs at least 1 example
    split_idx = max(1, int(len(examples) * 0.8))
    train_examples = examples[:split_idx]
    valid_examples = examples[split_idx:] if len(examples) > split_idx else examples[:1]

    (training_dir / "train.jsonl").write_text("\n".join(json.dumps(e) for e in train_examples))
    (training_dir / "valid.jsonl").write_text("\n".join(json.dumps(e) for e in valid_examples))

    # Save batch metadata for auditing
    timestamp = int(time.time())
    batch_meta = {
        "timestamp": timestamp,
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conversations_used": len(used_conv_ids),
        "training_examples": len(train_examples),
        "validation_examples": len(valid_examples),
    }
    (training_dir / f"batch_{timestamp}.json").write_text(json.dumps(batch_meta, indent=2))

    # Don't mark as used yet — wait until training actually succeeds.
    # Save the IDs so the caller can mark them after success.
    (training_dir / "pending_ids.json").write_text(json.dumps(used_conv_ids))

    return str(training_dir)


def mark_training_complete(training_data_dir: str):
    """Mark pending conversations as used. Call only after training succeeds."""
    pending_file = Path(training_data_dir) / "pending_ids.json"
    if pending_file.exists():
        conv_ids = json.loads(pending_file.read_text())
        _mark_conversations_used(conv_ids)
        pending_file.unlink()


def get_training_stats() -> dict:
    """Return statistics about training data history."""
    used_ids = _get_used_conversation_ids()
    training_dir = Path(config.LORA_TRAINING_DATA_DIR)
    batches = list(training_dir.glob("batch_*.json")) if training_dir.exists() else []
    return {
        "conversations_used_total": len(used_ids),
        "training_batches": len(batches),
    }
