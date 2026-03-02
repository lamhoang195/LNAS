"""Data loading for conversation datasets with binary labels."""

import json
import os
import random
from typing import Dict, List, Tuple


def _parse_conv(raw: List[Dict]) -> List[Dict]:
    """Convert [{"user": ...}, {"assistant": ...}, ...] to chat messages."""
    messages = []
    for turn in raw:
        if "user" in turn:
            messages.append({"role": "user", "content": turn["user"]})
        elif "assistant" in turn:
            messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages


def load_data(data_dir: str) -> List[Dict]:
    """Load all .jsonl files from directory.

    Each line must have:
      - ``label``: int (1 = harmful, 0 = benign)
      - ``conversation``  (single-key list of user/assistant dicts), OR
      - ``conversations`` (multi-turn version of the same)
    Returns a list of dicts: {"messages": [...], "label": int}
    """
    dataset = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                label = int(item["label"])
                raw = item.get("conversation") or item.get("conversations") or []
                messages = _parse_conv(raw)
                if messages:
                    dataset.append({"messages": messages, "label": label})
    return dataset


def load_data_split(
    data_dir: str,
    train_ratio: float = 0.80,
    eval_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List, List]:
    """Load all .jsonl files and split **per-file** into train/eval.

    Each file contributes train_ratio*100% to train and eval_ratio*100%
    to eval (the remaining fraction is discarded), so every file is
    represented in both sets.
    Returns (train_data, eval_data).
    """
    rng = random.Random(seed)
    train_all, eval_all = [], []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(data_dir, fname)
        file_items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                label = int(item["label"])
                raw = item.get("conversation") or item.get("conversations") or []
                messages = _parse_conv(raw)
                if messages:
                    file_items.append({"messages": messages, "label": label})

        if not file_items:
            continue

        rng.shuffle(file_items)
        n_eval  = max(1, int(len(file_items) * eval_ratio))
        n_train = int(len(file_items) * train_ratio)
        eval_all.extend(file_items[:n_eval])
        train_all.extend(file_items[n_eval:n_eval + n_train])

        print(f"  {fname}: {n_train} train / {n_eval} eval (of {len(file_items)} total)")

    rng.shuffle(train_all)
    rng.shuffle(eval_all)
    return train_all, eval_all
