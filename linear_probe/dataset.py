import json
import os
from typing import Dict, Iterable, List


def _parse_conv(raw: List[Dict]) -> List[Dict]:
    conversations = []
    for turn in raw:
        if "user" in turn and "assistant" in turn:
            conversations.append({"user": turn["user"]})
            conversations.append({"assistant": turn["assistant"]})
        elif "role" in turn and "content" in turn:
            # Handle messages style: {"role": "user", "content": "..."}
            if turn["role"] == "user":
                conversations.append({"user": turn["content"]})
            elif turn["role"] == "assistant":
                conversations.append({"assistant": turn["content"]})
        elif "user" in turn:
            conversations.append({"user": turn["user"]})
        elif "assistant" in turn:
            conversations.append({"assistant": turn["assistant"]})
    return conversations


def _iter_items_from_file(path: str) -> Iterable[Dict]:
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    with open(path, encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
            return

        if ext != ".json":
            return

        payload = json.load(f)
        if isinstance(payload, dict):
            # Support either a single sample or a container with a samples list.
            candidates = payload.get("data") if "data" in payload else [payload]
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []

        for item in candidates:
            if isinstance(item, dict):
                yield item


def load_data(data_path: str) -> List[Dict]:
    dataset = []

    # Identify if it's a file or directory
    if os.path.isfile(data_path):
        files_to_read = [data_path]
    elif os.path.isdir(data_path):
        files_to_read = [
            os.path.join(data_path, f)
            for f in sorted(os.listdir(data_path))
            if f.endswith((".jsonl", ".json"))
        ]
    else:
        raise ValueError(f"Path {data_path} is neither a file nor a directory.")

    for path in files_to_read:
        for item in _iter_items_from_file(path):
            if "label" not in item:
                continue
            label = float(item["label"])
            raw = item.get("conversation") or item.get("conversations") or []
            conversations = _parse_conv(raw)
            if conversations:
                sample_id = str(item.get("id", f"{os.path.basename(path)}:{len(dataset)}"))
                sample = {
                    "id": sample_id,
                    "label": label,
                    "conversations": conversations,
                }
                dataset.append(sample)

    if not dataset:
        raise ValueError(
            f"No valid samples found at {data_path}. "
            "Expected JSONL lines or JSON objects/lists with 'label' and conversation fields."
        )

    return dataset