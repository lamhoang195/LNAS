"""Activation extraction and caching for probe training."""

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class ActivationCollector:
    """Extract hidden-state activations from specified (or all) layers."""

    def __init__(self, model: nn.Module, target_layers: List[int] = None, device=None):
        self.model = model
        self.target_layers = target_layers or []
        self.device = device or next(model.parameters()).device

    def _resolve_layers(self, n_hidden_states: int) -> List[int]:
        """Empty target_layers = all transformer layers (skip embedding at [0])."""
        if self.target_layers:
            return self.target_layers
        return list(range(1, n_hidden_states))

    @torch.no_grad()
    def collect(self, input_ids, attention_mask, multi_layer=True) -> torch.Tensor:
        """
        Forward pass -> extract & build feature vector psi_t.

        Multi-layer: psi_t = [phi_t^(l1); phi_t^(l2); ...]  (concatenated)
        Single-layer: psi_t = phi_t^(l)

        Returns: (B, T, D)
        """
        out = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=attention_mask.to(self.model.device),
            output_hidden_states=True, return_dict=True,
        )
        layers = self._resolve_layers(len(out.hidden_states))

        if multi_layer and len(layers) > 1:
            return torch.cat([out.hidden_states[l].detach() for l in layers], dim=-1)
        return out.hidden_states[layers[0]].detach()


class ActivationCache:
    """Save/load pre-computed activations to disk."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save(self, activations, attention_mask, label, idx):
        torch.save({
            "activations": activations.cpu().half(),
            "attention_mask": attention_mask.cpu(),
            "label": label.cpu(),
        }, os.path.join(self.cache_dir, f"sample_{idx}.pt"))

    def exists(self, idx):
        return os.path.exists(os.path.join(self.cache_dir, f"sample_{idx}.pt"))

    def count(self):
        return len([f for f in os.listdir(self.cache_dir) if f.endswith(".pt")])


class CachedActivationDataset(Dataset):
    """Load pre-cached activations from disk."""

    def __init__(self, cache_dir: str):
        files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        self.paths = [os.path.join(cache_dir, f) for f in files]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], map_location="cpu", weights_only=True)
        return {
            "activations": data["activations"].float(),
            "attention_mask": data["attention_mask"],
            "label": data["label"],
        }


def cached_collate_fn(batch):
    """Left-pad cached activations."""
    max_len = max(b["activations"].size(0) for b in batch)
    acts, masks, labels = [], [], []
    for b in batch:
        pad = max_len - b["activations"].size(0)
        d = b["activations"].size(-1)
        acts.append(torch.cat([torch.zeros(pad, d), b["activations"]], dim=0))
        masks.append(torch.cat([torch.zeros(pad, dtype=b["attention_mask"].dtype),
                                b["attention_mask"]]))
        labels.append(b["label"])
    return {
        "activations": torch.stack(acts),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


def _item_to_text(tokenizer, item) -> str:
    """Convert a data item to a single string using the tokenizer's chat template."""
    messages = item["messages"]
    has_response = messages and messages[-1]["role"] == "assistant"
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not has_response,
        )
    return " ".join(m["content"] for m in messages)


@torch.no_grad()
def precompute_activations(
    model, tokenizer, data, target_layers,
    cache_dir, max_length=2048, multi_layer=True, batch_size=8,
):
    """Pre-compute and cache activations for the dataset (batched)."""
    collector = ActivationCollector(model, target_layers)
    cache = ActivationCache(cache_dir)

    # Only process samples that are not yet cached
    pending = [(idx, item) for idx, item in enumerate(data) if not cache.exists(idx)]
    if not pending:
        print(f"All {cache.count()} samples already cached in {cache_dir}")
        return

    n_batches = (len(pending) + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Caching activations"):
        chunk = pending[b * batch_size:(b + 1) * batch_size]
        texts = [_item_to_text(tokenizer, item) for _, item in chunk]

        enc = tokenizer(
            texts, return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        )
        # features: (B, T, D)
        features = collector.collect(enc["input_ids"], enc["attention_mask"], multi_layer)

        for i, (idx, item) in enumerate(chunk):
            cache.save(
                activations=features[i],
                attention_mask=enc["attention_mask"][i],
                label=torch.tensor(item["label"], dtype=torch.float32),
                idx=idx,
            )

    print(f"Cached {cache.count()} samples to {cache_dir}")
