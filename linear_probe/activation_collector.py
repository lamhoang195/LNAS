from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

class ActivationCollector:
    def __init__(self, model: nn.Module, target_layers: Optional[List[int]] = None, device=None):
        self.model = model
        self.target_layers = target_layers or []
        self.device = device or next(model.parameters()).device

    def _resolve_layers(self, n_hidden_states: int) -> List[int]:
        if self.target_layers:
            resolved_layers = []
            for layer_idx in self.target_layers:
                resolved_idx = n_hidden_states + layer_idx if layer_idx < 0 else layer_idx
                if resolved_idx <= 0 or resolved_idx >= n_hidden_states:
                    raise ValueError(
                        f"Layer index {layer_idx} resolves to {resolved_idx}, "
                        f"but hidden_states has valid indices [1, {n_hidden_states - 1}]"
                    )
                resolved_layers.append(resolved_idx)
            return resolved_layers
        return list(range(1, n_hidden_states))

    @torch.no_grad()
    def collect(self, input_ids, attention_mask, multi_layer=True) -> torch.Tensor:
        input_ids = input_ids.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layers = self._resolve_layers(len(out.hidden_states))
        hidden_states = out.hidden_states
        selected_states = [hidden_states[layer_idx].detach() for layer_idx in layers]

        if multi_layer and len(selected_states) > 1:
            # Keep concatenation on one device so device_map="auto" still works.
            target_device = selected_states[0].device
            acts = torch.cat(
                [state if state.device == target_device else state.to(target_device) for state in selected_states],
                dim=-1,
            )
        else:
            acts = selected_states[0]

        del hidden_states
        del out

        return acts

def make_tokenized_collate_fn(tokenizer, max_length):
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        texts = []
        labels = []
        for item in batch:
            messages = []
            for turn in item["conversations"]:
                if "user" in turn:
                    messages.append({"role": "user", "content": turn["user"]})
                elif "assistant" in turn:
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            try:
                seq = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                seq = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            texts.append(seq)
            labels.append(item["label"])

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
    return collate_fn

class OnTheFlyLoader:
    """Iterator that computes activations per batch (on-the-fly), no precomputation.

    Pipeline: batch from DataLoader (input_ids, mask, labels)
      → LLM forward in collect() → yield {activations, attention_mask, labels}.
    Used for both training and evaluation so activations are never fully materialized.
    """
    def __init__(self, dataloader, collector: ActivationCollector, multi_layer: bool = True, return_cpu: bool = False):
        self.dataloader = dataloader
        self.collector = collector
        self.multi_layer = multi_layer
        self.return_cpu = return_cpu

    def __iter__(self):
        for batch in self.dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            device = self.collector.device
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            activations = self.collector.collect(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_layer=self.multi_layer
            )

            if self.return_cpu:
                activations = activations.cpu()
                attention_mask = attention_mask.cpu()
                labels = labels.cpu()

            yield {
                "activations": activations,
                "attention_mask": attention_mask,
                "labels": labels
            }

    def __len__(self):
        return len(self.dataloader)