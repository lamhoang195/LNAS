import torch
import torch.nn as nn
from typing import List, Dict, Any

class ActivationCollector:
    def __init__(self, model: nn.Module, target_layers: List[int] = None, device=None):
        self.model = model
        self.target_layers = target_layers or []
        self.device = device or next(model.parameters()).device

    def _resolve_layers(self, n_hidden_states: int) -> List[int]:
        if self.target_layers:
            return self.target_layers
        return list(range(1, n_hidden_states))

    @torch.no_grad()
    def collect(self, input_ids, attention_mask, multi_layer=True) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        layers = self._resolve_layers(len(out.hidden_states))
        hidden_states = out.hidden_states

        if multi_layer and len(layers) > 1:
            acts = torch.cat([hidden_states[l].detach() for l in layers], dim=-1)
        else:
            acts = hidden_states[layers[0]].detach()

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
            "labels": torch.tensor(labels, dtype=torch.float32)  # shape (B,)
        }
    return collate_fn

class OnTheFlyLoader:
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

            # Lên device cho input ở chung 1 chỗ
            device = next(self.collector.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

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