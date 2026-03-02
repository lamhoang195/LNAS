"""Streaming inference with EMA smoothing."""

import torch
from typing import Dict

from model import LinearProbe, EMASmoother
from activation_collector import ActivationCollector
from probe_config import ProbeConfig


class StreamingDetector:
    """
    Real-time harmful content detector using EMA smoothing.
    Cost: ~2Ld FLOPs/token (negligible vs model forward pass).
    """

    def __init__(self, probe, model, tokenizer, target_layers,
                 ema_alpha=0.1, threshold=0.5, multi_layer=True):
        self.probe = probe.eval()
        self.tokenizer = tokenizer
        self.collector = ActivationCollector(model, target_layers)
        self.ema = EMASmoother(alpha=ema_alpha)
        self.threshold = threshold
        self.multi_layer = multi_layer
        self._token_count = 0

    def reset(self):
        """Reset state for new conversation."""
        self.ema.reset()
        self._token_count = 0

    @torch.no_grad()
    def process_prompt(self, input_ids, attention_mask=None) -> Dict:
        """Process full prompt, return per-token probabilities."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        features = self.collector.collect(input_ids, attention_mask, self.multi_layer)
        logits = self.probe(features).squeeze(0)

        self.reset()
        probs = []
        for t in range(logits.size(0)):
            _, prob = self.ema.update(logits[t].item())
            probs.append(prob)
            self._token_count += 1

        max_prob = max(probs)
        return {
            "is_harmful": max_prob >= self.threshold,
            "max_probability": max_prob,
            "probabilities": probs,
            "token_count": self._token_count,
        }

    @torch.no_grad()
    def process_token(self, input_ids, attention_mask=None) -> Dict:
        """Process a single new token (streaming generation)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        features = self.collector.collect(input_ids, attention_mask, self.multi_layer)
        logit = self.probe(features)[0, -1].item()
        smoothed, prob = self.ema.update(logit)
        self._token_count += 1

        return {
            "is_harmful": prob >= self.threshold,
            "probability": prob,
            "smoothed_logit": smoothed,
            "token_count": self._token_count,
        }

    @torch.no_grad()
    def detect(self, text: str) -> Dict:
        """Convenience: classify text as harmful/safe."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        return self.process_prompt(enc["input_ids"], enc["attention_mask"])


def load_detector(config: ProbeConfig, checkpoint_path=None):
    """Load a trained detector from checkpoint."""
    from train import load_base_model

    ckpt = torch.load(
        checkpoint_path or config.checkpoint_path,
        map_location="cpu", weights_only=True,
    )
    probe = LinearProbe(ckpt["hidden_dim"])
    probe.load_state_dict(ckpt["probe_state_dict"])

    model, tokenizer = load_base_model(config)
    return StreamingDetector(
        probe, model, tokenizer, ckpt["target_layers"],
        config.ema_alpha, config.threshold, ckpt.get("multi_layer", True),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    cfg = ProbeConfig.load(args.config) if args.config else ProbeConfig()
    if args.threshold:
        cfg.threshold = args.threshold

    result = load_detector(cfg, args.checkpoint).detect(args.text)
    print(f"{'HARMFUL' if result['is_harmful'] else 'SAFE'} "
          f"(p={result['max_probability']:.4f})")
