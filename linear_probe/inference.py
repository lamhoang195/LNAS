"""Streaming inference với EMA (detection) hoặc CumMax (gating) smoothing.

Gating mode (gate_mode=True):
    - Dùng CumMaxSmoother: xác suất chỉ tăng, không giảm → không bỏ sót jailbreak.
    - gate_threshold thấp hơn để kích hoạt AlphaSteer sớm.
    - AlphaSteer null-space projection lo false positive sau.

Detection mode (gate_mode=False):
    - Dùng EMASmoother: làm mượt giảm false positive.
    - threshold tiêu chuẩn.
"""

import torch
from typing import Dict

from model import LinearProbe, EMASmoother, CumMaxSmoother
from activation_collector import ActivationCollector
from probe_config import ProbeConfig


class StreamingDetector:
    """
    Detector thời gian thực hỗ trợ hai chế độ:

    gate_mode=True  → Tầng 1 gating cho AlphaSteer:
        - CumMaxSmoother: p_t = max_{τ≤t} σ(z̄_τ), chỉ tăng, không giảm.
        - gate_threshold thấp (~0.3): high recall, bắt sớm, AlphaSteer xử lý false positive.
        - process_token trả về `gate` → AlphaSteer kiểm tra để áp h_t+1.

    gate_mode=False → Detection độc lập:
        - EMASmoother: làm mượt giảm false positive.
        - threshold chuẩn (~0.5).
    """

    def __init__(self, probe, model, tokenizer, target_layers,
                 ema_alpha=0.1, threshold=0.5, multi_layer=True,
                 gate_mode=False, gate_threshold=0.3):
        self.probe = probe.eval()
        self.tokenizer = tokenizer
        self.collector = ActivationCollector(model, target_layers)
        self.gate_mode = gate_mode
        # gate_mode: CumMax (không bỏ sót); detection: EMA (giảm FP)
        self.smoother = CumMaxSmoother() if gate_mode else EMASmoother(alpha=ema_alpha)
        # Ngưỡng kích hoạt: thấp cho gating, chuẩn cho detection
        self.threshold = gate_threshold if gate_mode else threshold
        self.multi_layer = multi_layer
        self._token_count = 0

    def reset(self):
        """Reset state for new conversation."""
        self.smoother.reset()
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
        gates = []
        for t in range(logits.size(0)):
            _, score = self.smoother.update(logits[t].item())
            probs.append(score)
            gates.append(score >= self.threshold)
            self._token_count += 1

        max_prob = max(probs)
        return {
            # is_harmful: True nếu bất kỳ token nào vượt ngưỡng
            "is_harmful": max_prob >= self.threshold,
            "gate": any(gates),       # kích hoạt AlphaSteer khi gate_mode=True
            "max_probability": max_prob,
            "probabilities": probs,
            "token_count": self._token_count,
        }

    @torch.no_grad()
    def process_token(self, input_ids, attention_mask=None) -> Dict:
        """Process a single new token (streaming generation).

        Returns `gate=True` kật hiệu AlphaSteer cần can thiệp token tiếp theo.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        features = self.collector.collect(input_ids, attention_mask, self.multi_layer)
        logit = self.probe(features)[0, -1].item()
        score, prob = self.smoother.update(logit)
        self._token_count += 1
        triggered = prob >= self.threshold

        return {
            "is_harmful": triggered,
            "gate": triggered,         # AlphaSteer kiểm tra field này
            "probability": prob,
            "smoothed_logit": score,
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
    gate_mode = ckpt.get("gate_mode", False) or config.gate_mode
    return StreamingDetector(
        probe, model, tokenizer, ckpt["target_layers"],
        config.ema_alpha, config.threshold, ckpt.get("multi_layer", True),
        gate_mode=gate_mode,
        gate_threshold=ckpt.get("gate_threshold", config.gate_threshold),
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
