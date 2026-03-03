import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxWeightedBCELoss(nn.Module):
    def __init__(self, temperature: float = 1.0, window_size: int = 16):
        super().__init__()
        self.tau = temperature
        self.M = window_size

    def forward(self, smoothed_logits, labels, attention_mask):
        """
        Args:
            smoothed_logits: (B, T) — z_bar_t after SwIM
            labels: (B,) — binary or soft labels
            attention_mask: (B, T)
        Returns:
            Scalar loss
        """
        # Valid positions: at least M content tokens seen (full window)
        content_pos = attention_mask.cumsum(dim=1)
        valid = (content_pos >= self.M).float() * attention_mask

        # Skip samples with T_i < M (no valid positions)
        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]
        v = valid[has_valid]
        y = labels[has_valid]

        # Per-token BCE
        probs = torch.sigmoid(sl)
        bce = F.binary_cross_entropy(
            probs, y.unsqueeze(1).expand_as(probs), reduction="none"
        )

        # Softmax weights over valid positions only
        scaled = (sl / self.tau).masked_fill(v == 0, float("-inf"))
        weights = F.softmax(scaled, dim=1)

        return (weights * bce).sum(dim=1).mean()


class GatingLoss(nn.Module):
    """Loss cho gating mode (Tầng 1): SoftmaxWeightedBCE + Recall Penalty.

    Recall penalty buộc probe luôn gắn cờ ít nhất một token với xác suất
    cao hơn gate_threshold trên mọi chuỗi độc hại → không bỏ sót jailbreak.

    L_total = L_SWiM_BCE + β * L_recall_penalty

    Với L_recall_penalty = Σ_{i: y=1} max(0, θ_1 - max_t σ(z̄_t)):
    - Nếu probe đã dự đoán ≥ θ_1 tại ít nhất 1 token → penalty = 0
    - Nếu không → bị phạt tỷ lệ với mức thiếu hụt
    """

    def __init__(self, temperature: float = 1.0, window_size: int = 16,
                 recall_penalty_weight: float = 0.5, gate_threshold: float = 0.3):
        super().__init__()
        self.bce_loss = SoftmaxWeightedBCELoss(temperature, window_size)
        self.beta = recall_penalty_weight
        self.theta = gate_threshold
        self.M = window_size

    def forward(self, smoothed_logits, labels, attention_mask):
        """
        Args:
            smoothed_logits: (B, T) — z_bar_t after SwIM
            labels: (B,) — binary labels
            attention_mask: (B, T)
        Returns:
            Scalar loss
        """
        bce = self.bce_loss(smoothed_logits, labels, attention_mask)

        # Recall penalty: chỉ áp dụng trên chuỗi độc hại (y=1)
        harmful_mask = labels.bool()
        if not harmful_mask.any():
            return bce

        content_pos = attention_mask.cumsum(dim=1)
        valid = (content_pos >= self.M).float() * attention_mask

        # Max sigmoid over valid positions for each harmful sample
        sl_harmful = smoothed_logits[harmful_mask]
        v_harmful = valid[harmful_mask]
        masked = sl_harmful.masked_fill(v_harmful == 0, float("-inf"))
        max_probs = torch.sigmoid(masked.max(dim=1).values)  # (B_harm,)

        # penalty = max(0, θ_1 - max_prob) → 0 khi đã đủ tự tin
        penalty = torch.clamp(self.theta - max_probs, min=0.0).mean()

        return bce + self.beta * penalty
