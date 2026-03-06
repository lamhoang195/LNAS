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


class CumulativeMaxLoss(nn.Module):
    """Cumulative maximum loss (Section C.1).

    p(y=1|x_{1:t}) = max_{τ≤t} σ(z̃_τ)

    Uses the cumulative max of probe probabilities as the predictor
    for the full sequence label. Gradient only flows through the
    token position with the highest score.
    """

    def __init__(self, temperature: float = 1.0, window_size: int = 16):
        super().__init__()
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
        content_pos = attention_mask.cumsum(dim=1)
        valid = (content_pos >= self.M).float() * attention_mask

        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]
        v = valid[has_valid]
        y = labels[has_valid]

        # max over logits at valid positions, then sigmoid
        masked_logits = sl.masked_fill(v == 0, float("-inf"))
        max_logits = masked_logits.max(dim=1).values  # (B,)
        max_probs = torch.sigmoid(max_logits)

        loss = F.binary_cross_entropy(max_probs, y, reduction="mean")
        return loss


class AnnealedCumulativeMaxLoss(nn.Module):
    """Annealed cumulative max loss (Section C.1).

    p(y=1|x_{1:t}) = (1-ω)·σ(z̃_t) + ω·max_{τ≤t} σ(z̃_τ)

    ω starts at 0 and linearly increases to 1 during training.
    Early training has stable gradients from direct predictions,
    then transitions to cumulative max for streaming classification.
    """

    def __init__(self, temperature: float = 1.0, window_size: int = 16,
                 total_steps: int = 1000):
        super().__init__()
        self.tau = temperature
        self.M = window_size
        self.total_steps = max(total_steps, 1)

    def forward(self, smoothed_logits, labels, attention_mask, current_step: int):
        """
        Args:
            smoothed_logits: (B, T) — z_bar_t after SwIM
            labels: (B,) — binary or soft labels
            attention_mask: (B, T)
            current_step: current training step for annealing ω
        Returns:
            Scalar loss
        """
        omega = min(current_step / self.total_steps, 1.0)

        content_pos = attention_mask.cumsum(dim=1)
        valid = (content_pos >= self.M).float() * attention_mask

        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]
        v = valid[has_valid]
        y = labels[has_valid]

        # Direct probe probability at each position
        direct_probs = torch.sigmoid(sl)  # (B, T)

        # Cumulative max probability
        masked_for_cummax = direct_probs.masked_fill(v == 0, 0.0)
        cummax_probs, _ = masked_for_cummax.cummax(dim=1)  # (B, T)

        # Interpolated prediction: (1-ω)·direct + ω·cummax
        interp_probs = (1 - omega) * direct_probs + omega * cummax_probs  # (B, T)

        # Per-token BCE on interpolated predictions
        bce = F.binary_cross_entropy(
            interp_probs, y.unsqueeze(1).expand_as(interp_probs), reduction="none"
        )

        # Softmax weights (same as SoftmaxWeightedBCE) over valid positions
        scaled = (sl / self.tau).masked_fill(v == 0, float("-inf"))
        weights = F.softmax(scaled, dim=1)

        return (weights * bce).sum(dim=1).mean()
