import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build_full_window_mask


def _select_valid_rows(smoothed_logits, labels, attention_mask, window_size: int):
    valid = build_full_window_mask(attention_mask, window_size)
    has_valid = valid.any(dim=1)
    if not has_valid.any():
        return None, None, None

    return smoothed_logits[has_valid], valid[has_valid], labels[has_valid].to(dtype=smoothed_logits.dtype)


class SoftmaxWeightedBCELoss(nn.Module):
    def __init__(self, temperature: float = 1.0, window_size: int = 16):
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
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
        sl, valid, y = _select_valid_rows(smoothed_logits, labels, attention_mask, self.M)
        if sl is None:
            return smoothed_logits.sum() * 0.0

        not_valid = ~valid

        bce = F.binary_cross_entropy_with_logits(
            sl, y.unsqueeze(1).expand_as(sl), reduction="none"
        ).masked_fill_(not_valid, 0.0)

        # The softmax weighting focuses the loss on the most harmful positions.
        scaled = (sl / self.tau).masked_fill_(not_valid, float("-inf"))
        weights = F.softmax(scaled, dim=1)

        return (weights * bce).sum(dim=1).mean()


class CumulativeMaxLoss(nn.Module):
    """Cumulative maximum loss (Section C.1).

    p(y=1|x_{1:t}) = max_{τ≤t} σ(z̃_τ)

    Uses the cumulative max of probe probabilities as the predictor
    for the full sequence label. Gradient only flows through the
    token position with the highest score.
    """

    def __init__(self, window_size: int = 16):
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
        sl, valid, y = _select_valid_rows(smoothed_logits, labels, attention_mask, self.M)
        if sl is None:
            return smoothed_logits.sum() * 0.0

        valid_logits = sl.masked_fill(~valid, float("-inf"))
        max_prob = torch.sigmoid(valid_logits).max(dim=1).values.clamp(1e-7, 1 - 1e-7)

        return F.binary_cross_entropy(max_prob, y, reduction="mean")


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

        sl, valid, y = _select_valid_rows(smoothed_logits, labels, attention_mask, self.M)
        if sl is None:
            return smoothed_logits.sum() * 0.0

        not_valid = ~valid

        direct_probs = torch.sigmoid(sl)

        # The cumulative branch matches the streaming "stop if ever harmful" decision rule.
        cummax_probs = direct_probs.masked_fill(not_valid, 0.0).cummax(dim=1).values

        interp_probs = torch.lerp(direct_probs, cummax_probs, omega).clamp_(1e-7, 1 - 1e-7)

        bce = F.binary_cross_entropy(
            interp_probs,
            y.unsqueeze(1).expand_as(interp_probs),
            reduction="none"
        ).masked_fill_(not_valid, 0.0)

        valid_counts = valid.sum(dim=1).clamp(min=1)
        return (bce.sum(dim=1) / valid_counts).mean()