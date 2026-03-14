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
        am_bool = attention_mask.bool()  # cast once to avoid repeated alloc
        content_pos = attention_mask.float().cumsum(dim=1)  # float cumsum is faster on GPU
        
        # Ensure sequences smaller than M are not ignored by evaluating their last valid token
        seq_lens = attention_mask.sum(dim=1, keepdim=True)
        valid_threshold = torch.clamp(seq_lens, max=self.M)
        
        valid = (content_pos >= valid_threshold) & am_bool

        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]  # new tensor (boolean indexing = copy)
        v = valid[has_valid]
        y = labels[has_valid]
        not_v = ~v  # cache: used twice below

        bce = F.binary_cross_entropy_with_logits(
            sl, y.unsqueeze(1).expand_as(sl), reduction="none"
        ).masked_fill_(not_v, 0.0)  # in-place: output of bce_with_logits is a fresh tensor

        # divide before mask → 1 tensor allocation instead of 2
        scaled = (sl / self.tau).masked_fill_(not_v, float("-inf"))
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
        am_bool = attention_mask.bool()
        content_pos = attention_mask.float().cumsum(dim=1)  # float cumsum is faster on GPU
        
        seq_lens = attention_mask.sum(dim=1, keepdim=True)
        valid_threshold = torch.clamp(seq_lens, max=self.M)
        valid = (content_pos >= valid_threshold) & am_bool

        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]
        v = valid[has_valid]
        y = labels[has_valid]

        # sigmoid output is saved for its own backward, so masked_fill must be out-of-place
        # clamp to avoid log(0) = -inf when sigmoid saturates to exactly 1.0 in float32
        max_prob = torch.sigmoid(sl).masked_fill(~v, 0.0).max(dim=1).values.clamp(1e-7, 1 - 1e-7)

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

        am_bool = attention_mask.bool()
        content_pos = attention_mask.float().cumsum(dim=1)  # float cumsum is faster on GPU
        
        seq_lens = attention_mask.sum(dim=1, keepdim=True)
        valid_threshold = torch.clamp(seq_lens, max=self.M)
        valid = (content_pos >= valid_threshold) & am_bool

        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0

        sl = smoothed_logits[has_valid]  # new tensor (boolean indexing = copy)
        v = valid[has_valid]
        y = labels[has_valid]
        not_v = ~v  # cache: used twice below

        direct_probs = torch.sigmoid(sl)  # (B, T)

        # sigmoid output saved for backward → masked_fill must be out-of-place
        cummax_probs = direct_probs.masked_fill(not_v, 0.0).cummax(dim=1).values  # (B, T)

        interp_probs = torch.lerp(direct_probs, cummax_probs, omega).clamp_(1e-7, 1 - 1e-7)  # (B, T)

        bce = F.binary_cross_entropy(
            interp_probs,
            y.unsqueeze(1).expand_as(interp_probs),
            reduction="none"
        ).masked_fill_(not_v, 0.0)  # in-place: bce output is a fresh tensor

        # normalize by valid token count to avoid penalizing long sequences more
        valid_counts = v.sum(dim=1).clamp(min=1)
        return (bce.sum(dim=1) / valid_counts).mean()