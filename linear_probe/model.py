"""
Linear Probe with SwIM (training) and EMA (inference) smoothing.

Probe: p(y=1|x_{1:t}) = sigma(W^T psi_t + b)
SwIM:  z_bar_t = (1/M) * sum_{k=0}^{M-1} z_{t-k}
EMA:   z_bar_t = alpha * z_t + (1 - alpha) * z_bar_{t-1}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def as_bool_mask(mask: torch.Tensor, reference: torch.Tensor = None) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D attention mask, got shape {tuple(mask.shape)}")
    bool_mask = mask.to(dtype=torch.bool)
    if reference is not None and bool_mask.shape != reference.shape:
        raise ValueError(
            f"Mask shape must match reference shape, got {tuple(bool_mask.shape)} vs {tuple(reference.shape)}"
        )
    return bool_mask


def build_full_window_mask(attention_mask: torch.Tensor, window_size: int) -> torch.Tensor:
    """Only score positions that have a full M-token window, except short sequences."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    bool_mask = as_bool_mask(attention_mask)
    content_pos = bool_mask.to(dtype=torch.long).cumsum(dim=1)
    seq_lens = bool_mask.sum(dim=1, keepdim=True)

    # For long sequences, train only where a full M-token window exists.
    full_window = content_pos >= window_size
    # For short sequences (T < M), keep all valid content positions.
    short_sequence = seq_lens < window_size

    return bool_mask & (full_window | short_sequence)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.linear(activations).squeeze(-1)


class SwIMSmoother(nn.Module):
    def __init__(self, window_size=16):
        super().__init__()
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.M = int(window_size)

    def forward(self, logits, mask=None):
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape (B, T), got {tuple(logits.shape)}")

        M = self.M
        mask_f = None
        if mask is not None:
            mask_f = as_bool_mask(mask, logits).to(dtype=logits.dtype)

        pad_logits = F.pad(
            logits if mask_f is None else logits * mask_f, (M, 0)
        )
        logit_cumsum = pad_logits.cumsum(dim=1)
        window_sum = logit_cumsum[:, M:] - logit_cumsum[:, :-M]

        if mask_f is None:
            return window_sum / M

        pad_mask = F.pad(mask_f, (M, 0))
        mask_cumsum = pad_mask.cumsum(dim=1)
        window_count = mask_cumsum[:, M:] - mask_cumsum[:, :-M]
        return window_sum / (window_count + 1e-6)


class EMASmoother:
    def __init__(self, alpha: float = 0.1):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = 0.0

    @staticmethod
    def smooth_logits(
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape (B, T), got {tuple(logits.shape)}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        B, _ = logits.shape
        if mask is None:
            bool_mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            bool_mask = as_bool_mask(mask, logits)

        ema_logits = torch.zeros_like(logits)
        ema_state = torch.zeros(B, device=logits.device, dtype=logits.dtype)

        for t in range(logits.size(1)):
            is_valid = bool_mask[:, t]
            z_t = logits[:, t]
            ema_state = torch.where(
                is_valid,
                alpha * z_t + (1 - alpha) * ema_state,
                ema_state,
            )
            ema_logits[:, t] = torch.where(is_valid, ema_state, ema_logits[:, t])

        return ema_logits

    @staticmethod
    def max_ema_logits(
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        bool_mask = torch.ones_like(logits, dtype=torch.bool) if mask is None else as_bool_mask(mask, logits)
        ema_logits = EMASmoother.smooth_logits(logits, bool_mask, alpha)
        return ema_logits.masked_fill(~bool_mask, float("-inf")).max(dim=1).values

    def reset(self):
        self.value = 0.0

    def update(self, logit: float):
        self.value = self.alpha * logit + (1 - self.alpha) * self.value
        prob = 1 / (1 + math.exp(-self.value))
        return self.value, prob