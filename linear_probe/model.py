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


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.linear(activations).squeeze(-1)


class SwIMSmoother(nn.Module):
    def __init__(self, window_size=16):
        super().__init__()
        self.M = window_size

    def forward(self, logits, mask=None):
        B, T = logits.shape
        M = self.M

        pad_logits = F.pad(
            logits if mask is None else logits * mask, (M, 0)
        )
        logit_cumsum = pad_logits.cumsum(dim=1)
        window_sum = logit_cumsum[:, M:] - logit_cumsum[:, :-M]

        if mask is None:
            return window_sum / M

        pad_mask = F.pad(mask, (M, 0))
        mask_cumsum = pad_mask.cumsum(dim=1)
        window_count = mask_cumsum[:, M:] - mask_cumsum[:, :-M]
        return window_sum / (window_count + 1e-6)


class EMASmoother:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0

    @staticmethod
    def max_ema_logits(
        logits: torch.Tensor,
        mask: torch.Tensor = None,
        alpha: float = 0.1,
    ) -> torch.Tensor:

        if logits.ndim != 2:
            raise ValueError(f"Expected logits shape (B, T), got {tuple(logits.shape)}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        B, T = logits.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=logits.device)
        else:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask shape must match logits shape, got {tuple(mask.shape)} vs {tuple(logits.shape)}"
                )
            mask = mask.bool()

        max_ema_logits = torch.full((B,), float("-inf"), device=logits.device, dtype=logits.dtype)
        ema_state = torch.zeros(B, device=logits.device, dtype=logits.dtype)

        for t in range(T):
            is_valid = mask[:, t]
            z_t = logits[:, t]
            ema_state = torch.where(
                is_valid,
                alpha * z_t + (1 - alpha) * ema_state,
                ema_state,
            )
            max_ema_logits = torch.where(
                is_valid,
                torch.maximum(max_ema_logits, ema_state),
                max_ema_logits,
            )

        return max_ema_logits

    def reset(self):
        self.value = 0.0

    def update(self, logit: float):
        self.value = self.alpha * logit + (1 - self.alpha) * self.value
        prob = 1 / (1 + math.exp(-self.value))
        return self.value, prob