"""
Linear Probe with SwIM (training) and EMA (inference) smoothing.

Probe: p(y=1|x_{1:t}) = sigma(W^T psi_t + b)
SwIM:  z_bar_t = (1/M) * sum_{k=0}^{M-1} z_{t-k}
EMA:   z_bar_t = alpha * z_t + (1 - alpha) * z_bar_{t-1}
"""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """z_t = W^T psi_t + b, where psi_t is the activation vector."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T) raw logits."""
        return self.linear(activations).squeeze(-1)


class SwIMSmoother(nn.Module):
    """Sliding Window Mean logit smoothing for training."""

    def __init__(self, window_size: int = 16):
        super().__init__()
        self.M = window_size

    def forward(self, logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        (B, T) logits -> (B, T) smoothed logits.
        Padding positions (mask=0) contribute 0 to the sum.
        For valid loss positions (content_pos >= M), the full window
        falls within content, so division by M is exact.
        """
        B, T = logits.shape
        M = self.M
        if mask is not None:
            logits = logits * mask
        padded = torch.cat([torch.zeros(B, M, device=logits.device), logits], dim=1)
        cumsum = padded.cumsum(dim=1)
        return (cumsum[:, M:] - cumsum[:, :-M]) / M


class EMASmoother:
    """Exponential Moving Average smoother for inference (O(1) memory).

    Dùng cho detection mode: giảm false positive bằng cách làm mượt.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def update(self, logit: float) -> tuple:
        """Returns (smoothed_logit, probability)."""
        self.value = self.alpha * logit + (1 - self.alpha) * self.value
        prob = torch.sigmoid(torch.tensor(self.value)).item()
        return self.value, prob



