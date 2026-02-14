import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxWeightedBCELoss(nn.Module):
    '''
    Args:
        temperature (τ):  τ → 0 focuses on most-confident-harmful positions;
                          τ → ∞ gives uniform weights.
        window_size (M):  positions t < M are excluded (incomplete window). 
    '''
    def __init__(self, temperature: float = 1.0, window_size: int = 16):
        super().__init__()
        self.tau = temperature
        self.m_token = window_size

    def forward(self, smoothed_logits, labels, attention_mask):
        '''
        Args:
            smoothed_logits: (B, T) — z_bar_t after SwIM
            labels: (B,) — binary or soft labels
            attention_mask: (B, T)
        Returns:    
            Scalar loss
        '''
        # Valid positions: at least M content tokens seen (full window)
        content_pos = attention_mask.cumsum(dim=1)
        valid = (content_pos >= self.m_token).float() * attention_mask

        # Skip samples with T_i < m_token (no valid positions)
        has_valid = valid.any(dim=1)
        if not has_valid.any():
            return smoothed_logits.sum() * 0.0
        sl = smoothed_logits[has_valid]
        v = valid[has_valid]
        y = labels[has_valid]

        # Per-token BCE
        probs = torch.sigmoid(sl)
        bce = F.binary_cross_entropy(
            probs, y.unsqueeze(1).expand_as(probs), reduction='none'
        )

        # Softmax weights over valid positions only
        scaled = (sl / self.tau).masked_fill(v == 0, float('-inf'))
        weights = F.softmax(scaled, dim=1)

        return (weights * bce).sum(dim=1).mean()