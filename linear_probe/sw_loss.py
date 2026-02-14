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
        self.window_size = window_size

    def forward(self, smoothed_logits, labels, attention_mask):
        