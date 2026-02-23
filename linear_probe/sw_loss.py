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
        # 1. Xác định các vị trí hợp lệ (t >= M)
        content_pos = attention_mask.cumsum(dim=1)
        valid_mask = (content_pos >= self.m_token) & (attention_mask == 1)
        
        # 2. Xử lý Case 1: T_i >= M
        # Tính weights bằng softmax có mask
        scaled_logits = smoothed_logits / self.tau
        scaled_logits = scaled_logits.masked_fill(~valid_mask, float('-inf'))
        weights = F.softmax(scaled_logits, dim=1) # (B, T)
        
        # Tính BCE cho từng token (chưa khử sample-wise)
        # labels: (B,) -> (B, T) để tính bce cho mỗi vị trí t
        target = labels.unsqueeze(1).expand_as(smoothed_logits).float()
        bce_all = F.binary_cross_entropy_with_logits(smoothed_logits, target, reduction='none')
        
        loss_case1 = (weights * bce_all).sum(dim=1) # (B,)

        # 3. Xử lý Case 2: T_i < M
        # Tính trung bình logits của các token thực (attention_mask == 1)
        masked_logits = smoothed_logits.masked_fill(attention_mask == 0, 0.0)
        sum_logits = masked_logits.sum(dim=1)
        count_tokens = attention_mask.sum(dim=1).clamp(min=1)
        mean_logits = sum_logits / count_tokens
        
        loss_case2 = F.binary_cross_entropy_with_logits(mean_logits, labels.float(), reduction='none')

        # 4. Hợp nhất 2 case
        has_valid_window = valid_mask.any(dim=1)
        final_loss = torch.where(has_valid_window, loss_case1, loss_case2)

        return final_loss.mean()