"""
Smoke test — kiểm tra shapes và logic không cần GPU / LLM thật.
Chạy: python linear_probe/test_shapes.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

# ─── 1. Labels shape từ collate_fn (mock) ───────────────────────────────────
print("=== Test 1: Labels shape từ collate_fn ===")
labels_list = [1.0, 0.0, 1.0, 0.0]
labels_tensor = torch.tensor(labels_list, dtype=torch.float32)
assert labels_tensor.shape == (4,), f"Expected (4,), got {labels_tensor.shape}"
print(f"  labels shape: {labels_tensor.shape}  ✓")

# ─── 2. SwIMSmoother ─────────────────────────────────────────────────────────
print("\n=== Test 2: SwIMSmoother ===")
from model import SwIMSmoother

M = 3
smoother = SwIMSmoother(window_size=M)

# Sequence [1,2,3,4,5], non-masked
logits = torch.tensor([[1., 2., 3., 4., 5.]])  # (1, T=5)
smoothed = smoother(logits)
print(f"  Input  : {logits[0].tolist()}")
print(f"  Smoothed: {smoothed[0].tolist()}")
# Positions t=0,1 → window doesn't have M tokens yet, but smoother returns values for all positions
# t=2: mean(0,1,2)=2.0, t=3: mean(1,2,3)=3.0, t=4: mean(2,3,4)=4.0
expected_from_M = [2.0, 3.0, 4.0]
got_from_M = smoothed[0, M-1:].tolist()
assert [round(x, 5) for x in got_from_M] == expected_from_M, f"Got {got_from_M}"
print(f"  Positions t>=M-1: {got_from_M}  ✓")

# ─── 3. SoftmaxWeightedBCELoss ───────────────────────────────────────────────
print("\n=== Test 3: SoftmaxWeightedBCELoss ===")
from sw_loss import SoftmaxWeightedBCELoss

B, T, M = 2, 10, 3
loss_fn = SoftmaxWeightedBCELoss(temperature=1.0, window_size=M)

smoothed_logits = torch.randn(B, T)
labels = torch.tensor([1.0, 0.0])          # (B,) — đúng shape
attention_mask = torch.ones(B, T)

loss = loss_fn(smoothed_logits, labels, attention_mask)
assert loss.shape == (), f"Expected scalar, got {loss.shape}"
assert not torch.isnan(loss), "Loss is NaN!"
print(f"  Loss value: {loss.item():.4f}  ✓")

# ─── 4. CumulativeMaxLoss ─────────────────────────────────────────────────────
print("\n=== Test 4: CumulativeMaxLoss ===")
from sw_loss import CumulativeMaxLoss

loss_fn2 = CumulativeMaxLoss(window_size=M)
loss2 = loss_fn2(smoothed_logits, labels, attention_mask)
assert not torch.isnan(loss2), "CumulativeMaxLoss is NaN!"
print(f"  Loss value: {loss2.item():.4f}  ✓")

# ─── 5. AnnealedCumulativeMaxLoss ────────────────────────────────────────────
print("\n=== Test 5: AnnealedCumulativeMaxLoss ===")
from sw_loss import AnnealedCumulativeMaxLoss

loss_fn3 = AnnealedCumulativeMaxLoss(temperature=1.0, window_size=M, total_steps=100)
loss3 = loss_fn3(smoothed_logits, labels, attention_mask, current_step=50)
assert not torch.isnan(loss3), "AnnealedCumMax is NaN!"
print(f"  Loss value: {loss3.item():.4f}  ✓")

# ─── 6. LinearProbe forward ─────────────────────────────────────────────────
print("\n=== Test 6: LinearProbe forward ===")
from model import LinearProbe

hidden_dim = 4096
probe = LinearProbe(hidden_dim)
activations = torch.randn(B, T, hidden_dim)
logits_out = probe(activations)
assert logits_out.shape == (B, T), f"Expected (B,T)=({B},{T}), got {logits_out.shape}"
print(f"  Output shape: {logits_out.shape}  ✓")

# ─── 7. Full forward pass: probe → smoother → loss ──────────────────────────
print("\n=== Test 7: Full forward (probe → smoother → loss) ===")
logits_raw = probe(activations)                      # (B, T)
smoothed_out = smoother(logits_raw, attention_mask)  # (B, T)
loss_full = loss_fn(smoothed_out, labels, attention_mask)
assert not torch.isnan(loss_full), "Full forward loss is NaN!"
print(f"  Full forward loss: {loss_full.item():.4f}  ✓")

# ─── 8. EMASmoother.max_ema_logits ─────────────────────────────────────────
print("\n=== Test 8: EMASmoother.max_ema_logits ===")
from model import EMASmoother

logits_2d = torch.randn(B, T)
result = EMASmoother.max_ema_logits(logits_2d, alpha=0.1)
assert result.shape == (B,), f"Expected (B,)=({B},), got {result.shape}"
print(f"  EMA max logits shape: {result.shape}  ✓")

# ─── 9. Short sequence (T < M) không crash ──────────────────────────────────
print("\n=== Test 9: Short sequence (T < M) ===")
short_logits = torch.randn(2, 2)  # T=2 < M=3
short_mask = torch.ones(2, 2)
short_labels = torch.tensor([1.0, 0.0])
loss_short = loss_fn(short_logits, short_labels, short_mask)
# Should not crash; short seqs use last valid token per code logic
print(f"  Short seq loss: {loss_short.item():.4f}  ✓")

# ─── 10. padding trên batch ─────────────────────────────────────────────────
print("\n=== Test 10: Padded batch (attention_mask có zeros) ===")
padded_logits = torch.randn(2, 8)
padded_mask = torch.tensor([
    [0, 0, 0, 1, 1, 1, 1, 1],  # 5 real tokens, 3 leading padding
    [0, 0, 1, 1, 1, 1, 1, 1],  # 6 real tokens, 2 leading padding
], dtype=torch.float32)
padded_labels = torch.tensor([1.0, 0.0])
loss_padded = loss_fn(padded_logits, padded_labels, padded_mask)
print(f"  Padded batch loss: {loss_padded.item():.4f}  ✓")

print("\n✅ ALL TESTS PASSED")
