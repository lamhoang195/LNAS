"""Evaluation: max smoothed probability as exchange-level prediction."""

from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from model import LinearProbe, SwIMSmoother


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    smoother: SwIMSmoother,
    dataloader: DataLoader,
    device: torch.device,
    window_size: int = 16,
    threshold: float = 0.5,
    gate_mode: bool = False,
) -> Dict[str, float]:
    """
    Exchange-level prediction: max smoothed probability over valid positions.
    Flagged as harmful if max probability >= threshold at any point.

    gate_mode=True: dùng threshold thấp hơn, ưu tiên recall cho Tầng 1.
    """
    probe.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        acts = batch["activations"].to(device)
        mask = batch["attention_mask"].to(device)

        smoothed = smoother(probe(acts), mask)

        # Valid: content position >= M (full window)
        content_pos = mask.cumsum(dim=1)
        valid = (content_pos >= window_size).float() * mask

        # Max logit over valid positions -> sigmoid
        smoothed = smoothed.masked_fill(valid == 0, float("-inf"))
        max_logits = smoothed.max(dim=1).values
        all_probs.append(torch.sigmoid(max_logits).cpu())
        all_labels.append(batch["labels"])

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold maximizing F1 (detection mode)."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.95, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def find_gate_threshold(
    probs: np.ndarray, labels: np.ndarray,
    min_recall: float = 0.95,
) -> float:
    """Tìm threshold cao nhất mà vẫn đảm bảo recall >= min_recall (gate mode).

    Gate mode ưu tiên recall: chọn ngưỡng cao nhất thoả recall >= min_recall.
    Ngưỡng cao hơn → ít false positive hơn → AlphaSteer được gọi ít hơn.
    """
    best_t = 0.1
    for t in np.arange(0.1, 0.90, 0.01):
        rec = recall_score(labels, (probs >= t).astype(int), zero_division=0)
        if rec >= min_recall:
            best_t = t  # giữ ngưỡng cao nhất thỏa mãn
    return best_t


if __name__ == "__main__":
    import argparse
    import os

    import torch
    from torch.utils.data import DataLoader

    from probe_config import ProbeConfig
    from model import LinearProbe, SwIMSmoother
    from activation_collector import (
        CachedActivationDataset, cached_collate_fn, precompute_activations,
    )
    from dataset import load_data_split, load_data

    parser = argparse.ArgumentParser(description="Evaluate a trained linear probe.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_probe.pt saved by train.py")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: find optimal from data)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = ProbeConfig.load(args.config) if args.config else ProbeConfig()
    if args.train_data: cfg.train_file = args.train_data
    if args.eval_data:  cfg.eval_file  = args.eval_data
    if args.batch_size: cfg.batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    probe = LinearProbe(ckpt["hidden_dim"]).to(device)
    probe.load_state_dict(ckpt["probe_state_dict"])
    smoother = SwIMSmoother(ckpt["window_size"]).to(device)
    target_layers = ckpt["target_layers"]
    multi_layer = ckpt.get("multi_layer", True)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"train F1={ckpt.get('best_f1', 0):.4f}")

    # 2. Build or reuse eval activation cache
    eval_cache = os.path.join(cfg.cache_root, "eval")
    if not os.path.exists(eval_cache) or not os.listdir(eval_cache):
        print("Building eval activation cache...")
        from train import load_base_model
        model, tokenizer = load_base_model(cfg)

        if cfg.eval_file and os.path.exists(cfg.eval_file) and os.listdir(cfg.eval_file):
            eval_data = load_data(cfg.eval_file)
        else:
            _, eval_data = load_data_split(cfg.train_file, eval_ratio=0.20, seed=cfg.seed)

        precompute_activations(
            model, tokenizer, eval_data, target_layers,
            eval_cache, cfg.max_sequence_length, multi_layer,
        )
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Reusing cached eval activations from {eval_cache}")

    # 3. Dataloader
    eval_loader = DataLoader(
        CachedActivationDataset(eval_cache), batch_size=cfg.batch_size,
        shuffle=False, collate_fn=cached_collate_fn, num_workers=2, pin_memory=True,
    )

    # 4. Collect probs & labels
    probe.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            acts = batch["activations"].to(device)
            mask = batch["attention_mask"].to(device)
            smoothed = smoother(probe(acts), mask)

            content_pos = mask.cumsum(dim=1)
            valid = (content_pos >= ckpt["window_size"]).float() * mask
            smoothed = smoothed.masked_fill(valid == 0, float("-inf"))
            max_logits = smoothed.max(dim=1).values
            all_probs.append(torch.sigmoid(max_logits).cpu())
            all_labels.append(batch["labels"])

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    from sklearn.metrics import precision_score, recall_score, f1_score

    # 5. Metrics at fixed threshold
    threshold = args.threshold if args.threshold is not None else cfg.threshold
    metrics = evaluate_probe(probe, smoother, eval_loader, device,
                             ckpt["window_size"], threshold,
                             gate_mode=ckpt.get("gate_mode", False))
    print(f"\n=== Eval @ threshold={threshold:.2f} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUROC    : {metrics['auroc']:.4f}")

    # 6. Find optimal threshold
    opt_t = find_optimal_threshold(probs, labels)
    opt_preds = (probs >= opt_t).astype(int)
    print(f"\n=== Optimal F1 threshold={opt_t:.2f} ===")
    print(f"  Precision: {precision_score(labels, opt_preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(labels, opt_preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(labels, opt_preds, zero_division=0):.4f}")

    # 7. Gate threshold recommendation (gate_mode)
    if ckpt.get("gate_mode", False):
        gate_t = find_gate_threshold(probs, labels, min_recall=0.95)
        gate_preds = (probs >= gate_t).astype(int)
        print(f"\n=== Gate threshold (recall ≥0.95) = {gate_t:.2f} ===")
        print(f"  Precision: {precision_score(labels, gate_preds, zero_division=0):.4f}")
        print(f"  Recall   : {recall_score(labels, gate_preds, zero_division=0):.4f}")
        print(f"  F1       : {f1_score(labels, gate_preds, zero_division=0):.4f}")
        print(f"  → Dùng gate_threshold={gate_t:.2f} khi deploy AlphaSteer (Tầng 2)")