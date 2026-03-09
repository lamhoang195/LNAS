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
) -> Dict[str, float]:
    """
    Exchange-level prediction: max smoothed probability over valid positions.
    Flagged as harmful if max probability >= threshold at any point.
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
    """Find threshold maximizing F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.95, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def ensemble_logits(
    z1: np.ndarray, z2: np.ndarray, alpha: float = 0.5,
) -> np.ndarray:
    """Weighted ensemble of two classifiers' logits.

    z_ensemble = α * z1 + (1 - α) * z2

    Section 5.2: combining probe with an external classifier
    improves robustness. Equal weights (α=0.5) are near-optimal.
    """
    return alpha * z1 + (1 - alpha) * z2


def find_optimal_ensemble_alpha(
    z1: np.ndarray, z2: np.ndarray, labels: np.ndarray,
    metric: str = "f1",
) -> float:
    """Grid-search α minimizing attack success rate (or maximizing F1).

    Returns optimal α in [0, 1].
    """
    best_score, best_alpha = -1.0, 0.5
    for alpha in np.arange(0.0, 1.01, 0.05):
        z_ens = ensemble_logits(z1, z2, alpha)
        from scipy.special import expit
        probs = expit(z_ens)
        preds = (probs >= 0.5).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


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
        eval_data = load_data(cfg.eval_file)
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
                             ckpt["window_size"], threshold)
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

    # 7. Ensemble note
    print("\n=== Ensemble ===")
    print("  Use ensemble_logits(z1, z2, alpha=0.5) to combine with external classifier.")
    print("  Use find_optimal_ensemble_alpha(z1, z2, labels) to optimize alpha.")