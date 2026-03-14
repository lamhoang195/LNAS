import argparse
import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from dataset import load_data
from probe_config import ProbeConfig
from model import LinearProbe, EMASmoother
from activation_collector import (
    ActivationCollector, OnTheFlyLoader, make_tokenized_collate_fn,
)


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    dataloader,
    collector: ActivationCollector,
    multi_layer: bool,
    device: torch.device,
    ema_alpha: float = 0.1,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    
    probe.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        if "activations" in batch:
            acts = batch["activations"].to(device, non_blocking=True)
        else:
            # Fallback: tính activations trực tiếp (on-the-fly) nếu batch chưa có sẵn
            acts = collector.collect(
                batch["input_ids"].to(device, non_blocking=True),
                batch["attention_mask"].to(device, non_blocking=True),
                multi_layer=multi_layer,
            )
        mask   = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].squeeze(-1) if batch["labels"].ndim > 1 else batch["labels"]  # ensure (B,)

        logits = probe(acts)

        max_ema_logits = EMASmoother.max_ema_logits(
            logits=logits,
            mask=mask,
            alpha=ema_alpha,
        )

        all_probs.append(torch.sigmoid(max_ema_logits).cpu())
        all_labels.append(labels.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics_from_probs(probs, labels, threshold)
    return metrics, probs, labels


def compute_metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "auroc":     roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


def _strip_orig_mod_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved from torch.compile-wrapped modules."""
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
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
    if metric != "f1":
        raise ValueError("Only metric='f1' is currently supported")

    best_score, best_alpha = -1.0, 0.5
    for alpha in np.arange(0.0, 1.01, 0.05):
        z_ens = ensemble_logits(z1, z2, alpha)
        probs = expit(z_ens)
        preds = (probs >= 0.5).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained linear probe.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_probe.pt saved by train.py")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data",  type=str, default=None)
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Decision threshold (default: find optimal from data)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--ema-alpha",  type=float, default=None,
                        help="EMA smoothing coefficient for inference (default: from checkpoint or 0.1)")
    args = parser.parse_args()

    cfg = ProbeConfig.load(args.config) if args.config else ProbeConfig()
    if args.train_data: cfg.train_file = args.train_data
    if args.eval_data:  cfg.eval_file  = args.eval_data
    if args.batch_size: cfg.batch_size = args.batch_size

    from train import load_base_model

    model, tokenizer = load_base_model(cfg)
    device = next(model.parameters()).device

    # 1. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    probe = LinearProbe(ckpt["hidden_dim"]).to(device)
    probe_state_dict = _strip_orig_mod_prefix(ckpt["probe_state_dict"])
    probe.load_state_dict(probe_state_dict)
    target_layers = ckpt["target_layers"]
    multi_layer   = ckpt.get("multi_layer", True)
    # EMA alpha: CLI arg > checkpoint value > default 0.1
    ema_alpha = args.ema_alpha if args.ema_alpha is not None else ckpt.get("ema_alpha", 0.1)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"train F1={ckpt.get('best_f1', 0):.4f}")
    print(f"Using EMA alpha={ema_alpha} for inference smoothing")
    collector = ActivationCollector(model, target_layers)

    # 2. Dataloader (on-the-fly tokenization)
    eval_data = load_data(cfg.eval_file)
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    if original_truncation_side != "left":
        tokenizer.truncation_side = "left"

    _eval_loader_base = DataLoader(
        eval_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=make_tokenized_collate_fn(tokenizer, cfg.max_sequence_length),
        num_workers=max(cfg.num_workers, 0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=(4 if cfg.num_workers > 0 else None),
    )
    eval_loader = OnTheFlyLoader(
        _eval_loader_base, collector, multi_layer=multi_layer, return_cpu=False
    )

    # 3. Evaluate with EMA smoothing (inference mode, per instruction Section 5.2)
    try:
        threshold = args.threshold if args.threshold is not None else 0.5
        metrics, probs, labels = evaluate_probe(
            probe,
            eval_loader,
            collector,
            multi_layer,
            device,
            ema_alpha=ema_alpha,
            threshold=threshold,
        )
    finally:
        if original_truncation_side != "left":
            tokenizer.truncation_side = original_truncation_side

    if args.threshold is None:
        threshold = find_optimal_threshold(probs, labels)
        metrics = compute_metrics_from_probs(probs, labels, threshold)
        print(f"No --threshold provided, using optimal F1 threshold={threshold:.2f}")

    # 5. Print metrics at fixed threshold
    print(f"\n=== Eval @ threshold={threshold:.2f} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUROC    : {metrics['auroc']:.4f}")

    # 6. Find optimal threshold
    opt_t     = find_optimal_threshold(probs, labels)
    opt_metrics = compute_metrics_from_probs(probs, labels, opt_t)
    print(f"\n=== Optimal F1 threshold={opt_t:.2f} ===")
    print(f"  Precision: {opt_metrics['precision']:.4f}")
    print(f"  Recall   : {opt_metrics['recall']:.4f}")
    print(f"  F1       : {opt_metrics['f1']:.4f}")

    # 7. Ensemble note
    print("\n=== Ensemble ===")
    print("  Use ensemble_logits(z1, z2, alpha=0.5) to combine with external classifier.")
    print("  Use find_optimal_ensemble_alpha(z1, z2, labels) to optimize alpha.")