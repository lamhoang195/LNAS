import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from typing import Dict, Tuple

from activation_collector import ActivationCollector, OnTheFlyLoader, make_tokenized_collate_fn
from dataset import load_data
from model import EMASmoother, LinearProbe
from probe_config import ProbeConfig


def _strip_orig_mod_prefix(state_dict):
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
        for key, value in state_dict.items()
    }


@torch.no_grad()
def evaluate_probe(
    probe: LinearProbe,
    dataloader: DataLoader,
    collector=None,
    multi_layer: bool = True,
    device: torch.device = None,
    ema_alpha: float = 0.1,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Exchange-level prediction uses the max EMA-smoothed logit over the sequence.
    This matches the streaming deployment path from the write-up.
    """
    del collector
    del multi_layer

    if device is None:
        device = next(probe.parameters()).device

    probe.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        acts = batch["activations"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]

        logits = probe(acts)
        max_logits = EMASmoother.max_ema_logits(logits, mask=mask, alpha=ema_alpha)
        all_probs.append(torch.sigmoid(max_logits).cpu())
        all_labels.append(labels.cpu())

    if not all_probs:
        raise ValueError("Evaluation dataloader is empty; no batches were produced.")

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }
    return metrics, probs, labels


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
    parser = argparse.ArgumentParser(description="Evaluate a trained linear probe.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_probe.pt saved by train.py")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Base model name (must match the model used in train.py)")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: find optimal from data)")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = ProbeConfig.load(args.config) if args.config else ProbeConfig()
    if args.model:       cfg.model_name = args.model
    if args.train_data: cfg.train_file = args.train_data
    if args.eval_data:  cfg.eval_file  = args.eval_data
    if args.batch_size: cfg.batch_size = args.batch_size

    # 1. Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    target_layers = ckpt["target_layers"]
    multi_layer = ckpt.get("multi_layer", True)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"train F1={ckpt.get('best_f1', 0):.4f}")

    # 2. Build on-the-fly eval loader so activation computation matches training.
    from train import load_base_model

    model, tokenizer = load_base_model(cfg)
    probe_device = next(model.parameters()).device
    probe = LinearProbe(ckpt["hidden_dim"]).to(probe_device)
    probe.load_state_dict(_strip_orig_mod_prefix(ckpt["probe_state_dict"]))

    eval_data = load_data(cfg.eval_file)
    collector = ActivationCollector(model, target_layers)

    workers = max(cfg.num_workers, 0)
    collate_fn = make_tokenized_collate_fn(tokenizer, cfg.max_sequence_length)
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
    )
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    if original_truncation_side != "left":
        tokenizer.truncation_side = "left"

    try:
        eval_loader = OnTheFlyLoader(
            DataLoader(eval_data, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs),
            collector,
            multi_layer=multi_layer,
            return_cpu=False,
        )

        # 3. Metrics at fixed threshold
        threshold = args.threshold if args.threshold is not None else cfg.threshold
        metrics, probs, labels = evaluate_probe(
            probe,
            eval_loader,
            collector,
            multi_layer,
            probe_device,
            ema_alpha=ckpt.get("ema_alpha", cfg.ema_alpha),
            threshold=threshold,
        )
    finally:
        if original_truncation_side != "left":
            tokenizer.truncation_side = original_truncation_side

    print(f"\n=== Eval @ threshold={threshold:.2f} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  AUROC    : {metrics['auroc']:.4f}")

    # 4. Find optimal threshold
    opt_t = find_optimal_threshold(probs, labels)
    opt_preds = (probs >= opt_t).astype(int)
    print(f"\n=== Optimal F1 threshold={opt_t:.2f} ===")
    print(f"  Precision: {precision_score(labels, opt_preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(labels, opt_preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(labels, opt_preds, zero_division=0):.4f}")

    # 5. Ensemble note
    print("\n=== Ensemble ===")
    print("  Use ensemble_logits(z1, z2, alpha=0.5) to combine with external classifier.")
    print("  Use find_optimal_ensemble_alpha(z1, z2, labels) to optimize alpha.")