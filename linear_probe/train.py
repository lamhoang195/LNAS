"""Training pipeline for the Linear Probe classifier."""

import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from probe_config import ProbeConfig
from model import LinearProbe, SwIMSmoother
from sw_loss import SoftmaxWeightedBCELoss
from activation_collector import (
    CachedActivationDataset, cached_collate_fn, precompute_activations,
)
from dataset import load_data, load_data_split
from eval import evaluate_probe


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(config: ProbeConfig):
    """Load LLM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype.get(config.dtype, torch.bfloat16),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def train(config: ProbeConfig):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model, tokenizer = load_base_model(config)

    # 2. Load data
    print("Loading data...")
    if config.eval_file and os.path.exists(config.eval_file) and os.listdir(config.eval_file):
        train_data = load_data(config.train_file)
        eval_data = load_data(config.eval_file)
    else:
        # Per-file split from train directory
        train_data, eval_data = load_data_split(
            config.train_file,
            train_ratio=config.train_ratio,
            eval_ratio=config.eval_ratio,
            seed=config.seed,
        )
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Cap sample counts if requested
    if config.max_train_samples > 0:
        random.shuffle(train_data)
        train_data = train_data[:config.max_train_samples]
    if config.max_eval_samples > 0:
        random.shuffle(eval_data)
        eval_data = eval_data[:config.max_eval_samples]
    if config.max_train_samples > 0 or config.max_eval_samples > 0:
        print(f"  After cap → Train: {len(train_data)} | Eval: {len(eval_data)}")
    multi_layer = len(config.layers) != 1
    train_cache = os.path.join(config.cache_root, "train")
    eval_cache = os.path.join(config.cache_root, "eval")

    for name, data, cdir in [("train", train_data, train_cache),
                              ("eval", eval_data, eval_cache)]:
        print(f"Caching {name} activations...")
        precompute_activations(
            model, tokenizer, data, config.layers,
            cdir, config.max_sequence_length, multi_layer,
            batch_size=config.cache_batch_size,
        )

    del model
    torch.cuda.empty_cache()

    # 4. Dataloaders
    train_loader = DataLoader(
        CachedActivationDataset(train_cache), batch_size=config.batch_size,
        shuffle=True, collate_fn=cached_collate_fn, num_workers=8, pin_memory=True,
    )
    eval_loader = DataLoader(
        CachedActivationDataset(eval_cache), batch_size=config.batch_size,
        shuffle=False, collate_fn=cached_collate_fn, num_workers=8, pin_memory=True,
    )

    # 5. Probe + optimizer
    hidden_dim = CachedActivationDataset(train_cache)[0]["activations"].shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    probe = LinearProbe(hidden_dim).to(device)
    smoother = SwIMSmoother(config.window_size).to(device)
    criterion = SoftmaxWeightedBCELoss(config.temperature, config.window_size)

    optimizer = AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 6. Training loop
    os.makedirs(config.save_dir, exist_ok=True)
    best_f1, patience_cnt = 0.0, 0
    print(f"\nTraining: {config.num_epochs} epochs, M={config.window_size}, "
          f"tau={config.temperature}, layers={config.layers or 'all'}\n")

    for epoch in range(config.num_epochs):
        probe.train()
        total_loss, n = 0.0, 0
        t0 = time.time()

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            acts = batch["activations"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = criterion(smoother(probe(acts), mask), labels, mask)
            (loss / config.gradient_accumulation_steps).backward()

            if (i + 1) % config.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            n += 1

        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Loss: {total_loss/max(n,1):.4f} - {time.time()-t0:.1f}s")

        # Evaluate
        if (epoch + 1) % config.eval_interval == 0:  # eval_interval from ProbeConfig
            metrics = evaluate_probe(
                probe, smoother, eval_loader, device, config.window_size)
            print(f"  Eval - Acc: {metrics['accuracy']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | "
                  f"Rec: {metrics['recall']:.4f} | AUROC: {metrics['auroc']:.4f}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                patience_cnt = 0
                torch.save({
                    "probe_state_dict": probe.state_dict(),
                    "hidden_dim": hidden_dim,
                    "target_layers": config.layers,
                    "multi_layer": multi_layer,
                    "window_size": config.window_size,
                    "best_f1": best_f1,
                    "epoch": epoch + 1,
                }, os.path.join(config.save_dir, "best_probe.pt"))
                print(f"  New best F1: {best_f1:.4f}")
            else:
                patience_cnt += 1
                if patience_cnt >= config.early_stop_patience:
                    print("  Early stopping.")
                    break

    print(f"\nDone. Best F1: {best_f1:.4f}")
    return probe


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--eval-data", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--eval-ratio",  type=float, default=None)
    parser.add_argument("--max-train-samples",  type=int,   default=None)
    parser.add_argument("--max-eval-samples",   type=int,   default=None)
    parser.add_argument("--cache-batch-size",   type=int,   default=None)
    parser.add_argument("--batch-size",         type=int,   default=None)
    args = parser.parse_args()

    cfg = ProbeConfig.load(args.config) if args.config else ProbeConfig()
    if args.model: cfg.model_name = args.model
    if args.layers: cfg.layers = args.layers
    if args.epochs: cfg.num_epochs = args.epochs
    if args.lr: cfg.learning_rate = args.lr
    if args.window_size: cfg.window_size = args.window_size
    if args.temperature: cfg.temperature = args.temperature
    if args.train_data:  cfg.train_file   = args.train_data
    if args.eval_data:   cfg.eval_file    = args.eval_data
    if args.train_ratio:        cfg.train_ratio        = args.train_ratio
    if args.eval_ratio:         cfg.eval_ratio         = args.eval_ratio
    if args.max_train_samples:  cfg.max_train_samples  = args.max_train_samples
    if args.max_eval_samples:   cfg.max_eval_samples   = args.max_eval_samples
    if args.cache_batch_size:   cfg.cache_batch_size   = args.cache_batch_size
    if args.batch_size:         cfg.batch_size         = args.batch_size

    train(cfg)

    train(cfg)
