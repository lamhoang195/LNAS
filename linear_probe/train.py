import contextlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from activation_collector import ActivationCollector, OnTheFlyLoader, make_tokenized_collate_fn
from dataset import load_data
from eval import evaluate_probe
from model import LinearProbe, SwIMSmoother
from probe_config import ProbeConfig
from sw_loss import AnnealedCumulativeMaxLoss, CumulativeMaxLoss, SoftmaxWeightedBCELoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_model(config: ProbeConfig):
    """Load LLM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype.get(config.dtype, torch.bfloat16),
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=config.attn_impl,
        )
    except (ValueError, ImportError):
        print(f"  attn_impl='{config.attn_impl}' không khả dụng, fallback sang sdpa")
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
    model, tokenizer = load_base_model(config)
    probe_device = next(model.parameters()).device
    collector = ActivationCollector(model, config.layers)
    multi_layer = len(config.layers) != 1

    print("Loading data...")
    train_data = load_data(config.train_file)
    eval_data = load_data(config.eval_file)
    print(f"  Train: {len(train_data)} | Eval: {len(eval_data)}")

    workers = max(config.num_workers, 0)
    collate_fn = make_tokenized_collate_fn(tokenizer, config.max_sequence_length)
    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(workers > 0),
        prefetch_factor=(4 if workers > 0 else None),
    )
    # DataLoader only tokenizes (safe for CUDA workers); OnTheFlyLoader
    # collects activations in the main process to avoid CUDA fork issues.
    _train_loader_base = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, **loader_kwargs)
    _eval_loader_base  = DataLoader(eval_data,  batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    train_loader = OnTheFlyLoader(_train_loader_base, collector, multi_layer=multi_layer, return_cpu=False)
    eval_loader  = OnTheFlyLoader(_eval_loader_base,  collector, multi_layer=multi_layer, return_cpu=False)

    # Infer probe input dimension from a single on-the-fly forward pass.
    sample_batch = next(iter(train_loader))
    hidden_dim = sample_batch["activations"].shape[-1]
    del sample_batch
    print(f"Hidden dim: {hidden_dim}")

    probe = LinearProbe(hidden_dim).to(probe_device)
    smoother = SwIMSmoother(config.window_size).to(probe_device)

    optimizer = AdamW(probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    if config.use_compile and hasattr(torch, "compile"):
        probe = torch.compile(probe)
        smoother = torch.compile(smoother, dynamic=True)
        print("  torch.compile enabled")

    use_amp = config.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    print(f"  AMP: {use_amp}")

    if config.loss_type == "cummax":
        criterion = CumulativeMaxLoss(config.window_size)
        print("  Loss: Cumulative Max")
    elif config.loss_type == "annealed_cummax":
        anneal_steps = len(train_loader) * config.num_epochs
        criterion = AnnealedCumulativeMaxLoss(config.temperature, config.window_size, anneal_steps)
        print(f"  Loss: Annealed Cumulative Max (total_steps={anneal_steps})")
    else:
        criterion = SoftmaxWeightedBCELoss(config.temperature, config.window_size)
        print("  Loss: Softmax-Weighted BCE")

    os.makedirs(config.save_dir, exist_ok=True)
    best_f1 = 0.0
    patience_cnt = 0

    print(
        f"\nTraining: {config.num_epochs} epochs, M={config.window_size}, "
        f"tau={config.temperature}, loss={config.loss_type}, "
        f"layers={config.layers or 'all'}\n"
    )

    global_step = 0
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    if original_truncation_side != "left":
        tokenizer.truncation_side = "left"

    try:
        for epoch in range(config.num_epochs):
            probe.train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
                # activations already collected by OnTheFlyLoader
                features = batch["activations"].to(probe_device, non_blocking=True)
                labels   = batch["labels"].to(probe_device, non_blocking=True)
                mask     = batch["attention_mask"].to(probe_device, non_blocking=True)

                with amp_ctx:
                    smoothed = smoother(probe(features), mask)
                    if config.loss_type == "annealed_cummax":
                        loss = criterion(smoothed, labels, mask, global_step)
                    else:
                        loss = criterion(smoothed, labels, mask)
                global_step += 1

                scaled_loss = loss / config.gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if (i + 1) % config.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                total_loss += loss.item()
                n_batches += 1

                del features, labels, mask

            if len(train_loader) % config.gradient_accumulation_steps != 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(probe.parameters(), config.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            print(
                f"Epoch {epoch + 1}/{config.num_epochs} - "
                f"Loss: {total_loss / max(n_batches, 1):.4f} - {time.time() - t0:.1f}s"
            )

            if (epoch + 1) % config.eval_interval == 0:
                metrics, _, _ = evaluate_probe(
                    probe,
                    eval_loader,
                    collector,
                    multi_layer,
                    probe_device,
                    ema_alpha=getattr(config, "ema_alpha", 0.1),
                )
                print(
                    f"  Eval - Acc: {metrics['accuracy']:.4f} | "
                    f"F1: {metrics['f1']:.4f} | Prec: {metrics['precision']:.4f} | "
                    f"Rec: {metrics['recall']:.4f} | AUROC: {metrics['auroc']:.4f}"
                )

                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    patience_cnt = 0
                    torch.save(
                        {
                            "probe_state_dict": probe.state_dict(),
                            "hidden_dim": hidden_dim,
                            "target_layers": config.layers,
                            "multi_layer": multi_layer,
                            "window_size": config.window_size,
                            "ema_alpha": config.ema_alpha,
                            "best_f1": best_f1,
                            "epoch": epoch + 1,
                            "loss_type": config.loss_type,
                        },
                        os.path.join(config.save_dir, "best_probe.pt"),
                    )
                    print(f"  New best F1: {best_f1:.4f}")
                else:
                    patience_cnt += 1
                    if patience_cnt >= config.early_stop_patience:
                        print("  Early stopping.")
                        break
    finally:
        if original_truncation_side != "left":
            tokenizer.truncation_side = original_truncation_side

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
    parser.add_argument("--batch-size",         type=int,   default=None)
    parser.add_argument("--loss-type", type=str, default=None,
                        choices=["softmax_weighted", "cummax", "annealed_cummax"])
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
    if args.batch_size:         cfg.batch_size         = args.batch_size
    if args.loss_type:          cfg.loss_type          = args.loss_type

    train(cfg)
