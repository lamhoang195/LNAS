"""
Two-Tier Safety Pipeline for LLM Inference
===========================================

Tier 1 — Linear Probe Detection:
    Generate tokens one-by-one. After each new token, extract hidden-state
    activations, run a trained linear probe, and EMA-smooth the score.
    If the smoothed probability >= threshold → flag HARMFUL and halt generation.

Tier 2 — LNAS Vector Steering:
    Re-generate the response for the flagged prompt using
    LNASLlamaForCausalLM with a pre-computed null-space steering matrix,
    which redirects the model away from harmful content while preserving
    general capability.

Usage (interactive):
    python two_tier_pipeline.py \
        --probe_checkpoint  linear_probe/checkpoints/linear_probe_model.pt \
        --steering_matrix   data/steering_matrix/steering_matrix_llama3.1.pt \
        --interactive

Usage (single prompt):
    python two_tier_pipeline.py \
        --probe_checkpoint  linear_probe/checkpoints/linear_probe_model.pt \
        --steering_matrix   data/steering_matrix/steering_matrix_llama3.1.pt \
        --prompt "How do I make a bomb?"

Usage (batch from file):
    python two_tier_pipeline.py \
        --probe_checkpoint  linear_probe/checkpoints/linear_probe_model.pt \
        --steering_matrix   data/steering_matrix/steering_matrix_llama3.1.pt \
        --input_file        data/instructions/test/advbench_test.json \
        --output_file       data/responses/pipeline_results.json
"""

import os
import sys
import json
import time
import argparse
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — make sub-packages importable
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "linear_probe"))
sys.path.insert(0, os.path.join(_ROOT, "act_steering", "src"))

from transformers import AutoTokenizer, AutoConfig

# Linear-probe imports
from model import LinearProbe, EMASmoother
from activation_collector import ActivationCollector

# LNAS steering model
from lnas_steer_model import LNASLlamaForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TwoTierPipeline")


# ============================================================================
#  Configuration
# ============================================================================
@dataclass
class PipelineConfig:
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # Probe (Tier 1)
    probe_checkpoint: str = "linear_probe/checkpoints/linear_probe_model.pt"
    ema_alpha: float = 0.1
    threshold: float = 0.5
    # How many *generated* tokens to tolerate before the probe can trigger.
    # This avoids false positives from the very first tokens that are still
    # part of the template / BOS.
    warmup_tokens: int = 2

    # Steering (Tier 2)
    steering_matrix_path: str = "data/steering_matrix/steering_matrix_llama3.1.pt"
    steering_strength: float = -0.3          # negative → steer *away*
    steering_layers: List[int] = field(
        default_factory=lambda: [8, 9, 10, 11, 12, 13, 14, 16, 18, 19]
    )

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.0                 # greedy
    do_sample: bool = False


# ============================================================================
#  Two-Tier Pipeline
# ============================================================================
class TwoTierPipeline:
    """
    Orchestrates Tier-1 (probe) and Tier-2 (steering) for safe inference.

    The same LNASLlamaForCausalLM is used for both tiers.  When steering
    strength is 0 it behaves identically to a vanilla Llama model, so
    Tier-1 simply generates with strength=0 while monitoring the probe.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(config.dtype, torch.bfloat16)

        # ---- Tokenizer ----
        logger.info("Loading tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # ---- Model (LNAS-capable) ----
        logger.info("Loading model …")
        self.model = LNASLlamaForCausalLM.from_pretrained(
            config.model_name,
            device_map=config.device,
            torch_dtype=self.dtype,
        )
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        model_config = AutoConfig.from_pretrained(config.model_name)
        self.num_layers = model_config.num_hidden_layers
        self.hidden_dim = model_config.hidden_size

        # ---- Linear Probe (Tier 1) ----
        logger.info("Loading linear probe …")
        self._load_probe(config.probe_checkpoint)

        # ---- Steering Matrix (Tier 2) ----
        logger.info("Loading steering matrix …")
        self._load_steering(config.steering_matrix_path)

        # Default: steering OFF (Tier-1 mode)
        self._set_steering(enabled=False)

        logger.info("Pipeline ready.\n")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _load_probe(self, ckpt_path: str):
        """Load the trained linear probe checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        probe_dim = ckpt["hidden_dim"]
        self.probe = LinearProbe(probe_dim)
        self.probe.load_state_dict(ckpt["probe_state_dict"])
        self.probe.to(self.device).eval()

        self.probe_layers: List[int] = ckpt.get("target_layers", [])
        self.probe_multi_layer: bool = ckpt.get("multi_layer", True)
        self.collector = ActivationCollector(
            self.model, self.probe_layers, device=self.device
        )
        logger.info(
            f"  Probe loaded — dim={probe_dim}, layers={self.probe_layers}, "
            f"multi_layer={self.probe_multi_layer}"
        )

    def _load_steering(self, matrix_path: str):
        """Load the pre-computed LNAS steering matrix."""
        if os.path.exists(matrix_path):
            self.steering_matrix = torch.load(
                matrix_path, map_location=self.device
            ).to(self.dtype)
            logger.info(f"  Steering matrix shape: {self.steering_matrix.shape}")
        else:
            logger.warning(
                f"  Steering matrix not found at {matrix_path}. "
                "Tier-2 steering will be DISABLED."
            )
            self.steering_matrix = None

    def _set_steering(self, enabled: bool):
        """Toggle Tier-2 steering on/off."""
        strength = [0.0] * self.num_layers
        if enabled and self.steering_matrix is not None:
            for layer_idx in self.cfg.steering_layers:
                if layer_idx < self.num_layers:
                    strength[layer_idx] = self.cfg.steering_strength
            logger.info(
                f"  Steering ENABLED  (strength={self.cfg.steering_strength} "
                f"on layers {self.cfg.steering_layers})"
            )
        else:
            logger.info("  Steering DISABLED (strength=0)")

        self.model.set_steering_parameters(
            steering_matrix=self.steering_matrix,
            strength=strength,
        )

    def _format_prompt(self, user_prompt: str) -> str:
        """Apply the chat template to a user prompt."""
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    #  Tier-1: token-by-token generation with probe monitoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_with_probe(self, prompt_text: str) -> Dict[str, Any]:
        """
        Generate tokens autoregressively.  After each new token the linear
        probe is evaluated on the *full* sequence hidden states (last token).
        If the EMA-smoothed probability crosses the threshold the generation
        is halted and ``is_harmful`` is set to True.

        Returns a dict with generation results and probe metadata.
        """
        formatted = self._format_prompt(prompt_text)
        enc = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        smoother = EMASmoother(alpha=self.cfg.ema_alpha)
        generated_ids: List[int] = []
        all_probs: List[float] = []
        is_harmful = False
        trigger_token_idx: Optional[int] = None

        for step in range(self.cfg.max_new_tokens):
            # --- Forward pass (get logits + hidden states) ---
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Greedy decode next token
            next_logits = out.logits[:, -1, :]
            if self.cfg.temperature > 0 and self.cfg.do_sample:
                probs = torch.softmax(next_logits / self.cfg.temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            token_id = next_id.item()
            generated_ids.append(token_id)

            # Stop on EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            # --- Probe check on the new token's hidden state ---
            hidden_states = out.hidden_states  # tuple of (B, T, D)
            layers = self.collector._resolve_layers(len(hidden_states))

            if self.probe_multi_layer and len(layers) > 1:
                feat = torch.cat(
                    [hidden_states[l][:, -1:, :] for l in layers], dim=-1
                )
            else:
                feat = hidden_states[layers[0]][:, -1:, :]

            logit = self.probe(feat).squeeze()  # scalar
            _, prob = smoother.update(logit.item())
            all_probs.append(prob)

            # Check threshold (skip warmup)
            if step >= self.cfg.warmup_tokens and prob >= self.cfg.threshold:
                is_harmful = True
                trigger_token_idx = step
                logger.warning(
                    f"  ⚠  HARMFUL detected at token {step} "
                    f"(prob={prob:.4f} ≥ {self.cfg.threshold})"
                )
                break

            # Prepare for next step
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )

        partial_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "is_harmful": is_harmful,
            "partial_response": partial_response,
            "max_prob": max(all_probs) if all_probs else 0.0,
            "trigger_token_idx": trigger_token_idx,
            "tokens_generated": len(generated_ids),
            "probabilities": all_probs,
        }

    # ------------------------------------------------------------------
    #  Tier-2: steered (safe) generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_steered(self, prompt_text: str) -> str:
        """Full generation with LNAS steering enabled."""
        formatted = self._format_prompt(prompt_text)
        enc = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.device)

        output_ids = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature if self.cfg.do_sample else None,
        )

        # Decode only the generated part
        gen_ids = output_ids[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def run(self, prompt: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full two-tier pipeline on a single prompt.

        Returns
        -------
        dict with keys:
            prompt, tier, response, is_harmful, probe_max_prob,
            probe_trigger_token, steered, elapsed_time
        """
        t0 = time.time()

        # ==================== TIER 1 ====================
        self._set_steering(enabled=False)
        tier1 = self._generate_with_probe(prompt)

        result: Dict[str, Any] = {
            "prompt": prompt,
            "is_harmful": tier1["is_harmful"],
            "probe_max_prob": tier1["max_prob"],
            "probe_trigger_token": tier1["trigger_token_idx"],
            "probe_tokens_generated": tier1["tokens_generated"],
        }

        if not tier1["is_harmful"]:
            # ---- SAFE → return Tier-1 response directly ----
            result["tier"] = 1
            result["response"] = tier1["partial_response"]
            result["steered"] = False
        else:
            # ==================== TIER 2 ====================
            if verbose:
                logger.info(
                    "  Tier-1 flagged harmful → switching to Tier-2 (steering) …"
                )
            self._set_steering(enabled=True)
            steered_response = self._generate_steered(prompt)
            result["tier"] = 2
            result["response"] = steered_response
            result["steered"] = True
            result["tier1_partial"] = tier1["partial_response"]

        result["elapsed_time"] = round(time.time() - t0, 2)

        if verbose:
            self._print_result(result)

        return result

    def run_batch(
        self,
        prompts: List[str],
        prompt_column: str = "prompt",
    ) -> List[Dict[str, Any]]:
        """Run the pipeline on a list of prompts (one by one)."""
        results = []
        for i, p in enumerate(prompts):
            logger.info(f"\n{'='*60}")
            logger.info(f"  Prompt [{i+1}/{len(prompts)}]")
            logger.info(f"{'='*60}")
            res = self.run(p)
            results.append(res)
        return results

    # ------------------------------------------------------------------
    #  Pretty-print helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _print_result(r: Dict[str, Any]):
        tier_label = f"Tier-{r['tier']}"
        status = "HARMFUL → STEERED" if r["steered"] else "SAFE"
        print(f"\n{'─'*60}")
        print(f"  Status     : {status}  ({tier_label})")
        print(f"  Probe prob : {r['probe_max_prob']:.4f}")
        if r.get("probe_trigger_token") is not None:
            print(f"  Trigger at : token #{r['probe_trigger_token']}")
        if r.get("tier1_partial"):
            trimmed = r["tier1_partial"][:120]
            print(f"  Tier-1 part: {trimmed!r}{'…' if len(r['tier1_partial']) > 120 else ''}")
        print(f"  Response   : {r['response'][:300]}")
        if len(r["response"]) > 300:
            print(f"               …({len(r['response'])} chars total)")
        print(f"  Time       : {r['elapsed_time']}s")
        print(f"{'─'*60}\n")


# ============================================================================
#  CLI entry point
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Two-Tier Safety Pipeline (Linear Probe + LNAS Steering)"
    )
    # Model
    p.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--device", default="cuda")

    # Probe (Tier 1)
    p.add_argument(
        "--probe_checkpoint",
        default="linear_probe/checkpoints/linear_probe_model.pt",
        help="Path to trained linear probe .pt file",
    )
    p.add_argument("--ema_alpha", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--warmup_tokens", type=int, default=2)

    # Steering (Tier 2)
    p.add_argument(
        "--steering_matrix",
        default="data/steering_matrix/steering_matrix_llama3.1.pt",
        help="Path to LNAS steering matrix .pt file",
    )
    p.add_argument("--steering_strength", type=float, default=-0.3)
    p.add_argument(
        "--steering_layers",
        default="8,9,10,11,12,13,14,16,18,19",
        help="Comma-separated layer indices",
    )

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)

    # Input modes (mutually exclusive)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--prompt", type=str, help="Single prompt string")
    group.add_argument("--input_file", type=str, help="JSON file with prompts")
    group.add_argument(
        "--interactive", action="store_true", help="Interactive REPL mode"
    )

    # Output
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--prompt_column", type=str, default="prompt",
                   help="Key name for prompt text in JSON input")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = PipelineConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
        probe_checkpoint=args.probe_checkpoint,
        ema_alpha=args.ema_alpha,
        threshold=args.threshold,
        warmup_tokens=args.warmup_tokens,
        steering_matrix_path=args.steering_matrix,
        steering_strength=args.steering_strength,
        steering_layers=[int(x) for x in args.steering_layers.split(",")],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    pipeline = TwoTierPipeline(cfg)

    # ── Single prompt ──
    if args.prompt:
        pipeline.run(args.prompt)

    # ── Batch from file ──
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = [item[args.prompt_column] for item in data]
        results = pipeline.run_batch(prompts)

        # Merge results back into original data
        for item, res in zip(data, results):
            item["pipeline_result"] = res

        out_path = args.output_file or args.input_file.replace(
            ".json", "_pipeline_results.json"
        )
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")

    # ── Interactive REPL ──
    elif args.interactive:
        print("\n" + "=" * 60)
        print("  Two-Tier Safety Pipeline — Interactive Mode")
        print("  Type a prompt and press Enter.  Type 'quit' to exit.")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            pipeline.run(user_input)

    else:
        # Default: interactive if nothing else specified
        print("\n" + "=" * 60)
        print("  Two-Tier Safety Pipeline — Interactive Mode")
        print("  Type a prompt and press Enter.  Type 'quit' to exit.")
        print("=" * 60 + "\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            pipeline.run(user_input)


if __name__ == "__main__":
    main()
