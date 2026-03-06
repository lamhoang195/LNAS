#!/bin/bash
# =============================================================
#  Two-Tier Pipeline — Quick-run script
# =============================================================
# Usage:
#   bash scripts/run_pipeline.sh                       # interactive
#   bash scripts/run_pipeline.sh --prompt "Tell me ..."
#   bash scripts/run_pipeline.sh --input_file data/instructions/test/advbench_test.json

cd "$(dirname "$0")/.."

PROBE_CKPT="linear_probe/checkpoints/linear_probe_model.pt"
STEER_MAT="data/steering_matrix/steering_matrix_llama3.1.pt"

python two_tier_pipeline.py \
    --model_name  meta-llama/Llama-3.1-8B-Instruct \
    --dtype       bfloat16 \
    --device      cuda:0 \
    --probe_checkpoint  "$PROBE_CKPT" \
    --ema_alpha   0.1 \
    --threshold   0.5 \
    --warmup_tokens 2 \
    --steering_matrix   "$STEER_MAT" \
    --steering_strength -0.3 \
    --steering_layers   "8,9,10,11,12,13,14,16,18,19" \
    --max_new_tokens    256 \
    --interactive \
    "$@"
