#!/bin/bash
# Thin wrapper around run_eval.sh for the oracle_ab_vrec K-vs-V experiment.
# Pins MAX_EXAMPLES + OUT_DIR (env-driven in run_eval.sh) and forces the
# untrained 8B base model, reencode_num=0, kvlink prompt. batch_size is 1
# (oracle_ab_vrec requires B=1; references run B=1 too for uniform OOM-safe
# numbers, matching sweep_ablation).
#
# Usage:
#   bash run_oracle_ab_vrec.sh <bench> <model> <mode> <pos> <kv_policy> [max_examples] [out_dir]
set -e

BENCH="$1"
MODEL="$2"
MODE="$3"
POS="${4:-0}"
KV_POLICY="${5:-oracle_ab_vrec}"
MAXEX="${6:-1000}"
OUT="${7:-result/oracle_ab_vrec}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export MAX_EXAMPLES="$MAXEX"
export OUT_DIR="$OUT"

# run_eval.sh positional: bench model mode pos bsz kv_policy modular_q_pos prompt_preset reencode
exec bash "$PROJECT_ROOT/scripts/evaluation/run_eval.sh" \
    "$BENCH" "$MODEL" "$MODE" "$POS" 1 "$KV_POLICY" summed_pos kvlink 0
