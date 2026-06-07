#!/bin/bash
# KV-cache swap experiment wrapper (baseline <-> recover_pos_enc).
#
# The two swap policies prefill the SAME sequence twice (full-causal baseline +
# block-diagonal recover_pos_enc, both contiguous positions), then splice a
# hybrid cache -- system-prefix+docs from one policy, the question band from the
# other -- and decode. Isolates whether a well-restored QUESTION cache can
# rescue a broken DOCUMENT cache (and vice versa).
#
#   swap_docbase_qrec  docs = baseline,        question = recover_pos_enc
#   swap_docrec_qbase  docs = recover_pos_enc, question = baseline
#
# baseline / recover_pos_enc are accepted too (ceiling / floor references).
# batch_size is forced to 1 (the swap splices a per-sample cache), reencode=0.
#
# Usage:
#   bash run_kvswap.sh <bench> <model> <mode> <pos> <kv_policy> <prompt_preset> [max_examples] [out_dir]
set -e

BENCH="$1"
MODEL="$2"
MODE="$3"
POS="${4:-0}"
KV_POLICY="${5:-swap_docbase_qrec}"
PRESET="${6:-kvlink}"
MAXEX="${7:-1000}"
OUT="${8:-result/kvswap}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Qwen3 needs the transformers>=4.51 venv (.venv-q3); Llama uses the default
# .venv (4.46.3). run_eval.sh honours the PYTHON env override.
case "$MODEL" in
    *Qwen3*|*qwen3*) export PYTHON="$PROJECT_ROOT/.venv-q3/bin/python" ;;
esac

export MAX_EXAMPLES="$MAXEX"
export OUT_DIR="$OUT"

# run_eval.sh positional: bench model mode pos bsz kv_policy modular_q_pos prompt_preset reencode
exec bash "$PROJECT_ROOT/scripts/evaluation/run_eval.sh" \
    "$BENCH" "$MODEL" "$MODE" "$POS" 1 "$KV_POLICY" summed_pos "$PRESET" 0
