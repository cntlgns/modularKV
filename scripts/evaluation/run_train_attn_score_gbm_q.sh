#!/bin/bash
# Clean block_qa question-row GBM trainer (CPU-only sklearn). SLURM allocation
# (rtx3090) just isolates cores + memory; GPU left idle.
set -e
IN_DIR="${1:-analysis/attention_score_analysis/attn_raw_q_results}"
OUT_DIR="${2:-analysis/attention_score_analysis/attn_score_q_models}"

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate

echo "============================================="
echo "  Trainer:   attn_score_gbm_q (question rows)"
echo "  In dir:    $IN_DIR"
echo "  Out dir:   $OUT_DIR"
echo "  Host:      $(hostname)  CPU=$(nproc)"
echo "============================================="

python -u scripts/evaluation/train_attn_score_gbm_q.py \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR"
