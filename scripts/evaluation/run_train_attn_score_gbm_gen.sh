#!/bin/bash
# Args (all optional, passed straight through to train_attn_score_gbm_gen.py):
#   $1 IN_DIR       (default analysis/attention_score_analysis/attn_raw_gen_results)
#   $2 OUT_DIR      (default analysis/attention_score_analysis/attn_score_gen_models)
#
# Note: HistGradientBoostingRegressor from sklearn is CPU-only. This script
# is intended for a SLURM allocation (e.g. rtx3090) where the GPU is left
# idle and we only need isolated CPU cores + memory off the login node.
set -e
IN_DIR="${1:-analysis/attention_score_analysis/attn_raw_gen_results}"
OUT_DIR="${2:-analysis/attention_score_analysis/attn_score_gen_models}"

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate

echo "============================================="
echo "  Trainer:   attn_score_gbm_gen"
echo "  In dir:    $IN_DIR"
echo "  Out dir:   $OUT_DIR"
echo "  Host:      $(hostname)  CPU=$(nproc)  GPU(idle)=$(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")')"
echo "============================================="

python -u scripts/evaluation/train_attn_score_gbm_gen.py \
    --in_dir "$IN_DIR" \
    --out_dir "$OUT_DIR"
