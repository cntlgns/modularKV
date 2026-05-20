#!/bin/bash
# Run one (model, dataset) attention-analysis job on a single GPU.
#
# Usage:
#   bash scripts/evaluation/run_attn_analysis.sh <model> <dataset> [n_samples]
#
#   <model>      HF id, e.g. meta-llama/Llama-3.2-1B-Instruct
#   <dataset>    nq | hqa_gold | hqa_full | wiki | musique
#   [n_samples]  default 100
set -e

MODEL="$1"
DATASET="$2"
N="${3:-100}"

if [ -z "$MODEL" ] || [ -z "$DATASET" ]; then
    echo "Usage: bash run_attn_analysis.sh <model> <dataset> [n_samples]" >&2
    exit 1
fi

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
echo "============================================="
echo "  Model:      $MODEL"
echo "  Dataset:    $DATASET"
echo "  Samples:    $N"
echo "  GPU:        $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="

python scripts/evaluation/attn_analysis.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_samples "$N" \
    --policies baseline recover_pos_enc modular recover_cross_attn_oracle_pos \
    --out_dir attn_analysis_results \
    --device cuda \
    --dtype bfloat16
