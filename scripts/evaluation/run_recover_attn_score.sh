#!/bin/bash
set -e
MODEL="$1"
DATASET="$2"
N="${3:-100}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "============================================="
echo "  Model:    $MODEL"
echo "  Dataset:  $DATASET"
echo "  Samples:  $N"
echo "  GPU:      $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="
python scripts/evaluation/eval_recover_attn_score.py \
    --model "$MODEL" --dataset "$DATASET" --n_samples "$N" \
    --policies baseline recover_pos_enc recover_attn_score \
    --gbm_dir attn_score_models \
    --out_dir recover_attn_score_results \
    --device cuda --dtype bfloat16
