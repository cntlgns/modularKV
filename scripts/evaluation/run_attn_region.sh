#!/bin/bash
# Run one (config, dataset) region-attention comparison job on a single GPU.
#
# Args:
#   $1 MODEL         HF id or local dir
#   $2 POLICY        baseline | recover_pos_enc
#   $3 CONFIG_NAME   tag for output filename (baseline / base_broken / kvlink0_step4000)
#   $4 DATASET       nq | hqa_gold | hqa_full | wiki | musique | block_qa
#   $5 N_SAMPLES     (default 40)
#   $6 MAX_NEW       (default 128)
set -e
MODEL="$1"
POLICY="$2"
CONFIG_NAME="$3"
DATASET="$4"
N_SAMPLES="${5:-40}"
MAX_NEW="${6:-128}"

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================="
echo "  Model:       $MODEL"
echo "  Policy:      $POLICY"
echo "  Config:      $CONFIG_NAME"
echo "  Dataset:     $DATASET"
echo "  N samples:   $N_SAMPLES"
echo "  Max new:     $MAX_NEW"
echo "  GPU:         $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="

python scripts/evaluation/attn_region_compare.py \
    --model "$MODEL" \
    --policy "$POLICY" \
    --config_name "$CONFIG_NAME" \
    --dataset "$DATASET" \
    --n_samples "$N_SAMPLES" \
    --max_new_tokens "$MAX_NEW" \
    --out_dir result/attn_region_compare \
    --device cuda --dtype bfloat16
