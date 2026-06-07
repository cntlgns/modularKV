#!/bin/bash
# Doc-block attention analysis (baseline vs recover_pos_enc) over 5 datasets.
#   $1 MODEL      (default meta-llama/Llama-3.2-1B-Instruct)
#   $2 N_SAMPLES  (default 10)
set -e
MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
N="${2:-10}"

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "============================================="
echo "  Model:     $MODEL"
echo "  N samples: $N"
echo "  GPU:       $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="

python scripts/evaluation/attn_doc_block.py \
    --model "$MODEL" \
    --datasets nq block_qa hqa_full wiki musique \
    --policies baseline recover_pos_enc \
    --n_samples "$N" \
    --out_dir result/attn_doc_block \
    --device cuda --dtype bfloat16
