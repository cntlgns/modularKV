#!/bin/bash
set -e
MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
N="${2:-10}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
python scripts/evaluation/attn_within_doc.py \
    --model "$MODEL" --datasets nq block_qa hqa_full wiki musique \
    --policies baseline recover_pos_enc --n_samples "$N" \
    --out_dir result/attn_within_doc --device cuda --dtype bfloat16
