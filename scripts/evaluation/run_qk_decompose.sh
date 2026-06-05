#!/bin/bash
set -e
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
python scripts/evaluation/attn_qk_decompose.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --datasets nq musique --policies baseline recover_pos_enc \
    --n_samples 8 --out_dir result/attn_qk_decompose --device cuda --dtype bfloat16
