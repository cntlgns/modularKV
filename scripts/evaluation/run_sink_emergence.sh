#!/bin/bash
set -e
MODEL="${1:-meta-llama/Llama-3.2-1B-Instruct}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')  MODEL=$MODEL"
python scripts/evaluation/attn_sink_emergence.py --model "$MODEL" \
    --datasets nq musique --policies baseline recover_pos_enc \
    --n_samples 6 --out_dir result/attn_sink_emergence --device cuda --dtype bfloat16
