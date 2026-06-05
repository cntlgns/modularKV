#!/bin/bash
set -e
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
for M in "$@"; do
  echo "############## $M ##############"
  python scripts/evaluation/attn_sink_emergence_generic.py --model "$M" \
      --n_samples 6 --out_dir result/attn_sink_emergence --device cuda --dtype bfloat16
done
