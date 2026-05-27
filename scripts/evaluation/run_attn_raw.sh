#!/bin/bash
set -e
MODEL="$1"
BENCH="$2"
N="${3:-100}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
echo "============================================="
echo "  Model:      $MODEL"
echo "  Benchmark:  $BENCH"
echo "  Samples:    $N"
echo "  GPU:        $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="
python scripts/evaluation/attn_raw_collect.py \
    --model "$MODEL" --benchmark "$BENCH" --n_samples "$N" \
    --out_dir attn_raw_results --device cuda --dtype bfloat16
