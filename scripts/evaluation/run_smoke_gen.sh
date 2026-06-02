#!/bin/bash
# Smoke test for the recover_attn_score_gen decode-only policy: run a handful of
# NQ examples under baseline / recover_pos_enc / recover_attn_score_gen and print
# EM/F1 so we can confirm the wiring runs end-to-end and produces sane output.
set -e
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
N="${2:-5}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "##### GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
for POLICY in baseline recover_pos_enc recover_attn_score_gen; do
    echo ""
    echo "################## policy=$POLICY (n=$N) ##################"
    MAX_EXAMPLES="$N" OUT_DIR="result/smoke_gen" \
        bash scripts/evaluation/run_eval.sh nq "$MODEL" all 0 1 "$POLICY"
done
