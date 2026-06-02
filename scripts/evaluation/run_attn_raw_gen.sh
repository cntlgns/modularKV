#!/bin/bash
# Args:
#   $1 MODEL          (HF hub id, e.g. meta-llama/Llama-3.1-8B-Instruct)
#   $2 SHARD_RANK     (0-indexed)
#   $3 SHARD_WORLD    (total number of shards)
#   $4 MAX_SAMPLES    (optional cap; empty/0 = use full split. train=17,998 test=2,000)
#   $5 RUN_TAG        (optional label appended to output parquet name for append-friendly runs)
#   $6 SPLIT          (train | test ; default train)
#   $7 ROWS           (gen | question ; default gen) — gen rows go to attn_raw_gen_results,
#                     question rows go to attn_raw_q_results
set -e
MODEL="$1"
SHARD_RANK="$2"
SHARD_WORLD="$3"
MAX_SAMPLES="${4:-0}"
# Sentinel handling: shell word-splitting eats an empty $5, which would shift $6
# (SPLIT) into the $5 slot. The launcher therefore passes "NONE" instead of an
# empty string for RUN_TAG and we translate back here.
RUN_TAG="${5:-NONE}"
[[ "$RUN_TAG" == "NONE" ]] && RUN_TAG=""
SPLIT="${6:-train}"
ROWS="${7:-gen}"

cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "$ROWS" == "question" ]]; then
    OUT_DIR="analysis/attention_score_analysis/attn_raw_q_results"
else
    OUT_DIR="analysis/attention_score_analysis/attn_raw_gen_results"
fi

echo "============================================="
echo "  Model:        $MODEL"
echo "  Rows:         ${ROWS}"
echo "  Split:        ${SPLIT}"
echo "  Shard:        ${SHARD_RANK}/${SHARD_WORLD}"
echo "  Max samples:  ${MAX_SAMPLES:-all}"
echo "  Out dir:      ${OUT_DIR}"
echo "  GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================="

CAP_ARG=""
if [[ "$MAX_SAMPLES" != "0" && -n "$MAX_SAMPLES" ]]; then
    CAP_ARG="--max_samples $MAX_SAMPLES"
fi
TAG_ARG=""
if [[ -n "$RUN_TAG" ]]; then
    TAG_ARG="--run_tag $RUN_TAG"
fi

python scripts/evaluation/attn_raw_collect_gen.py \
    --model "$MODEL" \
    --rows "$ROWS" \
    --split "$SPLIT" \
    --shard_rank "$SHARD_RANK" \
    --shard_world_size "$SHARD_WORLD" \
    --out_dir "$OUT_DIR" \
    --device cuda --dtype bfloat16 \
    $CAP_ARG $TAG_ARG
