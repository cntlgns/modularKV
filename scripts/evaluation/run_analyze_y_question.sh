#!/bin/bash
# Diagnose why y_question is harder to predict than sink/sys/docs.
# CPU-only (sklearn); SLURM allocation just isolates cores + memory.
set -e
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate

echo "============================================="
echo "  analyze_y_question"
echo "  Host: $(hostname)  CPU=$(nproc)"
echo "============================================="

python -u scripts/evaluation/analyze_y_question.py \
    --in_dir analysis/attention_score_analysis/attn_raw_gen_results \
    --gbm_dir analysis/attention_score_analysis/attn_score_gen_models \
    --out_dir analysis/attention_score_analysis/analyze_y_question
