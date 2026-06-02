#!/bin/bash
# Wrapper around run_eval.sh that sets the decode rescale strength
# (KVMOD_GEN_STRENGTH), the system-prompt preset, and a per-config OUT_DIR so
# different (policy, strength, prompt) combos don't overwrite each other.
# Args: BENCH MODEL POLICY STRENGTH N BATCH PROMPT
set -e
BENCH="$1"; MODEL="$2"; POLICY="$3"; STRENGTH="$4"; N="$5"; BATCH="${6:-1}"; PROMPT="${7:-kvlink}"
cd /data_fast/home/sihun/kvcache/KVLink
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KVMOD_GEN_STRENGTH="$STRENGTH"
export MAX_EXAMPLES="$N"
export OUT_DIR="result/gen_matrix/${BENCH}_${POLICY}_a${STRENGTH}_${PROMPT}_n${N}"
echo "##### BENCH=$BENCH POLICY=$POLICY STRENGTH=$STRENGTH PROMPT=$PROMPT N=$N BATCH=$BATCH"
echo "##### GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
# run_eval.sh: <bench> <model> <mode> [pos] [batch] [kv_policy] [modular_q_pos] [prompt_preset] [reencode]
bash scripts/evaluation/run_eval.sh "$BENCH" "$MODEL" all 0 "$BATCH" "$POLICY" summed_pos "$PROMPT" 0
