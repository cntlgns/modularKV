#!/bin/bash
# Run one KVLink evaluation experiment on a single GPU (pretrained model, no
# fine-tuning). Designed to be launched one-job-per-experiment by
# scripts/evaluation/sweep_ablation.py via the SLURM launcher.
#
# Usage:
#   bash scripts/evaluation/run_eval.sh <bench> <model> <mode> \
#                                       [pos] [batch_size] [attn_type] [reencode_num]
#
#   <bench>        nq | hqa | wiki | musique
#   <model>        HF model id, e.g. meta-llama/Llama-3.2-1B-Instruct
#   <mode>         all | gold
#   [pos]          NQ gold-doc position 0|4|9   (default: 0; NQ "all" only)
#   [batch_size]   eval batch size              (default: 8)
#   [attn_type]    standard | blocked           (default: standard)
#   [reencode_num] number of link tokens        (default: 0)
#
# Example:
#   bash scripts/evaluation/run_eval.sh nq meta-llama/Llama-3.2-1B-Instruct gold 0 16

set -e

BENCH="$1"
MODEL="$2"
MODE="$3"
POS="${4:-0}"
BATCH_SIZE="${5:-8}"
ATTN_TYPE="${6:-standard}"
REENCODE_NUM="${7:-0}"

if [ -z "$BENCH" ] || [ -z "$MODEL" ] || [ -z "$MODE" ]; then
    echo "Usage: bash run_eval.sh <bench> <model> <mode> [pos] [batch_size] [attn_type] [reencode_num]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Use the project's uv-built .venv (transformers 4.46.3 + CUDA torch). The
# SLURM launcher's part_to_py only rewrites a literal "python" in base_cmd,
# which is "bash" here, so the interpreter must be pinned explicitly.
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
    else
        echo "Error: $PROJECT_ROOT/.venv/bin/python not found. Build it with:"
        echo "       uv venv .venv --prompt KVLink && uv pip install -r requirements.txt"
        exit 1
    fi
fi

SCRIPT="scripts/evaluation/${BENCH}_eval.py"
if [ ! -f "$SCRIPT" ]; then
    echo "Error: unknown benchmark '$BENCH' (no $SCRIPT)"
    exit 1
fi
if [ "$BENCH" = "wiki" ] && [ ! -f "data/raw/2wikimultihop/dev.json" ]; then
    echo "Error: data/raw/2wikimultihop/dev.json missing. Clone it first:"
    echo "       git clone https://huggingface.co/datasets/xanhho/2WikiMultihopQA data/raw/2wikimultihop"
    exit 1
fi

GOLD_FLAG=""
[ "$MODE" = "gold" ] && GOLD_FLAG="--gold_only"
POS_FLAG=""
[ "$BENCH" = "nq" ] && POS_FLAG="--pos $POS"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "  Benchmark:  $BENCH"
echo "  Model:      $MODEL"
echo "  Mode:       $MODE"
echo "  NQ pos:     $POS"
echo "  Attn:       $ATTN_TYPE   Reencode: $REENCODE_NUM   Batch: $BATCH_SIZE"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PYTHON "$SCRIPT" \
    --ckpt_path "$MODEL" \
    --model_name "$MODEL" \
    --hf True \
    --batch_size "$BATCH_SIZE" \
    --attn_type "$ATTN_TYPE" \
    --reencode_num "$REENCODE_NUM" \
    $POS_FLAG \
    $GOLD_FLAG

echo "All done: $BENCH / $MODEL / $MODE"
