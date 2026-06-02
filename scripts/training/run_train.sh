#!/bin/bash
# Run KVLink finetuning on a single node with torchrun.
#
# Usage:
#   bash scripts/training/run_train.sh <config_name> [ngpu] [extra_args...]
#
#   <config_name>  one of the keys in CONFIG_DICT inside titan_trainer_kvlink.py:
#                    - data_original_step6k_bsz64_link_5_selective_ckpt
#                    - data_original_step6k_bsz64_link_5_full_ckpt
#                    - data_nosum_step6k_bsz64_link_5_full_ckpt
#                    - data_nosftmem_step6k_bsz64_link_5_full_ckpt
#                    - data_qaonly_step6k_bsz64_link_5_full_ckpt
#   [ngpu]         number of GPUs on this node (default: 8). 1B model fits on
#                  much fewer; with bsz=64 (global), local_bsz = 64/ngpu must
#                  also fit memory.
#   [extra_args]   forwarded to titan_trainer_kvlink.py (e.g. --use_wandb_for_log)
#
# Examples:
#   bash scripts/training/run_train.sh data_qaonly_step6k_bsz64_link_5_full_ckpt 4
#   NGPU=2 bash scripts/training/run_train.sh data_original_step6k_bsz64_link_5_selective_ckpt
#
# The runner pins the venv interpreter explicitly because launchers that pass
# base_cmd="bash" cannot substitute the python path through PART_TO_PY.

set -e

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: bash scripts/training/run_train.sh <config_name> [ngpu] [extra_args...]" >&2
    exit 1
fi
shift

NGPU="${NGPU:-${1:-8}}"
# If the second positional looks like an integer, consume it as ngpu.
case "$1" in
    ''|*[!0-9]*) ;;
    *) shift ;;
esac

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
    else
        echo "Error: $PROJECT_ROOT/.venv/bin/python not found." >&2
        exit 1
    fi
fi

# Sanity checks before burning GPU hours
if [ ! -f "data/titan_tokenizer/original/tokenizer.model" ]; then
    echo "Error: tokenizer missing. Run:" >&2
    echo "  $PYTHON src/data/titan_download_tokenizer.py --repo_id meta-llama/Llama-3.2-1B-Instruct \\" >&2
    echo "      --tokenizer_path original --local_dir data/titan_tokenizer/" >&2
    exit 1
fi
# Pick the base-model path based on config suffix. 8B is sharded into 4 files;
# 1B is a single safetensors. Loader (torchtune_model_checkpointer.py) accepts
# either a file or a directory of safetensors shards.
case "$CONFIG_NAME" in
    *_8b)
        MODEL_DIR="model_cache/Llama-3.1-8B-Instruct"
        if [ ! -f "$MODEL_DIR/model-00001-of-00004.safetensors" ]; then
            echo "Error: 8B base weights missing at $MODEL_DIR" >&2
            exit 1
        fi
        ;;
    *)
        if [ ! -f "model_cache/Llama-3.2-1B-Instruct/model.safetensors" ]; then
            echo "Error: base model missing at model_cache/Llama-3.2-1B-Instruct/model.safetensors" >&2
            exit 1
        fi
        ;;
esac

# Which data components does this config need?
case "$CONFIG_NAME" in
    *_qaonly_*)           NEED="qa_mem" ;;
    *_nosum_*)            NEED="text tulu sft_mem qa qa_mem" ;;
    *_nosftmem_*)         NEED="text tulu qa qa_mem xsum" ;;
    *)                    NEED="text tulu sft_mem qa qa_mem xsum" ;;  # original
esac
for d in $NEED; do
    case "$d" in
        text|text_mem|text_inst) p="dataset_cache/processed/fineweb/$d" ;;
        sft|sft_mem)             p="dataset_cache/processed/daringanteater/$d" ;;
        tulu)                    p="dataset_cache/processed/tulu/sft" ;;
        qa|qa_mem)               p="dataset_cache/processed/block_qa/$d" ;;
        xsum)                    p="dataset_cache/processed/xsum/xsum" ;;
    esac
    if [ ! -d "$p" ]; then
        echo "Error: required dataset missing for $CONFIG_NAME: $p" >&2
        echo "       Run the relevant scripts/data_process/*.py first." >&2
        exit 1
    fi
done

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LOG_RANK="${LOG_RANK:-0}"

echo "=============================================="
echo "  Config:  $CONFIG_NAME"
echo "  NGPU:    $NGPU"
echo "  Python:  $PYTHON"
echo "  Extra:   $@"
echo "=============================================="

# Use a unique rdzv endpoint per submission (port 0 = OS picks) so multiple
# concurrent jobs on the same node don't collide.
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NGPU" \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "$LOG_RANK" \
    --role rank \
    --tee 3 \
    titan_trainer_kvlink.py \
    --config_name "$CONFIG_NAME" \
    "$@"
