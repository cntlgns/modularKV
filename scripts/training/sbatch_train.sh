#!/bin/bash
# SLURM batch wrapper for KVLink finetuning. Submit one per config.
#
# Usage:
#   sbatch \
#       --partition=<partition> \
#       --gres=gpu:<N> \
#       --job-name=kvlink_<cfg> \
#       --output=slurm/kvlink_train/%j.out \
#       scripts/training/sbatch_train.sh <config_name>
#
# Required positional arg: config_name (see scripts/training/run_train.sh).
# Reads NGPU from SLURM_GPUS_ON_NODE; falls back to env NGPU or 8.

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

set -e

CONFIG_NAME="$1"
if [ -z "$CONFIG_NAME" ]; then
    echo "Usage: sbatch ... scripts/training/sbatch_train.sh <config_name> [extra_args...]" >&2
    exit 1
fi
shift

# sbatch copies this script to a spool dir, so $0 is not the repo path.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

# Resolve NGPU from SLURM if available
if [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export NGPU="$SLURM_GPUS_ON_NODE"
elif [ -n "${SLURM_GPUS:-}" ]; then
    export NGPU="$SLURM_GPUS"
fi

echo "[$(date)] node=$(hostname)  job=${SLURM_JOB_ID:-local}  config=$CONFIG_NAME  NGPU=${NGPU:-unset}  extra=$*"
nvidia-smi -L 2>/dev/null || true

bash scripts/training/run_train.sh "$CONFIG_NAME" "$@"
