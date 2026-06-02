#!/bin/bash
# SLURM wrapper for gpt_answer.py: Contriever retrieval (GPU) + gpt-4o-mini API calls.
# Submit:
#   sbatch --partition=rtx2080 --gres=gpu:1 --job-name=block_qa \
#       --output=logs/data_process/block_qa.%j.out scripts/data_process/sbatch_gpt_answer.sh

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
echo "PROJECT_ROOT=$PROJECT_ROOT"

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi -L 2>/dev/null || true

mkdir -p logs/data_process data/raw/block_qa
.venv/bin/python scripts/data_process/gpt_answer.py
echo "[$(date)] done"
