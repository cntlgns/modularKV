#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
set -e
cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID args=$*"
.venv/bin/python scripts/data_process/scan_one.py "$@"
echo "[$(date)] done"
