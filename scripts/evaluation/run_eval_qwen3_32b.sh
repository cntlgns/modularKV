#!/bin/bash
# Qwen3-32B sweep wrapper: transformers>=4.51 venv (.venv-q3), 1000 ex, single a100-80G.
export PYTHON=/data_fast/home/sihun/kvcache/KVLink/.venv-q3/bin/python
export MAX_EXAMPLES=1000
export OUT_DIR=result/qwen3_32b_sweep
exec bash /data_fast/home/sihun/kvcache/KVLink/scripts/evaluation/run_eval.sh "$@"
