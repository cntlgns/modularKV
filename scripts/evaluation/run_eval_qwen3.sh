#!/bin/bash
# Qwen3-8B sweep wrapper: uses the transformers>=4.51 venv (.venv-q3), 1000
# examples/benchmark, results in result/qwen3_sweep.
export PYTHON=/data_fast/home/sihun/kvcache/KVLink/.venv-q3/bin/python
export MAX_EXAMPLES=1000
export OUT_DIR=result/qwen3_sweep
exec bash /data_fast/home/sihun/kvcache/KVLink/scripts/evaluation/run_eval.sh "$@"
