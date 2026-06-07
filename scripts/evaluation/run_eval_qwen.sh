#!/bin/bash
# Qwen sweep wrapper: 1000 examples/benchmark, results in result/qwen_sweep.
export MAX_EXAMPLES=1000
export OUT_DIR=result/qwen_sweep
exec bash /data_fast/home/sihun/kvcache/KVLink/scripts/evaluation/run_eval.sh "$@"
