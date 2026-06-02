#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
PY="${PY:-.venv/bin/python}"
LOG_DIR="${LOG_DIR:-logs/data_process}"
mkdir -p "$LOG_DIR"

# Skip helper: only run if target dir is missing.
run_if_missing() {
    local target="$1"; shift
    local name="$1"; shift
    if [ -d "$target" ]; then
        echo "[skip $name] $target already exists"
        return 0
    fi
    echo "[run  $name] -> $target"
    "$@"
}

echo "=== xsum ==="
run_if_missing dataset_cache/processed/xsum/xsum xsum \
    $PY scripts/data_process/xsum.py --max_length=4096 --validation_size=1000 \
    > "$LOG_DIR/xsum.log" 2>&1

echo "=== daring_anteater ==="
run_if_missing dataset_cache/processed/daringanteater/sft_mem daring_anteater \
    $PY scripts/data_process/daring_anteater.py --max_length=4096 --validation_size=2000 \
    > "$LOG_DIR/daring_anteater.log" 2>&1

echo "=== tulu ==="
run_if_missing dataset_cache/processed/tulu/sft tulu \
    $PY scripts/data_process/tulu.py --max_length=4096 --validation_size=2000 \
    > "$LOG_DIR/tulu.log" 2>&1

echo "=== fineweb ==="
run_if_missing dataset_cache/processed/fineweb/text fineweb \
    $PY scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000 \
    > "$LOG_DIR/fineweb.log" 2>&1

echo "ALL DONE"
