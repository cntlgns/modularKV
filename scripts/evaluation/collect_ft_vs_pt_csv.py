"""Collect ft-vs-pt comparison numbers into a single CSV.

Reads .summary.json files from:
  - result/ablation/  (baseline + recover_pos_enc only; full-dataset eval of the
    pretrained Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct)
  - result/ft_vs_pt/  (this sweep: recover_pos_enc with N=1000 cap;
    pretrained 1B vs Llama-3.2-1B-Instruct-ft-step{2000,6000})

Output:
  - result/ft_vs_pt/comparison_long.csv  -- one row per measurement (tidy)
  - result/ft_vs_pt/comparison_4way.csv  -- (benchmark, state) groups of 4 rows
        (full_baseline / n1000_step2000_recover_pos_enc /
         n1000_step6000_recover_pos_enc /
         n1000_pretrained_recover_pos_enc) x 6 metric cols
         (short|acc,em,f1, kvlink|acc,em,f1). 'all'-state groups on top,
         goldonly groups at the bottom.
"""
import csv
import glob
import json
import os

PROJECT = "/data_fast/home/sihun/kvcache/KVLink"
ABL_DIR = f"{PROJECT}/result/ablation"
FTPT_DIR = f"{PROJECT}/result/ft_vs_pt"
KEEP_POLICIES = {"baseline", "recover_pos_enc"}
# 1B-only comparison: drop the 8B pretrained rows from the ablation source.
SKIP_MODELS = {"Llama-3.1-8B-Instruct"}

LONG_FIELDS = [
    "source", "benchmark", "state", "kv_policy", "prompt_preset",
    "model", "num_examples", "accuracy", "em", "f1",
]


def load(path):
    with open(path) as f:
        return json.load(f)


def short_model(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else x


def main():
    rows = []

    # Ablation: keep only baseline + recover_pos_enc (and skip modular q_pos suffixes,
    # which are already absent for these two policies).
    for p in sorted(glob.glob(f"{ABL_DIR}/*.summary.json")):
        d = load(p)
        if d.get("kv_policy") not in KEEP_POLICIES:
            continue
        if short_model(d["model"]) in SKIP_MODELS:
            continue
        rows.append({
            "source": "ablation_full",
            "benchmark": d["benchmark"],
            "state": d["state"],
            "kv_policy": d["kv_policy"],
            "prompt_preset": d.get("prompt_preset", "kvlink"),
            "model": short_model(d["model"]),
            "num_examples": d["num_examples"],
            "accuracy": fmt(d["accuracy"]),
            "em": fmt(d["em"]),
            "f1": fmt(d["f1"]),
        })

    # ft_vs_pt: all of these are recover_pos_enc with N=1000 cap (NQ effectively 500).
    for p in sorted(glob.glob(f"{FTPT_DIR}/*.summary.json")):
        d = load(p)
        if short_model(d["model"]) in SKIP_MODELS:
            continue
        rows.append({
            "source": "ft_vs_pt_n1000",
            "benchmark": d["benchmark"],
            "state": d["state"],
            "kv_policy": d["kv_policy"],
            "prompt_preset": d.get("prompt_preset", "kvlink"),
            "model": short_model(d["model"]),
            "num_examples": d["num_examples"],
            "accuracy": fmt(d["accuracy"]),
            "em": fmt(d["em"]),
            "f1": fmt(d["f1"]),
        })

    long_path = f"{FTPT_DIR}/comparison_long.csv"
    with open(long_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[long] {long_path}  ({len(rows)} rows)")

    write_4way(rows)


# The four conditions we want as comparable rows for each (benchmark, state).
# Each entry is (label, predicate). Order is the row order within a group.
CONDITIONS = [
    ("full_baseline",
     lambda r: r["source"] == "ablation_full"
                and r["kv_policy"] == "baseline"
                and r["model"] == "Llama-3.2-1B-Instruct"),
    ("n1000_step2000_recover_pos_enc",
     lambda r: r["source"] == "ft_vs_pt_n1000"
                and r["kv_policy"] == "recover_pos_enc"
                and r["model"] == "Llama-3.2-1B-Instruct-ft-step2000"),
    ("n1000_step6000_recover_pos_enc",
     lambda r: r["source"] == "ft_vs_pt_n1000"
                and r["kv_policy"] == "recover_pos_enc"
                and r["model"] == "Llama-3.2-1B-Instruct-ft-step6000"),
    ("n1000_pretrained_recover_pos_enc",
     lambda r: r["source"] == "ft_vs_pt_n1000"
                and r["kv_policy"] == "recover_pos_enc"
                and r["model"] == "Llama-3.2-1B-Instruct"),
]

# Benchmark display order (matches sweep_ft_vs_pt.BENCHMARKS).
BENCH_ORDER = {"NQ": 0, "hqa": 1, "wiki": 2, "musique": 3}
# Within a benchmark, NQ "all" sweeps positions; "all"/"goldonly" sort below.
STATE_ORDER = {"at0": 0, "at4": 1, "at9": 2, "all": 3, "goldonly": 9}


def state_group(state: str) -> int:
    # All retrieval-mode states (at*, all) go on top; goldonly at the bottom.
    return 1 if state == "goldonly" else 0


def write_4way(rows):
    # Pick by (condition, benchmark, state, preset) -> {acc,em,f1}.
    idx = {}
    keys = set()
    for r in rows:
        for label, pred in CONDITIONS:
            if pred(r):
                idx[(label, r["benchmark"], r["state"], r["prompt_preset"])] = (
                    r["accuracy"], r["em"], r["f1"]
                )
                keys.add((r["benchmark"], r["state"]))
                break

    def sort_key(k):
        b, s = k
        return (state_group(s), BENCH_ORDER.get(b, 99), STATE_ORDER.get(s, 99))

    out_path = f"{FTPT_DIR}/comparison_4way.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "state", "condition",
            "short|acc", "short|em", "short|f1",
            "kvlink|acc", "kvlink|em", "kvlink|f1",
        ])
        for bench, state in sorted(keys, key=sort_key):
            for label, _ in CONDITIONS:
                row = [bench, state, label]
                for preset in ("short", "kvlink"):
                    cells = idx.get((label, bench, state, preset))
                    if cells is None:
                        row += ["", "", ""]
                    else:
                        row += list(cells)
                w.writerow(row)
    print(f"[4way] {out_path}  ({len(keys)} groups x {len(CONDITIONS)} rows)")


if __name__ == "__main__":
    main()
