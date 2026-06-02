"""SLURM launcher: attention-distribution analysis across (model, dataset).

Submits one single-GPU job per (model, dataset) — each job runs all 4 KV
policies sequentially inside the script.

Matrix:
  - MODELS:   1B (rtx3090) + 8B (h100/a100)
  - DATASETS: nq, hqa_gold, hqa_full, wiki, musique
  - POLICIES: baseline, recover_pos_enc, modular, recover_cross_attn_oracle_pos
              (looped inside the python script)

Usage:
    source .venv/bin/activate
    python scripts/evaluation/sweep_attn_analysis.py --dry-run
    python scripts/evaluation/sweep_attn_analysis.py
"""
import os
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_attn_analysis.sh"
RESULT_DIR = f"{PROJECT_DIR}/attn_analysis_results"

MODELS_SMALL = ["meta-llama/Llama-3.2-1B-Instruct"]
MODELS_LARGE = ["meta-llama/Llama-3.1-8B-Instruct"]
DATASETS = ["nq", "hqa_gold", "hqa_full", "wiki", "musique"]
N_SAMPLES = 100


def out_json(model: str, dataset: str) -> str:
    slug = model.split("/")[-1]
    return os.path.join(RESULT_DIR, f"{dataset}__{slug}.json")


def build_combos(force: bool):
    combos_small = []
    combos_large = []
    for ds in DATASETS:
        for m in MODELS_SMALL:
            if not force and os.path.isfile(out_json(m, ds)):
                print(f"  [skip] {out_json(m, ds)}")
                continue
            combos_small.append(f"{SCRIPT_PATH} {m} {ds} {N_SAMPLES}")
        for m in MODELS_LARGE:
            if not force and os.path.isfile(out_json(m, ds)):
                print(f"  [skip] {out_json(m, ds)}")
                continue
            combos_large.append(f"{SCRIPT_PATH} {m} {ds} {N_SAMPLES}")
    return combos_small, combos_large


def submit(combos, partition, max_job_num=10):
    if not combos:
        return
    from slurm_launcher.sbatch_launcher import launch_tasks
    print(f"\nSubmitting {len(combos)} job(s) to {partition!r}...")
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": combos},
        partition=partition,
        qos="normal",
        timeout="0-6",
        job_name=f"attn_analysis_{partition}",
        max_job_num=max_job_num,
        part_to_py={partition: f"{PROJECT_DIR}/.venv/bin/python"},
    )


def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    force = "--force" in args
    only = ""
    if "--only" in args:
        i = args.index("--only")
        if i + 1 < len(args):
            only = args[i + 1]

    small, large = build_combos(force=force)
    if only:
        small = [c for c in small if only in c]
        large = [c for c in large if only in c]

    print(f"1B / rtx3090: {len(small)} jobs")
    for c in small:
        print(f"  bash {c}")
    print(f"8B / h100:    {len(large)} jobs")
    for c in large:
        print(f"  bash {c}")

    if dry:
        print("\nDry-run only.")
        return

    submit(small, partition="rtx3090", max_job_num=10)
    submit(large, partition="h100", max_job_num=5)


if __name__ == "__main__":
    main()
