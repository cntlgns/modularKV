"""SLURM launcher: region-attention comparison across 3 configs x 5 datasets.

Configs (model, policy, name):
  baseline           : Llama-3.1-8B-Instruct           @ baseline        (oracle, full cache)
  base_broken        : Llama-3.1-8B-Instruct           @ recover_pos_enc (broken cache)
  kvlink0_step4000   : Llama-3.1-8B-Instruct-ft-step4000 @ recover_pos_enc (finetuned)

Datasets: nq, hqa_gold, wiki, musique, block_qa.
Job count: 3 configs * 5 datasets = 15 (one GPU each).

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate   # slurm_launcher venv
    python scripts/evaluation/sweep_attn_region.py --dry-run
    python scripts/evaluation/sweep_attn_region.py            # submit
    python scripts/evaluation/sweep_attn_region.py --partition a100
"""
import argparse
import os

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_attn_region.sh"
OUT_DIR = f"{PROJECT_DIR}/result/attn_region_compare"

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FT_MODEL = f"{PROJECT_DIR}/model_cache/Llama-3.1-8B-Instruct-ft-step4000"

# (config_name, model, policy)
CONFIGS = [
    ("baseline", BASE_MODEL, "baseline"),
    ("base_broken", BASE_MODEL, "recover_pos_enc"),
    ("kvlink0_step4000", FT_MODEL, "recover_pos_enc"),
]
DATASETS = ["nq", "hqa_gold", "wiki", "musique", "block_qa"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--partition", default="h100")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--timeout", default="0-4", help="SLURM time limit (D-H)")
    ap.add_argument("--max_job_num", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="Resubmit even if the output JSON already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # wiki needs the 2wikimultihop dev.json on disk.
    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json"))

    combos = []
    for cname, model, policy in CONFIGS:
        for ds in DATASETS:
            if ds == "wiki" and wiki_missing:
                print(f"  [skip] wiki/{cname} (data missing)")
                continue
            out = os.path.join(OUT_DIR, f"{ds}__{cname}.json")
            if not args.force and os.path.isfile(out):
                print(f"  [skip] {os.path.basename(out)} (exists)")
                continue
            combos.append(
                f"{SCRIPT_PATH} {model} {policy} {cname} {ds} "
                f"{args.n_samples} {args.max_new}")

    print(f"\njobs to submit: {len(combos)}/{len(CONFIGS)*len(DATASETS)}  "
          f"(partition={args.partition}, n={args.n_samples}, max_new={args.max_new})")
    for c in combos:
        print(f"  bash {c}")
    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return
    if not combos:
        print("[done] nothing to submit (all JSONs already exist).")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": combos},
        partition=args.partition,
        qos=args.qos,
        timeout=args.timeout,
        job_name=f"attn_region_{args.partition}",
        max_job_num=args.max_job_num,
        part_to_py={args.partition: f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    main()
