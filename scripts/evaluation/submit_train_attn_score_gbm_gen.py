"""SLURM submission for the gen-token GBM trainer.

Runs on rtx3090 by default. The GPU sits idle — sklearn's
HistGradientBoostingRegressor is CPU-only — but the SLURM allocation gives us
isolated CPU cores + memory and keeps the trainer off the login node.

Switch to a CPU-only partition (e.g. `dept`) if your launcher allows
GPU-less allocations; this script uses the standard 1-GPU `param_option=1`
for simplicity.

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate
    python scripts/evaluation/submit_train_attn_score_gbm_gen.py --dry-run
    python scripts/evaluation/submit_train_attn_score_gbm_gen.py
"""
import argparse

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_train_attn_score_gbm_gen.sh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="analysis/attention_score_analysis/attn_raw_gen_results")
    ap.add_argument("--out_dir",
                    default="analysis/attention_score_analysis/attn_score_gen_models")
    ap.add_argument("--partition", default="rtx3090")
    ap.add_argument("--qos",       default="normal")
    ap.add_argument("--timeout",   default="0-8", help="SLURM time limit (D-H)")
    ap.add_argument("--dry-run",   action="store_true")
    args = ap.parse_args()

    combo = f"{SCRIPT_PATH} {args.in_dir} {args.out_dir}"
    print(f"submitting on {args.partition}:")
    print(f"  bash {combo}")
    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": [combo]},
        partition=args.partition,
        qos=args.qos,
        timeout=args.timeout,
        job_name=f"train_attn_gbm_gen_{args.partition}",
        max_job_num=1,
        part_to_py={args.partition: f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    main()
