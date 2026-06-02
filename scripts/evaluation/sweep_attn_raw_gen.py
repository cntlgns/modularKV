"""SLURM launcher: collect generation-token attention masses from block_qa.

Sharded run — one job per shard, all on the same partition. Default 2 H100s,
which finishes the full 17,998-sample block_qa train split in ~30-60 min/GPU
(seq_len ~800-1500, gen len ~10-50).

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate   # slurm_launcher venv (py3.9)
    python scripts/evaluation/sweep_attn_raw_gen.py --dry-run
    python scripts/evaluation/sweep_attn_raw_gen.py            # submits jobs
    python scripts/evaluation/sweep_attn_raw_gen.py --world 4  # 4 shards
"""
import argparse
import os

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_attn_raw_gen.sh"
ANALYSIS_DIR = f"{PROJECT_DIR}/analysis/attention_score_analysis"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2,
                    help="Number of shards / GPUs (default 2)")
    ap.add_argument("--rows", default="gen", choices=["gen", "question"],
                    help="gen rows -> attn_raw_gen_results; question rows -> attn_raw_q_results")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Cap on the chosen split (0 = all; train=17,998 test=2,000)")
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="block_qa split. 'test' (2,000 samples) is the heldout pool used "
                         "as an independent R^2 check on the trained regressor.")
    ap.add_argument("--run_tag", default="",
                    help="Optional label suffix on output parquet filenames, so a later "
                         "extra run accumulates next to the originals.")
    ap.add_argument("--partition", default="h100")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--timeout", default="0-3", help="SLURM time limit (D-H)")
    ap.add_argument("--max_job_num", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="Resubmit even if the shard's output parquet exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result_dir = os.path.join(
        ANALYSIS_DIR,
        "attn_raw_q_results" if args.rows == "question" else "attn_raw_gen_results",
    )
    tag_suffix = f"__{args.run_tag}" if args.run_tag else ""
    split_infix = "" if args.split == "train" else f"_{args.split}"
    combos = []
    for rank in range(args.world):
        slug = MODEL.split("/")[-1]
        out = os.path.join(
            result_dir,
            f"block_qa{split_infix}__{slug}__shard{rank}of{args.world}{tag_suffix}.parquet",
        )
        if not args.force and os.path.isfile(out):
            print(f"  [skip] {out}")
            continue
        # Sentinel "NONE" so an empty run_tag doesn't shell-word-split-shift the
        # split/rows args into the run_tag slot. The runner translates "NONE" back.
        tag_arg = args.run_tag if args.run_tag else "NONE"
        combos.append(
            f"{SCRIPT_PATH} {MODEL} {rank} {args.world} {args.max_samples} "
            f"{tag_arg} {args.split} {args.rows}"
        )

    print(f"\nshards to submit: {len(combos)}/{args.world}  "
          f"(partition={args.partition}, split={args.split}, rows={args.rows})")
    for c in combos:
        print(f"  bash {c}")
    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return
    if not combos:
        print("[done] nothing to submit (all parquets already exist).")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": combos},
        partition=args.partition,
        qos=args.qos,
        timeout=args.timeout,
        job_name=f"attn_raw_{args.rows}_{args.partition}",
        max_job_num=args.max_job_num,
        part_to_py={args.partition: f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    main()
