"""SLURM launcher: oracle_ab_vrec K-vs-V experiment on the untrained 8B base.

For each (benchmark, doc-mode) we run three policies on the SAME
meta-llama/Llama-3.1-8B-Instruct (no fine-tuning):

  * baseline          -- full causal cache (ceiling).
  * recover_pos_enc   -- block-diagonal broken cache (floor).
  * oracle_ab_vrec    -- question/decode rows use the BASELINE attention scores
                         (A_B from a parallel full-causal stream) but multiply
                         them by the recover_pos_enc VALUES. Isolates the
                         contribution of K (routing) vs V (content): if oracle
                         ~ baseline the damage is all in K; if oracle ~ broken
                         the damage is in V.

Benchmarks x modes: nq/hqa/wiki/musique x {all, gold}. NQ "all" uses gold-doc
position 0. 1000 examples each (NQ auto-clamps to its 500 local examples).

Job count = 4 benches x 2 modes x 3 policies = 24 (one GPU each, B=1).

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate     # slurm_launcher venv
    python scripts/evaluation/sweep_oracle_ab_vrec.py --dry-run
    python scripts/evaluation/sweep_oracle_ab_vrec.py            # submit
"""
import argparse
import os

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_oracle_ab_vrec.sh"
OUT_DIR = f"{PROJECT_DIR}/result/oracle_ab_vrec"

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SLUG = MODEL.split("/")[-1]
BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}

BENCHMARKS = ["nq", "hqa", "wiki", "musique"]
MODES = ["all", "gold"]
# baseline/recover_pos_enc are valid references (no patch -> stock sdpa). The
# three oracle_* policies use the dual-stream patched (eager) forward.
POLICIES = ["baseline", "recover_pos_enc",
            "oracle_ab_vrec", "oracle_arec_vbase", "oracle_ab_vbase"]


def partition_for(bench, mode, policy, default):
    """musique/all has the longest contexts; eager softmax (32,T,T) OOMs a 24GB
    3090 for the dual-stream oracles -> route those to a100 (80GB)."""
    if policy.startswith("oracle") and bench == "musique" and mode == "all":
        return "a100"
    return default


def result_stem(bench, mode, pos, policy):
    """MUST match the *_eval.py stem (re0, kvlink => no ppx suffix)."""
    if mode == "gold":
        state = "goldonly"
    elif bench == "nq":
        state = f"at{pos}"
    else:
        state = "all"
    return f"{BENCH_TOKEN[bench]}_{MODEL_SLUG}_{state}_kv-{policy}_re0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--partition", default="rtx3090")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--timeout", default="1-0", help="SLURM time limit (D-H)")
    ap.add_argument("--max_job_num", type=int, default=40)
    ap.add_argument("--exclude", default="radish")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json"))

    by_part = {}  # partition -> [combo, ...]
    for bench in BENCHMARKS:
        for mode in MODES:
            if bench == "wiki" and wiki_missing:
                print(f"  [skip] wiki/{mode} (data missing)")
                continue
            pos = 0
            for policy in POLICIES:
                stem = result_stem(bench, mode, pos, policy)
                out = os.path.join(OUT_DIR, f"{stem}.jsonl")
                if not args.force and os.path.isfile(out):
                    print(f"  [skip] {stem} (exists)")
                    continue
                part = partition_for(bench, mode, policy, args.partition)
                by_part.setdefault(part, []).append(
                    f"{SCRIPT_PATH} {bench} {MODEL} {mode} {pos} {policy} "
                    f"{args.max_examples} {OUT_DIR}")

    total = sum(len(v) for v in by_part.values())
    print(f"\njobs to submit: {total}  (max_examples={args.max_examples})")
    for part, combos in by_part.items():
        print(f" [{part}] {len(combos)} jobs")
        for c in combos:
            print(f"   bash {c}")
    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return
    if not total:
        print("[done] nothing to submit.")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks
    for part, combos in by_part.items():
        launch_tasks(
            param_option=1,
            base_cmd="bash",
            param_dict={"": combos},
            partition=part,
            exclude=args.exclude,
            qos=args.qos,
            timeout=args.timeout,
            job_name=f"oracle_abv_{part}",
            max_job_num=args.max_job_num,
            part_to_py={part: f"{PROJECT_DIR}/.venv/bin/python"},
        )


if __name__ == "__main__":
    main()
