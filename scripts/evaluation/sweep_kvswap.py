"""SLURM launcher: KV-cache swap experiment (baseline <-> recover_pos_enc).

For each (model, benchmark, mode, prompt) we run four policies on the SAME
pretrained model (no fine-tuning):

  * baseline           -- full-causal cache               (ceiling reference)
  * recover_pos_enc    -- block-diagonal broken cache      (floor reference)
  * swap_docbase_qrec  -- docs from baseline, QUESTION band from recover_pos_enc
  * swap_docrec_qbase  -- docs from recover_pos_enc, QUESTION band from baseline

The swaps prefill the same sequence twice (both contiguous positions), splice a
hybrid cache, and decode -- isolating whether a well-restored question cache can
rescue a broken document cache, and vice versa. batch_size=1 (per-sample splice),
reencode=0.

Models x benchmarks x modes x prompts x policies
  = 3 x 4 x 2 x 2 x 4 = 192 jobs (skip-if-exists). NQ "all" uses gold-doc
position 0 only. 1000 examples each (NQ "all" auto-clamps to its 500 locals).

Prompts: ``kvlink`` (verbose) + ``short`` (short-answer). Qwen3 is run under
.venv-q3 automatically by run_kvswap.sh.

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate     # slurm_launcher venv
    python scripts/evaluation/sweep_kvswap.py --dry-run
    python scripts/evaluation/sweep_kvswap.py                 # submit
    python scripts/evaluation/sweep_kvswap.py --only Qwen3    # filter
"""
import argparse
import os

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_kvswap.sh"
OUT_DIR = f"{PROJECT_DIR}/result/kvswap"

# "llama3.1-1b" -> the repo's 1B is Llama-3.2-1B-Instruct (Llama 3.1 has no 1B).
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]
BENCHMARKS = ["nq", "hqa", "wiki", "musique"]
MODES = ["all", "gold"]
PROMPT_PRESETS = ["kvlink", "short"]      # verbose + short-answer
POLICIES = [
    "baseline",
    "recover_pos_enc",
    "swap_docbase_qrec",
    "swap_docrec_qbase",
]

BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}


def result_stem(bench, model, mode, pos, policy, prompt_preset):
    """MUST match the *_eval.py stem (re0, no modular_q_pos suffix)."""
    model_slug = model.split("/")[-1]
    if mode == "gold":
        state = "goldonly"
    elif bench == "nq":
        state = f"at{pos}"
    else:
        state = "all"
    ppx = "" if prompt_preset == "kvlink" else f"_p-{prompt_preset}"
    return f"{BENCH_TOKEN[bench]}_{model_slug}_{state}_kv-{policy}{ppx}_re0"


def partition_for(model, bench, mode, default):
    """All jobs go to ``default`` (rtx3090). The cluster's rtx3090 partition has
    ~14 nodes and is usually idle, while a100 has a single node (zinger) that
    starves -- routing the 8B musique/all swaps there only created PENDING
    pile-ups. The swap holds 2-3 small caches (hundreds of MB) so 8B/musique
    fits a 24GB 3090 fine. Keep the hook in case a future model truly needs it."""
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_examples", type=int, default=1000)
    ap.add_argument("--partition", default="rtx3090")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--timeout", default="1-0", help="SLURM time limit (D-H)")
    # rtx3090 is plentiful + usually idle -> submit everything, don't throttle
    # into PENDING. (radish stays excluded via --exclude.)
    ap.add_argument("--max_job_num", type=int, default=500)
    ap.add_argument("--exclude", default="radish")
    ap.add_argument("--only", default="", help="substring filter on the command")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json"))

    by_part = {}  # partition -> [combo, ...]
    for model in MODELS:
        for bench in BENCHMARKS:
            for mode in MODES:
                if bench == "wiki" and wiki_missing:
                    print(f"  [skip] wiki/{mode} (data missing)")
                    continue
                pos = 0
                for preset in PROMPT_PRESETS:
                    for policy in POLICIES:
                        stem = result_stem(bench, model, mode, pos, policy, preset)
                        out = os.path.join(OUT_DIR, f"{stem}.jsonl")
                        if not args.force and os.path.isfile(out):
                            print(f"  [skip] {stem} (exists)")
                            continue
                        cmd = (f"{SCRIPT_PATH} {bench} {model} {mode} {pos} "
                               f"{policy} {preset} {args.max_examples} {OUT_DIR}")
                        if args.only and args.only not in cmd:
                            continue
                        part = partition_for(model, bench, mode, args.partition)
                        by_part.setdefault(part, []).append(cmd)

    total = sum(len(v) for v in by_part.values())
    print(f"\njobs to submit: {total}  (max_examples={args.max_examples})")
    for part, combos in by_part.items():
        print(f" [{part}] {len(combos)} jobs")
        for c in combos:
            print("   bash " + c.replace(PROJECT_DIR + "/scripts/evaluation/", ""))
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
            job_name=f"kvswap_{part}",
            max_job_num=args.max_job_num,
            part_to_py={part: f"{PROJECT_DIR}/.venv/bin/python"},
        )


if __name__ == "__main__":
    main()
