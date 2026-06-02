"""SLURM launcher for the attention-rescale downstream EM/F1 matrix.

Config set (5 correction configs + 2 references):
  refs (batched, fast, no rescale):
    baseline            full causal (upper bound)
    recover_pos_enc     blocked mask, no correction (lower bound)
  corrections (batch_size=1, eager + rescale):
    recover_attn_score_q    prefill question-row rescale (clean block_qa GBM)
    recover_attn_score_gen  decode gen-row rescale, alpha=0.3  (decode-only best)
    recover_attn_score_qg   combined prefill-q + decode-gen, alpha in {0.3,0.6,1.0}

Benchmarks: nq (pos 0), hqa, wiki, musique   (mode=all)
Prompts:    short, kvlink(=verbose)

Full matrix = 5 corrections x 4 bench x 2 prompt = 40 jobs (+ 2 refs x 4 x 2 = 16).

Dependency: the *_q and *_qg configs need the clean question GBM
(analysis/attention_score_analysis/attn_score_q_models). The gen and ref configs
do not. Use --group to fire in stages.

Usage:
    source ~/diffprotein/dplm/.venv/bin/activate
    python scripts/evaluation/submit_gen_matrix.py --group refs_decode --dry-run
    python scripts/evaluation/submit_gen_matrix.py --group refs_decode
    python scripts/evaluation/submit_gen_matrix.py --group q_qg      # after q GBM ready
    python scripts/evaluation/submit_gen_matrix.py --group all
"""
import argparse

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
RUN = f"{PROJECT_DIR}/scripts/evaluation/run_eval_str.sh"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BENCHES = ["nq", "hqa", "wiki", "musique"]
PROMPTS = ["short", "kvlink"]   # short = short-answer preset, kvlink = verbose

ALPHAS = ["0.1", "0.3", "0.6", "1.0"]

# (policy, strength, batch, needs_q_gbm)
CONFIGS = {
    # baseline / recover_pos_enc references (batched, fast, no q GBM needed).
    "refs": [
        ("baseline",        "1.0", 8, False),
        ("recover_pos_enc", "1.0", 8, False),
    ],
    # decode-only gen rescale, alpha sweep — 4 x 4bench x 2prompt = 32 jobs. No q GBM.
    "g": [("recover_attn_score_gen", a, 1, False) for a in ALPHAS],
    # question-only + combined(alpha sweep) — (1+4) x 4 x 2 = 40 jobs. Needs q GBM.
    "q_qg": (
        [("recover_attn_score_q", "1.0", 1, True)]
        + [("recover_attn_score_qg", a, 1, True) for a in ALPHAS]
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["refs", "g", "q_qg", "all"], default="all")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    ap.add_argument("--prompts", nargs="+", default=PROMPTS)
    ap.add_argument("--partition", default="rtx3090")
    ap.add_argument("--qos", default="normal")
    ap.add_argument("--timeout", default="0-12")
    ap.add_argument("--max_job_num", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups = ["refs", "g", "q_qg"] if args.group == "all" else [args.group]
    cfgs = [c for g in groups for c in CONFIGS[g]]

    combos = []
    for (policy, strength, batch, _need) in cfgs:
        for bench in args.benches:
            for prompt in args.prompts:
                combos.append(f"{RUN} {bench} {MODEL} {policy} {strength} {args.n} {batch} {prompt}")

    print(f"group={args.group}  jobs={len(combos)}  (n={args.n}, benches={args.benches}, prompts={args.prompts})")
    for c in combos:
        print(f"  bash {c}")
    if args.dry_run:
        print("\n[dry-run] not submitting.")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks
    launch_tasks(
        param_option=1, base_cmd="bash", param_dict={"": combos},
        partition=args.partition, qos=args.qos, timeout=args.timeout,
        job_name=f"gen_matrix_{args.group}", max_job_num=args.max_job_num,
        part_to_py={args.partition: f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    main()
