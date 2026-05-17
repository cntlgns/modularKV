"""
SLURM launcher: modularization ablation for a pretrained (non-fine-tuned) model.

Submits one single-GPU job per experiment in the matrix

    models x benchmarks x doc-modes x attn-types  (+ NQ gold-position sweep)

Each job runs scripts/evaluation/run_eval.sh, which loads the pretrained HF
model and evaluates it on one benchmark in one document mode.

Matrix (edit the constants below to change it):
  - MODELS:      Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct
  - BENCHMARKS:  nq, hqa, wiki, musique
  - MODES:       all (all retrieved docs), gold (gold/supporting docs only)
  - ATTN_TYPES:  standard (full causal), blocked (per-document segment mask
                 via make_segment_mask; no link tokens since reencode_num=0)
  - NQ "all":    sweeps gold-doc position in NQ_POS_SWEEP; every other
                 (bench, mode) pair is a single job.
  => 2 x 2attn x (nq:3+1 + hqa:2 + wiki:2 + musique:2) = 40 jobs (wiki
     auto-skipped until data/raw/2wikimultihop/dev.json exists).

Idempotent: each experiment writes a deterministic, ablation-unique
result/<stem>.jsonl (+ <stem>.summary.json with accuracy/timestamp) where
stem = {BENCH}_{model}_{state}_attn-{attn}_re{reencode}. A combo whose
.jsonl already exists is skipped (pass --force to rerun it anyway), so a
re-run only fills in missing/failed experiments.

Usage (activate the project .venv first — slurm_launcher + filelock and the
run_general.sh / run_general_supp.sh helpers were copied into it, and sbatch
resolves run_general.sh from PATH, so .venv/bin must be on PATH):
    source .venv/bin/activate
    python scripts/evaluation/sweep_ablation.py

    # Dry run (just print the matrix, no submission, any python works):
    python3 scripts/evaluation/sweep_ablation.py --dry-run

    # Re-run everything, ignoring existing results:
    python scripts/evaluation/sweep_ablation.py --force

    # Only build jobs whose command contains a substring:
    python3 scripts/evaluation/sweep_ablation.py --only 1B --dry-run
"""
import os
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_eval.sh"

# ─── Experiment matrix ────────────────────────────────────────────────────────
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
BENCHMARKS = ["nq", "hqa", "wiki", "musique"]
MODES = ["all", "gold"]
NQ_POS_SWEEP = [0, 4, 9]      # only for NQ "all docs"

ATTN_TYPES = ["standard", "blocked"]   # full-causal vs per-document block mask
REENCODE_NUM = 0              # 0 = no link tokens (clean untrained baseline)

# Eval batch size = 1 for everything: zero OOM risk on a 24GB 3090 regardless
# of model/benchmark context length. generation (200 new tokens) is the
# bottleneck so larger batches buy little; the longest job (8B wiki, ~12.5k
# examples) still finishes well within the 1-day timeout.
BATCH_SIZE = 1
# ──────────────────────────────────────────────────────────────────────────────

# Benchmark token used in result filenames (must match each *_eval.py).
BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}


def result_stem(bench: str, model: str, mode: str, pos: int, attn_type: str) -> str:
    """Deterministic result stem — MUST stay in sync with the *_eval.py scripts:
    result/{stem}.jsonl  (+ {stem}.summary.json)."""
    model_slug = model.split("/")[-1]
    if mode == "gold":
        state = "goldonly"
    elif bench == "nq":
        state = f"at{pos}"
    else:
        state = "all"
    return (f"{BENCH_TOKEN[bench]}_{model_slug}_{state}"
            f"_attn-{attn_type}_re{REENCODE_NUM}")


def build_combos(force: bool = False):
    """Expand the matrix into run_eval.sh command strings, skipping any
    experiment whose result/<stem>.jsonl already exists (unless force)."""
    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json")
    )
    combos = []
    for model in MODELS:
        for bench in BENCHMARKS:
            bsz = BATCH_SIZE
            for mode in MODES:
                if bench == "wiki" and wiki_missing:
                    print(f"  [skip] wiki/{mode}/{model} "
                          f"(data/raw/2wikimultihop/dev.json missing)")
                    continue
                if bench == "nq" and mode == "all":
                    positions = NQ_POS_SWEEP
                else:
                    positions = [0]
                for pos in positions:
                    for attn_type in ATTN_TYPES:
                        stem = result_stem(bench, model, mode, pos, attn_type)
                        done = os.path.isfile(
                            os.path.join(PROJECT_DIR, "result", f"{stem}.jsonl")
                        )
                        if done and not force:
                            print(f"  [skip] {stem} (result already exists; "
                                  f"--force to rerun)")
                            continue
                        combos.append(
                            f"{SCRIPT_PATH} {bench} {model} {mode} "
                            f"{pos} {bsz} {attn_type} {REENCODE_NUM}"
                        )
    return combos


def run_slurm(only: str = "", dry_run: bool = False, force: bool = False):
    combos = build_combos(force=force)
    if only:
        combos = [c for c in combos if only in c]

    if not combos:
        print("Nothing to submit.")
        return

    print(f"{len(combos)} jobs submitted.")
    # print(f"Matrix: {len(combos)} jobs")
    # for c in combos:
    #     print(f"  bash {c}")

    if dry_run:
        print("\nDry run only. Re-run without --dry-run (with the "
              "slurm_launcher env) to submit.")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks

    print(f"\nSubmitting {len(combos)} jobs to 'rtx3090'...")
    launch_tasks(
        param_option=1,                 # 1 GPU, 10 cpus, 60G mem
        base_cmd="bash",
        param_dict={"": combos},
        partition="rtx3090",
        exclude="radish",  # avoid known OOM-prone nodes
        qos="normal",
        timeout="1-0",
        job_name="kvlink_ablation",
        max_job_num=40,
        # part_to_py is a no-op here: select_env_wrap.py only rewrites a
        # literal "python" in base_cmd, and base_cmd is "bash". The eval
        # interpreter is pinned inside run_eval.sh ($PROJECT_ROOT/.venv).
        part_to_py={"rtx3090": f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry-run" in args
    force = "--force" in args
    only_val = ""
    if "--only" in args:
        i = args.index("--only")
        if i + 1 < len(args):
            only_val = args[i + 1]
    run_slurm(only=only_val, dry_run=dry, force=force)
