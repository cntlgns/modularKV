"""
SLURM launcher: modularization ablation for a pretrained (non-fine-tuned) model.

Submits one single-GPU job per experiment in the matrix

    models x benchmarks x doc-modes x kv-policies  (+ NQ gold-position sweep)

Each job runs scripts/evaluation/run_eval.sh, which loads the pretrained HF
model and evaluates it on one benchmark in one document mode.

Matrix (edit the constants below to change it):
  - MODELS:      Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct
  - BENCHMARKS:  nq, hqa, wiki, musique
  - MODES:       all (all retrieved docs), gold (gold/supporting docs only)
  - KV_POLICIES: baseline, recover_pos_enc, modular, recover_cross_attn,
                 recover_cross_attn_oracle_pos (== legacy standard/blocked +
                 3 new). reencode_num=0 (no link tokens).
  - MODULAR_Q_POS: summed_pos (+ optionally min_pos); only modular /
                 recover_cross_attn expand over it.
  - PROMPT_PRESETS: kvlink (default; stem unchanged) (+ optionally short =
                 short-answer system prompt, stem gets _p-short).
  - NQ "all":    sweeps gold-doc position in NQ_POS_SWEEP; every other
                 (bench, mode) pair is a single job.
  Job count = |MODELS| x |KV_POLICIES (+qpos expansion)| x sum over benches
  (wiki auto-skipped until data/raw/2wikimultihop/dev.json exists).

Idempotent: each experiment writes a deterministic, ablation-unique
result/ablation/<stem>.jsonl (+ <stem>.summary.json with accuracy/timestamp) where
stem = {BENCH}_{model}_{state}_kv-{policy}[-{qpos}][_p-{preset}]_re{reencode}. A combo whose
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
BENCHMARKS = ["wiki"] # "nq", "hqa", "wiki", "musique"
MODES = ["all", "gold"] # "all", "gold"
NQ_POS_SWEEP = [0, 4, 9]      # only for NQ "all docs"

# KV-cache modularization policies (names shared with the
# modularization_ablation repo). Legacy aliases standard==baseline,
# blocked==recover_pos_enc still accepted by the eval scripts.
KV_POLICIES = [
    "baseline",
    "recover_pos_enc",
    "modular",
    "recover_cross_attn",
    "recover_cross_attn_oracle_pos",
]
# Question RoPE placement; only modular / recover_cross_attn are affected, and
# only those policies expand over this list (others use summed_pos as a no-op).
MODULAR_Q_POS = ["summed_pos"]  # add "min_pos" to sweep both
# Policies whose stem carries a -{modular_q_pos} suffix. MUST stay in sync
# with src/kvmod/policies.py:POLICIES_WITH_MODULAR_Q_POS (not imported here so
# --dry-run works without torch).
_POLICIES_WITH_QPOS = {"modular", "recover_cross_attn"}
# System-prompt preset. 'kvlink' == original wording (stem unchanged, keeps
# the in-flight comparison valid); 'short' == short-answer prompt (EM/F1↑),
# stem gets a _p-short suffix. MUST stay in sync with
# src/kvmod/prompt.py:PROMPT_PRESETS.
PROMPT_PRESETS = ["kvlink", "short"]  # both system-prompt presets
REENCODE_NUM = 0              # 0 = no link tokens (clean untrained baseline)

# Eval batch size = 1 for everything: zero OOM risk on a 24GB 3090 regardless
# of model/benchmark context length. generation (200 new tokens) is the
# bottleneck so larger batches buy little; the longest job (8B wiki, ~12.5k
# examples) still finishes well within the 1-day timeout.
BATCH_SIZE = 1
# ──────────────────────────────────────────────────────────────────────────────

# Benchmark token used in result filenames (must match each *_eval.py).
BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}


def result_stem(bench: str, model: str, mode: str, pos: int,
                 policy: str, modular_q_pos: str,
                 prompt_preset: str = "kvlink") -> str:
    """Deterministic result stem — MUST stay in sync with the *_eval.py scripts:
    result/ablation/{stem}.jsonl  (+ {stem}.summary.json)."""
    model_slug = model.split("/")[-1]
    if mode == "gold":
        state = "goldonly"
    elif bench == "nq":
        state = f"at{pos}"
    else:
        state = "all"
    qpos = f"-{modular_q_pos}" if policy in _POLICIES_WITH_QPOS else ""
    ppx = "" if prompt_preset == "kvlink" else f"_p-{prompt_preset}"
    return (f"{BENCH_TOKEN[bench]}_{model_slug}_{state}"
            f"_kv-{policy}{qpos}{ppx}_re{REENCODE_NUM}")


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
                    for policy in KV_POLICIES:
                        # Only reset-to-zero policies vary with modular_q_pos;
                        # others run once (summed_pos as a no-op).
                        qpos_list = (
                            MODULAR_Q_POS
                            if policy in _POLICIES_WITH_QPOS
                            else ["summed_pos"]
                        )
                        for qpos in qpos_list:
                            for preset in PROMPT_PRESETS:
                                stem = result_stem(
                                    bench, model, mode, pos, policy, qpos,
                                    preset,
                                )
                                done = os.path.isfile(
                                    os.path.join(
                                        PROJECT_DIR, "result", "ablation",
                                        f"{stem}.jsonl"
                                    )
                                )
                                if done and not force:
                                    print(f"  [skip] {stem} (result already "
                                          f"exists; --force to rerun)")
                                    continue
                                combos.append(
                                    f"{SCRIPT_PATH} {bench} {model} {mode} "
                                    f"{pos} {bsz} {policy} {qpos} {preset} "
                                    f"{REENCODE_NUM}"
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
        max_job_num=80,
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
