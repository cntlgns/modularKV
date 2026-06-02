"""Focused sweep: KVLink0 8B finetune (step-2000) eval.

Fixed axes (per user request 2026-05-29):
  - model:         model_cache/Llama-3.1-8B-Instruct-ft-step2000
                   (DCP step-2000 of data_original_step6k_bsz64_link_0_full_ckpt_8b,
                    converted to HF via convert_dcp_to_hf.py with 8B args)
  - kv_policy:     recover_pos_enc (only; matches link_0 = reencode_num=0)
  - prompt_preset: kvlink AND short
  - mode:          {all, gold}      (full retrieval AND gold-only)
  - NQ position:   0                (single position; no lost-in-the-middle sweep)
  - reencode_num:  0
  - max_examples:  1000             (per benchmark; clamped to len(dataset))

Job count: 1 model * 2 presets * 4 benches * 2 modes = 16
Combos whose .jsonl already exists in result/ft_vs_pt_8b/ are skipped
(use --force to rerun them).

Usage (activate the project .venv first):
    source .venv/bin/activate
    python scripts/evaluation/sweep_8b_step2000.py            # submit
    python3 scripts/evaluation/sweep_8b_step2000.py --dry-run # just print
    python  scripts/evaluation/sweep_8b_step2000.py --force   # rerun even if done
"""
import os
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_eval.sh"
OUT_DIR = "result/ft_vs_pt_8b"

MODELS = [
    f"{PROJECT_DIR}/model_cache/Llama-3.1-8B-Instruct-ft-step2000",
]
BENCHMARKS = ["nq", "hqa", "wiki", "musique"]
MODES = ["all", "gold"]           # gold = gold/supporting docs only
NQ_POS_SWEEP = [0]                # single position (no lost-in-the-middle sweep)
KV_POLICY = "recover_pos_enc"
MODULAR_Q_POS = "summed_pos"      # no-op for recover_pos_enc
PROMPT_PRESETS = ["kvlink", "short"]
REENCODE_NUM = 0
MAX_EXAMPLES = 1000
BATCH_SIZE = 1

BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}


def result_stem(bench: str, model: str, mode: str, pos: int, preset: str) -> str:
    """Same stem convention as sweep_ft_vs_pt.result_stem (recover_pos_enc has no qpos suffix)."""
    model_slug = model.rstrip("/").split("/")[-1]
    if mode == "gold":
        state = "goldonly"
    elif bench == "nq":
        state = f"at{pos}"
    else:
        state = "all"
    ppx = "" if preset == "kvlink" else f"_p-{preset}"
    return (f"{BENCH_TOKEN[bench]}_{model_slug}_{state}"
            f"_kv-{KV_POLICY}{ppx}_re{REENCODE_NUM}")


def build_combos(force: bool = False):
    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json")
    )
    combos = []
    for model in MODELS:
        for bench in BENCHMARKS:
            if bench == "wiki" and wiki_missing:
                print(f"  [skip] wiki/{model} (data/raw/2wikimultihop/dev.json missing)")
                continue
            for mode in MODES:
                # NQ "all" sweeps gold position; everything else is a single pos=0 job.
                if mode == "all" and bench == "nq":
                    positions = NQ_POS_SWEEP
                else:
                    positions = [0]
                for pos in positions:
                    for preset in PROMPT_PRESETS:
                        stem = result_stem(bench, model, mode, pos, preset)
                        done = os.path.isfile(
                            os.path.join(PROJECT_DIR, OUT_DIR, f"{stem}.jsonl")
                        )
                        if done and not force:
                            print(f"  [skip] {stem} (already exists; --force to rerun)")
                            continue
                        combos.append(
                            f"{SCRIPT_PATH} {bench} {model} {mode} "
                            f"{pos} {BATCH_SIZE} {KV_POLICY} {MODULAR_Q_POS} "
                            f"{preset} {REENCODE_NUM}"
                        )
    return combos


def run_slurm(only: str = "", dry_run: bool = False, force: bool = False):
    combos = build_combos(force=force)
    if only:
        combos = [c for c in combos if only in c]

    if not combos:
        print("Nothing to submit.")
        return

    print(f"{len(combos)} jobs will run.")
    for c in combos:
        print(f"  bash {c}")

    if dry_run:
        print("\nDry run only. Re-run without --dry-run to submit.")
        return

    from slurm_launcher.sbatch_launcher import launch_tasks

    # Forward MAX_EXAMPLES + OUT_DIR by baking them into base_cmd as env-prefix
    # assignments (survives select_env_wrap.py's literal "python" rewrite).
    base_cmd = f"MAX_EXAMPLES={MAX_EXAMPLES} OUT_DIR={OUT_DIR} bash"

    print(f"\nSubmitting {len(combos)} jobs to 'rtx3090'...")
    launch_tasks(
        param_option=1,                 # 1 GPU, 10 cpus, 60G mem
        base_cmd=base_cmd,
        param_dict={"": combos},
        partition="rtx3090",
        exclude="radish",
        qos="normal",
        timeout="1-0",
        job_name="kvlink_8b_step2000",
        max_job_num=16,
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
