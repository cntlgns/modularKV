"""Fill in the missing pretrained-8B wiki cells: baseline + recover_pos_enc,
both modes, both prompt presets. Mirrors the 8B PT entries already present in
result/ablation/ for NQ/hqa/musique.

Job count: 1 model * 1 bench(wiki) * 2 modes * 2 presets * 2 policies = 8
Output: result/ablation/  (matches the existing PT 8B result location, so
the comparison table can pick them up directly).
"""
import os
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_eval.sh"
OUT_DIR = "result/ablation"

MODELS = ["meta-llama/Llama-3.1-8B-Instruct"]   # HF id -> slug "Llama-3.1-8B-Instruct"
BENCHMARKS = ["wiki"]
MODES = ["all", "gold"]
KV_POLICIES = ["baseline", "recover_pos_enc"]
MODULAR_Q_POS = "summed_pos"
PROMPT_PRESETS = ["kvlink", "short"]
REENCODE_NUM = 0
MAX_EXAMPLES = 1000
BATCH_SIZE = 1

BENCH_TOKEN = {"wiki": "wiki"}


def result_stem(bench: str, model: str, mode: str, preset: str, policy: str) -> str:
    model_slug = model.rstrip("/").split("/")[-1]
    state = "goldonly" if mode == "gold" else "all"
    ppx = "" if preset == "kvlink" else f"_p-{preset}"
    # recover_pos_enc / baseline have no qpos suffix in the existing convention.
    return (f"{BENCH_TOKEN[bench]}_{model_slug}_{state}"
            f"_kv-{policy}{ppx}_re{REENCODE_NUM}")


def build_combos(force: bool = False):
    if not os.path.isfile(os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json")):
        print("Error: wiki data missing")
        return []
    combos = []
    for model in MODELS:
        for bench in BENCHMARKS:
            for mode in MODES:
                for policy in KV_POLICIES:
                    for preset in PROMPT_PRESETS:
                        stem = result_stem(bench, model, mode, preset, policy)
                        done = os.path.isfile(os.path.join(PROJECT_DIR, OUT_DIR, f"{stem}.jsonl"))
                        if done and not force:
                            print(f"  [skip] {stem} (exists)")
                            continue
                        combos.append(
                            f"{SCRIPT_PATH} {bench} {model} {mode} "
                            f"0 {BATCH_SIZE} {policy} {MODULAR_Q_POS} "
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
        print("\nDry run only.")
        return
    from slurm_launcher.sbatch_launcher import launch_tasks
    base_cmd = f"MAX_EXAMPLES={MAX_EXAMPLES} OUT_DIR={OUT_DIR} bash"
    print(f"\nSubmitting {len(combos)} jobs to 'rtx3090'...")
    launch_tasks(
        param_option=1,
        base_cmd=base_cmd,
        param_dict={"": combos},
        partition="rtx3090",
        exclude="radish",
        qos="normal",
        timeout="1-0",
        job_name="kvlink_8b_pt_wiki",
        max_job_num=8,
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
