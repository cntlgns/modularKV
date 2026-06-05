"""SLURM launcher: Qwen3-32B (non-thinking) baseline vs recover_pos_enc, 4 benchmarks, 1000
examples each, all+gold modes, short+verbose(kvlink) prompts. One rtx3090 job
per experiment. Mirrors sweep_ablation.py but for the Qwen pretrained model and
capped at MAX_EXAMPLES=1000 (set inside run_eval_qwen3_32b.sh) -> result/qwen_sweep/.

Matrix = 4 bench x 2 mode x 2 policy x 2 prompt = 32 jobs (NQ "all" uses pos 0
only, not the 0/4/9 position sweep).

    source .venv/bin/activate
    python scripts/evaluation/sweep_qwen.py --dry-run   # inspect matrix
    python scripts/evaluation/sweep_qwen.py             # submit
"""
import os
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_eval_qwen3_32b.sh"
OUT_SUBDIR = "qwen3_32b_sweep"

MODELS = ["Qwen/Qwen3-32B"]
BENCHMARKS = ["nq", "hqa", "wiki", "musique"]
MODES = ["all", "gold"]
NQ_POS_SWEEP = [0]                 # single position (no 0/4/9 sweep here)
KV_POLICIES = ["baseline", "recover_pos_enc"]
MODULAR_Q_POS = ["summed_pos"]     # no-op for baseline/recover_pos_enc
_POLICIES_WITH_QPOS = {"modular", "recover_cross_attn"}
PROMPT_PRESETS = ["kvlink", "short"]   # verbose + short-answer
REENCODE_NUM = 0
BATCH_SIZE = 1

BENCH_TOKEN = {"nq": "NQ", "hqa": "hqa", "wiki": "wiki", "musique": "musique"}


def result_stem(bench, model, mode, pos, policy, modular_q_pos, prompt_preset):
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


def build_combos(force=False):
    wiki_missing = not os.path.isfile(
        os.path.join(PROJECT_DIR, "data", "raw", "2wikimultihop", "dev.json"))
    combos = []
    for model in MODELS:
        for bench in BENCHMARKS:
            for mode in MODES:
                if bench == "wiki" and wiki_missing:
                    print(f"  [skip] wiki/{mode} (data missing)"); continue
                positions = NQ_POS_SWEEP if (bench == "nq" and mode == "all") else [0]
                for pos in positions:
                    for policy in KV_POLICIES:
                        qpos_list = (MODULAR_Q_POS if policy in _POLICIES_WITH_QPOS
                                     else ["summed_pos"])
                        for qpos in qpos_list:
                            for preset in PROMPT_PRESETS:
                                stem = result_stem(bench, model, mode, pos, policy, qpos, preset)
                                done = os.path.isfile(os.path.join(
                                    PROJECT_DIR, "result", OUT_SUBDIR, f"{stem}.jsonl"))
                                if done and not force:
                                    print(f"  [skip] {stem} (exists)"); continue
                                combos.append(
                                    f"{SCRIPT_PATH} {bench} {model} {mode} "
                                    f"{pos} {BATCH_SIZE} {policy} {qpos} {preset} {REENCODE_NUM}")
    return combos


def run_slurm(only="", dry_run=False, force=False):
    combos = build_combos(force=force)
    if only:
        combos = [c for c in combos if only in c]
    if not combos:
        print("Nothing to submit."); return
    print(f"{len(combos)} jobs:")
    for c in combos:
        print("  bash " + c.replace(PROJECT_DIR + "/scripts/evaluation/", ""))
    if dry_run:
        print("\nDry run only."); return
    from slurm_launcher.sbatch_launcher import launch_tasks
    print(f"\nSubmitting {len(combos)} jobs to 'rtx3090'...")
    launch_tasks(
        param_option=1, base_cmd="bash", param_dict={"": combos},
        partition="a100", qos="normal", timeout="1-0",
        job_name="qwen3_32b_sweep", max_job_num=80,
        part_to_py={"rtx3090": f"{PROJECT_DIR}/.venv/bin/python"},
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    only_val = ""
    if "--only" in args:
        i = args.index("--only")
        only_val = args[i + 1] if i + 1 < len(args) else ""
    run_slurm(only=only_val, dry_run="--dry-run" in args, force="--force" in args)
