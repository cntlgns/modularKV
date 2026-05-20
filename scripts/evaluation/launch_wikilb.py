"""One-off launcher: LongBench-2wikimqa control runs (baseline only).

Submits 2 single-GPU jobs (Llama-3.2-1B + Llama-3.1-8B, all+standard) to test
whether the paper's Table-1 1B 2WikiMQA number reflects a LongBench-style eval
set. Mirrors sweep_ablation.py's launch_tasks config. --dry-run to preview.
"""
import sys

PROJECT_DIR = "/data_fast/home/sihun/kvcache/KVLink"
SCRIPT_PATH = f"{PROJECT_DIR}/scripts/evaluation/run_eval.sh"

# run_eval.sh args: <bench> <model> <mode> <pos> <bsz> <kv_policy> <modular_q_pos> <prompt_preset> <reencode_num>
COMBOS = [
    f"{SCRIPT_PATH} wikilb meta-llama/Llama-3.2-1B-Instruct all 0 1 standard summed_pos kvlink 0",
    f"{SCRIPT_PATH} wikilb meta-llama/Llama-3.1-8B-Instruct all 0 1 standard summed_pos kvlink 0",
]

if __name__ == "__main__":
    dry = "--dry-run" in sys.argv[1:]
    for c in COMBOS:
        print(f"  bash {c}")
    if dry:
        print("\nDry run only.")
        sys.exit(0)
    from slurm_launcher.sbatch_launcher import launch_tasks
    print(f"\nSubmitting {len(COMBOS)} jobs to 'rtx3090'...")
    launch_tasks(
        param_option=1,
        base_cmd="bash",
        param_dict={"": COMBOS},
        partition="rtx3090",
        exclude="radish",
        qos="normal",
        timeout="1-0",
        job_name="kvlink_wikilb",
        max_job_num=2,
        part_to_py={"rtx3090": f"{PROJECT_DIR}/.venv/bin/python"},
    )
