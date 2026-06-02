"""Convert a TorchTitan DCP checkpoint to an HF-loadable model directory.

Pipeline:
  DCP dir (step-N/) --dcp_to_torch--> .pt (torchtune state_dict)
  .pt --tune_to_hf--> HF state_dict
  HF state_dict + Llama-3.2-1B-Instruct config --save_pretrained--> out_dir

Usage:
    python scripts/evaluation/convert_dcp_to_hf.py \
        --dcp_dir run_logs/data_original_step6k_bsz64_link_0_full_ckpt/checkpoints/step-2000 \
        --out_dir model_cache/Llama-3.2-1B-Instruct-ft-step2000 \
        --base_model meta-llama/Llama-3.2-1B-Instruct
"""
import argparse
import os
import tempfile

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtune.models.convert_weights import tune_to_hf
from transformers import AutoTokenizer, LlamaForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp_dir", required=True, help="DCP checkpoint directory (contains .metadata + __k_0.distcp)")
    ap.add_argument("--out_dir", required=True, help="Destination HF model directory")
    ap.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF id used for architecture/config + tokenizer")
    ap.add_argument("--num_heads", type=int, default=32)
    ap.add_argument("--num_kv_heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        pt_path = os.path.join(tmp, "checkpoint.pt")
        print(f"[1/3] dcp_to_torch_save: {args.dcp_dir} -> {pt_path}")
        dcp_to_torch_save(args.dcp_dir, pt_path)

        print(f"[2/3] loading + tune_to_hf (heads={args.num_heads}, kv_heads={args.num_kv_heads}, dim={args.dim})")
        sd = torch.load(pt_path, weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        if "output.weight" not in sd and "tok_embeddings.weight" in sd:
            sd["output.weight"] = sd["tok_embeddings.weight"]
        hf_sd = tune_to_hf(state_dict=sd, num_heads=args.num_heads,
                           num_kv_heads=args.num_kv_heads, dim=args.dim)

    print(f"[3/3] loading {args.base_model}, applying finetuned weights, saving to {args.out_dir}")
    model = LlamaForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    if missing:
        print(f"  WARN: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    model.save_pretrained(args.out_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(args.out_dir)

    print(f"Done. HF model dir: {args.out_dir}")


if __name__ == "__main__":
    main()
