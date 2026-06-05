"""Iterate on the Qwen chat template on a FEW samples before sweeping.

Self-contained: builds inputs with the chosen ChatFormat, prefills under a
policy (build_policy_prefill), greedy-decodes with the format's gen_prompt +
stop tokens (no dependence on inference.py's Llama-hardcoded answer split), and
prints the answer + substring/em/f1. Lets us confirm Qwen actually stops and
gives short answers under both baseline and recover_pos_enc.

  python scripts/evaluation/verify_qwen_template.py --model Qwen/Qwen2.5-7B-Instruct \
      --bench nq --n 5 --prompt_preset short --max_new_tokens 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datasets as hfds
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.kvmod.prompt import build_segmented_inputs, get_chat_format  # noqa: E402
from src.kvmod.policies import build_policy_prefill  # noqa: E402
from src.utils.metrics import score_sample  # noqa: E402


def load_bench(bench, n):
    out = []
    if bench == "nq":
        ds = hfds.load_dataset("json", data_files=str(ROOT / "data/raw/nq/nq-open-10_0.jsonl"),
                               split="train").select(range(n))
        for ex in ds:
            out.append((ex["question"], [(c["title"], c["text"]) for c in ex["ctxs"][:10]],
                        ex["answers"]))
    elif bench == "hqa":
        ds = hfds.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").select(range(n))
        for ex in ds:
            docs = [(t, "".join(s)) for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
            out.append((ex["question"], docs, [ex["answer"]]))
    elif bench == "musique":
        ds = hfds.load_dataset("dgslibisey/MuSiQue", split="validation").select(range(n))
        for ex in ds:
            docs = [(p["title"], p["paragraph_text"]) for p in ex["paragraphs"]]
            out.append((ex["question"], docs, [ex["answer"]]))
    return out


@torch.no_grad()
def gen_one(model, tok, fmt, q, docs, policy, mem_start, mem_end, sts, max_new, preset):
    device = next(model.parameters()).device
    seg = build_segmented_inputs(tok, q, docs, prompt_preset=preset, reencode_num=0,
                                 mem_start=mem_start, mem_end=mem_end,
                                 special_token_start=sts, chat_format=fmt)
    input_ids = torch.tensor([seg["input_ids"]], device=device)
    segment_ids = torch.tensor([seg["segment_ids"]], device=device)
    pad_mask = (segment_ids != -1).long()
    pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
    mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
            if pf.attention_mask_4d is not None else pad_mask)
    out = model(input_ids=input_ids, attention_mask=mask,
                position_ids=pf.position_ids.to(device),
                past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    gp = tok(fmt.gen_prompt, add_special_tokens=False)["input_ids"]
    G = len(gp)
    pos = pf.next_position.to(device).unsqueeze(1) + torch.arange(G, device=device)
    attn = torch.cat([pad_mask, torch.ones((1, G), device=device, dtype=pad_mask.dtype)], 1)
    out = model(input_ids=torch.tensor([gp], device=device), attention_mask=attn,
                position_ids=pos, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1)
    cur = pos[:, -1] + 1
    stop_ids = {tok.convert_tokens_to_ids(s) for s in fmt.stop_strings}
    stop_ids |= {tok.eos_token_id}
    gen = []
    for _ in range(max_new):
        t = int(nxt[0]);
        if t in stop_ids:
            break
        gen.append(t)
        attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], 1)
        out = model(input_ids=nxt.unsqueeze(1), attention_mask=attn,
                    position_ids=cur.unsqueeze(1), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1); cur = cur + 1
    return tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bench", default="nq", choices=["nq", "hqa", "musique"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--policies", nargs="+", default=["baseline", "recover_pos_enc"])
    ap.add_argument("--prompt_preset", default="short", choices=["short","kvlink"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    fmt = get_chat_format(args.model)
    print(f"[fmt={fmt.name}] preset={args.prompt_preset} gen_prompt={fmt.gen_prompt!r} stops={fmt.stop_strings}")
    vocab = len(tok)
    mem_start, mem_end, sts = (vocab - 2, vocab - 1, vocab - 2 - 1024)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

    data = load_bench(args.bench, args.n)
    agg = {p: {"substring": 0.0, "em": 0.0, "f1": 0.0} for p in args.policies}
    for i, (q, docs, gold) in enumerate(data):
        print(f"\n===== [{i}] Q: {q}")
        print(f"      gold: {gold}")
        for p in args.policies:
            ans = gen_one(model, tok, fmt, q, docs, p, mem_start, mem_end, sts, args.max_new_tokens, args.prompt_preset)
            sc = score_sample(ans, gold)
            for k in agg[p]:
                agg[p][k] += sc[k] / len(data)
            print(f"  [{p:16s}] sub={sc['substring']:.0f} em={sc['em']:.0f} f1={sc['f1']:.2f} | {ans!r}")
    print("\n==== MEAN over", len(data), "samples ====")
    for p in args.policies:
        a = agg[p]
        print(f"  {p:16s}: substring={a['substring']:.3f} em={a['em']:.3f} f1={a['f1']:.3f}")


if __name__ == "__main__":
    main()
