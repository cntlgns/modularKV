"""Mechanistic q.k^T decomposition of the attention sink: baseline(BOS) vs
recover_pos_enc(doc-first), per layer & head, broken down by RoPE freq-pair.

For doc-token query rows we recompute the EXACT post-RoPE q,k that each layer
uses (from output_hidden_states + the layer's input_layernorm / q_proj / k_proj
+ the model's rotary_emb), then decompose the sink logit

    logit_sink = (q_i . k_sink) / sqrt(head_dim)

into the head_dim/2 RoPE frequency pairs  (pair j = dims {j, j+d/2}, j=0 is the
LOWEST frequency = slowest rotation = the "front"/positional component; j=d/2-1
is the highest). This tells us whether the large sink logit comes from a few
low-frequency (positional) pairs, or from specific high-magnitude feature dims
(massive-activation style), and whether the SAME pairs drive it for the BOS sink
(baseline) and the per-doc "Document" sink (recover_pos_enc).

sink key per query row i:
    baseline         : global key 0  (<|begin_of_text|>)
    recover_pos_enc  : the first token of i's own document (rel-pos 0, "Document")

Outputs per (layer, head): sink logit, mean non-sink (own-doc) logit, the 32
per-pair contributions to the sink logit, |k_sink| per dim, and (recover) the
softmax sink probability as a sanity check against attn_within_doc.

Datasets: nq, musique (mechanism is dataset-invariant; 2 are enough).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from scripts.evaluation.attn_analysis import LOADERS  # noqa: E402
from scripts.evaluation.attn_region_compare import load_block_qa  # noqa: E402

ALL_LOADERS = dict(LOADERS)
ALL_LOADERS["block_qa"] = load_block_qa


def recompute_qk(model, hidden_states, position_ids, layer):
    """Return post-RoPE q (1,Hq,T,d), k repeated to Hq heads (1,Hq,T,d) for `layer`."""
    lyr = model.model.layers[layer]
    attn = lyr.self_attn
    h = lyr.input_layernorm(hidden_states[layer])          # (1,T,H)
    B, T, _ = h.shape
    nh, nkv, d = attn.config.num_attention_heads, attn.config.num_key_value_heads, attn.head_dim
    q = attn.q_proj(h).view(B, T, nh, d).transpose(1, 2)   # (1,nh,T,d)
    k = attn.k_proj(h).view(B, T, nkv, d).transpose(1, 2)  # (1,nkv,T,d)
    cos, sin = model.model.rotary_emb(h, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k = k.repeat_interleave(nh // nkv, dim=1)               # GQA -> (1,nh,T,d)
    return q, k


@torch.no_grad()
def run(model, samples, policy, device):
    n_layers = model.config.num_hidden_layers
    nh = model.config.num_attention_heads
    d = model.model.layers[0].self_attn.head_dim
    half = d // 2
    scale = 1.0 / math.sqrt(d)

    sink_logit = np.zeros((n_layers, nh)); nonsink_logit = np.zeros((n_layers, nh))
    pair_contrib = np.zeros((n_layers, nh, half))          # to sink logit
    k_absdim = np.zeros((n_layers, nh, d))                 # |k_sink| per dim
    sink_prob = np.zeros((n_layers, nh))                   # recover sanity
    cnt = 0

    for s in samples:
        input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
        pad_mask = (segment_ids != -1).long()
        pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
        mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
                if pf.attention_mask_4d is not None else pad_mask)
        pos = pf.position_ids.to(device)
        out = model(input_ids=input_ids, attention_mask=mask, position_ids=pos,
                    use_cache=False, output_hidden_states=True, output_attentions=False)
        hs = out.hidden_states          # tuple len n_layers+1; hs[L]=input to layer L

        for L in range(n_layers):
            q, k = recompute_qk(model, hs, pos, L)          # (1,nh,T,d)
            q0, k0 = q[0].float(), k[0].float()             # (nh,T,d)
            for ds, de in s.doc_spans:
                L_ = de - ds
                if L_ <= 0:
                    continue
                qr = q0[:, ds:de, :]                        # (nh, L_, d)
                s_idx = (0 if policy == "baseline" else ds)
                ksink = k0[:, s_idx, :]                     # (nh, d)
                cd = qr * ksink.unsqueeze(1) * scale        # (nh, L_, d) per-dim contrib
                # sink logit + pair contributions (mean over rows)
                sl = cd.sum(-1)                             # (nh, L_)
                sink_logit[L] += sl.sum(1).cpu().numpy()
                pc = cd[..., :half] + cd[..., half:]        # (nh, L_, half)
                pair_contrib[L] += pc.sum(1).cpu().numpy()
                k_absdim[L] += ksink.abs().cpu().numpy() * L_
                # non-sink own-doc keys mean logit (exclude rel-0)
                kr = k0[:, ds:de, :]                        # (nh, L_, d)
                full = torch.einsum('hid,hjd->hij', qr, kr) * scale  # (nh,L_,L_)
                ii = torch.arange(L_, device=device)
                causal = (ii.unsqueeze(0) >= ii.unsqueeze(1)) & (ii.unsqueeze(0) > 0)  # j<=i, j>0
                cnt_ns = causal.sum(1).clamp_min(1)
                ns = (full.masked_fill(~causal.unsqueeze(0), 0.0).sum(-1) / cnt_ns)  # (nh,L_)
                nonsink_logit[L] += ns.sum(1).cpu().numpy()
                # recover sink prob (softmax over own-doc causal keys)
                cmask = ii.unsqueeze(0) <= ii.unsqueeze(1)
                probs = full.masked_fill(~cmask.unsqueeze(0), -float('inf')).softmax(-1)
                sp = probs[:, 1:, 0]                        # rows i>0, prob on rel-0
                sink_prob[L] += sp.sum(1).cpu().numpy()
            if L == 0:
                pass
        cnt += sum(de - ds for ds, de in s.doc_spans)

    return {
        "sink_logit": (sink_logit / cnt),
        "nonsink_logit": (nonsink_logit / cnt),
        "pair_contrib": (pair_contrib / cnt),
        "k_absdim": (k_absdim / cnt),
        "sink_prob": (sink_prob / max(cnt, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["nq", "musique"])
    ap.add_argument("--policies", nargs="+", default=["baseline", "recover_pos_enc"])
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--out_dir", default="result/attn_qk_decompose")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    out_path = out_dir / f"qk_decompose__{slug}.json"

    tok = AutoTokenizer.from_pretrained(args.model); tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
    inv_freq = model.model.rotary_emb.inv_freq.float().cpu().numpy()   # (half,)

    # accumulate across datasets (mechanism is dataset-invariant) -> weighted mean
    agg = {}
    for ds in args.datasets:
        samples = ALL_LOADERS[ds](args.n_samples, tok)
        for pol in args.policies:
            r = run(model, samples, pol, args.device)
            if pol not in agg:
                agg[pol] = {k: [] for k in r}
            for k, v in r.items():
                agg[pol][k].append(v)
            print(f"[{ds}/{pol}] done", flush=True)
    res = {pol: {k: np.mean(np.stack(v), 0) for k, v in d.items()} for pol, d in agg.items()}

    payload = {"model": args.model, "inv_freq": inv_freq.tolist(),
               "policies": {p: {k: v.tolist() for k, v in d.items()} for p, d in res.items()}}
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[done] {out_path}")

    # ---- console: per-layer sink strength + freq-band breakdown ---- #
    half = inv_freq.shape[0]
    bands = [(0, 8), (8, 16), (16, 24), (24, 32)]
    for pol in args.policies:
        R = res[pol]
        sl = R["sink_logit"].mean(1); ns = R["nonsink_logit"].mean(1); sp = R["sink_prob"].mean(1)
        pc = R["pair_contrib"].mean(1)             # (L, half) head-averaged
        print(f"\n==== {pol} : per-layer ====")
        print(" L | sink_logit nonsink  gap | sinkProb | pair-contrib by freq band (low->high)")
        for L in range(sl.shape[0]):
            bb = [pc[L, a:b].sum() for a, b in bands]
            tot = pc[L].sum()
            frac = [f"{x/tot:+.2f}" if abs(tot) > 1e-6 else " . " for x in bb]
            print(f"L{L:2d}| {sl[L]:9.2f} {ns[L]:7.2f} {sl[L]-ns[L]:5.2f} | "
                  f"{sp[L]:6.3f}  | lo[{frac[0]} {frac[1]} {frac[2]} {frac[3]}]hi  (tot={tot:.2f})")


if __name__ == "__main__":
    main()
