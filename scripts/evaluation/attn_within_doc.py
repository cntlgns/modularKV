"""WITHIN-document attention: where does a doc token's mass land, per layer & HEAD.

Follow-up to attn_doc_block.py. Under recover_pos_enc a doc token attends 100%
within its own segment (sink/sys/other-docs all masked). This script asks WHERE
inside the document that mass goes, and whether a NEW local sink forms at the
document's first token. It keeps HEAD resolution (no head-averaging in the
reduction) so per-head specialization shows up.

For a query at relative position i inside its doc (doc-local keys j, 0<=j<=i),
we partition the within-doc mass into 4 disjoint regions:

    doc_first  j == 0  and i > 0   (the doc's first token -- local-sink candidate)
    self       j == i              (the diagonal)
    recent     i-R <= j < i, j>0   (recency window, R=8)
    mid        0 <  j < i-R        (older within-doc body)

so doc_first + self + recent + mid == within_total (the own_doc mass). We also
record, per (layer, head): sink0 (= global key 0 mass; the baseline attention
sink, masked-out under recover_pos_enc) and within_total. external = 1 - sink0
- within_total - ... is reported as residual (sys-minus-sink + other docs).

Output: per (layer, head) arrays for baseline & recover_pos_enc, plus console
tables (overall, per-layer, notable heads).

Datasets: nq, hqa_full, wiki, musique, block_qa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from scripts.evaluation.attn_analysis import Sample, LOADERS  # noqa: E402
from scripts.evaluation.attn_region_compare import load_block_qa  # noqa: E402

ALL_LOADERS = dict(LOADERS)
ALL_LOADERS["block_qa"] = load_block_qa

# region order in the per-head record
REC_NAMES = ["sink0", "within", "doc_first", "self", "recent", "mid", "entropy"]
RECENT_R = 8


class WithinDocCollector:
    """Per (layer, head) running sums of within-doc region masses over doc rows."""

    def __init__(self, n_layers: int, n_heads: int):
        self.n_layers, self.n_heads = n_layers, n_heads
        self.handles = []
        self._s: Sample | None = None
        self._li = 0
        K = len(REC_NAMES)
        self.sum = np.zeros((n_layers, n_heads, K), dtype=np.float64)
        self.cnt = np.zeros((n_layers,), dtype=np.float64)  # doc rows seen per layer

    def attach(self, model):
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin(self, sample):
        self._s = sample
        self._li = 0

    def _hook(self, module, inputs, output):
        if self._s is None:
            return
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        li = self._li
        self._li += 1
        a = attn[0].float()                # (H, T, T)
        H, T, _ = a.shape
        eps = 1e-12
        acc = np.zeros((H, len(REC_NAMES)), dtype=np.float64)
        n_rows = 0
        for ds, de in s.doc_spans:
            de = min(de, T)
            L = de - ds
            if L <= 0:
                continue
            sink0 = a[:, ds:de, 0]                       # (H, L)
            r = a[:, ds:de, ds:de]                       # (H, L, L) within-doc
            # causal band masks over (L, L); j=col(key), i=row(query)
            ii = torch.arange(L, device=a.device).unsqueeze(1)   # (L,1) query
            jj = torch.arange(L, device=a.device).unsqueeze(0)   # (1,L) key
            causal = jj <= ii
            self_m = (jj == ii)
            first_m = (jj == 0) & (ii > 0)
            recent_m = (jj >= (ii - RECENT_R)) & (jj < ii) & (jj > 0)
            mid_m = (jj > 0) & (jj < (ii - RECENT_R))
            within = (r * causal).sum(-1)                # (H, L)
            v_self = (r * self_m).sum(-1)
            v_first = (r * first_m).sum(-1)
            v_recent = (r * recent_m).sum(-1)
            v_mid = (r * mid_m).sum(-1)
            # within-doc renormalized entropy (per row, then summed)
            rn = (r * causal)
            rn = rn / rn.sum(-1, keepdim=True).clamp_min(eps)
            ent = -(rn.clamp_min(eps) * rn.clamp_min(eps).log()).sum(-1)   # (H, L)
            stack = torch.stack(
                [sink0, within, v_first, v_self, v_recent, v_mid, ent], dim=-1)  # (H,L,7)
            acc += stack.sum(dim=1).double().cpu().numpy()                 # sum over rows
            n_rows += L
        self.sum[li] += acc
        self.cnt[li] += n_rows

    def per_head_mean(self) -> np.ndarray:
        """(n_layers, n_heads, K)."""
        out = np.full_like(self.sum, np.nan)
        for li in range(self.n_layers):
            if self.cnt[li] > 0:
                out[li] = self.sum[li] / self.cnt[li]
        return out


@torch.no_grad()
def run(model, samples, policy, n_layers, n_heads, device):
    col = WithinDocCollector(n_layers, n_heads)
    col.attach(model)
    try:
        for s in samples:
            input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
            segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
            pad_mask = (segment_ids != -1).long()
            pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
            mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
                    if pf.attention_mask_4d is not None else pad_mask)
            col.begin(s)
            model(input_ids=input_ids, attention_mask=mask,
                  position_ids=pf.position_ids.to(device),
                  use_cache=False, output_attentions=True)
            col.begin(None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        col.detach()
    return col.per_head_mean()


def idx(name):
    return REC_NAMES.index(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--datasets", nargs="+",
                    default=["nq", "block_qa", "hqa_full", "wiki", "musique"])
    ap.add_argument("--policies", nargs="+", default=["baseline", "recover_pos_enc"])
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--out_dir", default="result/attn_within_doc")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    out_path = out_dir / f"within_doc__{slug}.json"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[info] model={args.model} n_layers={n_layers} n_heads={n_heads} "
          f"R={RECENT_R} n={args.n_samples}")

    payload: Dict = {"model": args.model, "rec_names": REC_NAMES, "recent_R": RECENT_R,
                     "n_layers": n_layers, "n_heads": n_heads, "results": {}}

    for ds in args.datasets:
        samples = ALL_LOADERS[ds](args.n_samples, tok)
        print(f"\n[{ds}] {len(samples)} samples", flush=True)
        payload["results"][ds] = {}
        for pol in args.policies:
            phm = run(model, samples, pol, n_layers, n_heads, args.device)  # (L,H,K)
            payload["results"][ds][pol] = phm.tolist()
            o = np.nanmean(phm, axis=(0, 1))   # over layers & heads
            print(f"  [{pol:16s}] sink0={o[idx('sink0')]:.3f} within={o[idx('within')]:.3f} "
                  f"|| doc_first={o[idx('doc_first')]:.3f} self={o[idx('self')]:.3f} "
                  f"recent={o[idx('recent')]:.3f} mid={o[idx('mid')]:.3f} "
                  f"H={o[idx('entropy')]:.2f}", flush=True)

    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()
