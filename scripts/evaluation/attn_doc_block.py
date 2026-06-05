"""Where does a DOCUMENT-BLOCK token send its attention? baseline vs recover_pos_enc.

Each document is rendered as ``Document [k](Title: T) text\n`` and assigned its
own ``segment_id`` (>=1). During prefill a doc token is a *query row*; this
script measures, for those rows, how the attention mass splits over four
key-regions:

    sink        key 0                                  (the attention-sink token)
    sys         keys [0, sys_end)  (= segment 0 prefix; includes the sink)
    own_doc     keys in the token's OWN doc span        (within-block causal past)
    other_docs  keys in every OTHER doc span            (cross-doc)

For a doc row the only causally-reachable keys live in ``sys`` U all-doc-spans
(the question band comes later and is causally masked), so

    sys + own_doc + other_docs == 1     (per softmax row).

  * baseline         full causal -> a doc token sees the prefix and every
                     PRECEDING document, so ``other_docs`` > 0.
  * recover_pos_enc  block-diagonal segment mask -> a doc token may attend only
                     to its own segment and segment 0, so ``other_docs`` == 0 by
                     construction; that mass must land on ``sys`` / ``own_doc``.

We forward each sample once (prefill only -- doc rows live entirely in the
prefill) with ``output_attentions=True`` and a per-layer forward hook that
reduces the (H, T, T) attention to per-region SUMS over (heads x doc-rows),
so nothing large is retained. Aggregated per layer and overall, per policy,
per dataset.

Datasets: nq, hqa_gold, hqa_full, wiki, musique, block_qa.

Output: JSON at <out_dir>/doc_block_attn__<model_slug>.json plus a console table.
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
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from scripts.evaluation.attn_analysis import Sample, LOADERS, _assemble  # noqa: E402
from scripts.evaluation.attn_region_compare import load_block_qa  # noqa: E402

ALL_LOADERS = dict(LOADERS)
ALL_LOADERS["block_qa"] = load_block_qa

REGION_NAMES = ["sink", "sys", "own_doc", "other_docs"]


class DocRowCollector:
    """Forward hook reducing each layer's attn to doc-row region SUMS.

    Accumulates, per layer: a (4,) sum over (heads x doc-rows) of region masses
    and a scalar count (= heads * #doc-rows), so the running mean is
    ``sum / count``. One collector spans all samples of one (dataset, policy).
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.handles = []
        self._s: Sample | None = None
        self._layer_iter = 0
        self.sum = [np.zeros(len(REGION_NAMES), dtype=np.float64) for _ in range(n_layers)]
        self.cnt = [0 for _ in range(n_layers)]
        # diagnostic: how close does sys+own+other get to 1.0 (softmax row sum)?
        self.row_total_sum = [0.0 for _ in range(n_layers)]

    def attach(self, model):
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def begin(self, sample: Sample):
        self._s = sample
        self._layer_iter = 0

    def _hook(self, module, inputs, output):
        if self._s is None:
            return
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        li = self._layer_iter
        self._layer_iter += 1

        a = attn[0].float()                  # (H, T, T)
        T = a.shape[-1]
        H = a.shape[0]
        _, sys_end = s.sys_span
        docs_mask = torch.zeros(T, dtype=torch.bool, device=a.device)
        for ds, de in s.doc_spans:
            docs_mask[ds:min(de, T)] = True

        # Gather query rows = all doc tokens, region masses per (H, row).
        acc = np.zeros(len(REGION_NAMES), dtype=np.float64)
        n_rows = 0
        row_total = 0.0
        for ds, de in s.doc_spans:
            de = min(de, T)
            if ds >= de:
                continue
            r = a[:, ds:de, :]               # (H, dl, T)
            sink = r[..., 0]                 # (H, dl)
            sys_m = r[..., 0:sys_end].sum(-1)
            own = r[..., ds:de].sum(-1)      # own span; future keys are 0 (causal)
            alldocs = r[..., docs_mask].sum(-1)
            other = alldocs - own
            stacked = torch.stack([sink, sys_m, own, other], dim=-1)  # (H, dl, 4)
            acc += stacked.sum(dim=(0, 1)).double().cpu().numpy()
            dl = de - ds
            n_rows += dl
            row_total += float((sys_m + own + other).sum().item())
        self.sum[li] += acc
        self.cnt[li] += H * n_rows
        self.row_total_sum[li] += row_total

    def per_layer_mean(self) -> np.ndarray:
        out = np.full((self.n_layers, len(REGION_NAMES)), np.nan)
        for li in range(self.n_layers):
            if self.cnt[li] > 0:
                out[li] = self.sum[li] / self.cnt[li]
        return out

    def row_total_mean(self) -> np.ndarray:
        out = np.full(self.n_layers, np.nan)
        for li in range(self.n_layers):
            if self.cnt[li] > 0:
                out[li] = self.row_total_sum[li] / self.cnt[li]
        return out


@torch.no_grad()
def run_dataset_policy(model, samples, policy, n_layers, device):
    col = DocRowCollector(n_layers)
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
            model(
                input_ids=input_ids,
                attention_mask=mask,
                position_ids=pf.position_ids.to(device),
                use_cache=False,
                output_attentions=True,
            )
            col.begin(None)  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        col.detach()
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--datasets", nargs="+",
                    default=["nq", "block_qa", "hqa_full", "wiki", "musique"])
    ap.add_argument("--policies", nargs="+",
                    default=["baseline", "recover_pos_enc"])
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--out_dir", default="result/attn_doc_block")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split("/")[-1]
    out_path = out_dir / f"doc_block_attn__{model_slug}.json"

    print(f"[info] model={args.model} dtype={args.dtype} n={args.n_samples}")
    print(f"[info] datasets={args.datasets} policies={args.policies}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[info] n_layers={n_layers}")

    payload: Dict = {
        "model": args.model, "datasets": args.datasets, "policies": args.policies,
        "n_samples": args.n_samples, "region_names": REGION_NAMES,
        "n_layers": n_layers, "results": {},
    }

    for ds_name in args.datasets:
        samples = ALL_LOADERS[ds_name](args.n_samples, tok)
        lens = [len(s.input_ids) for s in samples]
        ndocs = [len(s.doc_spans) for s in samples]
        print(f"\n[{ds_name}] {len(samples)} samples; "
              f"seq_len mean={sum(lens)/len(lens):.0f} "
              f"docs mean={sum(ndocs)/len(ndocs):.1f}", flush=True)
        payload["results"][ds_name] = {
            "n_samples": len(samples), "seq_len_mean": sum(lens) / len(lens),
            "n_docs_mean": sum(ndocs) / len(ndocs), "policies": {},
        }
        for pol in args.policies:
            col = run_dataset_policy(model, samples, pol, n_layers, args.device)
            plm = col.per_layer_mean()                  # (n_layers, 4)
            overall = np.nanmean(plm, axis=0)           # (4,)
            rtm = col.row_total_mean()
            payload["results"][ds_name]["policies"][pol] = {
                "per_layer_mean": plm.tolist(),
                "overall_mean": overall.tolist(),
                "row_total_mean_overall": float(np.nanmean(rtm)),
            }
            print(f"  [{pol:16s}] sink={overall[0]:.3f} sys={overall[1]:.3f} "
                  f"own_doc={overall[2]:.3f} other_docs={overall[3]:.3f} "
                  f"(row_sum~{np.nanmean(rtm):.3f})", flush=True)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[done] {out_path}")

    # ---- console summary table ---- #
    print("\n" + "=" * 78)
    print(f"DOC-BLOCK ATTENTION (mean over layers, heads, doc-rows, {args.n_samples} samples/ds)")
    print("=" * 78)
    hdr = f"{'dataset':10s} {'policy':16s} {'sink':>6s} {'sys':>7s} {'own_doc':>8s} {'other_docs':>11s}"
    print(hdr)
    print("-" * 78)
    for ds_name in args.datasets:
        for pol in args.policies:
            o = payload["results"][ds_name]["policies"][pol]["overall_mean"]
            print(f"{ds_name:10s} {pol:16s} {o[0]:6.3f} {o[1]:7.3f} {o[2]:8.3f} {o[3]:11.3f}")
        print("-" * 78)


if __name__ == "__main__":
    main()
