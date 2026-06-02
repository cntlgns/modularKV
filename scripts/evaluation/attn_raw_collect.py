"""Collect raw (x, y) data points for regression on baseline-policy attention.

For each (sample, layer, head, question_token_index) tuple, record:
  features (x):
    benchmark, sample_id (str) ; layer (int), head (int) ; qi (int) ;
    qabs (int, absolute pos in seq) ; qprev_len (int, == qi) ;
    sys_len, docs_total_len, n_docs, mean_doc_len, min_doc_len, max_doc_len,
    seq_len, q_len, q_rel (= qi / max(q_len-1,1))
  targets (y):
    y_sink   = α[head, qi, 0]                       (mass on first token)
    y_sys    = Σ_{k ∈ [0, sys_end)} α[head, qi, k]  (mass on system prefix)
    y_docs   = Σ_{k ∈ all-doc-spans} α[head, qi, k] (mass on all docs)

Only the `baseline` policy is run (full causal, contiguous positions).
Output: a single Parquet file per (model, benchmark).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402

from scripts.evaluation.attn_analysis import (
    LOADERS, _spans, _assemble, MEM_START, MEM_END, SPECIAL_TOKEN_START,
)  # noqa: E402


class RawCollector:
    """Hook-based collector that buffers per-layer (head, qi)-resolved targets,
    then materializes a tidy DataFrame per sample on demand."""

    def __init__(self):
        self.handles = []
        self._buf: List[np.ndarray] = []  # one (H, q_len, 3) per layer in order
        self._s = None

    def attach(self, model):
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self, sample):
        self._buf = []
        self._s = sample

    def _hook(self, module, inputs, output):
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        q_lo, q_hi = s.question_span
        aq = attn[0, :, q_lo:q_hi, :].float()  # (H, q_len, T)
        H, q_len, T = aq.shape

        sys_end = s.sys_span[1]
        # docs_mask: bool over T marking doc tokens
        docs_mask = torch.zeros(T, dtype=torch.bool, device=aq.device)
        for ds, de in s.doc_spans:
            docs_mask[ds:de] = True

        y_sink = aq[..., 0]                          # (H, q_len)
        y_sys  = aq[..., :sys_end].sum(-1)            # (H, q_len)
        y_docs = aq[..., docs_mask].sum(-1)           # (H, q_len)
        stacked = torch.stack([y_sink, y_sys, y_docs], dim=-1)  # (H, q_len, 3)
        self._buf.append(stacked.detach().to("cpu").numpy())

    def to_rows(self, benchmark: str):
        s = self._s
        n_layers = len(self._buf)
        if n_layers == 0:
            return None
        H, q_len, _ = self._buf[0].shape
        sys_lo, sys_hi = s.sys_span
        q_lo, q_hi = s.question_span
        doc_lens = [de - ds for (ds, de) in s.doc_spans]
        docs_total_len = int(sum(doc_lens))
        n_docs = len(doc_lens)
        sys_len = int(sys_hi - sys_lo)
        seq_len = len(s.input_ids)

        # Stack to (n_layers, H, q_len, 3)
        arr = np.stack(self._buf, axis=0)
        N = n_layers * H * q_len
        layer_grid, head_grid, qi_grid = np.meshgrid(
            np.arange(n_layers, dtype=np.int16),
            np.arange(H,        dtype=np.int16),
            np.arange(q_len,    dtype=np.int16),
            indexing="ij",
        )
        layer_col = layer_grid.reshape(-1)
        head_col  = head_grid.reshape(-1)
        qi_col    = qi_grid.reshape(-1)
        qabs_col  = (qi_col.astype(np.int32) + q_lo).astype(np.int32)
        y_sink = arr[..., 0].reshape(-1)
        y_sys  = arr[..., 1].reshape(-1)
        y_docs = arr[..., 2].reshape(-1)

        df = pd.DataFrame({
            "benchmark": np.full(N, benchmark, dtype=object),
            "sample_id": np.full(N, s.sample_id, dtype=object),
            "layer":     layer_col,
            "head":      head_col,
            "qi":        qi_col,
            "qabs":      qabs_col,
            "qprev_len": qi_col.astype(np.int16),
            "sys_len":   np.int16(sys_len),
            "docs_total_len": np.int32(docs_total_len),
            "n_docs":    np.int16(n_docs),
            "mean_doc_len": np.float32(np.mean(doc_lens) if doc_lens else 0),
            "min_doc_len":  np.int16(min(doc_lens) if doc_lens else 0),
            "max_doc_len":  np.int16(max(doc_lens) if doc_lens else 0),
            "seq_len":   np.int32(seq_len),
            "q_len":     np.int16(q_len),
            "q_rel":     (qi_col.astype(np.float32) / max(q_len - 1, 1)),
            "y_sink":    y_sink.astype(np.float32),
            "y_sys":     y_sys.astype(np.float32),
            "y_docs":    y_docs.astype(np.float32),
        })
        return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--benchmark", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--out_dir", default="attn_raw_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    out_path = out_dir / f"{args.benchmark}__{slug}__baseline.parquet"
    print(f"[info] model={args.model} bench={args.benchmark} n={args.n_samples}")
    print(f"[info] out={out_path}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    print(f"[info] n_layers={model.config.num_hidden_layers}")

    samples = LOADERS[args.benchmark](args.n_samples, tok)
    lens = [len(s.input_ids) for s in samples]
    print(f"[info] seq_lens min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    collector = RawCollector()
    collector.attach(model)

    dfs = []
    try:
        with torch.no_grad():
            for si, s in enumerate(samples):
                print(f"[{si+1}/{len(samples)}] {s.sample_id} len={len(s.input_ids)}", flush=True)
                # Build baseline prefill inputs
                device = next(model.parameters()).device
                input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
                segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
                pad_mask = (segment_ids != -1).long()
                pf = build_policy_prefill(segment_ids, "baseline", dtype=model.dtype)
                collector.reset(s)
                _ = model(
                    input_ids=input_ids,
                    attention_mask=pad_mask,
                    position_ids=pf.position_ids.to(device),
                    use_cache=False,
                    output_attentions=True,
                )
                df = collector.to_rows(args.benchmark)
                if df is not None:
                    dfs.append(df)
    finally:
        collector.detach()

    big = pd.concat(dfs, ignore_index=True)
    print(f"[info] total rows: {len(big):,}")
    print(big.head().to_string())
    big.to_parquet(out_path, index=False, compression="zstd")
    print(f"[done] wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
