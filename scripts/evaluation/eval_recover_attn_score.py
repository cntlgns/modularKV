"""Evaluate recover_attn_score against baseline + recover_pos_enc.

For each (sample, policy):
  1. Build (input_ids, segment_ids) like attn_analysis.py.
  2. If policy == recover_attn_score:
       - Compute per-(layer, head, qi) features
       - Predict (sink, sys, docs) targets with the saved GBM models
       - Build RescaleContext and set it; run forward with the patched attn.
     Else:
       - Clear RescaleContext; run forward with the patched attn (no-op).
  3. Collect per-layer aggregates via hook (head-mean + q-token-mean):
     mass_sys, mass_qprev, mass_first, mass_docs_total, mass_per_doc,
     entropy, entropy_excl_first.

Saves JSON in the same schema as attn_analysis.py so existing plotter works.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from src.kvmod.attn_rescale import (
    RescaleContext, install_rescale_patch, set_rescale_context, build_region_idx,
)  # noqa: E402

from scripts.evaluation.attn_analysis import (
    LOADERS, HookCollector, MEM_START, MEM_END, SPECIAL_TOKEN_START,
)  # noqa: E402

CONT_FEATS = [
    "qprev_len", "docs_total_len", "n_docs", "mean_doc_len",
    "min_doc_len", "max_doc_len", "q_len", "seq_len", "q_rel", "qabs",
    "log_seq_len", "log1p_docs_total_len", "log1p_qabs", "docs_to_seq_ratio",
]


def build_features_for_sample(s, n_layers: int, n_heads: int) -> np.ndarray:
    """Return (n_layers * n_heads * q_len, n_cont+2) feature matrix in the
    order expected by the saved GBM models: cont feats then [layer, head].
    """
    q_lo, q_hi = s.question_span
    q_len = q_hi - q_lo
    if q_len == 0:
        return np.zeros((0, len(CONT_FEATS) + 2))
    doc_lens = [de - ds for (ds, de) in s.doc_spans]
    docs_total_len = float(sum(doc_lens))
    n_docs = float(len(doc_lens))
    mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0
    min_doc  = float(min(doc_lens)) if doc_lens else 0.0
    max_doc  = float(max(doc_lens)) if doc_lens else 0.0
    sys_len = float(s.sys_span[1] - s.sys_span[0])
    seq_len = float(len(s.input_ids))
    log_seq_len = float(np.log(seq_len))
    log1p_docs = float(np.log1p(docs_total_len))

    qi_arr   = np.arange(q_len, dtype=np.float32)
    qabs_arr = qi_arr + float(q_lo)
    q_rel_arr = qi_arr / max(q_len - 1, 1)
    log1p_qabs = np.log1p(qabs_arr)

    # per-qi block (cont feats only) — same for every (layer, head)
    qi_block = np.stack([
        qi_arr,                                          # qprev_len
        np.full(q_len, docs_total_len, dtype=np.float32),
        np.full(q_len, n_docs, dtype=np.float32),
        np.full(q_len, mean_doc, dtype=np.float32),
        np.full(q_len, min_doc, dtype=np.float32),
        np.full(q_len, max_doc, dtype=np.float32),
        np.full(q_len, q_len, dtype=np.float32),
        np.full(q_len, seq_len, dtype=np.float32),
        q_rel_arr,
        qabs_arr,
        np.full(q_len, log_seq_len, dtype=np.float32),
        np.full(q_len, log1p_docs, dtype=np.float32),
        log1p_qabs,
        np.full(q_len, docs_total_len / seq_len, dtype=np.float32),
    ], axis=1)  # (q_len, n_cont)

    # broadcast to (n_layers, n_heads, q_len, n_cont+2)
    out = np.empty((n_layers, n_heads, q_len, len(CONT_FEATS) + 2), dtype=np.float32)
    out[..., :len(CONT_FEATS)] = qi_block[None, None, :, :]
    layer_grid = np.arange(n_layers, dtype=np.float32)[:, None, None]
    head_grid  = np.arange(n_heads,  dtype=np.float32)[None, :, None]
    out[..., len(CONT_FEATS)]     = np.broadcast_to(layer_grid, (n_layers, n_heads, q_len))
    out[..., len(CONT_FEATS) + 1] = np.broadcast_to(head_grid,  (n_layers, n_heads, q_len))
    return out.reshape(-1, len(CONT_FEATS) + 2)


def predict_targets(gbms, X) -> np.ndarray:
    """Predict (sink, sys, docs) targets. Returns (N, 3) clipped to [0, 1]."""
    out = np.stack([gbms["sink"].predict(X), gbms["sys"].predict(X), gbms["docs"].predict(X)], axis=-1)
    return np.clip(out, 0.0, 1.0)


def run_policy(model, s, policy: str, collector: HookCollector,
               gbms=None, n_layers=None, n_heads=None):
    device = next(model.parameters()).device
    input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
    pad_mask = (segment_ids != -1).long()
    # use recover_pos_enc's policy prefill for recover_attn_score (same mask/pos)
    pp_policy = "recover_pos_enc" if policy == "recover_attn_score" else policy
    pf = build_policy_prefill(segment_ids, pp_policy, dtype=model.dtype)
    mask = pf.attention_mask_4d.to(device=device, dtype=model.dtype) if pf.attention_mask_4d is not None else pad_mask

    if policy == "recover_attn_score":
        X = build_features_for_sample(s, n_layers, n_heads)
        preds = predict_targets(gbms, X).reshape(n_layers, n_heads, -1, 3)
        targets = torch.from_numpy(preds).to(device=device, dtype=torch.float32)
        region_idx = build_region_idx(
            T=len(s.input_ids), sys_end=s.sys_span[1],
            doc_spans=s.doc_spans, device=device,
        )
        ctx = RescaleContext(
            targets=targets, q_lo=s.question_span[0], q_hi=s.question_span[1],
            sys_end=s.sys_span[1], region_idx=region_idx,
        )
        set_rescale_context(ctx)
    else:
        set_rescale_context(None)

    collector.reset(s, mode="single")
    try:
        # output_attentions=False prevents HF from accumulating all per-layer
        # attn_weights in the model output (which would OOM for 8B+long ctx).
        # The hook on self_attn still receives attn_weights via the patched
        # forward's return tuple (which never None-outs them).
        _ = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=pf.position_ids.to(device),
            use_cache=False,
            output_attentions=False,
        )
    finally:
        set_rescale_context(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--policies", nargs="+",
                    default=["baseline", "recover_pos_enc", "recover_attn_score"])
    ap.add_argument("--gbm_dir", default="attn_score_models")
    ap.add_argument("--out_dir", default="recover_attn_score_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    out_path = out_dir / f"{args.dataset}__{slug}.json"

    print(f"[info] model={args.model} dataset={args.dataset} n={args.n_samples}")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"[info] n_layers={n_layers} n_heads={n_heads}")

    n_patched = install_rescale_patch(model)
    print(f"[info] installed rescale patch on {n_patched} attention modules")

    gbms = {
        "sink": joblib.load(Path(args.gbm_dir) / "sink.joblib")["model"],
        "sys":  joblib.load(Path(args.gbm_dir) / "sys.joblib")["model"],
        "docs": joblib.load(Path(args.gbm_dir) / "docs.joblib")["model"],
    }
    print(f"[info] loaded GBM models from {args.gbm_dir}")

    samples = LOADERS[args.dataset](args.n_samples, tok)
    lens = [len(s.input_ids) for s in samples]
    print(f"[info] seq_lens min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    collector = HookCollector(); collector.attach(model)
    results: Dict[str, List[List[dict]]] = {p: [] for p in args.policies}
    try:
        with torch.no_grad():
            for si, s in enumerate(samples):
                print(f"[{si+1}/{len(samples)}] {s.sample_id} len={len(s.input_ids)}", flush=True)
                for p in args.policies:
                    run_policy(model, s, p, collector,
                               gbms=gbms, n_layers=n_layers, n_heads=n_heads)
                    if len(collector.layer_aggs) != n_layers:
                        print(f"  [warn] {p}: got {len(collector.layer_aggs)} layer aggs")
                    results[p].append([asdict(a) for a in collector.layer_aggs])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.detach()

    payload = {
        "model": args.model, "dataset": args.dataset,
        "samples": [s.sample_id for s in samples],
        "seq_lens": lens,
        "n_docs": [len(s.doc_spans) for s in samples],
        "gold_idx_list": [s.gold_doc_indices for s in samples],
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
