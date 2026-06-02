"""Attention-distribution analysis across KV-cache policies (GPU, hook-based).

For each (dataset, sample, policy):
  1. Assemble (input_ids, segment_ids) like the eval scripts (reencode_num=0).
  2. Build policy attention_mask_4d + position_ids via build_policy_prefill.
  3. Forward with eager attention. We attach a forward hook to each layer's
     self_attn that consumes attn_weights and stores ONLY per-layer aggregates
     (region masses + entropies), discarding the full attention tensor — so
     8B + long context still fits on a single GPU.
  4. For recover_cross_attn_oracle_pos we run a two-stage prefill (stage 1:
     prefix+docs full-causal; stage 2: question against the relocalized cache)
     and aggregate from stage 2 only.

Per-layer aggregates (one row per question token, averaged over heads & tokens):
  mass_sys, mass_first, mass_qprev, mass_docs_total, mass_per_doc (list),
  gold_mask (list[bool]), entropy, entropy_excl_first.

JSON layout:
  {samples, seq_lens, n_docs, gold_idx_list,
   results: {policy: [[per-layer dict, ...] per sample]}}

Datasets supported: nq, hqa_gold, hqa_full, wiki, musique.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import datasets as hfds
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from src.kvmod.prompt import build_segmented_inputs  # noqa: E402
from src.kvmod.inference import _relocalize_doc_keys  # noqa: E402

POLICIES_DEFAULT = ("baseline", "recover_pos_enc", "modular", "recover_cross_attn_oracle_pos")
MEM_START = 128254
MEM_END = 128255
SPECIAL_TOKEN_START = 128011


# -------------------------- sample assembly -------------------------------- #

@dataclass
class Sample:
    sample_id: str
    input_ids: List[int]
    segment_ids: List[int]
    doc_spans: List[Tuple[int, int]]
    gold_doc_indices: List[int]      # indices into doc_spans that are gold
    sys_span: Tuple[int, int]
    question_span: Tuple[int, int]


def _spans(seg: List[int]) -> Dict:
    n = len(seg)
    i = 0
    while i < n and seg[i] == 0:
        i += 1
    sys_end = i
    docs: List[Tuple[int, int]] = []
    while i < n:
        s = seg[i]
        if s >= 1:
            j = i
            while j < n and seg[j] == s:
                j += 1
            docs.append((i, j))
            i = j
        else:
            i += 1
    q_start = docs[-1][1] if docs else sys_end
    return {"sys": (0, sys_end), "docs": docs, "question": (q_start, n)}


def _assemble(tokenizer, sample_id, question, docs, gold_indices) -> Sample:
    seg = build_segmented_inputs(
        tokenizer, question, docs,
        prompt_preset="kvlink", reencode_num=0,
        mem_start=MEM_START, mem_end=MEM_END,
        special_token_start=SPECIAL_TOKEN_START,
    )
    sp = _spans(seg["segment_ids"])
    return Sample(sample_id, seg["input_ids"], seg["segment_ids"],
                  sp["docs"], gold_indices, sp["sys"], sp["question"])


def load_nq(n: int, tokenizer) -> List[Sample]:
    path = ROOT / "data/raw/nq/nq-open-10_0.jsonl"
    ds = hfds.load_dataset("json", data_files=str(path), split="train").select(range(n))
    out = []
    for k, ex in enumerate(ds):
        docs = [(c["title"], c["text"]) for c in ex["ctxs"][:10]]
        gold = [0]  # data file lays out ground-truth at index 0
        out.append(_assemble(tokenizer, f"nq_{k}", ex["question"], docs, gold))
    return out


def load_hotpotqa(n: int, tokenizer, gold_only: bool) -> List[Sample]:
    ds = hfds.load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").select(range(n))
    out = []
    for k, ex in enumerate(ds):
        gold_titles = set(ex["supporting_facts"]["title"])
        docs, gold = [], []
        for j, title in enumerate(ex["context"]["title"]):
            is_gold = title in gold_titles
            if gold_only and not is_gold:
                continue
            if is_gold:
                gold.append(len(docs))
            docs.append((title, "".join(ex["context"]["sentences"][j])))
        sid = f"hqa{'g' if gold_only else 'f'}_{k}"
        out.append(_assemble(tokenizer, sid, ex["question"], docs, gold))
    return out


def load_wiki(n: int, tokenizer) -> List[Sample]:
    path = ROOT / "data/raw/2wikimultihop/dev.json"
    ds = hfds.load_dataset("json", data_files=str(path), split="train").select(range(n))
    out = []
    for k, ex in enumerate(ds):
        gold_titles = {sf["title"] for sf in ex["supporting_facts"]}
        docs, gold = [], []
        for j in range(len(ex["context"])):
            title = ex["context"][j]["title"]
            text = " ".join(ex["context"][j]["sentences"])
            if title in gold_titles:
                gold.append(len(docs))
            docs.append((title, text))
        out.append(_assemble(tokenizer, f"wiki_{k}", ex["question"], docs, gold))
    return out


def load_musique(n: int, tokenizer) -> List[Sample]:
    ds = hfds.load_dataset("dgslibisey/MuSiQue", split="validation").select(range(n))
    out = []
    for k, ex in enumerate(ds):
        docs, gold = [], []
        for j, para in enumerate(ex["paragraphs"]):
            if para.get("is_supporting"):
                gold.append(len(docs))
            docs.append((para["title"], para["paragraph_text"]))
        out.append(_assemble(tokenizer, f"mus_{k}", ex["question"], docs, gold))
    return out


LOADERS = {
    "nq":       load_nq,
    "hqa_gold": lambda n, tok: load_hotpotqa(n, tok, gold_only=True),
    "hqa_full": lambda n, tok: load_hotpotqa(n, tok, gold_only=False),
    "wiki":     load_wiki,
    "musique":  load_musique,
}


# -------------------------- hook-based collector --------------------------- #

@dataclass
class PerLayerAgg:
    mass_sys: float
    mass_qprev: float
    mass_first: float
    mass_docs_total: float
    mass_per_doc: List[float]
    entropy: float
    entropy_excl_first: float


class HookCollector:
    """Forward-hook on each layer's self_attn that aggregates attention from
    question rows to predefined regions, without retaining the full attn tensor.
    """

    def __init__(self):
        self.handles = []
        self.layer_aggs: List[PerLayerAgg] = []
        self._s: Sample | None = None
        # Stage-2 (oracle): question rows are at q-axis index [0, q_len),
        # key axis covers the WHOLE concatenated cache => key indices 0..T-1
        # already align with sample's segment layout. For single-stage prefill
        # we slice question rows out of the (T, T) matrix.
        self._mode = "single"   # "single" | "stage2"

    def attach(self, model):
        # Llama: model.model.layers[i].self_attn
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self, sample: Sample, mode: str):
        self.layer_aggs = []
        self._s = sample
        self._mode = mode

    def _hook(self, module, inputs, output):
        # Llama eager: output = (attn_output, attn_weights, past_kv) OR
        # (attn_output, attn_weights). attn_weights: (B, H, q, k) or None.
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        q_lo, q_hi = s.question_span
        sy_lo, sy_hi = s.sys_span
        q_len = q_hi - q_lo

        if self._mode == "single":
            aq = attn[0, :, q_lo:q_hi, :]   # (H, q_len, T)
        else:
            # stage2: attn is (B, H, q_len, T) where columns 0..T-1 align with
            # the full sample (cache columns + new query columns).
            aq = attn[0]                    # (H, q_len, T)

        aq = aq.float()
        am = aq.mean(dim=(0, 1))            # (T,) head-mean & q-token-mean
        eps = 1e-12
        # Entropy: row-wise then averaged
        ent = (-(aq.clamp_min(eps) * aq.clamp_min(eps).log()).sum(-1)).mean().item()
        # Entropy excluding the first key: zero out col 0, renormalize per row
        aq_e = aq.clone()
        aq_e[..., 0] = 0.0
        renorm = aq_e.sum(-1, keepdim=True).clamp_min(eps)
        aq_e = aq_e / renorm
        ent_excl = (-(aq_e.clamp_min(eps) * aq_e.clamp_min(eps).log()).sum(-1)).mean().item()

        mass_sys = am[sy_lo:sy_hi].sum().item()
        mass_first = am[0].item()

        mq = 0.0
        for qi in range(q_len):
            hi = q_lo + qi
            if hi > q_lo:
                mq += aq[:, qi, q_lo:hi].sum(-1).mean().item()
        mq /= max(q_len, 1)

        per_doc = [am[ds:de].sum().item() for (ds, de) in s.doc_spans]
        m_total = float(sum(per_doc))

        self.layer_aggs.append(PerLayerAgg(
            mass_sys=mass_sys, mass_qprev=mq, mass_first=mass_first,
            mass_docs_total=m_total, mass_per_doc=per_doc,
            entropy=ent, entropy_excl_first=ent_excl,
        ))


# -------------------------- runner ----------------------------------------- #

def run_single_stage(model, s: Sample, policy: str, collector: HookCollector):
    device = next(model.parameters()).device
    input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
    pad_mask = (segment_ids != -1).long()
    pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
    mask = pf.attention_mask_4d.to(device=device, dtype=model.dtype) if pf.attention_mask_4d is not None else pad_mask

    collector.reset(s, mode="single")
    _ = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=pf.position_ids.to(device),
        use_cache=False,
        output_attentions=True,
    )


def run_oracle(model, s: Sample, collector: HookCollector):
    """Two-stage prefill for recover_cross_attn_oracle_pos. Aggregates from stage 2."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
    pad_mask = (segment_ids != -1).to(device)
    pf = build_policy_prefill(segment_ids, "recover_cross_attn_oracle_pos", dtype=model.dtype)
    q_start = pf.question_start[0]
    pos_g = pf.position_ids.to(device)

    # Stage 1: prefix + docs with full causal — DO NOT aggregate
    collector.reset(s, mode="ignore")
    stage1 = model(
        input_ids=input_ids[:, :q_start],
        attention_mask=pad_mask[:, :q_start],
        position_ids=pos_g[:, :q_start],
        past_key_values=DynamicCache(),
        use_cache=True,
        output_attentions=False,   # save mem
    )
    cache = stage1.past_key_values
    _relocalize_doc_keys(cache, pf.doc_spans[0], pos_g[:, :q_start], model)

    # Stage 2: question against relocalized cache — aggregate.
    q_end = input_ids.shape[1]
    q_len = q_end - q_start
    q_mask = torch.ones((1, q_end), dtype=pad_mask.dtype, device=device)
    collector.reset(s, mode="stage2")
    _ = model(
        input_ids=input_ids[:, q_start:q_end],
        attention_mask=q_mask,                       # (1, q_end) – HF builds proper 2D causal
        position_ids=pos_g[:, q_start:q_end],
        past_key_values=cache,
        use_cache=True,
        output_attentions=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES_DEFAULT))
    ap.add_argument("--out_dir", default="attn_analysis_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.split("/")[-1]
    out_path = out_dir / f"{args.dataset}__{model_slug}.json"

    print(f"[info] model={args.model} device={args.device} dtype={args.dtype} dataset={args.dataset} n={args.n_samples}")
    print(f"[info] out={out_path}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[info] n_layers={n_layers}")

    samples = LOADERS[args.dataset](args.n_samples, tok)
    lens = [len(s.input_ids) for s in samples]
    print(f"[info] seq_lens min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")

    collector = HookCollector()
    collector.attach(model)

    results: Dict[str, List[List[dict]]] = {p: [] for p in args.policies}
    try:
        with torch.no_grad():
            for si, s in enumerate(samples):
                print(f"[{si+1}/{len(samples)}] {s.sample_id} len={len(s.input_ids)} docs={len(s.doc_spans)} gold={s.gold_doc_indices}", flush=True)
                for p in args.policies:
                    if p == "recover_cross_attn_oracle_pos":
                        run_oracle(model, s, collector)
                    else:
                        run_single_stage(model, s, p, collector)
                    if len(collector.layer_aggs) != n_layers:
                        print(f"  [warn] policy={p} got {len(collector.layer_aggs)} layer aggs, expected {n_layers}")
                    results[p].append([asdict(a) for a in collector.layer_aggs])
    finally:
        collector.detach()

    payload = {
        "model": args.model,
        "dataset": args.dataset,
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
