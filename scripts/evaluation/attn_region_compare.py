"""Region-attention comparison: baseline(oracle) vs broken-cache vs KVLink0 ft.

Motivation
----------
The ``recover_attn_score`` rescaler was trained on attention masses collected
under the **baseline** policy (intact KV cache, full cross-attention). But it is
applied at inference under ``recover_pos_enc``, where the doc band is
block-diagonal (no cross-doc / no doc->prefix attention) — so the cache the
rescaler "corrects" is already a different object than the one it learned from.

This script measures, per layer, how QUESTION rows and GENERATION (decode) rows
distribute their attention over 5 regions, for one (model, policy) config:

    region 0  sink      key 0
    region 1  sys       keys [0, sys_end)            (includes the sink)
    region 2  docs      keys in the union of doc spans
    region 3  question  keys [q_lo, q_hi)            (own causal past for q-rows)
    region 4  rest      everything else (gen-prev + self for gen rows)

Run one config per process (one model + one policy + one dataset):

    baseline   : meta-llama/Llama-3.1-8B-Instruct          @ baseline
    base_broken: meta-llama/Llama-3.1-8B-Instruct          @ recover_pos_enc
    kvlink0    : .../Llama-3.1-8B-Instruct-ft-step4000      @ recover_pos_enc

Generation rows come from each config's OWN greedy decode (the model decodes its
own answer under its own policy, exactly like eval), so the gen-row attention is
faithful to what that config actually does at inference. Both the prefill
(question-row) forward and the decode loop use ``output_attentions=True`` with a
per-layer forward hook that reduces the attention tensor to region masses on the
fly, so 8B + long context still fits on a single GPU.

Datasets: nq, hqa_gold, hqa_full, wiki, musique, block_qa.

Output: JSON at  <out_dir>/<dataset>__<config_name>.json  with per-layer mean
region masses for question rows and gen rows, ready for attn_region_plot.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import datasets as hfds
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from src.kvmod.prompt import build_segmented_inputs  # noqa: E402
# Reuse the exact eval-style sample loaders from attn_analysis.py.
from scripts.evaluation.attn_analysis import (  # noqa: E402
    Sample, LOADERS, _assemble, MEM_START, MEM_END, SPECIAL_TOKEN_START,
)

GENPREFIX_STR = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
BLOCK_QA_DEFAULT_PATH = (
    "/storage/sihun/kvcache/KVLink/dataset_cache/processed/block_qa/qa"
)

# 4 raw targets recorded per row; sys already includes the sink.
TARGET_NAMES = ["sink", "sys", "docs", "question"]


# --------------------------- block_qa loader -------------------------------- #

def load_block_qa(n: int, tokenizer, split: str = "test",
                  path: str = BLOCK_QA_DEFAULT_PATH) -> List[Sample]:
    """block_qa held-out split as eval-style Samples (no gold labels -> [])."""
    ds = hfds.load_from_disk(path)[split]
    n = min(n, len(ds))
    out: List[Sample] = []
    for k in range(n):
        ex = ds[k]
        docs = [(d["title"], d["text"]) for d in ex["documents"]]
        out.append(_assemble(tokenizer, f"bqa_{k}", ex["question"], docs, []))
    return out


ALL_LOADERS = dict(LOADERS)
ALL_LOADERS["block_qa"] = load_block_qa


# --------------------------- region collector ------------------------------- #

class RegionCollector:
    """Forward hook that reduces each layer's attn tensor to region masses.

    Two phases share one collector instance per sample:

      phase="question": slice question rows [q_lo, q_hi) out of the (T, T)
                        prefill attention; average over heads + question tokens.
      phase="gen":      every decode forward contributes its new row(s); we
                        accumulate per-layer region-mass SUMS and a count
                        (heads * rows) across all decode steps, then average.

    Region key-spans (sink/sys/docs/question) are absolute indices into the
    prefill cache [0, T_prefill); decode appends gen keys after that, which all
    fall into "rest".
    """

    def __init__(self):
        self.handles = []
        self._s: Sample | None = None
        self._phase = "idle"          # "question" | "gen" | "idle"
        self._layer_iter = 0          # which layer the next hook call is
        # question phase: one (n_targets,) mean per layer
        self._q_means: List[np.ndarray] = []
        # gen phase: running sums + counts per layer
        self._g_sum: List[np.ndarray] = []
        self._g_cnt: List[int] = []

    def attach(self, model):
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self, sample: Sample, n_layers: int):
        self._s = sample
        self._q_means = []
        self._g_sum = [np.zeros(len(TARGET_NAMES), dtype=np.float64) for _ in range(n_layers)]
        self._g_cnt = [0 for _ in range(n_layers)]

    def begin(self, phase: str):
        self._phase = phase
        self._layer_iter = 0

    def _regions(self, a: torch.Tensor) -> torch.Tensor:
        """a: (H, n_rows, T) -> (H, n_rows, 4) masses [sink, sys, docs, question]."""
        s = self._s
        q_lo, q_hi = s.question_span
        sy_lo, sy_hi = s.sys_span
        T = a.shape[-1]
        docs_mask = torch.zeros(T, dtype=torch.bool, device=a.device)
        for ds, de in s.doc_spans:
            de = min(de, T)
            if ds < T:
                docs_mask[ds:de] = True
        y_sink = a[..., 0]
        y_sys = a[..., sy_lo:sy_hi].sum(-1)
        y_docs = a[..., docs_mask].sum(-1)
        y_q = a[..., q_lo:min(q_hi, T)].sum(-1)
        return torch.stack([y_sink, y_sys, y_docs, y_q], dim=-1)

    def _hook(self, module, inputs, output):
        if self._phase == "idle":
            return
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        li = self._layer_iter
        self._layer_iter += 1

        if self._phase == "question":
            q_lo, q_hi = s.question_span
            a = attn[0, :, q_lo:q_hi, :].float()       # (H, q_len, T)
            r = self._regions(a)                       # (H, q_len, 4)
            self._q_means.append(r.mean(dim=(0, 1)).cpu().numpy())
        else:  # gen: attn rows are the freshly-forwarded tokens
            a = attn[0].float()                        # (H, n_rows, T)
            r = self._regions(a)                       # (H, n_rows, 4)
            H, n_rows, _ = r.shape
            self._g_sum[li] += r.sum(dim=(0, 1)).double().cpu().numpy()
            self._g_cnt[li] += H * n_rows

    # ---- per-sample finalizers ---- #
    def question_layers(self) -> np.ndarray | None:
        if not self._q_means:
            return None
        return np.stack(self._q_means, axis=0)         # (n_layers, 4)

    def gen_layers(self) -> np.ndarray | None:
        if not self._g_cnt or all(c == 0 for c in self._g_cnt):
            return None
        out = np.full((len(self._g_cnt), len(TARGET_NAMES)), np.nan)
        for li, (ssum, cnt) in enumerate(zip(self._g_sum, self._g_cnt)):
            if cnt > 0:
                out[li] = ssum / cnt
        return out


# --------------------------- decode driver ---------------------------------- #

@torch.no_grad()
def run_sample(model, tok, s: Sample, policy: str, collector: RegionCollector,
               genprefix_ids: List[int], stop_ids: set, max_new_tokens: int,
               n_layers: int, modular_q_pos: str = "summed_pos"):
    """Prefill (capture question rows) + greedy decode (capture gen rows) under
    ``policy``. baseline and recover_pos_enc are both single-prefill (no reloc),
    which is all this comparison needs."""
    device = next(model.parameters()).device
    input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
    segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
    pad_mask = (segment_ids != -1).long()
    pf = build_policy_prefill(segment_ids, policy, modular_q_pos=modular_q_pos,
                              dtype=model.dtype)
    if pf.relocalize:
        raise ValueError(f"policy {policy} needs two-stage prefill; not supported here")
    prefill_mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
                    if pf.attention_mask_4d is not None else pad_mask)

    collector.reset(s, n_layers)

    # --- prefill: capture question rows --- #
    collector.begin("question")
    out = model(
        input_ids=input_ids,
        attention_mask=prefill_mask,
        position_ids=pf.position_ids.to(device),
        past_key_values=DynamicCache(),
        use_cache=True,
        output_attentions=True,
    )
    cache = out.past_key_values

    # --- greedy decode: capture gen rows (genprefix + generated) --- #
    G = len(genprefix_ids)
    genprefix = torch.tensor([genprefix_ids], dtype=torch.long, device=device)
    arangeG = torch.arange(G, device=device)
    pos = pf.next_position.to(device).unsqueeze(1) + arangeG          # (1, G)
    attn_mask = torch.cat(
        [pad_mask.to(device), torch.ones((1, G), device=device, dtype=pad_mask.dtype)],
        dim=1,
    )
    collector.begin("gen")
    out = model(
        input_ids=genprefix,
        attention_mask=attn_mask,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        output_attentions=True,
    )
    cache = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1)
    cur_pos = pos[:, -1] + 1

    for _ in range(max_new_tokens):
        t = int(next_tok[0].item())
        if t in stop_ids:
            break
        attn_mask = torch.cat(
            [attn_mask, torch.ones((1, 1), device=device, dtype=attn_mask.dtype)], dim=1)
        collector.begin("gen")
        out = model(
            input_ids=next_tok.unsqueeze(1),
            attention_mask=attn_mask,
            position_ids=cur_pos.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
        cache = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        cur_pos = cur_pos + 1

    collector.begin("idle")
    return collector.question_layers(), collector.gen_layers()


# --------------------------- runner ----------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF id or local dir (base model or ft-step4000 checkpoint)")
    ap.add_argument("--policy", required=True,
                    choices=["baseline", "recover_pos_enc"])
    ap.add_argument("--config_name", required=True,
                    help="tag for output filename, e.g. baseline / base_broken / kvlink0_step4000")
    ap.add_argument("--dataset", required=True, choices=list(ALL_LOADERS.keys()))
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--modular_q_pos", default="summed_pos",
                    choices=["summed_pos", "min_pos"])
    ap.add_argument("--out_dir", default="result/attn_region_compare")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}__{args.config_name}.json"
    print(f"[info] model={args.model} policy={args.policy} config={args.config_name}")
    print(f"[info] dataset={args.dataset} n={args.n_samples} max_new={args.max_new_tokens}")
    print(f"[info] out={out_path}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[info] n_layers={n_layers}")

    genprefix_ids = tok(GENPREFIX_STR, add_special_tokens=False)["input_ids"]
    stop_ids = {tok.convert_tokens_to_ids("<|eot_id|>"),
                tok.convert_tokens_to_ids("<|end_of_text|>")}

    samples = ALL_LOADERS[args.dataset](args.n_samples, tok)
    lens = [len(s.input_ids) for s in samples]
    print(f"[info] {len(samples)} samples; seq_len min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)/len(lens):.0f}")

    collector = RegionCollector()
    collector.attach(model)

    q_acc: List[np.ndarray] = []
    g_acc: List[np.ndarray] = []
    n_docs_list, gold_idx_list = [], []
    try:
        with torch.no_grad():
            for si, s in enumerate(samples):
                if si % 10 == 0 or si == len(samples) - 1:
                    print(f"[{si+1}/{len(samples)}] {s.sample_id} len={len(s.input_ids)} "
                          f"docs={len(s.doc_spans)}", flush=True)
                ql, gl = run_sample(
                    model, tok, s, args.policy, collector, genprefix_ids,
                    stop_ids, args.max_new_tokens, n_layers, args.modular_q_pos)
                if ql is not None:
                    q_acc.append(ql)
                if gl is not None:
                    g_acc.append(gl)
                n_docs_list.append(len(s.doc_spans))
                gold_idx_list.append(s.gold_doc_indices)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.detach()

    # Stack -> (n_samples, n_layers, 4); nanmean over samples for gen.
    q_arr = np.stack(q_acc, axis=0) if q_acc else None
    g_arr = np.stack(g_acc, axis=0) if g_acc else None

    def summarize(arr):
        if arr is None:
            return None
        return {
            "mean": np.nanmean(arr, axis=0).tolist(),   # (n_layers, 4)
            "std": np.nanstd(arr, axis=0).tolist(),
            "n": int(arr.shape[0]),
        }

    payload = {
        "model": args.model,
        "policy": args.policy,
        "config_name": args.config_name,
        "dataset": args.dataset,
        "target_names": TARGET_NAMES,
        "n_layers": n_layers,
        "n_samples": len(samples),
        "seq_lens": lens,
        "n_docs": n_docs_list,
        "gold_idx_list": gold_idx_list,
        "question_rows": summarize(q_arr),
        "gen_rows": summarize(g_arr),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[done] {out_path}  q_n={payload['question_rows']['n'] if q_arr is not None else 0} "
          f"g_n={payload['gen_rows']['n'] if g_arr is not None else 0}")


if __name__ == "__main__":
    main()
