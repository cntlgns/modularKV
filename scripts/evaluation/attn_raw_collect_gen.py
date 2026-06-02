"""Collect raw (x, y) data points for regression on baseline-policy attention,
GENERATION-token rows (teacher-forced).

For each (sample, layer, head, gi) tuple, record:
  features (x):
    sample_id (str), layer (int), head (int), gi (int) = generation step index
    q_len, sys_len, n_docs, docs_total_len, mean_doc_len, seq_len   (sample-level)
  targets (y, all attention masses from gen-row gi over the WHOLE causal context):
    y_sink     = α[head, gi, 0]                            (sink token)
    y_sys      = Σ_{k ∈ [0, sys_end)} α[head, gi, k]       (sys prefix incl sink)
    y_docs     = Σ_{k ∈ ∪doc_spans} α[head, gi, k]         (all docs)
    y_question = Σ_{k ∈ [q_lo, q_hi)} α[head, gi, k]       (question span)
  → rest = 1 - (y_sys + y_docs + y_question - y_sink)      [reconstructed]

Data source: block_qa training split (FiD-retrieved Wikipedia QA, GPT-4o-mini
generated answers in the `generated` field). NOT the eval benchmarks
(nq/hotpotqa/2wiki/musique) to avoid fitting the rescaler on the test
distribution.

Sequence layout for each sample:
  [system + [mem_start] + docs + [mem_end] + question]   ← prompt (build_segmented_inputs)
  [<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n]  ← genprefix
  [tokenize(generated)]                                  ← teacher-forced answer

gi = 0 indexes the first row AFTER question (= first genprefix token); rows
[g_lo, g_hi) are the gen rows. Forward runs ONCE under baseline policy
(full causal, contiguous positions); attentions extracted via per-layer hook.

Output: Parquet per shard at
  analysis/attention_score_analysis/attn_raw_gen_results/block_qa[_test]__<model>__shard{r}of{w}.parquet

Multi-GPU: shard the dataset with --shard_rank / --shard_world_size; launch
N independent jobs.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from src.kvmod.prompt import build_segmented_inputs  # noqa: E402

# Same special tokens as scripts/evaluation/attn_analysis.py.
MEM_START = 128254
MEM_END = 128255
SPECIAL_TOKEN_START = 128011

# Llama-3 chat genprefix appended after the question and before the answer.
GENPREFIX_STR = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

BLOCK_QA_DEFAULT_PATH = "/storage/sihun/kvcache/KVLink/dataset_cache/processed/block_qa/qa"


# --------------------------- sample assembly -------------------------------- #

@dataclass
class GenSample:
    sample_id: str
    input_ids: List[int]           # [prompt + genprefix + generated]
    segment_ids: List[int]
    doc_spans: List[Tuple[int, int]]
    sys_span: Tuple[int, int]
    question_span: Tuple[int, int]  # [q_lo, q_hi)
    gen_span: Tuple[int, int]       # [g_lo=q_hi, g_hi)


def _spans(seg: List[int]):
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


def _assemble(tokenizer, sample_id: str, question: str,
              docs: List[Tuple[str, str]], generated: str,
              genprefix_ids: List[int], include_gen: bool = True) -> GenSample:
    seg = build_segmented_inputs(
        tokenizer, question, docs,
        prompt_preset="kvlink", reencode_num=0,
        mem_start=MEM_START, mem_end=MEM_END,
        special_token_start=SPECIAL_TOKEN_START,
    )
    prompt_ids = seg["input_ids"]
    prompt_segs = seg["segment_ids"]
    sp = _spans(prompt_segs)

    if include_gen:
        # Teacher-force the genprefix + generated answer onto the prompt.
        gen_text_ids = tokenizer(generated, add_special_tokens=False)["input_ids"]
        g_lo = len(prompt_ids)
        full_ids = prompt_ids + genprefix_ids + gen_text_ids
        g_hi = len(full_ids)
        # Gen rows get segment_id 0 (so baseline = full causal applies trivially).
        full_segs = prompt_segs + [0] * (g_hi - g_lo)
    else:
        # Question-row mode: prompt-only forward. Question rows are causal and
        # never see gen tokens, so appending the answer would not change their
        # attention — we skip it for speed and to avoid needing `generated`.
        full_ids = prompt_ids
        full_segs = prompt_segs
        g_lo = g_hi = len(prompt_ids)  # empty gen span

    return GenSample(
        sample_id=sample_id,
        input_ids=full_ids,
        segment_ids=full_segs,
        doc_spans=sp["docs"],
        sys_span=sp["sys"],
        question_span=sp["question"],
        gen_span=(g_lo, g_hi),
    )


def load_block_qa(path: str, split: str, shard_rank: int, shard_world: int,
                  max_samples: int | None) -> List[dict]:
    ds = hfds.load_from_disk(path)[split]
    n = len(ds)
    if max_samples is not None:
        n = min(n, max_samples)
    # Stride sharding so each shard sees a roughly uniform slice of the dataset.
    idxs = list(range(shard_rank, n, shard_world))
    return [ds[i] for i in idxs]


# --------------------------- hook-based collector --------------------------- #

class GenRawCollector:
    """Hook on each layer's self_attn that records region attention masses for
    a row slice of attn_weights, aggregated per head and per row index.

    rows_mode="gen":      slice [g_lo, g_hi); 4 targets (sink, sys, docs, question).
    rows_mode="question": slice [q_lo, q_hi); 3 targets (sink, sys, docs). The
                          question span is the row's own causal past here, so it
                          is NOT a separate region — it falls into "rest" exactly
                          like the original question-token regression."""

    def __init__(self, rows_mode: str = "gen"):
        self.rows_mode = rows_mode
        self.handles = []
        self._buf: List[np.ndarray] = []  # one (H, n_rows, n_targets) per layer
        self._s: GenSample | None = None

    def attach(self, model):
        for layer in model.model.layers:
            self.handles.append(layer.self_attn.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self, sample: GenSample):
        self._buf = []
        self._s = sample

    def _hook(self, module, inputs, output):
        attn = output[1] if isinstance(output, tuple) and len(output) >= 2 else None
        if attn is None:
            return
        s = self._s
        q_lo, q_hi = s.question_span
        sy_lo, sy_hi = s.sys_span

        if self.rows_mode == "question":
            lo, hi = q_lo, q_hi
        else:
            lo, hi = s.gen_span
        a = attn[0, :, lo:hi, :].float()  # (H, n_rows, T)
        H, n_rows, T = a.shape

        docs_mask = torch.zeros(T, dtype=torch.bool, device=a.device)
        for ds, de in s.doc_spans:
            docs_mask[ds:de] = True

        y_sink = a[..., 0]                  # (H, n_rows)
        y_sys  = a[..., :sy_hi].sum(-1)      # includes sink
        y_docs = a[..., docs_mask].sum(-1)
        if self.rows_mode == "question":
            stacked = torch.stack([y_sink, y_sys, y_docs], dim=-1)
        else:
            y_question = a[..., q_lo:q_hi].sum(-1)
            stacked = torch.stack([y_sink, y_sys, y_docs, y_question], dim=-1)
        self._buf.append(stacked.detach().to("cpu").numpy())

    def to_rows(self) -> pd.DataFrame | None:
        s = self._s
        n_layers = len(self._buf)
        if n_layers == 0:
            return None
        H, n_rows, _ = self._buf[0].shape
        if n_rows <= 0:
            return None
        sys_len = int(s.sys_span[1] - s.sys_span[0])
        q_len = int(s.question_span[1] - s.question_span[0])
        doc_lens = [de - ds for (ds, de) in s.doc_spans]
        docs_total_len = int(sum(doc_lens))
        n_docs = len(doc_lens)
        mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0
        seq_len = len(s.input_ids)

        arr = np.stack(self._buf, axis=0)  # (n_layers, H, n_rows, n_targets)
        N = n_layers * H * n_rows
        layer_grid, head_grid, pos_grid = np.meshgrid(
            np.arange(n_layers, dtype=np.int16),
            np.arange(H,        dtype=np.int16),
            np.arange(n_rows,   dtype=np.int16),
            indexing="ij",
        )
        # Position column name differs by mode: gen rows -> gi, question rows -> qi.
        pos_col = "qi" if self.rows_mode == "question" else "gi"
        cols = {
            "sample_id":      np.full(N, s.sample_id, dtype=object),
            "layer":          layer_grid.reshape(-1),
            "head":           head_grid.reshape(-1),
            pos_col:          pos_grid.reshape(-1).astype(np.int16),
            "q_len":          np.int16(q_len),
            "sys_len":        np.int16(sys_len),
            "n_docs":         np.int16(n_docs),
            "docs_total_len": np.int32(docs_total_len),
            "mean_doc_len":   np.float32(mean_doc),
            "seq_len":        np.int32(seq_len),
            "y_sink":         arr[..., 0].reshape(-1).astype(np.float32),
            "y_sys":          arr[..., 1].reshape(-1).astype(np.float32),
            "y_docs":         arr[..., 2].reshape(-1).astype(np.float32),
        }
        if self.rows_mode != "question":
            cols["y_question"] = arr[..., 3].reshape(-1).astype(np.float32)
        return pd.DataFrame(cols)


# --------------------------- runner ----------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--block_qa_path", default=BLOCK_QA_DEFAULT_PATH)
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="Which block_qa split to collect from. 'test' is the 2,000-sample "
                         "held-out split, useful as an independent R^2 check on the trained "
                         "regressor before running downstream QA eval.")
    ap.add_argument("--rows", default="gen", choices=["gen", "question"],
                    help="Which rows to collect attention for. 'gen' = generated/decode "
                         "rows [g_lo,g_hi) with 4 targets (sink/sys/docs/question), needs "
                         "the teacher-forced answer. 'question' = prompt question rows "
                         "[q_lo,q_hi) with 3 targets (sink/sys/docs), prompt-only forward "
                         "(no answer needed) — clean block_qa replacement for the original "
                         "eval-benchmark-trained question regressor.")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Cap on the chosen-split size; None = use all (train 17,998 / test 2,000).")
    ap.add_argument("--shard_rank", type=int, default=0)
    ap.add_argument("--shard_world_size", type=int, default=1)
    ap.add_argument("--out_dir",
                    default="analysis/attention_score_analysis/attn_raw_gen_results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--flush_every", type=int, default=500,
                    help="Flush accumulated dataframes to a temp parquet every N samples.")
    ap.add_argument("--run_tag", default="",
                    help="Optional label appended to the output filename, so multiple "
                         "runs (e.g. additional samples, different cutoffs) accumulate "
                         "side-by-side instead of overwriting. Example: --run_tag v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    tag_suffix = f"__{args.run_tag}" if args.run_tag else ""
    # Default split=train keeps the legacy filename (block_qa__<model>__shardRofW.parquet)
    # so already-collected train shards aren't renamed. Test split gets a 'test' infix.
    split_infix = "" if args.split == "train" else f"_{args.split}"
    out_path = out_dir / (
        f"block_qa{split_infix}__{slug}__shard{args.shard_rank}of{args.shard_world_size}"
        f"{tag_suffix}.parquet"
    )
    print(f"[info] model={args.model}  device={args.device}  dtype={args.dtype}  rows={args.rows}")
    print(f"[info] split={args.split}  shard={args.shard_rank}/{args.shard_world_size}  out={out_path}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager",
    ).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    print(f"[info] n_layers={n_layers}  n_heads={n_heads}")

    genprefix_ids = tok(GENPREFIX_STR, add_special_tokens=False)["input_ids"]
    print(f"[info] genprefix tok len={len(genprefix_ids)}")

    raw = load_block_qa(args.block_qa_path, args.split, args.shard_rank,
                        args.shard_world_size, args.max_samples)
    print(f"[info] shard sample count = {len(raw)}  (split={args.split})")
    if not raw:
        print("[empty] no samples in this shard; exiting.")
        return

    # Build all GenSamples first (cheap, CPU-only).
    include_gen = (args.rows == "gen")
    samples: List[GenSample] = []
    for k, ex in enumerate(raw):
        # Gen rows need the teacher-forced answer; question rows don't.
        if include_gen and not ex.get("generated"):
            continue
        docs = [(d["title"], d["text"]) for d in ex["documents"]]
        s = _assemble(
            tok, f"bqa_{args.split}_{args.shard_rank}_{k}", ex["question"], docs,
            ex.get("generated", ""), genprefix_ids, include_gen=include_gen,
        )
        if include_gen and s.gen_span[1] <= s.gen_span[0]:
            continue
        if s.question_span[1] <= s.question_span[0]:
            continue
        samples.append(s)
    if not samples:
        print("[empty] no usable samples.")
        return

    lens = [len(s.input_ids) for s in samples]
    print(f"[info] seq_lens  min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")
    if include_gen:
        g_lens = [s.gen_span[1] - s.gen_span[0] for s in samples]
        print(f"[info] gen lens  min={min(g_lens)} max={max(g_lens)} mean={sum(g_lens)/len(g_lens):.1f}")
    else:
        q_lens = [s.question_span[1] - s.question_span[0] for s in samples]
        print(f"[info] q lens    min={min(q_lens)} max={max(q_lens)} mean={sum(q_lens)/len(q_lens):.1f}")

    collector = GenRawCollector(rows_mode=args.rows)
    collector.attach(model)

    dfs: List[pd.DataFrame] = []
    flushed_parts: List[Path] = []

    def flush(part_idx: int):
        if not dfs:
            return
        big = pd.concat(dfs, ignore_index=True)
        part_path = out_dir / f"{out_path.stem}.part{part_idx:03d}.parquet"
        big.to_parquet(part_path, index=False, compression="zstd")
        flushed_parts.append(part_path)
        print(f"  [flush] {part_path.name}  rows={len(big):,}  "
              f"size={os.path.getsize(part_path)/1e6:.1f} MB", flush=True)
        dfs.clear()

    part_idx = 0
    try:
        with torch.no_grad():
            for si, s in enumerate(samples):
                if si % 25 == 0 or si == len(samples) - 1:
                    span = (s.gen_span if include_gen else s.question_span)
                    print(f"[{si+1}/{len(samples)}] {s.sample_id} "
                          f"seq={len(s.input_ids)} rows={span[1]-span[0]}",
                          flush=True)
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
                df = collector.to_rows()
                if df is not None:
                    dfs.append(df)

                if (si + 1) % args.flush_every == 0:
                    flush(part_idx)
                    part_idx += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        collector.detach()
        flush(part_idx)

    # Merge all parts into one final parquet.
    if flushed_parts:
        merged = pd.concat([pd.read_parquet(p) for p in flushed_parts], ignore_index=True)
        merged.to_parquet(out_path, index=False, compression="zstd")
        for p in flushed_parts:
            p.unlink()
        print(f"[done] wrote {out_path}  rows={len(merged):,}  "
              f"size={os.path.getsize(out_path)/1e6:.1f} MB")
    else:
        print("[done] nothing flushed (empty shard).")


if __name__ == "__main__":
    main()
