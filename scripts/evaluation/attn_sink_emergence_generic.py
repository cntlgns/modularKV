"""Model-AGNOSTIC version of attn_sink_emergence: does the per-doc sink feature
(massive activation written by an early MLP) appear in non-Llama models too?

Builds input_ids/segment_ids without any Llama-specific reserved tokens, so it
runs on Qwen2/Qwen3/Mistral/etc (any HF decoder with model.model.layers having
.self_attn and .mlp). For each model it:
  1. assembles  [BOS?] system | Doc1 | Doc2 | ... | question   with segment_ids
     (seg 0 = system/question, seg k = doc k) -- a doc = "Document [k](Title: T) text\n".
  2. runs baseline (plain causal) and recover_pos_enc (block-diagonal segment mask
     via build_policy_prefill, which is itself model-agnostic).
  3. taps each layer's attn_out and mlp_out + the residual stream, at token 0
     (global sink) and the doc-first tokens (candidate local sinks).
  4. auto-detects the "massive" residual dims and reports WHICH layer/component
     writes them.

Run e.g. on meta-llama/Llama-3.2-1B-Instruct (sanity) and Qwen/Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import datasets as hfds
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.kvmod.policies import build_policy_prefill  # noqa: E402

SYSTEM = ("You are a helpful AI assistant. Answer the question based on the "
          "reference documents below.")


def load_raw(n):
    """Return list of (question, [(title,text),...]) from nq + musique raw data."""
    out = []
    nq = hfds.load_dataset("json", data_files=str(ROOT / "data/raw/nq/nq-open-10_0.jsonl"),
                           split="train").select(range(n))
    for ex in nq:
        out.append((ex["question"], [(c["title"], c["text"]) for c in ex["ctxs"][:10]]))
    mus = hfds.load_dataset("dgslibisey/MuSiQue", split="validation").select(range(n))
    for ex in mus:
        out.append((ex["question"], [(p["title"], p["paragraph_text"]) for p in ex["paragraphs"]]))
    return out


def build_generic(tok, question, docs):
    ids, seg = [], []
    if tok.bos_token_id is not None:
        ids.append(tok.bos_token_id); seg.append(0)
    sids = tok(SYSTEM, add_special_tokens=False)["input_ids"]
    ids += sids; seg += [0] * len(sids)
    doc_spans = []
    for k, (title, text) in enumerate(docs):
        s = f"Document [{k+1}](Title: {title}) {text}\n"
        t = tok(s, add_special_tokens=False)["input_ids"]
        doc_spans.append((len(ids), len(ids) + len(t)))
        ids += t; seg += [k + 1] * len(t)
    q = tok("\nQuestion: " + question + "\nAnswer:", add_special_tokens=False)["input_ids"]
    ids += q; seg += [0] * len(q)
    return ids, seg, doc_spans


class Tap:
    def __init__(self, model):
        self.attn, self.mlp, self.handles = {}, {}, []
        for L, lyr in enumerate(model.model.layers):
            self.handles.append(lyr.self_attn.register_forward_hook(self._mk(self.attn, L)))
            self.handles.append(lyr.mlp.register_forward_hook(self._mk(self.mlp, L)))

    def _mk(self, store, L):
        def hook(m, i, o):
            store[L] = (o[0] if isinstance(o, tuple) else o).detach()[0].float()
        return hook

    def clear(self):
        self.attn.clear(); self.mlp.clear()

    def detach(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run(model, tok, raw, policy, device):
    nL = model.config.num_hidden_layers
    Hd = model.config.hidden_size
    tap = Tap(model)
    rb = np.zeros((nL + 1, Hd)); rd = np.zeros((nL + 1, Hd))
    ad = np.zeros((nL, Hd)); md = np.zeros((nL, Hd))
    ab = np.zeros((nL, Hd)); mb = np.zeros((nL, Hd))
    ns = 0
    for question, docs in raw:
        ids, seg, doc_spans = build_generic(tok, question, docs)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([seg], dtype=torch.long, device=device)
        pad_mask = (segment_ids != -1).long()
        pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
        mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
                if pf.attention_mask_4d is not None else pad_mask)
        tap.clear()
        out = model(input_ids=input_ids, attention_mask=mask,
                    position_ids=pf.position_ids.to(device),
                    use_cache=False, output_hidden_states=True)
        hs = out.hidden_states
        df = [ds for ds, de in doc_spans]
        for L in range(nL + 1):
            rb[L] += hs[L][0, 0].float().cpu().numpy()
            rd[L] += hs[L][0, df].float().mean(0).cpu().numpy()
        for L in range(nL):
            ab[L] += tap.attn[L][0].cpu().numpy(); mb[L] += tap.mlp[L][0].cpu().numpy()
            ad[L] += tap.attn[L][df].mean(0).cpu().numpy()
            md[L] += tap.mlp[L][df].mean(0).cpu().numpy()
        ns += 1
    tap.detach()
    f = 1.0 / ns
    return dict(resid_bos=rb*f, resid_doc=rd*f, attn_doc=ad*f, mlp_doc=md*f,
                attn_bos=ab*f, mlp_bos=mb*f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--out_dir", default="result/attn_sink_emergence")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
    nL = model.config.num_hidden_layers
    print(f"[info] {args.model} arch={model.config.architectures} layers={nL} "
          f"hidden={model.config.hidden_size} bos={tok.bos_token_id}")

    raw = load_raw(args.n_samples)
    res = {p: run(model, tok, raw, p, args.device) for p in ["baseline", "recover_pos_enc"]}
    json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in res.items()},
              open(out_dir / f"sink_emergence_generic__{slug}.json", "w"))

    # massive dims = outlier dims of residual at BOS (max over layers)
    rb = res["baseline"]["resid_bos"]
    peak = np.abs(rb).max(0)
    massive = np.argsort(-peak)[:4]
    med = np.median(np.abs(rb))
    print(f"\n[{slug}] median|resid|={med:.3f}  massive dims={[int(x) for x in massive]} "
          f"peak|val|={[round(float(peak[x]),1) for x in massive]} "
          f"(ratio to median ~{round(float(peak[massive[0]])/max(med,1e-6))}x)")

    for pol in ["baseline", "recover_pos_enc"]:
        R = res[pol]
        print(f"\n  === {pol}: 'Document'-token residual at massive dims ===")
        # auto-detect writer layer/component per dim (largest single increment)
        for d in massive:
            a = R["attn_doc"][:, d]; m = R["mlp_doc"][:, d]
            La, Lm = int(np.argmax(np.abs(a))), int(np.argmax(np.abs(m)))
            who = ("MLP", Lm, m[Lm]) if abs(m[Lm]) >= abs(a[La]) else ("ATTN", La, a[La])
            finalv = R["resid_doc"][-1, d]
            peakv = R["resid_doc"][:, d][np.argmax(np.abs(R["resid_doc"][:, d]))]
            print(f"    d{int(d):4d}: peak resid={peakv:+8.1f} | written by {who[0]} @ layer {who[1]} "
                  f"(+{who[2]:+.1f})")
        # compact per-layer trace of dim massive[0]
        d0 = massive[0]
        tr = " ".join(f"L{L}:{R['resid_doc'][L,d0]:+.0f}" for L in range(min(nL+1, 8)))
        print(f"    trace d{int(d0)} (recover) first layers: {tr} ...")


if __name__ == "__main__":
    main()
