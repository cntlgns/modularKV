"""WHERE does the per-doc sink feature get written into the 'Document' token?

BOS is a sink from its embedding (e_bos -> W_K gives a massive key at L0). The
'Document' token (doc-first) is NOT trained as a sink, yet from L2 its key
becomes massive. This script traces the residual stream at the doc-first token
to find which component (which layer's attention vs MLP output) writes the
massive activation that later produces the sink key.

Per layer L the Llama block does:
    h = resid + attn_out(LN1(resid));  resid2 = h;  out = resid2 + mlp_out(LN2(resid2))
We capture, at token 0 (BOS) and at every doc-first ('Document') position:
    - resid_in[L]  = hidden_states[L]            (residual entering layer L)
    - attn_out[L]  = self_attn output            (its additive contribution)
    - mlp_out[L]   = mlp output                  (its additive contribution)
Then we find the residual 'massive' dims (outlier dims at BOS), and track, at the
Document token, when they appear and whether attn or mlp of which layer writes
them. Run for baseline and recover_pos_enc (the residual stream of the Document
token is policy-dependent from L1 on, since attention masking differs).

Output: console trace + JSON.  Default model: 1B (mechanism is scale-invariant).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kvmod.policies import build_policy_prefill  # noqa: E402
from scripts.evaluation.attn_analysis import LOADERS  # noqa: E402
from scripts.evaluation.attn_region_compare import load_block_qa  # noqa: E402

ALL_LOADERS = dict(LOADERS); ALL_LOADERS["block_qa"] = load_block_qa


class Tap:
    """Capture per-layer attn_out and mlp_out additive contributions."""
    def __init__(self, model):
        self.attn = {}; self.mlp = {}; self.handles = []
        for L, lyr in enumerate(model.model.layers):
            self.handles.append(lyr.self_attn.register_forward_hook(self._mk(self.attn, L)))
            self.handles.append(lyr.mlp.register_forward_hook(self._mk(self.mlp, L)))

    def _mk(self, store, L):
        def hook(m, i, o):
            store[L] = (o[0] if isinstance(o, tuple) else o).detach()[0].float()  # (T,H)
        return hook

    def clear(self):
        self.attn.clear(); self.mlp.clear()

    def detach(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run(model, samples, policy, device):
    n_layers = model.config.num_hidden_layers
    tap = Tap(model)
    # accumulators: resid/attn/mlp magnitude vectors at BOS and at Document, per layer
    sum_resid_bos = np.zeros((n_layers + 1, model.config.hidden_size))
    sum_resid_doc = np.zeros((n_layers + 1, model.config.hidden_size))
    sum_attn_doc = np.zeros((n_layers, model.config.hidden_size))
    sum_mlp_doc = np.zeros((n_layers, model.config.hidden_size))
    sum_attn_bos = np.zeros((n_layers, model.config.hidden_size))
    sum_mlp_bos = np.zeros((n_layers, model.config.hidden_size))
    n_doc = 0; n_s = 0
    for s in samples:
        input_ids = torch.tensor([s.input_ids], dtype=torch.long, device=device)
        segment_ids = torch.tensor([s.segment_ids], dtype=torch.long, device=device)
        pad_mask = (segment_ids != -1).long()
        pf = build_policy_prefill(segment_ids, policy, dtype=model.dtype)
        mask = (pf.attention_mask_4d.to(device=device, dtype=model.dtype)
                if pf.attention_mask_4d is not None else pad_mask)
        tap.clear()
        out = model(input_ids=input_ids, attention_mask=mask,
                    position_ids=pf.position_ids.to(device),
                    use_cache=False, output_hidden_states=True)
        hs = out.hidden_states  # len n_layers+1 ; hs[L][0] = (T,H)
        docfirst = [ds for ds, de in s.doc_spans]
        for L in range(n_layers + 1):
            sum_resid_bos[L] += hs[L][0, 0].float().cpu().numpy()
            sum_resid_doc[L] += hs[L][0, docfirst].float().mean(0).cpu().numpy()
        for L in range(n_layers):
            sum_attn_bos[L] += tap.attn[L][0].cpu().numpy()
            sum_mlp_bos[L] += tap.mlp[L][0].cpu().numpy()
            sum_attn_doc[L] += tap.attn[L][docfirst].mean(0).cpu().numpy()
            sum_mlp_doc[L] += tap.mlp[L][docfirst].mean(0).cpu().numpy()
        n_s += 1
    tap.detach()
    f = 1.0 / n_s
    return dict(resid_bos=sum_resid_bos * f, resid_doc=sum_resid_doc * f,
                attn_doc=sum_attn_doc * f, mlp_doc=sum_mlp_doc * f,
                attn_bos=sum_attn_bos * f, mlp_bos=sum_mlp_bos * f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--datasets", nargs="+", default=["nq", "musique"])
    ap.add_argument("--policies", nargs="+", default=["baseline", "recover_pos_enc"])
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--out_dir", default="result/attn_sink_emergence")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(args.model); tok.pad_token_id = 128004
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
    nL = model.config.num_hidden_layers

    res = {}
    for pol in args.policies:
        samples = []
        for ds in args.datasets:
            samples += ALL_LOADERS[ds](args.n_samples, tok)
        res[pol] = run(model, samples, pol, args.device)
        print(f"[{pol}] done over {len(samples)} samples", flush=True)

    json.dump({k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in res.items()},
              open(out_dir / f"sink_emergence__{slug}.json", "w"))

    # ----- analysis ----- #
    # massive dims: outlier dims of the residual at BOS (max over layers)
    rb = res["baseline"]["resid_bos"]
    peak = np.abs(rb).max(0)                       # (H,)
    massive = np.argsort(-peak)[:4]
    med = np.median(np.abs(rb))
    print(f"\n[massive residual dims @ BOS] {[int(x) for x in massive]} "
          f"|val|={[round(float(peak[x]),1) for x in massive]}  (median|resid|={med:.2f})")

    for pol in args.policies:
        R = res[pol]
        print(f"\n===== {pol} : residual at the 'Document' token, massive dims {list(map(int,massive))} =====")
        print(" L | " + " | ".join(f"d{int(d)}: resid (attn+mlp)" for d in massive))
        for L in range(nL):
            cells = []
            for d in massive:
                rv = R["resid_doc"][L, d]
                a = R["attn_doc"][L, d]; m = R["mlp_doc"][L, d]
                cells.append(f"{rv:+7.1f} ({a:+5.1f}+{m:+5.1f})")
            print(f"L{L:2d}| " + " | ".join(cells))
        # final residual row
        cells = [f"{R['resid_doc'][nL, d]:+7.1f}" for d in massive]
        print(f"L{nL:2d}| " + " | ".join(f"{c:>22s}" for c in cells) + "   (final)")

    # BOS trajectory for contrast (baseline)
    R = res["baseline"]
    print(f"\n===== baseline : residual at BOS, massive dims (for contrast) =====")
    for L in [0, 1, 2, 3]:
        cells = [f"d{int(d)}={R['resid_bos'][L,d]:+.1f}" for d in massive]
        print(f"  L{L} resid_in: " + "  ".join(cells))
    print(f"[done] {out_dir / f'sink_emergence__{slug}.json'}")


if __name__ == "__main__":
    main()
