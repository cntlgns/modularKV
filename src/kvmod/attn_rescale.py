"""recover_attn_score: post-softmax question-row attention rescaler.

Same mask + position layout as ``recover_pos_enc`` (block-diagonal doc band,
contiguous global positions). The difference is that, inside each layer's
eager attention forward, the question rows of ``attn_weights`` are rescaled
to match per-(layer, head, qi) sink/sys/docs mass targets predicted by a
trained regressor. Mass split into 4 disjoint regions:

  region 0 = sink             = key index 0
  region 1 = sys-excl-sink    = key indices [1, sys_end)
  region 2 = docs             = union of doc spans
  region 3 = rest             = qprev + self diagonal (= 1 - sys - docs)

Predicted targets are ``(t_sink, t_sys, t_docs)``; ``t_sys_excl = t_sys - t_sink``,
``t_rest = 1 - t_sys - t_docs``. Each is clipped to >= 0 and renormalized to
sum to 1, then per-region multipliers ``t_r / current_r_mass`` are applied,
followed by a row-wise renormalization (safety).

The patched ``LlamaAttention.forward`` is installed via
``install_rescale_patch(model)``. It is a no-op while ``_CTX`` (the rescale
context) is None — so the same patched forward can serve baseline,
recover_pos_enc, etc., simply by NOT setting a context.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaAttention, apply_rotary_pos_emb, repeat_kv,
)


# --------------------------- per-sample context --------------------------- #

@dataclass
class RescaleContext:
    """Predictions + region indices for one prefill call (one sample)."""
    # (n_layers, n_heads, q_len, 3) -- t_sink, t_sys, t_docs (in [0, 1])
    targets: torch.Tensor
    q_lo: int
    q_hi: int
    sys_end: int
    # (T,) long tensor: 0=sink, 1=sys_excl, 2=docs, 3=rest. Pre-built so the
    # patched forward can broadcast multipliers via gather/advanced index.
    region_idx: torch.Tensor


@dataclass
class DecodeRescaleContext:
    """Predictions + region layout for the DECODE phase of one sample.

    Used to rescale generation-token query rows (1+ rows per decode forward).
    Region scheme is 5-way (the question span is now a distinct region the gen
    token attends back to):
        0=sink  1=sys_excl  2=docs  3=question  4=rest(=genprefix+gen-prev+self)
    """
    # (max_gi, n_layers, n_heads, 4) -- t_sink, t_sys(incl sink), t_docs, t_question
    targets: torch.Tensor
    prompt_len: int                  # == q_hi; first gen position
    sys_end: int
    q_lo: int
    q_hi: int
    doc_spans: list                  # list[(ds, de)] within [0, prompt_len)
    # (prompt_len,) long: region id for the fixed prompt keys. Keys >= prompt_len
    # are region 4 (rest), filled on demand for the current key length.
    prompt_region_idx: torch.Tensor
    # Correction strength in [0, 1]: interpolate per-region between the current
    # (measured) distribution and the GBM target. 0 = no-op (== recover_pos_enc),
    # 1 = full correction. Lets us measure the dose-response and avoid the
    # over-correction that a full per-layer attention rewrite can cause.
    strength: float = 1.0


# module-level contexts (set before forward, cleared after)
_CTX: Optional[RescaleContext] = None
_DECODE_CTX: Optional[DecodeRescaleContext] = None
_DECODE_START_GI: int = 0


def set_rescale_context(ctx: Optional[RescaleContext]):
    global _CTX
    _CTX = ctx


def get_rescale_context() -> Optional[RescaleContext]:
    return _CTX


def set_decode_context(ctx: Optional[DecodeRescaleContext]):
    global _DECODE_CTX
    _DECODE_CTX = ctx


def set_decode_start_gi(gi: int):
    """gi of the FIRST query row in the next decode forward. Subsequent rows in
    that forward are gi+1, gi+2, ... (handles the multi-token genprefix pass)."""
    global _DECODE_START_GI
    _DECODE_START_GI = int(gi)


def build_region_idx(T: int, sys_end: int, doc_spans, device) -> torch.Tensor:
    """Tag each key with its region id: 0/1/2/3 (4-region, prefill question)."""
    idx = torch.full((T,), 3, dtype=torch.long, device=device)  # default = rest
    idx[1:sys_end] = 1                                          # sys excl sink
    idx[0] = 0                                                  # sink
    for ds, de in doc_spans:
        idx[ds:de] = 2                                          # docs
    return idx


def build_region_idx5(prompt_len: int, sys_end: int, q_lo: int, q_hi: int,
                      doc_spans, device) -> torch.Tensor:
    """Tag prompt keys [0, prompt_len) with the 5-region id:
       0=sink 1=sys_excl 2=docs 3=question (4=rest is the default for keys
       beyond prompt_len, added at apply time)."""
    idx = torch.full((prompt_len,), 4, dtype=torch.long, device=device)  # default rest
    idx[1:sys_end] = 1
    idx[0] = 0
    for ds, de in doc_spans:
        idx[ds:de] = 2
    idx[q_lo:q_hi] = 3
    return idx


# --------------------------- patched forward ------------------------------ #

def _patched_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
    **kwargs,
):
    """Copy of LlamaAttention.forward (eager) with a rescale step inserted
    between softmax(attn_weights) and the V matmul.
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view  (bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs,
        )

    key_states_r   = repeat_kv(key_states, self.num_key_value_groups)
    value_states_r = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states_r.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states_r.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # === RESCALE QUESTION ROWS (prefill) ================================= #
    ctx = _CTX
    if ctx is not None and self.layer_idx is not None:
        attn_weights = _apply_rescale(attn_weights, ctx, self.layer_idx)
    # === RESCALE GENERATION ROWS (decode) =============================== #
    dctx = _DECODE_CTX
    if dctx is not None and self.layer_idx is not None:
        attn_weights = _apply_decode_rescale(attn_weights, dctx, _DECODE_START_GI, self.layer_idx)
    # ===================================================================== #

    # NB: we ALWAYS return attn_weights (never None-out) so that downstream
    # forward hooks on self_attn can collect them even when the top-level
    # model.forward is called with output_attentions=False. (HF only
    # accumulates them in all_self_attns when output_attentions=True, so
    # passing False at the top still saves the per-layer accumulation memory.)
    attn_weights_for_hook = attn_weights
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states_r)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights_for_hook, past_key_value


def _apply_rescale(attn_weights: torch.Tensor, ctx: RescaleContext, layer_idx: int) -> torch.Tensor:
    """Rescale question rows in-place-style; returns new tensor (same dtype/device)."""
    B, H, T_q, T = attn_weights.shape
    q_lo, q_hi = ctx.q_lo, ctx.q_hi
    if q_lo >= q_hi or q_hi > T_q:
        return attn_weights

    region_idx = ctx.region_idx.to(attn_weights.device)
    # targets for this layer: (H, q_len, 3)
    t = ctx.targets[layer_idx].to(device=attn_weights.device, dtype=torch.float32)
    if t.shape[0] != H:
        # GQA model: targets might be saved at num_attention_heads which equals
        # H after repeat_kv. They should match here.
        raise RuntimeError(f"targets head dim {t.shape[0]} != attn H {H}")

    aq = attn_weights[:, :, q_lo:q_hi, :].float()  # (B, H, q_len, T)

    # Current mass per region — use scatter_add into 4 buckets.
    # mass[..., r] = sum_{k: region_idx[k]==r} aq[..., k]
    region_one_hot = nn.functional.one_hot(region_idx, num_classes=4).to(aq.dtype)  # (T, 4)
    mass = aq @ region_one_hot                       # (B, H, q_len, 4)

    # Targets per region
    t_sink     = t[..., 0]                            # (H, q_len)
    t_sys_full = t[..., 1]
    t_docs     = t[..., 2]
    t_sys_excl = (t_sys_full - t_sink).clamp_min(0.0)
    t_rest     = (1.0 - t_sink - t_sys_excl - t_docs).clamp_min(0.0)
    tgt = torch.stack([t_sink, t_sys_excl, t_docs, t_rest], dim=-1)  # (H, q_len, 4)
    tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(1e-8)
    tgt = tgt.unsqueeze(0)                            # (1, H, q_len, 4)

    # Per-region multipliers
    eps = 1e-8
    mult4 = tgt / mass.clamp_min(eps)                 # (B, H, q_len, 4)
    # Expand to per-key multipliers via region_idx gather
    mult = mult4[..., region_idx]                     # (B, H, q_len, T)

    rescaled = aq * mult
    rescaled = rescaled / rescaled.sum(-1, keepdim=True).clamp_min(eps)  # row renorm

    # In-place: avoid 575 MB clone for 8B + T=3000.
    attn_weights[:, :, q_lo:q_hi, :] = rescaled.to(attn_weights.dtype)
    return attn_weights


def _apply_decode_rescale(attn_weights: torch.Tensor, ctx: DecodeRescaleContext,
                          start_gi: int, layer_idx: int) -> torch.Tensor:
    """Rescale decode (generation) query rows over 5 regions.

    attn_weights: (B=1, H, Q, T). Q is the number of query rows in this forward
    (Q=len(genprefix) for the first decode pass, Q=1 afterward). Query row j has
    gi = start_gi + j and attends causally to keys [0, prompt_len + start_gi + j].
    """
    B, H, Q, T = attn_weights.shape
    if B != 1:
        return attn_weights
    device = attn_weights.device
    max_gi = ctx.targets.shape[0]

    # region_idx over the current key axis [0, T): prompt part fixed, rest=4 beyond.
    region_idx = torch.full((T,), 4, dtype=torch.long, device=device)
    pl = min(ctx.prompt_len, T)
    region_idx[:pl] = ctx.prompt_region_idx[:pl].to(device)
    one_hot = nn.functional.one_hot(region_idx, num_classes=5).to(torch.float32)  # (T, 5)

    eps = 1e-8
    alpha = float(getattr(ctx, "strength", 1.0))
    for j in range(Q):
        gi = start_gi + j
        if gi >= max_gi:
            continue
        t = ctx.targets[gi, layer_idx].to(device=device, dtype=torch.float32)  # (H, 4)
        aq = attn_weights[0, :, j, :].float()                                   # (H, T)
        mass = aq @ one_hot                                                     # (H, 5) current region dist

        t_sink = t[:, 0]; t_sys = t[:, 1]; t_docs = t[:, 2]; t_q = t[:, 3]
        t_sys_excl = (t_sys - t_sink).clamp_min(0.0)
        t_rest     = (1.0 - t_sys - t_docs - t_q).clamp_min(0.0)
        tgt = torch.stack([t_sink, t_sys_excl, t_docs, t_q, t_rest], dim=-1)    # (H, 5)
        tgt = tgt / tgt.sum(-1, keepdim=True).clamp_min(eps)

        # Interpolate target toward current by strength alpha (alpha=1 -> full
        # target, alpha=0 -> identity). Working in distribution space keeps the
        # per-region multiplier = eff/current well-behaved.
        cur = mass / mass.sum(-1, keepdim=True).clamp_min(eps)
        eff = (1.0 - alpha) * cur + alpha * tgt
        mult5 = eff / mass.clamp_min(eps)                                        # (H, 5)
        mult  = mult5[:, region_idx]                                             # (H, T)
        resc = aq * mult
        resc = resc / resc.sum(-1, keepdim=True).clamp_min(eps)
        attn_weights[0, :, j, :] = resc.to(attn_weights.dtype)
    return attn_weights


# --------------------------- install / uninstall -------------------------- #

def install_rescale_patch(model):
    """Bind the patched forward to every LlamaAttention in the model."""
    n = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = types.MethodType(_patched_attention_forward, module)
            n += 1
    return n


# --------------------------- pipeline helpers ----------------------------- #
# These wire the rescale machinery into generate_for_policy with one call:
#   _ensure_rescale_ready(model)             — lazy patch install + GBM load
#   _build_and_set_context_for_batch(...)    — per-sample feature -> targets
# Default GBM dir is "attn_score_models" relative to CWD (run_eval.sh runs
# from the project root).

_CONT_FEATS = [
    "qprev_len", "docs_total_len", "n_docs", "mean_doc_len",
    "min_doc_len", "max_doc_len", "q_len", "seq_len", "q_rel", "qabs",
    "log_seq_len", "log1p_docs_total_len", "log1p_qabs", "docs_to_seq_ratio",
]


def _spans_from_segment_ids(seg):
    """Recover (sys_span, doc_spans, question_span) from a flat segment list."""
    n = len(seg)
    i = 0
    while i < n and seg[i] == 0:
        i += 1
    sys_end = i
    docs = []
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
    # right-pad (-1) chops the question; identify q_hi as last non-pad index
    q_hi = n
    while q_hi > 0 and seg[q_hi - 1] == -1:
        q_hi -= 1
    q_lo = docs[-1][1] if docs else sys_end
    return (0, sys_end), docs, (q_lo, q_hi)


def _build_features(sys_span, doc_spans, question_span, seq_len, n_layers, n_heads):
    import numpy as np
    q_lo, q_hi = question_span
    q_len = q_hi - q_lo
    if q_len <= 0:
        return np.zeros((0, len(_CONT_FEATS) + 2), dtype=np.float32)
    doc_lens = [de - ds for (ds, de) in doc_spans]
    docs_total_len = float(sum(doc_lens))
    n_docs = float(len(doc_lens))
    mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0
    min_doc = float(min(doc_lens)) if doc_lens else 0.0
    max_doc = float(max(doc_lens)) if doc_lens else 0.0
    sys_len_v = float(sys_span[1] - sys_span[0])
    seq_len_v = float(seq_len)
    log_seq_len = float(np.log(seq_len_v))
    log1p_docs = float(np.log1p(docs_total_len))

    qi_arr = np.arange(q_len, dtype=np.float32)
    qabs_arr = qi_arr + float(q_lo)
    q_rel_arr = qi_arr / max(q_len - 1, 1)
    log1p_qabs = np.log1p(qabs_arr)

    qi_block = np.stack([
        qi_arr,
        np.full(q_len, docs_total_len, dtype=np.float32),
        np.full(q_len, n_docs, dtype=np.float32),
        np.full(q_len, mean_doc, dtype=np.float32),
        np.full(q_len, min_doc, dtype=np.float32),
        np.full(q_len, max_doc, dtype=np.float32),
        np.full(q_len, q_len, dtype=np.float32),
        np.full(q_len, seq_len_v, dtype=np.float32),
        q_rel_arr,
        qabs_arr,
        np.full(q_len, log_seq_len, dtype=np.float32),
        np.full(q_len, log1p_docs, dtype=np.float32),
        log1p_qabs,
        np.full(q_len, docs_total_len / seq_len_v, dtype=np.float32),
    ], axis=1)

    out = np.empty((n_layers, n_heads, q_len, len(_CONT_FEATS) + 2), dtype=np.float32)
    out[..., :len(_CONT_FEATS)] = qi_block[None, None, :, :]
    layer_grid = np.arange(n_layers, dtype=np.float32)[:, None, None]
    head_grid = np.arange(n_heads, dtype=np.float32)[None, :, None]
    out[..., len(_CONT_FEATS)] = np.broadcast_to(layer_grid, (n_layers, n_heads, q_len))
    out[..., len(_CONT_FEATS) + 1] = np.broadcast_to(head_grid, (n_layers, n_heads, q_len))
    return out.reshape(-1, len(_CONT_FEATS) + 2)


def _ensure_rescale_ready(model, gbm_dir: str = "attn_score_models"):
    """Lazy: install attention patch + load 3 GBM regressors. Cached on model."""
    import os
    if not getattr(model, "_rescale_patch_installed", False):
        install_rescale_patch(model)
        model._rescale_patch_installed = True
    if not hasattr(model, "_rescale_gbms"):
        import joblib
        gbm_dir = os.environ.get("KVMOD_GBM_DIR", gbm_dir)
        from pathlib import Path
        p = Path(gbm_dir)
        if not p.exists():
            raise FileNotFoundError(
                f"recover_attn_score: GBM dir {p} not found. "
                "Set KVMOD_GBM_DIR or place sink.joblib/sys.joblib/docs.joblib there."
            )
        model._rescale_gbms = {
            "sink": joblib.load(p / "sink.joblib")["model"],
            "sys":  joblib.load(p / "sys.joblib")["model"],
            "docs": joblib.load(p / "docs.joblib")["model"],
        }


def _build_and_set_context_for_batch(model, segment_ids_row, n_layers, n_heads, device):
    """For a single-row batch: derive spans, predict 3 targets, set ctx."""
    import numpy as np
    sys_span, doc_spans, q_span = _spans_from_segment_ids(segment_ids_row)
    seq_len = sum(1 for s in segment_ids_row if s != -1)
    X = _build_features(sys_span, doc_spans, q_span, seq_len, n_layers, n_heads)
    if X.shape[0] == 0:
        set_rescale_context(None)
        return
    gbms = model._rescale_gbms
    preds = np.stack([gbms["sink"].predict(X),
                      gbms["sys"].predict(X),
                      gbms["docs"].predict(X)], axis=-1)
    preds = np.clip(preds, 0.0, 1.0).reshape(n_layers, n_heads, q_span[1] - q_span[0], 3)
    targets = torch.from_numpy(preds).to(device=device, dtype=torch.float32)
    region_idx = build_region_idx(T=seq_len, sys_end=sys_span[1],
                                   doc_spans=doc_spans, device=device)
    set_rescale_context(RescaleContext(
        targets=targets, q_lo=q_span[0], q_hi=q_span[1],
        sys_end=sys_span[1], region_idx=region_idx,
    ))


# ============ CLEAN question-row prefill rescaler (block_qa GBM) =========== #
# Feature schema MUST match train_attn_score_gbm_q.py CONT_FEATS + CAT_FEATS.
# This replaces the legacy _build_features/_build_and_set_context_for_batch
# (which used the 14-feat eval-benchmark-trained models, attn_score_models/) with
# the clean block_qa-trained question GBMs (attn_score_q_models/).
_Q_CONT_FEATS = [
    "qi", "q_len", "n_docs", "docs_total_len", "mean_doc_len", "sys_len",
    "pre_q_len", "cur_pos", "qi_to_q_len", "q_to_docs_ratio",
]
_Q_DEFAULT_DIR = "analysis/attention_score_analysis/attn_score_q_models"


def _ensure_q_rescale_ready(model, gbm_dir: str = _Q_DEFAULT_DIR):
    """Lazy: install attention patch + load the 3 clean question GBM regressors."""
    import os
    if not getattr(model, "_rescale_patch_installed", False):
        install_rescale_patch(model)
        model._rescale_patch_installed = True
    if not hasattr(model, "_q_rescale_gbms"):
        import joblib
        from pathlib import Path
        gbm_dir = os.environ.get("KVMOD_Q_GBM_DIR", gbm_dir)
        p = Path(gbm_dir)
        if not p.exists():
            raise FileNotFoundError(
                f"recover_attn_score_q/_qg: clean question GBM dir {p} not found. "
                "Set KVMOD_Q_GBM_DIR or place sink/sys/docs.joblib there."
            )
        model._q_rescale_gbms = {
            k: joblib.load(p / f"{k}.joblib")["model"] for k in ("sink", "sys", "docs")
        }


def _build_q_features(sys_span, doc_spans, q_span, n_layers, n_heads):
    """Per-(layer, head, qi) feature matrix matching train_attn_score_gbm_q.py."""
    import numpy as np
    q_lo, q_hi = q_span
    q_len = q_hi - q_lo
    if q_len <= 0:
        return np.zeros((0, len(_Q_CONT_FEATS) + 2), dtype=np.float32)
    doc_lens = [de - ds for (ds, de) in doc_spans]
    docs_total_len = float(sum(doc_lens))
    n_docs = float(len(doc_lens))
    mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0
    sys_len = float(sys_span[1] - sys_span[0])
    pre_q_len = sys_len + docs_total_len

    qi_arr = np.arange(q_len, dtype=np.float32)
    cont = np.stack([
        qi_arr,
        np.full(q_len, q_len, np.float32),
        np.full(q_len, n_docs, np.float32),
        np.full(q_len, docs_total_len, np.float32),
        np.full(q_len, mean_doc, np.float32),
        np.full(q_len, sys_len, np.float32),
        np.full(q_len, pre_q_len, np.float32),
        pre_q_len + qi_arr,                                  # cur_pos
        qi_arr / max(q_len, 1),                              # qi_to_q_len
        np.full(q_len, q_len / max(docs_total_len, 1.0), np.float32),  # q_to_docs_ratio
    ], axis=1)  # (q_len, 10)

    n_cont = cont.shape[1]
    out = np.empty((n_layers, n_heads, q_len, n_cont + 2), dtype=np.float32)
    out[..., :n_cont] = cont[None, None, :, :]
    layer_grid = np.arange(n_layers, dtype=np.float32)[:, None, None]
    head_grid = np.arange(n_heads, dtype=np.float32)[None, :, None]
    out[..., n_cont] = np.broadcast_to(layer_grid, (n_layers, n_heads, q_len))
    out[..., n_cont + 1] = np.broadcast_to(head_grid, (n_layers, n_heads, q_len))
    return out.reshape(-1, n_cont + 2)


def _build_and_set_q_context_for_batch(model, segment_ids_row, n_layers, n_heads, device):
    """Predict 3 clean question targets and set the prefill RescaleContext (4-region)."""
    import numpy as np
    sys_span, doc_spans, q_span = _spans_from_segment_ids(segment_ids_row)
    X = _build_q_features(sys_span, doc_spans, q_span, n_layers, n_heads)
    if X.shape[0] == 0:
        set_rescale_context(None)
        return
    g = model._q_rescale_gbms
    preds = np.stack([g["sink"].predict(X), g["sys"].predict(X), g["docs"].predict(X)], axis=-1)
    q_len = q_span[1] - q_span[0]
    preds = np.clip(preds, 0.0, 1.0).reshape(n_layers, n_heads, q_len, 3)
    targets = torch.from_numpy(preds).to(device=device, dtype=torch.float32)
    seq_len = sum(1 for s in segment_ids_row if s != -1)
    region_idx = build_region_idx(T=seq_len, sys_end=sys_span[1],
                                   doc_spans=doc_spans, device=device)
    set_rescale_context(RescaleContext(
        targets=targets, q_lo=q_span[0], q_hi=q_span[1],
        sys_end=sys_span[1], region_idx=region_idx,
    ))


# ===================== DECODE (generation-row) rescaler ==================== #
# Feature schema MUST match train_attn_score_gbm_gen.py CONT_FEATS + CAT_FEATS.
_GEN_CONT_FEATS = [
    "gi", "q_len", "n_docs", "docs_total_len", "mean_doc_len", "sys_len",
    "prompt_len", "cur_pos", "q_to_prompt_ratio", "q_to_docs_ratio", "gi_to_q_len",
]
_GEN_DEFAULT_DIR = "analysis/attention_score_analysis/attn_score_gen_models"


def _ensure_gen_rescale_ready(model, gbm_dir: str = _GEN_DEFAULT_DIR):
    """Lazy: install attention patch + load the 4 gen-token GBM regressors."""
    import os
    if not getattr(model, "_rescale_patch_installed", False):
        install_rescale_patch(model)
        model._rescale_patch_installed = True
    if not hasattr(model, "_gen_rescale_gbms"):
        import joblib
        from pathlib import Path
        gbm_dir = os.environ.get("KVMOD_GEN_GBM_DIR", gbm_dir)
        p = Path(gbm_dir)
        if not p.exists():
            raise FileNotFoundError(
                f"recover_attn_score_gen: gen GBM dir {p} not found. "
                "Set KVMOD_GEN_GBM_DIR or place sink/sys/docs/question.joblib there."
            )
        model._gen_rescale_gbms = {
            k: joblib.load(p / f"{k}.joblib")["model"]
            for k in ("sink", "sys", "docs", "question")
        }


def build_gen_decode_context(model, segment_ids_row, n_layers, n_heads,
                             max_gi, device) -> DecodeRescaleContext:
    """Derive spans from a single sample's segment_ids, predict the entire
    (gi, layer, head) -> (sink, sys, docs, question) target grid in ONE batched
    GBM call (amortized — no per-decode-step prediction), and return the context.
    """
    import numpy as np
    sys_span, doc_spans, q_span = _spans_from_segment_ids(segment_ids_row)
    sys_end = sys_span[1]
    q_lo, q_hi = q_span
    q_len = q_hi - q_lo
    doc_lens = [de - ds for (ds, de) in doc_spans]
    docs_total_len = float(sum(doc_lens))
    n_docs = float(len(doc_lens))
    mean_doc = float(np.mean(doc_lens)) if doc_lens else 0.0
    sys_len = float(sys_end)
    prompt_len = int(q_hi)  # question is the last prompt segment

    gi_arr = np.arange(max_gi, dtype=np.float32)
    cont = np.stack([
        gi_arr,
        np.full(max_gi, q_len, np.float32),
        np.full(max_gi, n_docs, np.float32),
        np.full(max_gi, docs_total_len, np.float32),
        np.full(max_gi, mean_doc, np.float32),
        np.full(max_gi, sys_len, np.float32),
        np.full(max_gi, float(prompt_len), np.float32),
        float(prompt_len) + gi_arr,                                   # cur_pos
        np.full(max_gi, q_len / max(prompt_len, 1), np.float32),      # q_to_prompt_ratio
        np.full(max_gi, q_len / max(docs_total_len, 1.0), np.float32),# q_to_docs_ratio
        gi_arr / max(q_len, 1),                                        # gi_to_q_len
    ], axis=1)  # (max_gi, 11)

    n_cont = cont.shape[1]
    X = np.empty((max_gi, n_layers, n_heads, n_cont + 2), dtype=np.float32)
    X[..., :n_cont] = cont[:, None, None, :]
    layer_grid = np.arange(n_layers, dtype=np.float32)[None, :, None]
    head_grid  = np.arange(n_heads,  dtype=np.float32)[None, None, :]
    X[..., n_cont]     = np.broadcast_to(layer_grid, (max_gi, n_layers, n_heads))
    X[..., n_cont + 1] = np.broadcast_to(head_grid,  (max_gi, n_layers, n_heads))
    Xf = X.reshape(-1, n_cont + 2)

    g = model._gen_rescale_gbms
    preds = np.stack([g["sink"].predict(Xf), g["sys"].predict(Xf),
                      g["docs"].predict(Xf), g["question"].predict(Xf)], axis=-1)
    preds = np.clip(preds, 0.0, 1.0).reshape(max_gi, n_layers, n_heads, 4)
    targets = torch.from_numpy(preds).to(device=device, dtype=torch.float32)

    prompt_region_idx = build_region_idx5(
        prompt_len=prompt_len, sys_end=sys_end, q_lo=q_lo, q_hi=q_hi,
        doc_spans=doc_spans, device=device,
    )
    import os
    strength = float(os.environ.get("KVMOD_GEN_STRENGTH", "1.0"))
    return DecodeRescaleContext(
        targets=targets, prompt_len=prompt_len, sys_end=sys_end,
        q_lo=q_lo, q_hi=q_hi, doc_spans=doc_spans,
        prompt_region_idx=prompt_region_idx, strength=strength,
    )
