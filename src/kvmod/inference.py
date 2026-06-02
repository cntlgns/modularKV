"""Policy-aware prefill + batched greedy decode for KVLink eval.

Consolidates the prefill -> generate -> decode block that was copy-pasted
byte-for-byte across nq/hqa/wiki/musique_eval.py. One entry point,
``generate_for_policy``, handles all 5 KV-cache policies:

  * 4 batched policies (baseline / recover_pos_enc / modular /
    recover_cross_attn) -- one batched prefill with the policy's mask +
    position_ids, then a batched greedy loop with explicit position_ids so
    reset-to-zero policies continue decoding from the right place.

  * recover_cross_attn_oracle_pos -- a two-stage prefill (prefix+docs ->
    rotate cached doc K to per-doc local RoPE positions -> question). The
    question must read the *rotated* doc K, so it cannot share a single
    prefill pass with the docs. Document spans / question start vary per row
    (different doc lengths), so this policy is run row-by-row; the other four
    stay fully batched.

Greedy decode is deterministic, so ``baseline`` reproduces the legacy
``standard`` (full ``model.generate`` greedy) results token-for-token.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from .policies import build_policy_prefill, PolicyPrefill


# --------------------------------------------------------------------------- #
# Post-prefill doc-K relocalization (global RoPE -> per-doc local RoPE).
# Ported from modularkv.inference; adapted to per-row doc spans and the
# transformers 4.46 legacy DynamicCache (key_cache / value_cache lists).
# --------------------------------------------------------------------------- #

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rotary_emb(model: PreTrainedModel):
    inner = getattr(model, "model", model)
    return inner.rotary_emb


@torch.no_grad()
def _relocalize_doc_keys(
    past_key_values,
    doc_spans: list[tuple[int, int]],
    position_ids_global: torch.Tensor,  # (1, L)
    model: PreTrainedModel,
) -> None:
    """Rotate cached doc K from global RoPE positions to per-doc local
    positions (each doc restarts at 0). Values / prefix K are untouched.

        K_new = R(p_local) R(p_global)^-1 K_old = R(p_local - p_global) K_old
    """
    if not doc_spans:
        return
    rotary = _rotary_emb(model)
    key_cache = past_key_values.key_cache
    if not key_cache or key_cache[0].numel() == 0:
        return
    cache_len = key_cache[0].shape[-2]
    dev = key_cache[0].device
    pos_g = position_ids_global.to(dev)[:, :cache_len]

    per_doc = []
    for ds, de in doc_spans:
        if de <= ds or ds >= cache_len:
            continue
        de = min(de, cache_len)
        p_g = pos_g[:, ds:de].to(torch.long)
        p_l = torch.arange(de - ds, device=dev).unsqueeze(0)
        delta = (p_l - p_g).to(torch.long)
        cos, sin = rotary(key_cache[0], delta)  # (1, doc_len, head_dim)
        per_doc.append((ds, de, cos, sin))

    for k in key_cache:
        if k.numel() == 0:
            continue
        for ds, de, cos, sin in per_doc:
            r = cos.shape[-1]
            cos_b = cos.to(k.dtype).unsqueeze(1)  # (1,1,doc_len,r)
            sin_b = sin.to(k.dtype).unsqueeze(1)
            sl = k[:, :, ds:de, :r]
            k[:, :, ds:de, :r] = sl * cos_b + _rotate_half(sl) * sin_b


# --------------------------------------------------------------------------- #
# Batched greedy decode
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _greedy_decode(
    model: PreTrainedModel,
    cache,
    genprefix: torch.Tensor,       # (B, G) chat "assistant" turn tokens
    decode_pad_mask: torch.Tensor,  # (B, cache_len) 1 for real cached tokens
    start_position: torch.Tensor,   # (B,) position id of the first genprefix tok
    pad_id: int,
    stop_ids: set[int],
    max_new_tokens: int,
    track_gen_decode: bool = False,
) -> list[list[int]]:
    """Forward ``genprefix`` against ``cache`` then greedy-argmax until every
    row hits a stop id (or ``max_new_tokens``). Returns generated ids per row,
    *including* the genprefix tokens (caller decodes + splits, mirroring the
    original ``outputs[i, input_ids.size(1):]`` slice).

    When ``track_gen_decode`` is set, the module-level decode rescale step is
    advanced before each forward so the patched attention applies the gen-row
    rescale to the right gi (genprefix rows are gi=0..G-1, then gi=G, G+1, ...).
    """
    device = next(model.parameters()).device
    B, G = genprefix.shape

    if track_gen_decode:
        from .attn_rescale import set_decode_start_gi
    else:
        set_decode_start_gi = None

    arangeG = torch.arange(G, device=device)
    pos = start_position.to(device).unsqueeze(1) + arangeG  # (B, G)
    attn = torch.cat(
        [decode_pad_mask.to(device), torch.ones((B, G), device=device, dtype=decode_pad_mask.dtype)],
        dim=1,
    )
    if set_decode_start_gi is not None:
        set_decode_start_gi(0)  # genprefix rows occupy gi 0..G-1
    out = model(
        input_ids=genprefix.to(device),
        attention_mask=attn,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
    )
    cache = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1)  # (B,)
    cur_pos = pos[:, -1] + 1
    cur_gi = G  # next generated token's gi

    seqs: list[list[int]] = [genprefix[b].tolist() for b in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        for b in range(B):
            if not done[b]:
                t = int(next_tok[b].item())
                if t in stop_ids:
                    done[b] = True
                else:
                    seqs[b].append(t)
        if bool(done.all()):
            break
        feed = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)
        attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
        if set_decode_start_gi is not None:
            set_decode_start_gi(cur_gi)
        out = model(
            input_ids=feed.unsqueeze(1),
            attention_mask=attn,
            position_ids=cur_pos.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        cur_pos = cur_pos + 1
        cur_gi += 1
    return seqs


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate_for_policy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    input_ids: torch.Tensor,        # (B, T) right-padded
    pad_mask: torch.Tensor,         # (B, T) 1 real / 0 pad
    segment_ids: torch.Tensor,      # (B, T) 0 prefix/q, >=1 doc, -1 pad
    policy: str,
    generation_token_ids: torch.Tensor,  # (G,) chat assistant-turn prefix
    stop_token_ids,
    max_new_tokens: int = 200,
    modular_q_pos: str = "summed_pos",
    disable_doc_k_relocalization: bool = False,
) -> list[str]:
    """Run one batch under ``policy`` and return the decoded responses.

    The returned strings are the decoded continuation (genprefix + generated)
    with the original eval split applied:
    ``... .split("assistant<|end_header_id|>")[-1].strip().split("<|eot_id|>")[0]``.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    pad_mask = pad_mask.to(device)
    segment_ids = segment_ids.to(device)
    B = input_ids.shape[0]
    G = generation_token_ids.shape[0]
    genprefix = generation_token_ids.to(device).unsqueeze(0).expand(B, G).contiguous()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    stop_ids = set(int(x) for x in stop_token_ids)

    # `recover_attn_score`: same mask/position as `recover_pos_enc`, plus a
    # per-(layer, head, qi) attention-row rescale driven by a trained
    # regressor (see src/kvmod/attn_rescale.py). The patched LlamaAttention
    # forward is a no-op when the rescale context is unset.
    rescale_active = (policy == "recover_attn_score")
    # `recover_attn_score_gen`: recover_pos_enc mask/position, plus a 5-region
    # rescale of each GENERATION (decode) row driven by the gen-token GBMs. The
    # question span is a distinct region. prefill question rows are NOT rescaled
    # here (decode-only correction). See src/kvmod/attn_rescale.py.
    rescale_gen_active = (policy == "recover_attn_score_gen")
    # clean question-only (prefill) rescale, and combined (prefill question +
    # decode gen) rescale — both use the block_qa-trained GBMs.
    rescale_q_active  = (policy == "recover_attn_score_q")
    rescale_qg_active = (policy == "recover_attn_score_qg")
    if rescale_active:
        from .attn_rescale import (
            _ensure_rescale_ready, _build_and_set_context_for_batch,
            set_rescale_context,
        )
        _ensure_rescale_ready(model)
        effective_policy = "recover_pos_enc"
    elif rescale_gen_active:
        from .attn_rescale import (
            _ensure_gen_rescale_ready, build_gen_decode_context,
            set_decode_context, set_rescale_context,
        )
        _ensure_gen_rescale_ready(model)
        set_rescale_context(None)  # no prefill question rescale in decode-only mode
        effective_policy = "recover_pos_enc"
        if B != 1:
            raise NotImplementedError(
                "recover_attn_score_gen currently requires batch_size=1; got B=%d" % B
            )
    elif rescale_q_active or rescale_qg_active:
        from .attn_rescale import (
            _ensure_q_rescale_ready, _build_and_set_q_context_for_batch,
            set_rescale_context, set_decode_context,
        )
        _ensure_q_rescale_ready(model)
        if rescale_qg_active:
            from .attn_rescale import _ensure_gen_rescale_ready, build_gen_decode_context
            _ensure_gen_rescale_ready(model)
        set_decode_context(None)
        effective_policy = "recover_pos_enc"
        if B != 1:
            raise NotImplementedError(
                "%s currently requires batch_size=1; got B=%d" % (policy, B)
            )
    else:
        # Make sure any leftover ctx from a prior call is cleared.
        try:
            from .attn_rescale import set_rescale_context, set_decode_context
            set_rescale_context(None)
            set_decode_context(None)
        except Exception:
            pass
        effective_policy = policy

    pf: PolicyPrefill = build_policy_prefill(
        segment_ids, effective_policy, modular_q_pos=modular_q_pos, dtype=model.dtype
    )

    if rescale_active:
        # Build per-row rescale context; for B>1, prediction is per-row so we
        # serialize. The existing sweep uses B=1 so this is the common path.
        if B != 1:
            raise NotImplementedError(
                "recover_attn_score currently requires batch_size=1; got B=%d" % B
            )
        _build_and_set_context_for_batch(
            model, segment_ids[0].tolist(),
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            device=device,
        )

    if rescale_q_active or rescale_qg_active:
        # Prefill question rescale with the clean block_qa question GBM.
        from .attn_rescale import _build_and_set_q_context_for_batch
        _build_and_set_q_context_for_batch(
            model, segment_ids[0].tolist(),
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            device=device,
        )

    if rescale_gen_active or rescale_qg_active:
        # Amortized: predict the whole (gi, layer, head) target grid once. The
        # context is built here but only ACTIVATED after prefill (below), so the
        # decode rescale never touches prefill rows (sys/docs/question).
        from .attn_rescale import build_gen_decode_context
        dctx = build_gen_decode_context(
            model, segment_ids[0].tolist(),
            n_layers=model.config.num_hidden_layers,
            n_heads=model.config.num_attention_heads,
            max_gi=G + max_new_tokens + 1,
            device=device,
        )

    if not pf.relocalize:
        # ---- single batched prefill -------------------------------------- #
        if pf.attention_mask_4d is not None:
            prefill_mask = pf.attention_mask_4d.to(device=device, dtype=model.dtype)
        else:
            prefill_mask = pad_mask
        # Decode context MUST be unset during prefill (the patched forward would
        # otherwise apply the gen-row rescale to prompt rows). The prefill
        # question rescale (_CTX, recover_attn_score / _qg) is separately gated
        # by q_hi > T_q inside _apply_rescale and is safe to leave set.
        from .attn_rescale import set_decode_context as _set_dctx
        _set_dctx(None)
        out = model(
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=pf.position_ids.to(device),
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        decode_rescale = rescale_gen_active or rescale_qg_active
        if decode_rescale:
            _set_dctx(dctx)  # activate only for the decode loop
        seqs = _greedy_decode(
            model, out.past_key_values, genprefix, pad_mask,
            pf.next_position, pad_id, stop_ids, max_new_tokens,
            track_gen_decode=decode_rescale,
        )
    else:
        # ---- recover_cross_attn_oracle_pos: row-by-row two-stage --------- #
        seqs = []
        for b in range(B):
            seqs.extend(
                _run_oracle_row(
                    model, input_ids[b], pad_mask[b], pf, b,
                    genprefix[b:b + 1], pad_id, stop_ids, max_new_tokens,
                    disable_doc_k_relocalization,
                )
            )

    # Clear rescale contexts (no-op if not used).
    if rescale_active or rescale_q_active or rescale_qg_active:
        from .attn_rescale import set_rescale_context
        set_rescale_context(None)
    if rescale_gen_active or rescale_qg_active:
        from .attn_rescale import set_decode_context
        set_decode_context(None)

    decoded = [tokenizer.decode(s) for s in seqs]
    return [
        d.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        .strip()
        .split("<|eot_id|>")[0]
        for d in decoded
    ]


@torch.no_grad()
def _run_oracle_row(
    model, ids_row, pad_row, pf: PolicyPrefill, b: int,
    genprefix1, pad_id, stop_ids, max_new_tokens,
    disable_reloc: bool,
) -> list[list[int]]:
    """Two-stage prefill + greedy decode for one row of the oracle policy.

    Stage 1: forward [0 : q_start) (system prefix + docs) with baseline-style
    causal attention + contiguous positions. Stage 2: rotate cached doc K to
    per-doc local positions, then forward [q_start : q_end) (the question)
    against the rotated cache before decoding.
    """
    device = next(model.parameters()).device
    q_start = pf.question_start[b]
    # q_end = first pad (right-padded) -> count of real tokens.
    q_end = int(pad_row.sum().item())
    pos_g = pf.position_ids[b:b + 1].to(device)  # contiguous (1, T)

    ids = ids_row.unsqueeze(0).to(device)
    stage1 = model(
        input_ids=ids[:, :q_start],
        attention_mask=pad_row[:q_start].unsqueeze(0).to(device),
        position_ids=pos_g[:, :q_start],
        past_key_values=DynamicCache(),
        use_cache=True,
    )
    cache = stage1.past_key_values

    if not disable_reloc:
        _relocalize_doc_keys(cache, pf.doc_spans[b], pos_g[:, :q_start], model)

    # Stage 2: question tokens [q_start:q_end) against the rotated cache.
    q_mask = torch.ones((1, q_end), device=device, dtype=pad_row.dtype)
    stage2 = model(
        input_ids=ids[:, q_start:q_end],
        attention_mask=q_mask,
        position_ids=pos_g[:, q_start:q_end],
        past_key_values=cache,
        use_cache=True,
    )
    cache = stage2.past_key_values
    return _greedy_decode(
        model, cache, genprefix1, q_mask,
        pf.next_position[b:b + 1], pad_id, stop_ids, max_new_tokens,
    )
