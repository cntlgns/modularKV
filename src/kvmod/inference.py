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
) -> list[list[int]]:
    """Forward ``genprefix`` against ``cache`` then greedy-argmax until every
    row hits a stop id (or ``max_new_tokens``). Returns generated ids per row,
    *including* the genprefix tokens (caller decodes + splits, mirroring the
    original ``outputs[i, input_ids.size(1):]`` slice)."""
    device = next(model.parameters()).device
    B, G = genprefix.shape
    arangeG = torch.arange(G, device=device)
    pos = start_position.to(device).unsqueeze(1) + arangeG  # (B, G)
    attn = torch.cat(
        [decode_pad_mask.to(device), torch.ones((B, G), device=device, dtype=decode_pad_mask.dtype)],
        dim=1,
    )
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

    pf: PolicyPrefill = build_policy_prefill(
        segment_ids, policy, modular_q_pos=modular_q_pos, dtype=model.dtype
    )

    if not pf.relocalize:
        # ---- single batched prefill -------------------------------------- #
        if pf.attention_mask_4d is not None:
            prefill_mask = pf.attention_mask_4d.to(device=device, dtype=model.dtype)
        else:
            prefill_mask = pad_mask
        out = model(
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=pf.position_ids.to(device),
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        seqs = _greedy_decode(
            model, out.past_key_values, genprefix, pad_mask,
            pf.next_position, pad_id, stop_ids, max_new_tokens,
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
