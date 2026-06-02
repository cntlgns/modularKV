"""Per-policy ``position_ids`` + attention-mask construction for KVLink eval.

The eval scripts build a ``segment_ids`` tensor per batch where, for each row:

    seg == 0    system prefix / link tokens / question (mem_end + prompt)
    seg >= 1    document ``seg`` (1-indexed; doc k -> seg k)
    seg == -1   right-padding

From that we recover, per row, the system-prefix span, each document span,
and the trailing question span, then assign ``position_ids`` and pick the
attention pattern according to the chosen KV-cache policy. This mirrors
``modularkv.masking.build_prefill_inputs`` exactly (same 5 policies, same
``modular_q_pos`` semantics) so the two repos stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.titan_preprocessor import make_segment_mask

KV_CACHE_POLICIES = (
    "baseline",
    "recover_pos_enc",
    "modular",
    "recover_cross_attn",
    "recover_cross_attn_oracle_pos",
    "recover_attn_score",
    "recover_attn_score_gen",
    "recover_attn_score_q",
    "recover_attn_score_qg",
)

# Doc band isolated (no cross-doc, no prefix attention) -> use the 4-D
# segment mask. recover_pos_enc / modular isolate the doc band;
# recover_pos_enc is the legacy ``blocked``. baseline / recover_cross_attn /
# oracle use plain causal attention (the 2-D pad mask). recover_attn_score
# inherits recover_pos_enc's mask + position scheme and adds a post-softmax
# question-row rescale driven by a trained regressor (see
# src/kvmod/attn_rescale.py).
_USE_SEGMENT_MASK = {
    "baseline": False,
    "recover_pos_enc": True,
    "modular": True,
    "recover_cross_attn": False,
    "recover_cross_attn_oracle_pos": False,
    "recover_attn_score": True,
    "recover_attn_score_gen": True,
    "recover_attn_score_q": True,
    "recover_attn_score_qg": True,
}

# Doc positions during prefill: "contiguous" (global 0..T-1) or
# "reset_to_zero" (each doc + the system prefix restart at 0).
_DOC_POS_SCHEME = {
    "baseline": "contiguous",
    "recover_pos_enc": "contiguous",
    "modular": "reset_to_zero",
    "recover_cross_attn": "reset_to_zero",
    "recover_cross_attn_oracle_pos": "contiguous",
    "recover_attn_score": "contiguous",
    "recover_attn_score_gen": "contiguous",
    "recover_attn_score_q": "contiguous",
    "recover_attn_score_qg": "contiguous",
}

# Post-prefill rotation of cached doc K to per-doc local RoPE positions.
_RELOCALIZE_DOC_K = {p: False for p in KV_CACHE_POLICIES}
_RELOCALIZE_DOC_K["recover_cross_attn_oracle_pos"] = True

MODULAR_Q_POS_CHOICES = ("summed_pos", "min_pos")

# ``modular_q_pos`` only changes anything for reset-to-zero policies. Derived
# from _DOC_POS_SCHEME so a new reset-zero policy picks it up automatically.
POLICIES_WITH_MODULAR_Q_POS: tuple[str, ...] = tuple(
    p for p, s in _DOC_POS_SCHEME.items() if s == "reset_to_zero"
)


@dataclass
class PolicyPrefill:
    """Everything the inference driver needs to prefill + decode one batch.

    Shapes are ``(B, T)`` where ``T`` is the padded batch length.

    ``attention_mask_4d`` is the additive segment+causal mask (``None`` ->
    use the plain 2-D pad mask, i.e. ordinary causal attention).

    ``relocalize`` (only ``recover_cross_attn_oracle_pos``) drives a two-stage
    prefill: forward ``[0:question_start)`` first, rotate cached doc K to
    per-doc local positions, then forward ``[question_start:T)``.
    ``doc_spans[b]`` / ``prefix_len[b]`` / ``question_start[b]`` describe row
    ``b``'s layout (end indices exclusive).
    """

    policy: str
    attention_mask_4d: torch.Tensor | None
    position_ids: torch.Tensor  # (B, T)
    next_position: torch.Tensor  # (B,) first decoded token's position id
    relocalize: bool
    doc_spans: list[list[tuple[int, int]]]
    prefix_len: list[int]
    question_start: list[int]


def _row_segments(seg_row: torch.Tensor):
    """Return (sys_end, doc_spans, q_start, q_end) for one segment_ids row.

    ``sys_end`` = first doc-token index (end of system prefix, exclusive).
    ``doc_spans`` = list of (start, end) per doc, end exclusive.
    ``q_start`` = end of the last doc (start of the trailing question band).
    ``q_end``   = one past the last non-pad token.
    """
    seg = seg_row.tolist()
    n = len(seg)
    # Last non-pad (seg != -1) index; right-padded so this is contiguous, but
    # scan to stay robust to stray pads.
    q_end = n
    while q_end > 0 and seg[q_end - 1] == -1:
        q_end -= 1

    doc_spans: list[tuple[int, int]] = []
    i = 0
    while i < q_end:
        s = seg[i]
        if s >= 1:
            j = i
            while j < q_end and seg[j] == s:
                j += 1
            doc_spans.append((i, j))
            i = j
        else:
            i += 1

    if not doc_spans:
        # No documents: treat the whole row as one contiguous causal block.
        return q_end, [], q_end, q_end

    sys_end = doc_spans[0][0]
    q_start = doc_spans[-1][1]
    return sys_end, doc_spans, q_start, q_end


def build_policy_prefill(
    segment_ids: torch.Tensor,
    policy: str,
    *,
    modular_q_pos: str = "summed_pos",
    dtype: torch.dtype = torch.bfloat16,
) -> PolicyPrefill:
    """Build ``position_ids`` + attention mask + reloc metadata for one batch.

    ``segment_ids`` is the ``(B, T)`` LongTensor produced by the eval
    collator (0 = prefix/link/question, >=1 = doc id, -1 = pad).
    """
    if policy not in KV_CACHE_POLICIES:
        raise ValueError(
            f"Unknown kv_policy {policy!r}; expected one of {KV_CACHE_POLICIES}"
        )
    if modular_q_pos not in MODULAR_Q_POS_CHOICES:
        raise ValueError(
            f"Unknown modular_q_pos {modular_q_pos!r}; "
            f"expected one of {MODULAR_Q_POS_CHOICES}"
        )

    device = segment_ids.device
    B, T = segment_ids.shape
    scheme = _DOC_POS_SCHEME[policy]
    reset = scheme == "reset_to_zero"

    # Contiguous positions = arange over valid tokens; pad positions are
    # arbitrary (they are masked out). HF's default (cumsum-1 of the pad mask)
    # is identical for the right-padded layout, so ``baseline`` stays
    # byte-for-byte the legacy ``standard`` behaviour.
    position_ids = (
        torch.arange(T, device=device).unsqueeze(0).expand(B, T).clone()
    )
    next_position = torch.empty(B, dtype=torch.long, device=device)
    doc_spans_all: list[list[tuple[int, int]]] = []
    prefix_len_all: list[int] = []
    q_start_all: list[int] = []

    for b in range(B):
        sys_end, doc_spans, q_start, q_end = _row_segments(segment_ids[b])
        doc_spans_all.append(doc_spans)
        prefix_len_all.append(sys_end)
        q_start_all.append(q_start)

        if not reset:
            next_position[b] = q_end
            continue

        if any(
            segment_ids[b, k].item() == 0
            for s, e in zip([sys_end] + [d[1] for d in doc_spans],
                            [d[0] for d in doc_spans] + [q_start])
            for k in range(s, e)
        ):
            # A seg-0 island sits between docs (only happens with link tokens,
            # reencode_num > 0). Reset-zero positioning for that case is not
            # part of the modularization ablation (which fixes reencode_num=0)
            # and modularkv has no equivalent -> refuse rather than guess.
            raise ValueError(
                f"reset-zero policy {policy!r} needs contiguous doc band "
                f"(reencode_num=0); row {b} has seg-0 tokens between docs"
            )

        pos = position_ids[b]
        pos[:sys_end] = torch.arange(sys_end, device=device)
        max_doc_len = 0
        total_doc_len = 0
        for ds, de in doc_spans:
            dl = de - ds
            pos[ds:de] = torch.arange(dl, device=device)
            max_doc_len = max(max_doc_len, dl)
            total_doc_len += dl
        if modular_q_pos == "summed_pos":
            q_pos0 = sys_end + total_doc_len
        else:  # "min_pos"
            q_pos0 = max(sys_end, max_doc_len)
        q_len = q_end - q_start
        pos[q_start:q_end] = q_pos0 + torch.arange(q_len, device=device)
        next_position[b] = q_pos0 + q_len

    attention_mask_4d = None
    if _USE_SEGMENT_MASK[policy]:
        attention_mask_4d = make_segment_mask(
            source_segments=segment_ids,
            target_segments=segment_ids,
            dtype=dtype,
            add_causal_lm_mask=True,
        ).unsqueeze(1)

    return PolicyPrefill(
        policy=policy,
        attention_mask_4d=attention_mask_4d,
        position_ids=position_ids,
        next_position=next_position,
        relocalize=_RELOCALIZE_DOC_K[policy],
        doc_spans=doc_spans_all,
        prefix_len=prefix_len_all,
        question_start=q_start_all,
    )
