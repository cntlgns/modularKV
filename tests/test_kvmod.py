"""Correctness tests for src/kvmod (the 5 KV-cache policies).

Two layers:

  * ``test_policy_*`` -- pure-logic asserts on ``build_policy_prefill``
    (no model): position_ids / mask presence / reloc metadata per policy,
    including the reset-to-zero ``modular_q_pos`` variants and the
    seg-0-between-docs guard.

  * ``test_model_*`` -- Llama-3.2-1B smoke (skipped if the model can't be
    loaded). Verifies the consolidated path reproduces the *legacy* eval:
    ``baseline`` == old ``standard`` (full ``model.generate`` greedy,
    token-for-token); ``recover_cross_attn_oracle_pos`` with relocalization
    disabled == ``baseline``; and that the doc-K rotation is exactly
    invertible (rotate local->global recovers baseline K).

Run:  .venv/bin/python -m pytest tests/test_kvmod.py -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kvmod import (  # noqa: E402
    build_policy_prefill,
    build_segmented_inputs,
    generate_for_policy,
)
from src.kvmod.inference import _relocalize_doc_keys, _rotate_half  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Two rows, right-padded. Layout per row:
#   seg 0 = system prefix / question tail, seg k>=1 = doc k, seg -1 = pad.
# Row A (len 11, no pad):  P=3  doc1=2  doc2=4  Qtail=2
# Row B (len 7 + 4 pad):   P=2  doc1=3            Qtail=2
SEG = torch.tensor(
    [
        [0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, -1, -1, -1, -1],
    ],
    dtype=torch.long,
)


def test_policy_baseline():
    pf = build_policy_prefill(SEG, "baseline")
    assert pf.attention_mask_4d is None and not pf.relocalize
    assert torch.equal(pf.position_ids[0], torch.arange(11))
    assert torch.equal(pf.position_ids[1][:7], torch.arange(7))
    assert pf.next_position.tolist() == [11, 7]


def test_policy_recover_pos_enc():
    pf = build_policy_prefill(SEG, "recover_pos_enc")
    # Doc band isolated -> 4-D segment mask; positions stay contiguous.
    assert pf.attention_mask_4d is not None
    assert pf.attention_mask_4d.shape == (2, 1, 11, 11)
    assert torch.equal(pf.position_ids[0], torch.arange(11))
    assert pf.next_position.tolist() == [11, 7]


def test_policy_modular_summed_and_min():
    pf = build_policy_prefill(SEG, "modular", modular_q_pos="summed_pos")
    assert pf.attention_mask_4d is not None  # isolated doc band
    # Row A: sys 0..2 | doc1 0..1 | doc2 0..3 | q at sys_end+sum_doc=3+6=9
    assert pf.position_ids[0].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3, 9, 10]
    assert pf.next_position[0].item() == 11
    # Row B: sys 0..1 | doc1 0..2 | q at 2+3=5
    assert pf.position_ids[1][:7].tolist() == [0, 1, 0, 1, 2, 5, 6]
    assert pf.next_position[1].item() == 7

    pf_min = build_policy_prefill(SEG, "modular", modular_q_pos="min_pos")
    # Row A: q at max(sys_end=3, max_doc_len=4) = 4
    assert pf_min.position_ids[0].tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 5]
    assert pf_min.next_position[0].item() == 6


def test_policy_recover_cross_attn():
    pf = build_policy_prefill(SEG, "recover_cross_attn")
    # Full causal (no 4-D mask) but reset-to-zero positions == modular's.
    assert pf.attention_mask_4d is None and not pf.relocalize
    ref = build_policy_prefill(SEG, "modular")
    assert torch.equal(pf.position_ids, ref.position_ids)
    assert torch.equal(pf.next_position, ref.next_position)


def test_policy_oracle_metadata():
    pf = build_policy_prefill(SEG, "recover_cross_attn_oracle_pos")
    assert pf.relocalize and pf.attention_mask_4d is None
    assert torch.equal(pf.position_ids[0], torch.arange(11))  # contiguous
    assert pf.doc_spans[0] == [(3, 5), (5, 9)]
    assert pf.doc_spans[1] == [(2, 5)]
    assert pf.prefix_len == [3, 2]
    assert pf.question_start == [9, 5]


def test_reset_zero_rejects_seg0_between_docs():
    bad = torch.tensor([[0, 0, 1, 1, 0, 2, 2, 0]], dtype=torch.long)
    build_policy_prefill(bad, "baseline")  # contiguous: fine
    for p in ("modular", "recover_cross_attn"):
        with pytest.raises(ValueError, match="contiguous doc band"):
            build_policy_prefill(bad, p)


def test_crossrepo_policy_spec_parity():
    """KVLink's policy tables must agree with the sibling
    modularization_ablation repo (modularkv.masking) so the two codebases'
    numbers are comparable. Skipped if that repo isn't checked out."""
    mk_src = "/data_fast/home/sihun/kvcache/modularization_ablation/src"
    if not os.path.isdir(mk_src):
        pytest.skip("modularization_ablation repo not present")
    sys.path.insert(0, mk_src)
    try:
        from modularkv import masking as m  # noqa: WPS433
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"modularkv import failed: {e}")

    from src.kvmod import policies as p

    assert set(p.KV_CACHE_POLICIES) == set(m.KV_CACHE_POLICIES)
    assert p.MODULAR_Q_POS_CHOICES == m.MODULAR_Q_POS_CHOICES
    assert set(p.POLICIES_WITH_MODULAR_Q_POS) == set(
        m.POLICIES_WITH_MODULAR_Q_POS
    )
    for pol in p.KV_CACHE_POLICIES:
        # KVLink uses the 4-D segment mask exactly when modularkv isolates the
        # doc band (no cross-doc AND no prefix attention move together).
        assert p._USE_SEGMENT_MASK[pol] == (
            not m._CROSS_DOC_ATTN[pol]
        ), pol
        assert m._CROSS_DOC_ATTN[pol] == m._DOC_ATTENDS_PREFIX[pol], pol
        assert p._DOC_POS_SCHEME[pol] == m._DOC_POS_SCHEME[pol], pol
        assert (
            p._RELOCALIZE_DOC_K[pol]
            == m._POST_PREFILL_DOC_K_LOCAL_ROTATION[pol]
        ), pol


# --------------------------------------------------------------------------- #
# Prompt builder: kvlink preset must reproduce the old hardcoded assembly
# byte-for-byte; short must differ ONLY in the system prefix.
# --------------------------------------------------------------------------- #

_MEM_START, _MEM_END, _SPECIAL = 128254, 128255, 128011


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        t = AutoTokenizer.from_pretrained(MODEL_ID)
        t.pad_token_id = 128004
        t.pad_token = "<|finetune_right_pad_id|>"
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {e}")
    return t


def _legacy_assembly(tokenizer, question, docs, reencode_num):
    """Verbatim pre-refactor preprocess_fn body (kvlink wording)."""
    system = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a intelligent AI assistant. Please answer questions based "
        "on the user's instruction. Below are some reference documents that "
        "may help you in answering the user's question.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    memory_list = [
        f"Document [{j + 1}](Title: {t}) {x}\n" for j, (t, x) in enumerate(docs)
    ]
    ids = tokenizer(system, add_special_tokens=False)["input_ids"] + [_MEM_START]
    seg = [0] * len(ids)
    for mem_id, st in enumerate(memory_list):
        tem = tokenizer(st, add_special_tokens=False)["input_ids"]
        seg = seg + [mem_id + 1] * len(tem) + [0] * reencode_num
        for k in range(reencode_num):
            tem = tem + [_SPECIAL + reencode_num * mem_id + k]
        ids = ids + tem
    pid = tokenizer(question, add_special_tokens=False)["input_ids"]
    ids = ids + [_MEM_END] + pid
    seg = seg + [0] + [0] * len(pid)
    return ids, seg


@pytest.mark.parametrize("reencode_num", [0, 3])
def test_prompt_kvlink_byte_identical_to_legacy(tok, reencode_num):
    docs = [("Sky", "The sky is blue."), ("Sun", "The sun is hot.")]
    q = "What color is the sky?"
    old_ids, old_seg = _legacy_assembly(tok, q, docs, reencode_num)
    new = build_segmented_inputs(
        tok, q, docs, prompt_preset="kvlink", reencode_num=reencode_num,
        mem_start=_MEM_START, mem_end=_MEM_END, special_token_start=_SPECIAL,
    )
    assert new["input_ids"] == old_ids
    assert new["segment_ids"] == old_seg


def test_prompt_short_differs_only_in_system_prefix(tok):
    docs = [("Sky", "The sky is blue."), ("Sun", "The sun is hot.")]
    q = "What color is the sky?"
    a = build_segmented_inputs(
        tok, q, docs, prompt_preset="kvlink", reencode_num=0,
        mem_start=_MEM_START, mem_end=_MEM_END, special_token_start=_SPECIAL,
    )
    b = build_segmented_inputs(
        tok, q, docs, prompt_preset="short", reencode_num=0,
        mem_start=_MEM_START, mem_end=_MEM_END, special_token_start=_SPECIAL,
    )
    # From the first doc token onward (seg >= 1), the two presets must be
    # byte-identical: only the leading system prefix changes.
    da = a["segment_ids"].index(1)
    db = b["segment_ids"].index(1)
    assert a["input_ids"][da:] == b["input_ids"][db:]
    assert a["segment_ids"][da:] == b["segment_ids"][db:]
    # Sanity: the system prefix actually differs (preset took effect).
    assert a["input_ids"][:da] != b["input_ids"][:db]


# --------------------------------------------------------------------------- #
# Model smoke (skipped offline / if the weights are unavailable).
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def model_tok():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        # Mirror the eval scripts' Llama-3.2 pad-token setup.
        tok.pad_token_id = 128004
        tok.pad_token = "<|finetune_right_pad_id|>"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"model unavailable: {e}")
    return model, tok


def _toy_batch(tok, device):
    """Two QA rows in KVLink's segment layout (reencode_num=0)."""
    mem_start, mem_end = 128254, 128255
    sys_txt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
    )
    docs = [
        ["Document [1](Title: Sky) The sky is blue.\n",
         "Document [2](Title: Grass) Grass is green.\n"],
        ["Document [1](Title: Sun) The sun is hot.\n"],
    ]
    qs = ["What color is the sky?", "Is the sun hot?"]
    rows = []
    for dlist, q in zip(docs, qs):
        ids = tok(sys_txt, add_special_tokens=False)["input_ids"] + [mem_start]
        seg = [0] * len(ids)
        for k, d in enumerate(dlist):
            t = tok(d, add_special_tokens=False)["input_ids"]
            ids += t
            seg += [k + 1] * len(t)
        qid = tok(q, add_special_tokens=False)["input_ids"]
        ids += [mem_end] + qid
        seg += [0] * (1 + len(qid))
        rows.append((ids, seg))
    maxlen = max(len(r[0]) for r in rows)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_ids, segment_ids, pad_mask = [], [], []
    for ids, seg in rows:
        r = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * r)
        segment_ids.append(seg + [-1] * r)
        pad_mask.append([1] * len(ids) + [0] * r)
    return (
        torch.tensor(input_ids, device=device),
        torch.tensor(segment_ids, device=device),
        torch.tensor(pad_mask, device=device),
    )


def _legacy_standard(model, tok, input_ids, pad_mask, genprefix, stop_ids):
    """Reproduce the pre-refactor 'standard' path: prefill + model.generate
    greedy + the same decode/split."""
    from transformers import GenerationConfig

    cfg = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=20,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.convert_tokens_to_ids("<|eot_id|>"),
    )
    pre = model(input_ids=input_ids, attention_mask=pad_mask)
    B = input_ids.size(0)
    gp = genprefix.unsqueeze(0).repeat(B, 1)
    gen_in = torch.cat([input_ids, gp], dim=1)
    am = torch.cat([pad_mask, torch.ones_like(gp)], dim=1)
    out = model.generate(
        input_ids=gen_in, attention_mask=am, use_cache=True,
        generation_config=cfg, past_key_values=pre.past_key_values,
        tokenizer=tok,
    )
    seqs = [tok.decode(out[i, input_ids.size(1):].tolist()) for i in range(B)]
    return [
        s.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        .strip().split("<|eot_id|>")[0]
        for s in seqs
    ]


def test_model_baseline_matches_legacy(model_tok):
    model, tok = model_tok
    dev = next(model.parameters()).device
    input_ids, segment_ids, pad_mask = _toy_batch(tok, dev)
    gp = torch.tensor(
        tok("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False)["input_ids"],
        device=dev,
    )
    stop = {tok.convert_tokens_to_ids("<|eot_id|>"),
            tok.convert_tokens_to_ids("<|end_of_text|>")}
    legacy = _legacy_standard(model, tok, input_ids, pad_mask, gp, stop)
    new = generate_for_policy(
        model, tok, input_ids=input_ids, pad_mask=pad_mask,
        segment_ids=segment_ids, policy="baseline",
        generation_token_ids=gp, stop_token_ids=stop, max_new_tokens=20,
    )
    assert new == legacy, f"baseline != legacy standard\nnew={new}\nold={legacy}"


def test_model_oracle_reloc_off_equals_baseline(model_tok):
    model, tok = model_tok
    dev = next(model.parameters()).device
    input_ids, segment_ids, pad_mask = _toy_batch(tok, dev)
    gp = torch.tensor(
        tok("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False)["input_ids"],
        device=dev,
    )
    stop = {tok.convert_tokens_to_ids("<|eot_id|>"),
            tok.convert_tokens_to_ids("<|end_of_text|>")}
    kw = dict(
        model=model, tokenizer=tok, input_ids=input_ids, pad_mask=pad_mask,
        segment_ids=segment_ids, generation_token_ids=gp,
        stop_token_ids=stop, max_new_tokens=20,
    )
    base = generate_for_policy(policy="baseline", **kw)
    oracle_off = generate_for_policy(
        policy="recover_cross_attn_oracle_pos",
        disable_doc_k_relocalization=True, **kw,
    )
    assert oracle_off == base, (
        f"oracle(reloc off) != baseline\n{oracle_off}\n{base}"
    )


def test_doc_k_rotation_is_invertible(model_tok):
    """rotate global->local then local->global must recover the original K."""
    model, tok = model_tok
    dev = next(model.parameters()).device
    input_ids, _, pad_mask = _toy_batch(tok, dev)
    from transformers import DynamicCache

    row = input_ids[:1, : int(pad_mask[0].sum())]
    out = model(input_ids=row, past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    # Check the rotation MATH in float32 (bf16 double-rotation rounding ~4 ulps
    # is precision, not logic; the end-to-end oracle correctness is covered by
    # test_model_oracle_reloc_off_equals_baseline). modularkv checks this in
    # float32 too.
    cache.key_cache = [k.float() for k in cache.key_cache]
    cache.value_cache = [v.float() for v in cache.value_cache]
    k0 = [k.clone() for k in cache.key_cache]
    v0 = [v.clone() for v in cache.value_cache]
    spans = [(3, 6), (6, 10)]
    L = row.size(1)
    pos_g = torch.arange(L, device=dev).unsqueeze(0)

    _relocalize_doc_keys(cache, spans, pos_g, model)
    # Values untouched; doc K changed.
    for a, b in zip(cache.value_cache, v0):
        assert torch.equal(a, b)
    assert not torch.allclose(cache.key_cache[0], k0[0])

    # Inverse rotation (swap p_local<->p_global by negating the delta): a
    # second call with positions chosen so delta = +doc_global_start.
    rotary = model.model.rotary_emb
    for k in cache.key_cache:
        for ds, de in spans:
            p_l = torch.arange(de - ds, device=dev).unsqueeze(0)
            p_g = pos_g[:, ds:de]
            delta = (p_g - p_l).to(torch.long)  # inverse of the local move
            cos, sin = rotary(k, delta)
            r = cos.shape[-1]
            cb = cos.to(k.dtype).unsqueeze(1)
            sb = sin.to(k.dtype).unsqueeze(1)
            sl = k[:, :, ds:de, :r]
            k[:, :, ds:de, :r] = sl * cb + _rotate_half(sl) * sb
    for a, b in zip(cache.key_cache, k0):
        assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
