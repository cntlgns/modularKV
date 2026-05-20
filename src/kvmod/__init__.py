"""KV-cache modularization policies for KVLink eval.

Single source of truth for the 5 KV-cache policies probed across the
nq/hqa/wiki/musique eval scripts. The names and semantics are kept identical
to the sibling ``modularization_ablation`` repo (``src/modularkv``) so the two
codebases' numbers are directly comparable:

    baseline                       full prefill (causal, contiguous positions)
    recover_pos_enc                doc band isolated, contiguous positions
    modular                        doc band isolated, per-doc reset positions
    recover_cross_attn             full causal, per-doc reset positions
    recover_cross_attn_oracle_pos  baseline prefill, then cached doc-K rotated
                                   to per-doc local RoPE positions

Legacy ``--attn_type`` mapping (pre-refactor): ``standard`` == ``baseline``,
``blocked`` == ``recover_pos_enc``.
"""

from .policies import (
    KV_CACHE_POLICIES,
    MODULAR_Q_POS_CHOICES,
    POLICIES_WITH_MODULAR_Q_POS,
    PolicyPrefill,
    build_policy_prefill,
)
from .inference import generate_for_policy
from .prompt import (
    PROMPT_PRESETS,
    SYSTEM_PROMPTS,
    build_segmented_inputs,
)

__all__ = [
    "KV_CACHE_POLICIES",
    "MODULAR_Q_POS_CHOICES",
    "POLICIES_WITH_MODULAR_Q_POS",
    "PolicyPrefill",
    "build_policy_prefill",
    "generate_for_policy",
    "PROMPT_PRESETS",
    "SYSTEM_PROMPTS",
    "build_segmented_inputs",
]
