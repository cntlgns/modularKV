"""Short-answer QA metrics with SQuAD-style normalization.

Consolidated here from the previously-duplicated inline copies in
``scripts/evaluation/{nq,hqa,wiki,musique}_eval.py``. ``normalize_answer`` and
``best_subspan_em`` are byte-for-byte the original implementation (same
``regex`` module, same article/punctuation handling), so historical
``accuracy`` numbers are unchanged. ``exact_match`` / ``f1_score`` are added
for richer reporting (substring EM is only an upper bound on real accuracy).
"""

from __future__ import annotations

import string
from collections import Counter
from typing import Dict, List

import regex


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    """1.0 if any normalized reference is a substring of the normalized prediction.

    LongBench / "Lost in the Middle" style. Identical to the original inline
    implementation -- this is the metric reported as ``accuracy``.
    """
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


# Name used in the broader literature / the modularization_ablation package.
substring_match = best_subspan_em


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    """1.0 if the normalized prediction exactly equals any normalized reference."""
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(g) for g in ground_truths))


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Token-level SQuAD F1, taking the max over the reference list."""
    pt = normalize_answer(prediction).split()
    best = 0.0
    for g in ground_truths:
        gt = normalize_answer(g).split()
        if not pt and not gt:
            best = max(best, 1.0)
            continue
        if not pt or not gt:
            continue
        common = Counter(pt) & Counter(gt)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        precision = n_same / len(pt)
        recall = n_same / len(gt)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def score_sample(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    """All three metrics for one (prediction, references) pair."""
    return {
        "substring": best_subspan_em(prediction, ground_truths),
        "em": exact_match(prediction, ground_truths),
        "f1": f1_score(prediction, ground_truths),
    }
