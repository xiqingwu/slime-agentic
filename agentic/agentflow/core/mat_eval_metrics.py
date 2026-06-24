"""Evaluation metrics for MAT-Coding (plan §8).

Main metrics  : F1 / EM, reported overall and per split (simple / hard).
Mechanism     : Call Gain / Call Harm (from the disentangling paper arXiv:2602.01334)
                — does the tool *selectively* help? Gain = was wrong without the
                tool, right with it; Harm = was right without it, wrong with it.

All pure functions, unit-testable offline. F1/EM reuse the verbatim Visual-ARFT
primitives from ``mat_rewards`` so numbers stay comparable to the baseline.
Predictions are scored as the **max over all valid answers** (benchmark items can
list several acceptable answers).
"""

from __future__ import annotations

from .mat_rewards import compute_f1, exact_match_score


def score_one(pred: str, answers) -> dict:
    """F1 / EM of a prediction against one-or-many valid answers (max over answers)."""
    if isinstance(answers, str):
        answers = [answers]
    answers = [a for a in (answers or []) if a is not None]
    if not answers:
        return {"f1": 0.0, "em": 0}
    f1 = max(compute_f1(pred or "", str(a)) for a in answers)
    em = max(exact_match_score(pred or "", str(a)) for a in answers)
    return {"f1": f1, "em": int(em)}


def aggregate(records: list[dict]) -> dict:
    """Aggregate per-item {f1, em, split} into overall + per-split means.

    Returns {"overall": {...}, "simple": {...}, "hard": {...}} where each block is
    {"f1": mean, "em": mean, "n": count}. Splits with no items are omitted.
    """
    def _mean(items, key):
        return sum(it[key] for it in items) / len(items) if items else 0.0

    result = {
        "overall": {"f1": _mean(records, "f1"), "em": _mean(records, "em"), "n": len(records)},
    }
    for split in ("simple", "hard"):
        sub = [r for r in records if r.get("split") == split]
        if sub:
            result[split] = {"f1": _mean(sub, "f1"), "em": _mean(sub, "em"), "n": len(sub)}
    return result


def call_gain_harm(pairs: list[tuple[bool, bool]]) -> dict:
    """Call Gain / Call Harm from (baseline_correct, tool_correct) per item.

    - gain  = #(baseline wrong, tool right)   -> tool rescued a failure
    - harm  = #(baseline right, tool wrong)   -> tool broke a success
    - net   = gain - harm
    - gain_rate = gain / #(baseline wrong)    -> rescue rate among failures
    - harm_rate = harm / #(baseline right)    -> break rate among successes

    Correctness is the caller's call (typically EM == 1).
    """
    n = len(pairs)
    gain = sum(1 for b, t in pairs if (not b) and t)
    harm = sum(1 for b, t in pairs if b and (not t))
    baseline_wrong = sum(1 for b, _ in pairs if not b)
    baseline_right = sum(1 for b, _ in pairs if b)
    return {
        "n": n,
        "gain": gain,
        "harm": harm,
        "net": gain - harm,
        "gain_rate": gain / baseline_wrong if baseline_wrong else 0.0,
        "harm_rate": harm / baseline_right if baseline_right else 0.0,
        "baseline_acc": baseline_right / n if n else 0.0,
        "tool_acc": sum(1 for _, t in pairs if t) / n if n else 0.0,
    }
