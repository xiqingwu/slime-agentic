"""Path-independent MAT rewards and evaluation metrics."""

from __future__ import annotations

import re
from collections import Counter

from .protocol import STEP_PROBLEM, classify_step, extract_answer, extract_problems


def normalize(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^\w\s.-]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize(prediction) == normalize(answer))


def token_f1(prediction: str, answer: str) -> float:
    pred = normalize(prediction).split()
    gold = normalize(answer).split()
    if not pred or not gold:
        return float(pred == gold)
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, answers) -> dict[str, float]:
    if not isinstance(answers, list):
        answers = [answers]
    answers = [str(x) for x in answers if x is not None] or [""]
    return {
        "f1": max(token_f1(prediction, x) for x in answers),
        "em": max(exact_match(prediction, x) for x in answers),
    }


def aggregate_reward(final_output: str, planner_steps: list[dict], corruption_gt,
                     code_exec_oks: list[bool], answers, baseline_output: str | None = None) -> dict[str, float]:
    answer_score = score_answer(extract_answer(final_output), answers)
    typed = [step for step in planner_steps if step.get("type")]
    format_score = sum(classify_step(step.get("content", "")) is not None for step in typed) / max(len(typed), 1)
    problem_steps = [step for step in typed if step.get("type") == STEP_PROBLEM]
    expected = sorted(str(x).lower() for x in (corruption_gt or []))
    diagnosis = float(bool(problem_steps) and extract_problems(problem_steps[0].get("content", "")) == expected)
    if code_exec_oks:
        code_exec = sum(bool(x) for x in code_exec_oks) / len(code_exec_oks)
    else:
        code_exec = float(expected == ["none"])
    correction = 0.0
    if baseline_output is not None:
        baseline_em = score_answer(extract_answer(baseline_output), answers)["em"]
        if baseline_em == 0 and answer_score["em"] == 1:
            correction = 1.0
        elif baseline_em == 1 and answer_score["em"] == 0:
            correction = -1.0
    total = answer_score["f1"] + 0.05 * format_score + 0.1 * diagnosis + 0.05 * code_exec + 0.5 * correction
    return {
        "score": total,
        "outcome": answer_score["f1"],
        "em": answer_score["em"],
        "format": format_score,
        "diagnosis": diagnosis,
        "code_exec": code_exec,
        "correction": correction,
    }
