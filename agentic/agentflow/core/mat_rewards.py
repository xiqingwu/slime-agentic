"""Path-independent verifiable rewards for the MAT-Coding visual agentic task.

This module is the M1 ("Reward 先行") deliverable of
`docs/algorithms/plan_agentflow_visual.md` (方案 C, §4.0).

The four primitives copied verbatim from Visual-ARFT
(`Liuziyu77/Visual-RFT`, `Visual-ARFT/src/visual_arft/src/open_r1/grpo_agent_code.py`)
are: ``normalize``, ``compute_f1``, ``exact_match_score`` and ``extract_problems``.
They are kept byte-for-byte so our F1/EM/diagnosis numbers stay comparable to the
baseline.

The reward *functions* (``format_reward_step``, ``diagnosis_reward``,
``outcome_reward``, ``code_exec_reward``) are **adapted** from Visual-ARFT's
``format_reward`` / ``accuracy_reward``. The key difference (see plan §1.2.1):
Visual-ARFT branches on a per-step ground-truth ``solution`` (only valid for its
offline step-pre-split training). AgentFlow runs the model online and the model
walks its own path, so we make every signal **path-independent**: it depends only
on the model's emitted content plus trajectory-level GTs (the corruption label and
the final answer), never on a teacher-written per-step solution.
"""

from __future__ import annotations

import re
import string

# ── 1. Verbatim primitives from Visual-ARFT (do not edit — keep comparable) ──────

# Diagnosis label vocabulary, from SYSTEM_PROMPT_AGENT_CODE.
CORRUPTION_LABELS = (
    "rotation90", "rotation180", "dark", "overexposure",
    "blur", "noise", "crop", "none",
)


def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))


def extract_problems(text):
    match = re.search(r"<problem>\s*\{(.*?)\}\s*</problem>", text, re.DOTALL)
    if not match:
        return []

    content = match.group(1)
    # 提取所有用英文单引号包裹的单词
    problems = re.findall(r"'([^']+)'", content)
    return sorted(problems)


# ── 2. Step-format regexes (verbatim from Visual-ARFT format_reward) ─────────────

_PATTERN_ANSWER = r"<think>.*?</think>\s*<answer>.*?</answer>"
_PATTERN_CODE = r"<think>.*?</think>\s*<code>\s*```python(.*?)```.*?</code>"
_PATTERN_PROBLEM = (
    r"^<think>.*?</think>\s*<problem>\s*\{\s*'[^']+'\s*"
    r"(?:,\s*'[^']+'\s*)*\}\s*</problem>$"
)

# What kind of step did the model emit?  In online rollout there is no per-step GT
# step type, so we infer it from the content itself.
STEP_PROBLEM = "problem"
STEP_CODE = "code"
STEP_ANSWER = "answer"


def classify_step(content: str) -> str | None:
    """Return which single step type the content emitted, or None if ambiguous.

    Ambiguous = zero closing tags present, or more than one *distinct* type
    present (e.g. both <code> and <answer>) — both are malformed and score 0 on
    format.
    """
    present = [
        t for t, tag in (
            (STEP_PROBLEM, "<problem>"),
            (STEP_CODE, "<code>"),
            (STEP_ANSWER, "<answer>"),
        ) if tag in content
    ]
    if len(present) != 1:
        return None
    return present[0]


def format_reward_step(content: str) -> float:
    """Path-independent version of Visual-ARFT's format_reward.

    1.0 if the content matches *exactly one* valid step format
    (think+problem / think+code / think+answer), else 0.0.

    The repeated-tag penalty (any tag appearing >= 2 times -> 0.0) is preserved
    verbatim from the original.
    """
    if (
        content.count("<answer>") >= 2
        or content.count("<code>") >= 2
        or content.count("<think>") >= 2
        or content.count("<problem>") >= 2
    ):
        return 0.0

    step = classify_step(content)
    if step == STEP_ANSWER:
        return 1.0 if re.fullmatch(_PATTERN_ANSWER, content, re.DOTALL) else 0.0
    if step == STEP_CODE:
        return 1.0 if re.fullmatch(_PATTERN_CODE, content, re.DOTALL) else 0.0
    if step == STEP_PROBLEM:
        return 1.0 if re.fullmatch(_PATTERN_PROBLEM, content, re.DOTALL) else 0.0
    return 0.0


# ── 3. Path-independent reward signals (plan §4.0) ───────────────────────────────


def diagnosis_reward(content: str, corruption_gt) -> float:
    """Reward the <problem> diagnosis step against the *known* corruption label(s).

    ``corruption_gt`` is trajectory-level and known by construction (the
    corruption we programmatically applied / the original Visual-ARFT <problem>
    GT), so this is path-independent.

    Scoring is copied from Visual-ARFT accuracy_reward's <problem> branch:
    all-correct -> 1.0, rotation90/180 half-match -> 0.5, else 0.0.
    """
    ground_truth = sorted(corruption_gt) if not isinstance(corruption_gt, str) else sorted([corruption_gt])
    student_answer = extract_problems(content)

    reward = 0.0
    # Half correct (rotation only matched partially)
    if len(ground_truth) == 2 and "rotation90" in ground_truth:
        if len(student_answer) == 1 and "rotation90" in student_answer:
            reward = 0.5
    elif len(ground_truth) == 2 and "rotation180" in ground_truth:
        if len(student_answer) == 1 and "rotation180" in student_answer:
            reward = 0.5
    # All correct
    if reward == 0:
        reward = 1.0 if ground_truth == student_answer else 0.0
    return reward


def outcome_reward(final_answer: str, gt: str) -> float:
    """Credit the final answer against the trajectory-level ground truth.

    Uses ``max(F1, EM)`` rather than F1 alone. Visual-ARFT's ``normalize`` strips
    the articles a/an/the, which collapses single-letter multiple-choice answers
    like "A" to the empty string and makes ``compute_f1("A", "A") == 0`` (the
    plan §8 known pitfall — already handled on the *eval* side, but the training
    reward used raw F1 and inherited the bug). F1 stays the comparable primary
    signal; EM rescues those exact-match cases so a correct MC answer is never
    scored 0 during training.
    """
    if not final_answer or gt is None:
        return 0.0
    answer = _extract_answer_text(final_answer)
    gt = str(gt)
    return max(compute_f1(answer, gt), float(exact_match_score(answer, gt)))


def _extract_answer_text(content: str) -> str:
    """Pull the text inside <answer>...</answer>, else use the content as-is."""
    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    return m.group(1).strip() if m else content.strip()


def code_exec_reward(exec_ok: bool | None) -> float:
    """Reward a <code> step for actually executing and producing an output image.

    M1 stub: the real signal comes from the OpenCV tool's exec-success flag in M3
    (plan §4.0 / §5.3). Until then ``exec_ok`` is None and this returns 0.0.

    +1.0 when the cv2 code ran and emitted an output image, 0.0 on error/no output.
    This is strictly stronger than Visual-ARFT, which never executes code during
    training (it hard-codes the code step to 0.9).
    """
    if exec_ok is None:
        return 0.0
    return 1.0 if exec_ok else 0.0


# ── 4. Trajectory-level aggregation (plan §4.0) ──────────────────────────────────

# Default weights for combining the path-independent signals into one scalar the
# AgentFlow reward-amortization machinery (custom_convert) spreads across turns.
# Outcome is primary; format/diagnosis/code-exec are dense shaping and are kept
# *small* on purpose: they vary across rollouts of the same prompt, so under GRPO
# they enter the advantage directly. With the old (0.2/0.3/0.2) weights a wrong
# answer could still bank ~0.7 from shaping alone and steer the policy toward
# "well-formatted, tool-poking" behaviour instead of solving. Total shaping is now
# capped at 0.2 (« outcome 1.0) so it densifies the signal without dominating it.
DEFAULT_WEIGHTS = {
    "outcome": 1.0,
    "format": 0.05,
    "diagnosis": 0.1,
    "code_exec": 0.05,
    # Correction-targeted shaping (plan §1.3 / §2.1) — the *mechanism* behind the
    # thesis "fewer ineffective tool calls for higher F1". It compares the final
    # (with-tool) answer to a no-tool baseline on the same image: rescuing a failure
    # earns +1, breaking a success earns -1 (it can go negative — that's the point;
    # GRPO handles it). Weighted substantially so the policy is steered toward
    # *selective successful correction* rather than poking the tool indiscriminately.
    "correction": 0.5,
}


def correction_reward(final_answer: str, baseline_answer: str, gt: str) -> float:
    """+1 if the tool rescued a failure, -1 if it broke a success, else 0 (EM-based).

    ``baseline_answer`` is the model's no-tool answer on the same corrupted image
    (see ``mat_solver.baseline_answer``). Uses EM to match the eval Call Gain/Harm
    definition so the training signal and the reported metric agree.
    """
    if not gt or baseline_answer is None:
        return 0.0
    gt = str(gt)
    base_ok = bool(exact_match_score(_extract_answer_text(baseline_answer), gt))
    final_ok = bool(exact_match_score(_extract_answer_text(final_answer or ""), gt))
    if final_ok and not base_ok:
        return 1.0          # Call Gain: tool rescued a failure
    if base_ok and not final_ok:
        return -1.0         # Call Harm: tool broke a success
    return 0.0


def _is_none_corruption(corruption_gt) -> bool:
    """True when the trajectory's ground-truth corruption is exactly 'none'."""
    if corruption_gt is None:
        return False
    labels = [corruption_gt] if isinstance(corruption_gt, str) else list(corruption_gt)
    return labels == ["none"]


def aggregate_mat_reward(
    final_output: str,
    gt: str,
    planner_steps: list[dict] | None = None,
    corruption_gt=None,
    code_exec_oks: list[bool] | None = None,
    baseline_answer: str | None = None,
    weights: dict | None = None,
) -> dict:
    """Combine the path-independent signals into a single trajectory reward.

    Inputs (all trajectory-level, model-path-independent):
    - ``final_output``: the planner's final consolidated answer (for outcome F1).
    - ``gt``: trajectory ground-truth answer (``sample.label``).
    - ``planner_steps``: list of ``{"type": <problem|code|answer>, "content": str}``
      for each Planner turn — populated by the online solver in M3. Used for the
      mean format reward and to find the diagnosis (<problem>) step.
    - ``corruption_gt``: known corruption label(s) for the diagnosis reward.
    - ``code_exec_oks``: per code-step exec-success flags (M3 tool).
    - ``baseline_answer``: the model's no-tool answer on the same image (A4). When
      present, adds the correction-targeted term; when None it simply contributes 0.

    Components that lack their inputs contribute 0.0, so this gracefully degrades to
    an outcome-only reward. Returns a dict with each component, the weighted
    ``score``, and ``acc`` (EM of the final answer).
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    steps = planner_steps or []

    outcome = outcome_reward(final_output, gt)

    fmt_scores = [format_reward_step(s.get("content", "")) for s in steps]
    fmt = sum(fmt_scores) / len(fmt_scores) if fmt_scores else 0.0

    diag = 0.0
    if corruption_gt is not None:
        problem_steps = [s for s in steps if s.get("type") == STEP_PROBLEM]
        if problem_steps:
            diag = diagnosis_reward(problem_steps[0].get("content", ""), corruption_gt)

    exec_scores = [code_exec_reward(ok) for ok in (code_exec_oks or [])]
    if exec_scores:
        code_exec = sum(exec_scores) / len(exec_scores)
    elif _is_none_corruption(corruption_gt):
        # No code steps because none were needed: 'none' is the correct no-op path
        # (the solver's tip tells the model to answer directly). An empty mean would
        # score this 0.0 and make a correct no-code trajectory *lose* to one that
        # ran a pointless cv2 call — so credit it instead.
        code_exec = 1.0
    else:
        code_exec = 0.0

    # acc uses EM (not F1 == 1.0): free-form answers rarely hit F1 == 1.0, and the
    # normalize-strips-"A" issue (see outcome_reward) would make F1-based acc lie.
    answer_text = _extract_answer_text(final_output) if final_output else ""
    acc = bool(gt) and bool(exact_match_score(answer_text, str(gt)))

    correction = correction_reward(final_output, baseline_answer, gt) if baseline_answer is not None else 0.0

    score = (
        w["outcome"] * outcome
        + w["format"] * fmt
        + w["diagnosis"] * diag
        + w["code_exec"] * code_exec
        + w["correction"] * correction
    )
    return {
        "score": score,
        "acc": acc,
        "outcome": outcome,
        "format": fmt,
        "diagnosis": diag,
        "code_exec": code_exec,
        "correction": correction,
    }


async def mat_reward_func(args, sample, **kwargs) -> dict:
    """Slime ``--custom-rm-path`` entry point for MAT-Coding (rule-based, no judge).

    Reads the trajectory-level fields the solver/data-conversion populate:
    ``sample.label`` (gt), ``sample.metadata['final_output']``,
    ``sample.metadata['planner_steps']``, ``sample.metadata['corruption_gt']``,
    ``sample.metadata['code_exec_oks']``, ``sample.metadata['baseline_output']``
    (the no-tool answer, for the A4 correction term). All are optional except the
    answer, so this works end-to-end as soon as M3 fills them in.
    """
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = meta.get("final_output", "") or getattr(sample, "response", "") or ""
    gt = str(sample.label) if getattr(sample, "label", None) is not None else ""

    result = aggregate_mat_reward(
        final_output=final_output,
        gt=gt,
        planner_steps=meta.get("planner_steps"),
        corruption_gt=meta.get("corruption_gt"),
        code_exec_oks=meta.get("code_exec_oks"),
        baseline_answer=meta.get("baseline_output"),
    )
    result["gt"] = gt
    return result
