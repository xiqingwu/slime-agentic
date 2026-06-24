"""Offline unit tests for the MAT-Coding reward module (plan M1).

Run standalone (no pytest needed):
    python agentic/agentflow/tests/test_mat_rewards.py
Or with pytest:
    pytest agentic/agentflow/tests/test_mat_rewards.py
"""

import sys
from pathlib import Path

# Allow `from core.mat_rewards import ...` exactly like rollout.py does (cwd-agnostic).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

from core.mat_rewards import (  # noqa: E402
    compute_f1,
    exact_match_score,
    extract_problems,
    classify_step,
    format_reward_step,
    diagnosis_reward,
    outcome_reward,
    code_exec_reward,
    correction_reward,
    aggregate_mat_reward,
    mat_reward_func,
    weights_from_env,
    reward_component_means,
    mechanism_means,
    DEFAULT_WEIGHTS,
)
import os  # noqa: E402

THINK = "<think> reasoning here </think>"


# ── Verbatim primitives ──────────────────────────────────────────────────────────

def test_compute_f1():
    assert compute_f1("the cat", "the cat") == 1.0          # identical
    assert compute_f1("", "anything") == 0.0                # no overlap
    # articles/punctuation/case normalized away -> still perfect
    assert compute_f1("The Cat.", "a cat") == 1.0
    # partial overlap is between 0 and 1
    f1 = compute_f1("red big dog", "big dog")
    assert 0.0 < f1 < 1.0


def test_exact_match_score():
    assert exact_match_score("The Answer.", "answer") == 1
    assert exact_match_score("nope", "answer") == 0
    assert exact_match_score(None, "answer") == 0.0


def test_extract_problems_sorted_and_deduped_format():
    text = "<problem> {'noise', 'dark'} </problem>"
    assert extract_problems(text) == ["dark", "noise"]      # sorted
    assert extract_problems("no tags here") == []


# ── classify_step ────────────────────────────────────────────────────────────────

def test_classify_step():
    assert classify_step(f"{THINK} <problem> {{'dark'}} </problem>") == "problem"
    assert classify_step(f"{THINK} <code> x </code>") == "code"
    assert classify_step(f"{THINK} <answer> 42 </answer>") == "answer"
    # ambiguous: two distinct step types -> None
    assert classify_step(f"{THINK} <code>x</code><answer>y</answer>") is None
    # none present -> None
    assert classify_step(THINK) is None


# ── format_reward_step ───────────────────────────────────────────────────────────

def test_format_reward_valid_each_type():
    problem = f"{THINK}\n<problem> {{'dark'}} </problem>"
    code = f"{THINK}\n<code>\n```python\nimport cv2\n```\n</code>"
    answer = f"{THINK}\n<answer> 42 </answer>"
    assert format_reward_step(problem) == 1.0
    assert format_reward_step(code) == 1.0
    assert format_reward_step(answer) == 1.0


def test_format_reward_repeat_tag_penalty():
    dup = f"{THINK}<answer>a</answer><answer>b</answer>"
    assert format_reward_step(dup) == 0.0


def test_format_reward_malformed():
    assert format_reward_step("<answer> no think tag </answer>") == 0.0
    assert format_reward_step(f"{THINK} plain text, no step tag") == 0.0
    # code step missing the ```python fenced block
    assert format_reward_step(f"{THINK}\n<code> not fenced </code>") == 0.0


# ── diagnosis_reward (path-independent, uses corruption_gt) ───────────────────────

def test_diagnosis_all_correct():
    content = "<problem> {'dark', 'noise'} </problem>"
    assert diagnosis_reward(content, ["noise", "dark"]) == 1.0


def test_diagnosis_single_label_string_gt():
    content = "<problem> {'blur'} </problem>"
    assert diagnosis_reward(content, "blur") == 1.0


def test_diagnosis_rotation_half_credit():
    # GT has two labels incl rotation90; student names only rotation90 -> 0.5
    content = "<problem> {'rotation90'} </problem>"
    assert diagnosis_reward(content, ["rotation90", "dark"]) == 0.5
    content180 = "<problem> {'rotation180'} </problem>"
    assert diagnosis_reward(content180, ["rotation180", "blur"]) == 0.5


def test_diagnosis_wrong():
    content = "<problem> {'blur'} </problem>"
    assert diagnosis_reward(content, ["dark"]) == 0.0


# ── outcome_reward ───────────────────────────────────────────────────────────────

def test_outcome_reward():
    final = f"{THINK}\n<answer> a red apple </answer>"
    assert outcome_reward(final, "red apple") == 1.0   # F1=1 after normalize
    assert outcome_reward(final, "blue car") == 0.0
    # falls back to whole content when no <answer> tag
    assert outcome_reward("red apple", "red apple") == 1.0
    assert outcome_reward("", "x") == 0.0


def test_outcome_reward_single_letter_mc():
    # normalize() strips a/an/the, so raw F1("A","A") == 0; max(F1, EM) rescues
    # exact multiple-choice matches (plan §8 pitfall — now fixed on the train side).
    assert outcome_reward("<answer> A </answer>", "A") == 1.0
    assert outcome_reward("<answer> a </answer>", "a") == 1.0
    assert outcome_reward("<answer> the </answer>", "the") == 1.0
    assert outcome_reward("<answer> B </answer>", "A") == 0.0   # wrong choice still 0


# ── code_exec_reward (M1 stub) ───────────────────────────────────────────────────

def test_code_exec_reward_stub():
    assert code_exec_reward(None) == 0.0   # M1: real signal arrives in M3
    assert code_exec_reward(True) == 1.0
    assert code_exec_reward(False) == 0.0


# ── correction_reward (A4: Call Gain / Call Harm shaping) ────────────────────────

def test_correction_reward():
    gt = "42"
    wrong, right = "<answer> nope </answer>", "<answer> 42 </answer>"
    # baseline wrong, tool right -> rescued -> +1
    assert correction_reward(final_answer=right, baseline_answer=wrong, gt=gt) == 1.0
    # baseline right, tool wrong -> broke -> -1
    assert correction_reward(final_answer=wrong, baseline_answer=right, gt=gt) == -1.0
    # both right or both wrong -> 0 (tool made no difference)
    assert correction_reward(right, right, gt) == 0.0
    assert correction_reward(wrong, wrong, gt) == 0.0
    # no gt -> 0
    assert correction_reward(right, wrong, "") == 0.0


def test_aggregate_correction_in_score():
    # tool rescued a failure: outcome 1.0 + 0.5*(+1) correction (+ shaping)
    gain = aggregate_mat_reward(
        "<answer> 42 </answer>", "42", baseline_answer="<answer> nope </answer>",
    )
    assert gain["correction"] == 1.0
    assert gain["score"] == DEFAULT_WEIGHTS["outcome"] * 1.0 + DEFAULT_WEIGHTS["correction"] * 1.0

    # tool broke a success: outcome 0.0 + 0.5*(-1) -> net negative
    harm = aggregate_mat_reward(
        "<answer> nope </answer>", "42", baseline_answer="<answer> 42 </answer>",
    )
    assert harm["correction"] == -1.0
    assert harm["score"] < 0.0

    # no baseline -> correction omitted (0), score unchanged from outcome-only
    none = aggregate_mat_reward("<answer> 42 </answer>", "42")
    assert none["correction"] == 0.0


# ── aggregate_mat_reward ─────────────────────────────────────────────────────────

def _good_steps():
    return [
        {"type": "problem", "content": "<think>t</think>\n<problem> {'dark'} </problem>"},
        {"type": "code", "content": "<think>t</think>\n<code>\n```python\nimport cv2\n```\n</code>"},
        {"type": "answer", "content": "<think>t</think>\n<answer> red apple </answer>"},
    ]


def test_aggregate_degrades_to_outcome_only():
    # No steps / corruption / exec flags (pre-M3): only outcome contributes.
    r = aggregate_mat_reward("<answer> red apple </answer>", "red apple")
    assert r["outcome"] == 1.0
    assert r["format"] == 0.0 and r["diagnosis"] == 0.0 and r["code_exec"] == 0.0
    assert r["score"] == DEFAULT_WEIGHTS["outcome"] * 1.0
    assert r["acc"] is True


def test_aggregate_full_signals():
    r = aggregate_mat_reward(
        final_output="<answer> red apple </answer>",
        gt="red apple",
        planner_steps=_good_steps(),
        corruption_gt=["dark"],
        code_exec_oks=[True],
    )
    assert r["outcome"] == 1.0
    assert r["format"] == 1.0       # all three steps well-formed
    assert r["diagnosis"] == 1.0    # dark == dark
    assert r["code_exec"] == 1.0
    w = DEFAULT_WEIGHTS
    expected = w["outcome"] + w["format"] + w["diagnosis"] + w["code_exec"]
    assert abs(r["score"] - expected) < 1e-9


def test_aggregate_partial_format():
    steps = _good_steps()
    steps[1]["content"] = "<think>t</think> malformed code step"  # 1 of 3 bad
    r = aggregate_mat_reward(
        "<answer> red apple </answer>", "red apple",
        planner_steps=steps, corruption_gt=["dark"], code_exec_oks=[False],
    )
    assert abs(r["format"] - (2 / 3)) < 1e-9
    assert r["code_exec"] == 0.0    # exec failed


def test_shaping_cannot_outweigh_outcome():
    # Wrong answer, but perfect format + diagnosis + a real executed edit...
    wrong = aggregate_mat_reward(
        "<answer> wrong </answer>", "42",
        planner_steps=_good_steps(), corruption_gt=["dark"], code_exec_oks=[True],
    )
    # ...still scores below a bare correct answer carrying no shaping at all.
    correct_bare = aggregate_mat_reward("<answer> 42 </answer>", "42")
    assert wrong["outcome"] == 0.0
    assert wrong["score"] < correct_bare["score"]
    assert wrong["score"] <= 0.2 + 1e-9          # total shaping capped at 0.2


def test_aggregate_none_case_no_code_not_penalized():
    none_steps = [
        {"type": "problem", "content": "<think>t</think>\n<problem> {'none'} </problem>"},
        {"type": "answer", "content": "<think>t</think>\n<answer> 42 </answer>"},
    ]
    # 'none' = no fix needed; emitting no code is correct, so code_exec is credited.
    r = aggregate_mat_reward("<answer> 42 </answer>", "42",
                             planner_steps=none_steps, corruption_gt=["none"], code_exec_oks=[])
    assert r["code_exec"] == 1.0
    # but a real-corruption trajectory that ran no code is still penalized.
    r2 = aggregate_mat_reward("<answer> 42 </answer>", "42",
                              planner_steps=none_steps, corruption_gt=["dark"], code_exec_oks=[])
    assert r2["code_exec"] == 0.0


def test_aggregate_acc_uses_em():
    # F1 == 1.0 would be False here (normalize strips "A"), but EM-based acc is True.
    r = aggregate_mat_reward("<answer> A </answer>", "A")
    assert r["acc"] is True
    assert r["outcome"] == 1.0
    # genuinely wrong answer -> acc False
    assert aggregate_mat_reward("<answer> B </answer>", "A")["acc"] is False


def test_mat_reward_func_with_fake_sample():
    class FakeSample:
        label = "red apple"
        response = ""
        metadata = {
            "final_output": "<answer> red apple </answer>",
            "planner_steps": _good_steps(),
            "corruption_gt": ["dark"],
            "code_exec_oks": [True],
        }

    out = asyncio.run(mat_reward_func(args=None, sample=FakeSample()))
    assert out["acc"] is True
    assert out["gt"] == "red apple"
    assert out["score"] > 1.0       # outcome + shaping


def test_mat_reward_func_reads_baseline_output():
    class FakeSample:
        label = "red apple"
        response = ""
        metadata = {
            "final_output": "<answer> red apple </answer>",   # tool got it right
            "baseline_output": "<answer> blue car </answer>",  # no-tool was wrong
            "planner_steps": _good_steps(),
            "corruption_gt": ["dark"],
            "code_exec_oks": [True],
        }

    out = asyncio.run(mat_reward_func(args=None, sample=FakeSample()))
    assert out["correction"] == 1.0          # rescued -> Call Gain
    assert out["score"] > DEFAULT_WEIGHTS["outcome"] + DEFAULT_WEIGHTS["correction"] - 1e-9


# ── env-configurable weights + logging helpers (reward polish) ───────────────────

def test_weights_from_env():
    assert weights_from_env() == DEFAULT_WEIGHTS          # no env -> defaults
    os.environ["MAT_W_CORRECTION"] = "0.8"
    os.environ["MAT_W_OUTCOME"] = "not_a_number"          # bad value ignored
    try:
        w = weights_from_env()
        assert w["correction"] == 0.8
        assert w["outcome"] == DEFAULT_WEIGHTS["outcome"]  # unchanged (bad value)
    finally:
        del os.environ["MAT_W_CORRECTION"], os.environ["MAT_W_OUTCOME"]


def test_reward_component_means():
    dicts = [
        {"score": 1.0, "outcome": 1.0, "acc": True, "correction": 1.0},
        {"score": 0.0, "outcome": 0.0, "acc": False, "correction": -1.0},
    ]
    m = reward_component_means(dicts + [None, "junk"])     # non-dicts ignored
    assert m["outcome"] == 0.5 and m["score"] == 0.5
    assert m["acc"] == 0.5                                 # bool averaged
    assert m["correction"] == 0.0
    assert reward_component_means([]) == {}


def test_mechanism_means():
    metas = [
        {"planner_steps": [1, 2, 3], "code_exec_oks": [True]},        # tool used, ran ok
        {"planner_steps": [1, 2], "code_exec_oks": []},               # none-path, no tool
        {"planner_steps": [1, 2, 3], "code_exec_oks": [False, True]},  # tool used, 1/2 ok
    ]
    m = mechanism_means(metas)
    assert abs(m["avg_steps"] - (3 + 2 + 3) / 3) < 1e-9
    assert abs(m["tool_call_rate"] - 2 / 3) < 1e-9         # 2 of 3 ran code
    assert abs(m["code_exec_success_rate"] - 2 / 3) < 1e-9  # 2 ok of 3 exec attempts
    assert mechanism_means([]) == {}


# ── standalone runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
