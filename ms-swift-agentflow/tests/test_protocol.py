from mat_agentflow.protocol import (STEP_ANSWER, STEP_CODE, STEP_PROBLEM, build_tip,
                                    classify_step, extract_answer, extract_code, extract_problems)


def test_protocol_parsing():
    problem = "<think>x</think><problem>{'blur', 'rotation90'}</problem>"
    code = "<think>x</think><code>```python\nprint('x')\n```</code>"
    answer = "<think>x</think><answer>7500</answer>"
    assert classify_step(problem) == STEP_PROBLEM
    assert extract_problems(problem) == ["blur", "rotation90"]
    assert classify_step(code) == STEP_CODE
    assert "print" in extract_code(code)
    assert classify_step(answer) == STEP_ANSWER
    assert extract_answer(answer) == "7500"


def test_ambiguous_step_rejected():
    assert classify_step("<problem>{'blur'}</problem><answer>x</answer>") is None
    assert "no repair" in build_tip(["none"])
