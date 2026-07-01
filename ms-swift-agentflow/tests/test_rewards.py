from mat_agentflow.rewards import aggregate_reward, score_answer


def test_reward_components():
    steps = [
        {"type": "problem", "content": "<think>x</think><problem>{'blur'}</problem>"},
        {"type": "code", "content": "<think>x</think><code>fix</code>"},
        {"type": "answer", "content": "<think>x</think><answer>7500</answer>"},
    ]
    result = aggregate_reward("7500", steps, ["blur"], [True], ["7,500", "7500"])
    assert result["outcome"] == 1.0
    assert result["diagnosis"] == 1.0
    assert result["code_exec"] == 1.0
    assert result["score"] > 1.0


def test_multiple_answers_take_best():
    assert score_answer("42", ["forty two", "42"])["em"] == 1.0


def test_correction_rewards_rescue_and_penalizes_harm():
    rescue = aggregate_reward("42", [], [], [], ["42"], baseline_output="41")
    harm = aggregate_reward("41", [], [], [], ["42"], baseline_output="42")
    assert rescue["correction"] == 1.0
    assert harm["correction"] == -1.0
