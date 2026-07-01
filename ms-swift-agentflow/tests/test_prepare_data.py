from scripts.prepare_data import convert_agentflow, prepare_raw


ROWS = [
    {"type": "pre_problem", "image_path": "blur_1_proc.png", "problem": "<query>number?</query>",
     "solution": "<problem>{'blur'}</problem>", "gt": "<think>x</think><problem>{'blur'}</problem>"},
    {"type": "pre_code", "image_path": "blur_1_proc.png", "context": "diagnosed",
     "solution": "<code>fix</code>", "gt": ""},
    {"type": "pre_answer", "image_path": "blur_1_ori.png", "context": "fixed",
     "solution": "<answer>42</answer>", "gt": ""},
]


def test_raw_mat_emits_sft_and_rl(tmp_path):
    sft, rl = prepare_raw(ROWS, str(tmp_path))
    assert len(sft) == 3 and len(rl) == 1
    assert [x["step_type"] for x in sft] == ["problem", "code", "answer"]
    assert rl[0]["corruption_gt"] == ["blur"]
    assert rl[0]["solution"] == "42"
    assert rl[0]["messages"][1]["content"].startswith("<image>")


def test_existing_agentflow_jsonl_conversion(tmp_path):
    row = {"problem": "<image>\nq", "image_path": [str(tmp_path / "x.png")], "gt": "a",
           "metadata": {"corruption_gt": ["noise"], "answers": ["a", "A"]}}
    output = convert_agentflow([row])
    assert output[0]["answers"] == ["a", "A"]
    assert output[0]["images"][0].endswith("x.png")
