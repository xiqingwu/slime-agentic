"""Offline tests for the MAT-Coding data converter (plan §5.1).

Fixture rows are copied verbatim from the real HuggingFace dataset
(`laolao77/MAT`, `MAT-Training/rft_agent_code_1_2k.json`): two 3-step
trajectories (overexposure, rotation90) and one 2-step 'none' trajectory.

Run standalone:  python agentic/agentflow/tests/test_prepare_mat_data.py
Or with pytest:  pytest agentic/agentflow/tests/test_prepare_mat_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_mat_data import convert, trajectory_id, extract_answer  # noqa: E402

# ── Real rows from the dataset (abridged problem text) ───────────────────────────
FIXTURE = [
    # overexposure_60: pre_problem -> pre_code -> pre_answer (answer uses _ori)
    {"type": "pre_problem", "image_path": "overexposure_60_proc.png",
     "problem": "<query> what is the biggest number on this ruler? </query>",
     "solution": "<problem> {'overexposure'} </problem>",
     "gt": "<think> Whiteout. </think>\n<problem> {'overexposure'} </problem>"},
    {"type": "pre_code", "image_path": "overexposure_60_proc.png", "context": "...",
     "solution": "<code>\n```python\nimport cv2\n```\n</code>", "gt": "..."},
    {"type": "pre_answer", "image_path": "overexposure_60_ori.png", "context": "...",
     "solution": "<answer> 81 </answer>",
     "gt": "<think> ok. </think>\n<answer> 81 </answer>"},
    # none_79: pre_problem -> pre_answer (no code step)
    {"type": "pre_problem", "image_path": "none_79_proc.png",
     "problem": "<query> what is written in the image? </query>",
     "solution": "<problem> {'none'} </problem>",
     "gt": "<think> clean. </think>\n<problem> {'none'} </problem>"},
    {"type": "pre_answer", "image_path": "none_79_ori.png", "context": "...",
     "solution": "<answer> TAEST </answer>",
     "gt": "<think> clean. </think>\n<answer> TAEST </answer>"},
    # rotation90_21: 3 steps, non-ASCII answer
    {"type": "pre_problem", "image_path": "rotation90_21_proc.png",
     "problem": "<query> 从图中提取: 始发站 </query>",
     "solution": "<problem> {'rotation90'} </problem>",
     "gt": "<think> rotated. </think>\n<problem> {'rotation90'} </problem>"},
    {"type": "pre_code", "image_path": "rotation90_21_proc.png", "context": "...",
     "solution": "<code>\n```python\nimport cv2\n```\n</code>", "gt": "..."},
    {"type": "pre_answer", "image_path": "rotation90_21_ori.png", "context": "...",
     "solution": "<answer> {'始发站': '广安枢纽站'} </answer>",
     "gt": "<think> done. </think>\n<answer> {'始发站': '广安枢纽站'} </answer>"},
]


def test_trajectory_id_strips_suffix():
    assert trajectory_id("overexposure_60_proc.png") == "overexposure_60"
    assert trajectory_id("rotation90_21_ori.png") == "rotation90_21"
    assert trajectory_id("/data/imgs/none_79_proc.png") == "none_79"
    # no _proc/_ori suffix -> just drop extension
    assert trajectory_id("weird_name.png") == "weird_name"


def test_extract_answer():
    assert extract_answer("<answer> 81 </answer>") == "81"
    assert extract_answer("<think>x</think>\n<answer> TAEST </answer>") == "TAEST"
    assert extract_answer("no tags") == "no tags"
    assert extract_answer("") == ""


def test_convert_groups_trajectories():
    out = convert(FIXTURE)
    assert len(out) == 3                       # 8 rows -> 3 trajectories
    ids = [s["metadata"]["trajectory_id"] for s in out]
    assert ids == ["overexposure_60", "none_79", "rotation90_21"]  # order preserved


def test_convert_fields():
    out = convert(FIXTURE)
    over = out[0]
    # image_path is a LIST pointing at the corrupted (_proc) image
    assert over["image_path"] == ["overexposure_60_proc.png"]
    # problem carries the <image> placeholder
    assert over["problem"].startswith("<image>\n")
    assert "biggest number" in over["problem"]
    # outcome GT = final answer, corruption GT from <problem>
    assert over["gt"] == "81"
    assert over["metadata"]["corruption_gt"] == ["overexposure"]
    assert over["metadata"]["num_steps"] == 3
    # path lives only in image_path now (single source of truth, B4)
    assert "input_image_path" not in over["metadata"]
    assert over["metadata"]["clean_image_path"] == "overexposure_60_ori.png"


def test_convert_none_trajectory_two_steps():
    out = convert(FIXTURE)
    none = out[1]
    assert none["metadata"]["corruption_gt"] == ["none"]
    assert none["metadata"]["num_steps"] == 2     # no code step
    assert none["gt"] == "TAEST"


def test_convert_unicode_answer_preserved():
    out = convert(FIXTURE)
    rot = out[2]
    assert rot["metadata"]["corruption_gt"] == ["rotation90"]
    assert rot["gt"] == "{'始发站': '广安枢纽站'}"


def test_convert_image_root_and_no_placeholder():
    out = convert(FIXTURE, image_root="/data/imgs", add_placeholder=False)
    over = out[0]
    assert over["image_path"] == ["/data/imgs/overexposure_60_proc.png"]
    assert not over["problem"].startswith("<image>")


def test_convert_skips_incomplete_trajectory():
    # a lone pre_code with no problem/answer should be skipped
    rows = FIXTURE + [{"type": "pre_code", "image_path": "lonely_99_proc.png",
                       "solution": "<code> x </code>", "gt": "..."}]
    out = convert(rows)
    assert len(out) == 3
    assert all(s["metadata"]["trajectory_id"] != "lonely_99" for s in out)


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
