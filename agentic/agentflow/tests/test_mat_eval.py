"""Offline tests for the MAT eval suite: benchmark converter + metrics (plan §8).

Run standalone:  python agentic/agentflow/tests/test_mat_eval.py
Or with pytest:  pytest agentic/agentflow/tests/test_mat_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_mat_data import convert_benchmark  # noqa: E402
from core.mat_eval_metrics import (  # noqa: E402
    score_one, aggregate, aggregate_by_type, call_gain_harm, diagnosis_accuracy,
)

# Real rows from laolao77/MAT MAT-Benchmark/MAT-Coding.json (abridged question).
BENCH = [
    {"id": 1, "question": "大洋洲主要属于哪个板块", "answers": ["A"],
     "ori_image_path": "rotation90_0_ori.png", "processed_image_path": "rotation90_0_proc.png",
     "type": ["rotation90"], "split": "simple"},
    {"id": 2, "question": "what is written?", "answers": ["TAEST", "TEST"],
     "ori_image_path": "none_5_ori.png", "processed_image_path": "none_5_proc.png",
     "type": ["none"], "split": "simple"},
    {"id": 3, "question": "crop and read", "answers": ["42"],
     "ori_image_path": "crop_9_ori.png", "processed_image_path": "crop_9_proc.png",
     "type": ["crop"], "split": "hard"},
]


# ── benchmark converter ──────────────────────────────────────────────────────────

def test_convert_benchmark_fields():
    out = convert_benchmark(BENCH)
    assert len(out) == 3
    r0 = out[0]
    assert r0["image_path"] == ["rotation90_0_proc.png"]      # corrupted input, as list
    assert r0["problem"].startswith("<image>\n")
    assert r0["gt"] == "A"                                     # first answer
    assert r0["metadata"]["corruption_gt"] == ["rotation90"]
    assert r0["metadata"]["split"] == "simple"
    assert r0["metadata"]["answers"] == ["A"]
    # path lives only in image_path now (single source of truth, B4)
    assert "input_image_path" not in r0["metadata"]


def test_convert_benchmark_multi_answer_and_root():
    out = convert_benchmark(BENCH, image_root="/imgs")
    assert out[1]["metadata"]["answers"] == ["TAEST", "TEST"]
    assert out[1]["image_path"] == ["/imgs/none_5_proc.png"]
    assert out[1]["gt"] == "TAEST"


# ── score_one (max over answers) ─────────────────────────────────────────────────

def test_score_one_exact_and_f1():
    # NOTE: Visual-ARFT's normalize() strips articles, and "A" -> "a" (an article)
    # -> empty token set, so single-letter multiple-choice answers get F1=0 but EM=1.
    # EM is the meaningful metric for MC; this is faithful to the baseline's normalize.
    assert score_one("A", ["A"]) == {"f1": 0.0, "em": 1}
    # matches the 2nd of several valid answers -> EM 1 via max
    assert score_one("TEST", ["TAEST", "TEST"]) == {"f1": 1.0, "em": 1}
    assert score_one("wrong", ["A"]) == {"f1": 0.0, "em": 0}


def test_score_one_partial_f1():
    s = score_one("red big dog", ["big dog"])
    assert 0.0 < s["f1"] < 1.0 and s["em"] == 0


def test_score_one_empty():
    assert score_one("", []) == {"f1": 0.0, "em": 0}
    assert score_one(None, ["x"]) == {"f1": 0.0, "em": 0}


# ── aggregate by split ───────────────────────────────────────────────────────────

def test_aggregate_overall_and_splits():
    records = [
        {"f1": 1.0, "em": 1, "split": "simple"},
        {"f1": 0.0, "em": 0, "split": "simple"},
        {"f1": 0.5, "em": 0, "split": "hard"},
    ]
    agg = aggregate(records)
    assert agg["overall"]["n"] == 3
    assert abs(agg["overall"]["f1"] - 0.5) < 1e-9
    assert agg["simple"]["n"] == 2 and agg["simple"]["em"] == 0.5
    assert agg["hard"]["n"] == 1 and agg["hard"]["f1"] == 0.5


def test_aggregate_omits_absent_split():
    agg = aggregate([{"f1": 1.0, "em": 1, "split": "simple"}])
    assert "hard" not in agg and "simple" in agg


# ── per-corruption-type breakdown + diagnosis accuracy (eval polish) ─────────────

def test_aggregate_by_type():
    records = [
        {"f1": 1.0, "em": 1, "type": "blur"},
        {"f1": 0.0, "em": 0, "type": "blur"},
        {"f1": 1.0, "em": 1, "type": "rotation90+dark"},
        {"f1": 0.5, "em": 0, "type": None},        # missing type -> excluded
    ]
    by_type = aggregate_by_type(records)
    assert set(by_type) == {"blur", "rotation90+dark"}
    assert by_type["blur"] == {"f1": 0.5, "em": 0.5, "n": 2}
    assert by_type["rotation90+dark"]["em"] == 1.0


def test_diagnosis_accuracy():
    records = [
        {"diagnosis_correct": True}, {"diagnosis_correct": False},
        {"diagnosis_correct": True}, {},   # missing -> excluded
    ]
    assert abs(diagnosis_accuracy(records) - 2 / 3) < 1e-9
    assert diagnosis_accuracy([]) == 0.0


# ── Call Gain / Call Harm ────────────────────────────────────────────────────────

def test_call_gain_harm():
    # (baseline_correct, tool_correct)
    pairs = [
        (False, True),   # gain
        (False, True),   # gain
        (True, False),   # harm
        (True, True),    # unchanged-correct
        (False, False),  # unchanged-wrong
    ]
    m = call_gain_harm(pairs)
    assert m["gain"] == 2 and m["harm"] == 1 and m["net"] == 1
    assert m["n"] == 5
    assert abs(m["gain_rate"] - 2 / 3) < 1e-9     # 2 gains / 3 baseline-wrong
    assert abs(m["harm_rate"] - 1 / 2) < 1e-9     # 1 harm / 2 baseline-right
    assert abs(m["baseline_acc"] - 2 / 5) < 1e-9
    assert abs(m["tool_acc"] - 3 / 5) < 1e-9


def test_call_gain_harm_empty():
    m = call_gain_harm([])
    assert m["gain"] == 0 and m["harm"] == 0 and m["gain_rate"] == 0.0


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
