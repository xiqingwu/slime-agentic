"""Offline tests for programmatic MAT data synthesis (plan §3.6 / D1).

Pure planner/builder are tested directly; the end-to-end synthesize() uses real cv2
on synthetic clean images in a tmpdir. Run standalone:
    python agentic/agentflow/tests/test_synthesize_mat_data.py
Or with pytest:
    pytest agentic/agentflow/tests/test_synthesize_mat_data.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from synthesize_mat_data import plan_corruptions, build_row, synthesize, _load_generic_bases  # noqa: E402
from core.corruptions import ROTATIONS, SINGLE_CORRUPTIONS  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="mat_synth_test_")


def _clean_base(name="clean_a.png"):
    img = np.full((40, 60, 3), 100, dtype=np.uint8)
    img[:20, :30] = 240
    path = os.path.join(_TMP, name)
    cv2.imwrite(path, img)
    return {"clean_image_path": path, "question": "what is shown?", "answers": ["cat"]}


# ── plan_corruptions (pure) ──────────────────────────────────────────────────────

def test_plan_corruptions_mix_and_counts():
    rng = np.random.default_rng(0)
    plans = plan_corruptions(100, double_frac=0.3, none_frac=0.1, rng=rng)
    assert len(plans) == 100
    nones = [p for p in plans if p == ["none"]]
    doubles = [p for p in plans if len(p) == 2]
    singles = [p for p in plans if len(p) == 1 and p != ["none"]]
    assert len(nones) == 10 and len(doubles) == 30 and len(singles) == 60
    # every double = exactly one rotation + one (non-rotation) degradation, sorted
    for d in doubles:
        assert d == sorted(d)
        assert len(set(d) & set(ROTATIONS)) == 1
        assert len(set(d) - set(ROTATIONS)) == 1
    for s in singles:
        assert s[0] in SINGLE_CORRUPTIONS


def test_plan_corruptions_deterministic():
    a = plan_corruptions(50, rng=np.random.default_rng(7))
    b = plan_corruptions(50, rng=np.random.default_rng(7))
    assert a == b


# ── build_row (pure) ─────────────────────────────────────────────────────────────

def test_build_row_schema():
    row = build_row("what is it?", ["dog", "puppy"], "/x/blur_3_proc.png", ["blur"],
                    clean_path="/x/clean.png", traj_id="synth_3")
    assert row["problem"].startswith("<image>\n") and "what is it?" in row["problem"]
    assert row["image_path"] == ["/x/blur_3_proc.png"]      # list, single source of truth
    assert row["gt"] == "dog"                                # first answer
    assert row["metadata"]["corruption_gt"] == ["blur"]
    assert row["metadata"]["answers"] == ["dog", "puppy"]
    assert row["metadata"]["synthesized"] is True


def test_build_row_sorts_double_and_handles_empty_answers():
    row = build_row("q", [], "/x/p.png", ["rotation90", "dark"])
    assert row["metadata"]["corruption_gt"] == ["dark", "rotation90"]   # sorted
    assert row["gt"] == ""


# ── synthesize (real cv2, tmpdir) ────────────────────────────────────────────────

def test_synthesize_writes_images_and_rows():
    bases = [_clean_base("c1.png"), _clean_base("c2.png")]
    out_dir = os.path.join(_TMP, "synth_out")
    rows = synthesize(bases, num=12, out_image_dir=out_dir, seed=0)
    assert len(rows) == 12
    for r in rows:
        path = r["image_path"][0]
        assert os.path.exists(path) and cv2.imread(path) is not None  # valid corrupted image
        assert r["gt"] == "cat"                                       # inherited from clean base
        assert r["metadata"]["corruption_gt"] == sorted(r["metadata"]["corruption_gt"])
        assert r["metadata"]["synthesized"] is True


def test_synthesize_empty_bases_raises():
    try:
        synthesize([], num=4, out_image_dir=os.path.join(_TMP, "x"), seed=0)
        assert False, "should have raised"
    except ValueError:
        pass


# ── generic clean-VQA loader ─────────────────────────────────────────────────────

def test_load_generic_bases():
    import json
    p = os.path.join(_TMP, "clean.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"image": "a.png", "question": "q1", "answer": "x"}) + "\n")
        f.write(json.dumps({"image_path": ["b.png"], "question": "q2", "answers": ["y", "z"]}) + "\n")
        f.write(json.dumps({"image": "c.png"}) + "\n")   # missing q/answer -> skipped
    bases = _load_generic_bases(p, image_root="/imgs")
    assert len(bases) == 2
    assert bases[0] == {"clean_image_path": "/imgs/a.png", "question": "q1",
                        "answers": ["x"], "trajectory_id": None}
    assert bases[1]["clean_image_path"] == "/imgs/b.png" and bases[1]["answers"] == ["y", "z"]


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
