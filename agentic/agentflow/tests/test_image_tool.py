"""Offline tests for the OpenCV execution engine (plan M3 / §5.3).

Uses real cv2 on synthetic images. Run standalone:
    python agentic/agentflow/tests/test_image_tool.py
Or with pytest:
    pytest agentic/agentflow/tests/test_image_tool.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from core.image_tool import (  # noqa: E402
    extract_code,
    replace_paths,
    run_opencv_code,
    INPUT_PLACEHOLDER,
    OUTPUT_PLACEHOLDER,
)

_TMP = tempfile.mkdtemp(prefix="mat_imgtool_test_")


def _make_image(name: str, rotate: bool = False) -> str:
    """Write a small synthetic image (optionally rotated 90° CW) and return its path."""
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:20, :, 0] = 255          # top half red — asymmetric so rotation is detectable
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    path = os.path.join(_TMP, name)
    cv2.imwrite(path, img)
    return path


# ── extract_code ─────────────────────────────────────────────────────────────────

def test_extract_code_from_code_step():
    text = "<think>t</think>\n<code>\n```python\nimport cv2\n```\n</code>"
    assert extract_code(text) == "import cv2"


def test_extract_code_bare_fence_and_none():
    assert extract_code("```python\nx=1\n```") == "x=1"
    assert extract_code("no code here") is None


# ── replace_paths ────────────────────────────────────────────────────────────────

def test_replace_paths_both_quote_styles():
    code = f"a=cv2.imread('{INPUT_PLACEHOLDER}')\ncv2.imwrite(\"{OUTPUT_PLACEHOLDER}\", a)"
    out = replace_paths(code, "/real/in.png", "/real/out.png")
    assert "/real/in.png" in out and "/real/out.png" in out
    assert INPUT_PLACEHOLDER not in out and OUTPUT_PLACEHOLDER not in out


# ── run_opencv_code: success ─────────────────────────────────────────────────────

GOOD_ROTATE_FIX = (
    "<code>\n```python\nimport cv2\n"
    "img = cv2.imread('path_to_input_image.jpg')\n"
    "fixed = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)\n"
    "cv2.imwrite('path_to_output_image.jpg', fixed)\n"
    "print('done')\n```\n</code>"
)


def test_run_success_produces_restored_image():
    inp = _make_image("rot_in.png", rotate=True)
    out = os.path.join(_TMP, "rot_out.png")
    res = asyncio.run(run_opencv_code(GOOD_ROTATE_FIX, inp, out))
    assert res.success is True
    assert res.output_image_path == out
    assert os.path.exists(out)
    assert "done" in res.stdout
    # rotating back should restore original shape (40x60)
    restored = cv2.imread(out)
    assert restored.shape == (40, 60, 3)


def test_run_autogenerates_output_path():
    inp = _make_image("auto_in.png")
    code = (
        "import cv2\nimg=cv2.imread('path_to_input_image.jpg')\n"
        "cv2.imwrite('path_to_output_image.jpg', img)\n"
    )
    res = asyncio.run(run_opencv_code(code, inp))   # no output path given
    assert res.success is True
    assert res.output_image_path and os.path.exists(res.output_image_path)


# ── changed-detection: no-op copy vs real edit ───────────────────────────────────

def test_run_noop_copy_flagged_unchanged():
    """A plain read->write copy succeeds but must report changed=False."""
    inp = _make_image("noop_in.png")
    out = os.path.join(_TMP, "noop_out.png")
    code = ("import cv2\nimg=cv2.imread('path_to_input_image.jpg')\n"
            "cv2.imwrite('path_to_output_image.jpg', img)\n")
    res = asyncio.run(run_opencv_code(code, inp, out))
    assert res.success is True          # it did run and produce a valid image
    assert res.changed is False         # ...but nothing actually changed
    assert res.as_dict()["changed"] is False


def test_run_real_edit_flagged_changed():
    inp = _make_image("edit_in.png")
    out = os.path.join(_TMP, "edit_out.png")
    code = ("import cv2\nimg=cv2.imread('path_to_input_image.jpg')\n"
            "cv2.imwrite('path_to_output_image.jpg', cv2.bitwise_not(img))\n")  # invert -> differs
    res = asyncio.run(run_opencv_code(code, inp, out))
    assert res.success is True
    assert res.changed is True


# ── run_opencv_code: failure modes ───────────────────────────────────────────────

def test_run_runtime_error():
    inp = _make_image("err_in.png")
    out = os.path.join(_TMP, "err_out.png")
    bad = "import cv2\nraise ValueError('boom')\n"
    res = asyncio.run(run_opencv_code(bad, inp, out))
    assert res.success is False
    assert res.output_image_path is None
    assert "boom" in (res.error or "")
    assert not os.path.exists(out)


def test_run_no_output_written():
    inp = _make_image("noout_in.png")
    out = os.path.join(_TMP, "noout_out.png")
    code = "import cv2\nimg=cv2.imread('path_to_input_image.jpg')\nprint('read ok')\n"  # never writes
    res = asyncio.run(run_opencv_code(code, inp, out))
    assert res.success is False
    assert "no valid output image" in (res.error or "")


def test_run_missing_input():
    res = asyncio.run(run_opencv_code("import cv2", "/does/not/exist.png", "/tmp/x.png"))
    assert res.success is False
    assert "input image not found" in (res.error or "")


def test_run_no_code():
    inp = _make_image("nocode_in.png")
    res = asyncio.run(run_opencv_code("just prose, no fences", inp, "/tmp/y.png"))
    # treated as a bare snippet -> runs as python -> SyntaxError -> failure
    assert res.success is False


def test_run_timeout():
    inp = _make_image("to_in.png")
    out = os.path.join(_TMP, "to_out.png")
    code = "import time\ntime.sleep(5)\n"
    res = asyncio.run(run_opencv_code(code, inp, out, timeout=1))
    assert res.success is False
    assert "timed out" in (res.error or "")


# ── B2 hardening: isolated cwd + rlimits don't break normal exec ──────────────────

def test_run_isolated_cwd_contains_relative_writes():
    """A relative-path write by the snippet must NOT land in the caller's cwd."""
    inp = _make_image("cwd_in.png")
    out = os.path.join(_TMP, "cwd_out.png")
    marker = f"sentinel_{os.getpid()}.txt"
    cwd_before = os.getcwd()
    code = ("import cv2\nimg=cv2.imread('path_to_input_image.jpg')\n"
            f"open({marker!r}, 'w').write('x')\n"      # relative write -> isolated temp cwd
            "cv2.imwrite('path_to_output_image.jpg', img)\n")
    res = asyncio.run(run_opencv_code(code, inp, out))
    assert res.success is True                          # rlimits don't break normal cv2
    assert os.getcwd() == cwd_before
    assert not os.path.exists(os.path.join(cwd_before, marker))  # contained + cleaned up


# ── tool wrapper ─────────────────────────────────────────────────────────────────

def test_tool_wrapper_returns_dict():
    # Load the tool the way the executor does is heavier; import directly instead.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_tool_opencv_editor",
        Path(__file__).resolve().parents[1] / "tools" / "opencv_editor" / "tool.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    inp = _make_image("wrap_in.png")
    out = os.path.join(_TMP, "wrap_out.png")
    tool = mod.OpenCV_Editor_Tool()
    res = asyncio.run(tool.execute(code=GOOD_ROTATE_FIX.replace(
        "ROTATE_90_COUNTERCLOCKWISE", "ROTATE_90_CLOCKWISE"), input_image_path=inp, output_image_path=out))
    assert isinstance(res, dict)
    assert res["success"] is True
    assert res["output_image_path"] == out


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
