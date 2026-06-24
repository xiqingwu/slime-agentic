"""Offline tests for the warm forkserver executor (plan B3).

Exercises the same contract as ``image_tool.run_opencv_code`` (success / error /
no-op-changed / real-edit-changed / timeout) but through the forkserver path, with
real cv2 on synthetic images.

Run standalone:  python agentic/agentflow/tests/test_image_worker.py
Or with pytest:  pytest agentic/agentflow/tests/test_image_worker.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from core.image_worker import run_in_forkserver  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="mat_worker_test_")


def _make_image(name: str, rotate: bool = False) -> str:
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:20, :, 2] = 255
    if rotate:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    path = os.path.join(_TMP, name)
    cv2.imwrite(path, img)
    return path


ROTATE_FIX = (
    "<code>\n```python\nimport cv2\n"
    "img = cv2.imread('path_to_input_image.jpg')\n"
    "cv2.imwrite('path_to_output_image.jpg', cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))\n"
    "print('ok')\n```\n</code>"
)


def test_forkserver_success_and_changed():
    inp = _make_image("fk_rot_in.png", rotate=True)
    out = os.path.join(_TMP, "fk_rot_out.png")
    res = asyncio.run(run_in_forkserver(ROTATE_FIX, inp, out))
    assert res.success is True
    assert res.changed is True
    assert res.output_image_path == out
    assert cv2.imread(out).shape == (40, 60, 3)
    assert "ok" in res.stdout


def test_forkserver_noop_unchanged():
    inp = _make_image("fk_noop_in.png")
    out = os.path.join(_TMP, "fk_noop_out.png")
    code = ("import cv2\nimg=cv2.imread('path_to_input_image.jpg')\n"
            "cv2.imwrite('path_to_output_image.jpg', img)\n")
    res = asyncio.run(run_in_forkserver(code, inp, out))
    assert res.success is True and res.changed is False


def test_forkserver_runtime_error():
    inp = _make_image("fk_err_in.png")
    out = os.path.join(_TMP, "fk_err_out.png")
    res = asyncio.run(run_in_forkserver("import cv2\nraise ValueError('boom')\n", inp, out))
    assert res.success is False
    assert "boom" in (res.error or "")


def test_forkserver_no_output():
    inp = _make_image("fk_noout_in.png")
    out = os.path.join(_TMP, "fk_noout_out.png")
    res = asyncio.run(run_in_forkserver("import cv2\nx=1\n", inp, out))
    assert res.success is False
    assert "no valid output image" in (res.error or "")


def test_forkserver_missing_input():
    res = asyncio.run(run_in_forkserver("import cv2", "/does/not/exist.png", "/tmp/x.png"))
    assert res.success is False
    assert "input image not found" in (res.error or "")


def test_forkserver_timeout():
    inp = _make_image("fk_to_in.png")
    out = os.path.join(_TMP, "fk_to_out.png")
    res = asyncio.run(run_in_forkserver("import time\ntime.sleep(5)\n", inp, out, timeout=1))
    assert res.success is False
    assert "timed out" in (res.error or "")


def test_forkserver_reuses_warm_server():
    """Several sequential execs work (server stays warm across calls)."""
    inp = _make_image("fk_warm_in.png", rotate=True)
    for i in range(3):
        out = os.path.join(_TMP, f"fk_warm_out_{i}.png")
        res = asyncio.run(run_in_forkserver(ROTATE_FIX, inp, out))
        assert res.success is True


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
