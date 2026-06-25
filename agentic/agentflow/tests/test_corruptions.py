"""Offline tests for deterministic cv2 corruptions (plan §3.6 / D1).

Real cv2 + numpy on synthetic images. Run standalone:
    python agentic/agentflow/tests/test_corruptions.py
Or with pytest:
    pytest agentic/agentflow/tests/test_corruptions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from core.corruptions import (  # noqa: E402
    apply_corruption, apply_corruptions, SINGLE_CORRUPTIONS, ROTATIONS, DEGRADATIONS,
)


def _img():
    """A structured 40x60 image (so rotation/blur/brightness changes are detectable)."""
    img = np.full((40, 60, 3), 128, dtype=np.uint8)
    img[:20, :30, :] = 255   # bright quadrant -> asymmetric, has an edge
    return img


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_none_is_identity_copy():
    img = _img()
    out = apply_corruption(img, "none", _rng())
    assert np.array_equal(out, img)
    out[0, 0, 0] = 7
    assert img[0, 0, 0] != 7          # it's a copy, not the same array


def test_rotation90_swaps_shape():
    img = _img()
    out = apply_corruption(img, "rotation90", _rng())
    assert out.shape == (60, 40, 3)   # H/W swapped


def test_rotation180_same_shape_changed():
    img = _img()
    out = apply_corruption(img, "rotation180", _rng())
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_dark_darkens_and_overexposure_brightens():
    img = _img()
    assert apply_corruption(img, "dark", _rng()).mean() < img.mean()
    assert apply_corruption(img, "overexposure", _rng()).mean() > img.mean()


def test_blur_and_noise_change_image_same_shape():
    img = _img()
    for name in ("blur", "noise"):
        out = apply_corruption(img, name, _rng())
        assert out.shape == img.shape and not np.array_equal(out, img)


def test_double_corruption_rotation_plus_degrade():
    img = _img()
    out = apply_corruptions(img, ["rotation90", "dark"], _rng())
    assert out.shape == (60, 40, 3)   # rotated
    assert out.mean() < img.mean()    # and darkened


def test_deterministic_with_seed():
    img = _img()
    for name in ("dark", "noise", "blur"):
        a = apply_corruption(img, name, _rng(123))
        b = apply_corruption(img, name, _rng(123))
        assert np.array_equal(a, b)   # same seed -> same output


def test_unknown_corruption_raises():
    try:
        apply_corruption(_img(), "crop", _rng())   # crop intentionally unsupported
        assert False, "should have raised"
    except ValueError:
        pass


def test_vocab_split():
    assert set(ROTATIONS) == {"rotation90", "rotation180"}
    assert set(DEGRADATIONS) == {"dark", "overexposure", "blur", "noise"}
    assert set(ROTATIONS) | set(DEGRADATIONS) == set(SINGLE_CORRUPTIONS)


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
