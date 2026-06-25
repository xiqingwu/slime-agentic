"""Deterministic cv2 image corruptions for programmatic MAT data synthesis (plan §3.6).

Method C needs no teacher CoT — the diagnosis GT is just *which corruption we
applied*, and the outcome GT is inherited from the clean VQA answer. So we can mint
unlimited trajectory start points by taking a clean (image, question, answer) and
degrading the image with one of these known, parameterized cv2 ops.

Covered: rotation90, rotation180, dark, overexposure, blur, noise, none.
**Excluded: crop** — a faithful crop trajectory needs to know where the answer
region is (the model crops to zoom in), which we don't have for arbitrary VQA. Mix
synthesized data with the real MAT-Training set (which has crop) for crop coverage.

cv2 is imported lazily inside the ops so this module imports without it; numpy is a
hard dependency (already required by slime).
"""

from __future__ import annotations

import numpy as np

# Single-corruption vocabulary we can synthesize (subset of the MAT label set).
SINGLE_CORRUPTIONS = ("rotation90", "rotation180", "dark", "overexposure", "blur", "noise")
ROTATIONS = ("rotation90", "rotation180")
# Degradations usable as the non-rotation half of a double corruption.
DEGRADATIONS = tuple(c for c in SINGLE_CORRUPTIONS if c not in ROTATIONS)


def _u8(arr) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def _rotation90(img, rng):
    import cv2
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def _rotation180(img, rng):
    import cv2
    return cv2.rotate(img, cv2.ROTATE_180)


def _dark(img, rng):
    return _u8(img.astype(np.float32) * rng.uniform(0.30, 0.55))


def _overexposure(img, rng):
    return _u8(img.astype(np.float32) * rng.uniform(1.7, 2.8))


def _blur(img, rng):
    import cv2
    k = int(rng.choice([5, 7, 9, 11]))
    return cv2.GaussianBlur(img, (k, k), 0)


def _noise(img, rng):
    return _u8(img.astype(np.float32) + rng.normal(0.0, rng.uniform(15.0, 35.0), img.shape))


_FUNCS = {
    "rotation90": _rotation90,
    "rotation180": _rotation180,
    "dark": _dark,
    "overexposure": _overexposure,
    "blur": _blur,
    "noise": _noise,
}


def apply_corruption(img: np.ndarray, name: str, rng) -> np.ndarray:
    """Apply one corruption (randomized params from ``rng``); 'none' is identity."""
    if name == "none":
        return img.copy()
    if name not in _FUNCS:
        raise ValueError(f"unknown corruption {name!r}; known: {sorted(_FUNCS)} + 'none'")
    return _FUNCS[name](img, rng)


def apply_corruptions(img: np.ndarray, names, rng) -> np.ndarray:
    """Apply a sequence of corruptions (for double corruptions), in order."""
    out = img
    for name in names:
        out = apply_corruption(out, name, rng)
    return out
