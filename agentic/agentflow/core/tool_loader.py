"""Shared OpenCV-tool factory + timeout config for the MAT rollout & eval paths.

``rollout_mat`` (training) and ``eval_mat`` each used to carry their own importlib
spec dance to load the tool; this centralizes it into one place. The tool is
stateless (``execute`` takes everything as args), so a single cached instance is
shared across concurrent rollouts. The tool is also registered in the AgentFlow
executor's name-mapping for the general path — the MAT path uses this direct
factory instead, since it needs no LLM engine and thus no executor wiring.
"""

from __future__ import annotations

import os

_TOOL = None
DEFAULT_TOOL_TIMEOUT = 30  # seconds for one cv2 snippet


def load_opencv_tool():
    """Return the shared (cached) OpenCV editor tool instance."""
    global _TOOL
    if _TOOL is None:
        from tools.opencv_editor.tool import OpenCV_Editor_Tool

        _TOOL = OpenCV_Editor_Tool()
    return _TOOL


def tool_timeout_from_env() -> int:
    """cv2-exec timeout, overridable via ``MAT_TOOL_TIMEOUT`` (bad values -> default)."""
    try:
        return int(os.environ.get("MAT_TOOL_TIMEOUT", DEFAULT_TOOL_TIMEOUT))
    except ValueError:
        return DEFAULT_TOOL_TIMEOUT
