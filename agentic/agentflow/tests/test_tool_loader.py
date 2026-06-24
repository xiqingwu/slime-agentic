"""Offline tests for the shared OpenCV-tool loader (tools integration polish).

Run standalone:  python agentic/agentflow/tests/test_tool_loader.py
Or with pytest:  pytest agentic/agentflow/tests/test_tool_loader.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tool_loader import load_opencv_tool, tool_timeout_from_env, DEFAULT_TOOL_TIMEOUT  # noqa: E402


def test_load_opencv_tool_cached_and_usable():
    t1 = load_opencv_tool()
    t2 = load_opencv_tool()
    assert t1 is t2                       # cached single instance
    assert hasattr(t1, "execute")         # the real tool with an execute() coroutine


def test_tool_timeout_from_env():
    assert tool_timeout_from_env() == DEFAULT_TOOL_TIMEOUT
    os.environ["MAT_TOOL_TIMEOUT"] = "12"
    try:
        assert tool_timeout_from_env() == 12
        os.environ["MAT_TOOL_TIMEOUT"] = "bad"
        assert tool_timeout_from_env() == DEFAULT_TOOL_TIMEOUT   # bad value -> default
    finally:
        del os.environ["MAT_TOOL_TIMEOUT"]


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
