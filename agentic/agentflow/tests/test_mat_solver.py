"""Offline tests for the MAT-Coding online solver (plan M3 / §4.1).

The engine (LLM) and tool (cv2) are faked, so the full problem->code->answer loop
is exercised with no GPU/model/cv2. Verifies turn recording, metadata assembly
(planner_steps / code_exec_oks), tips injection, image feedback, and termination.

Run standalone:  python agentic/agentflow/tests/test_mat_solver.py
Or with pytest:  pytest agentic/agentflow/tests/test_mat_solver.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm_engine import GenerationOutput  # noqa: E402
from core.mat_solver import (  # noqa: E402
    MATSolver, build_tip, parse_step, extract_answer,
    build_baseline_messages, baseline_answer, BASELINE_PROMPT, single_flight,
    _TIP_CROP, _TIP_NONE,
)

THINK = "<think> reasoning </think>"
PROBLEM = f"{THINK}\n<problem> {{'dark'}} </problem>"
PROBLEM_NONE = f"{THINK}\n<problem> {{'none'}} </problem>"
PROBLEM_CROP = f"{THINK}\n<problem> {{'crop'}} </problem>"
CODE = (f"{THINK}\n<code>\n```python\nimport cv2\n"
        "img=cv2.imread('path_to_input_image.jpg')\n"
        "cv2.imwrite('path_to_output_image.jpg', img)\n```\n</code>")
ANSWER = f"{THINK}\n<answer> 42 </answer>"


class FakeEngine:
    def __init__(self, responses, mm=None):
        self.responses = list(responses)
        self.calls = []     # captured message lists
        self.mm = mm

    async def generate(self, messages, sampling_params=None):
        self.calls.append(messages)
        resp = self.responses.pop(0)
        return GenerationOutput(
            prompt_text="P", prompt_token_ids=[1, 2], response=resp,
            token_ids=[10, 11], log_probs=[-0.1, -0.2], finish_reason="stop",
            multimodal_train_inputs=self.mm,
        )


class FakeTool:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def execute(self, code, input_image_path, output_image_path=None, timeout=30):
        self.calls.append({"code": code, "in": input_image_path, "out": output_image_path, "timeout": timeout})
        return self.results.pop(0)


def _img_of(message_list):
    """Extract the image path from a solver-built user message."""
    for block in message_list[0]["content"]:
        if block["type"] == "image":
            return block["image"]
    return None


# ── pure helpers ─────────────────────────────────────────────────────────────────

def test_build_tip():
    assert build_tip(["crop"]) == _TIP_CROP
    assert build_tip(["none"]) == _TIP_NONE
    assert build_tip([]) == _TIP_NONE
    assert "dark" in build_tip(["dark"]) and "<tips>" in build_tip(["dark"])


def test_parse_step():
    assert parse_step(PROBLEM) == ("problem", ["dark"])
    t, code = parse_step(CODE)
    assert t == "code" and "cv2.imread" in code
    assert parse_step(ANSWER) == ("answer", "42")
    assert parse_step(f"{THINK} no tag") == (None, None)


def test_extract_answer():
    assert extract_answer(ANSWER) == "42"
    assert extract_answer("plain") == "plain"


# ── shared no-tool baseline (B5 / A4) ────────────────────────────────────────────

def test_build_baseline_messages():
    msgs = build_baseline_messages("what is it?", "/img.png")
    assert _img_of(msgs) == "/img.png"
    text = next(b["text"] for b in msgs[0]["content"] if b["type"] == "text")
    assert BASELINE_PROMPT in text and "what is it?" in text
    assert "<answer>" in text                     # asks for the answer tag


def test_baseline_answer_extracts_and_passes_params():
    engine = FakeEngine([f"{THINK}\n<answer> blue car </answer>"])
    pred = asyncio.run(baseline_answer(engine, "q", "/img.png", sampling_params={"temperature": 0.0}))
    assert pred == "blue car"                      # extracted from <answer>
    assert _img_of(engine.calls[0]) == "/img.png"  # one-shot on the corrupted image


# ── single_flight (R3: dedup identical baselines across a GRPO group) ─────────────

def test_single_flight_collapses_concurrent():
    registry, calls = {}, {"n": 0}

    async def factory():
        calls["n"] += 1
        await asyncio.sleep(0.01)   # hold the in-flight window open
        return "R"

    async def main():
        return await asyncio.gather(*[single_flight(registry, ("k",), factory) for _ in range(8)])

    results = asyncio.run(main())
    assert results == ["R"] * 8     # all 8 got the result
    assert calls["n"] == 1          # ...but it was computed once
    assert registry == {}           # cleaned up after completion


def test_single_flight_recomputes_after_completion():
    """Sequential (next-step) calls recompute — no stale cross-step caching."""
    registry, calls = {}, {"n": 0}

    async def factory():
        calls["n"] += 1
        return calls["n"]

    async def main():
        a = await single_flight(registry, "k", factory)
        b = await single_flight(registry, "k", factory)
        return a, b

    assert asyncio.run(main()) == (1, 2)


def test_single_flight_propagates_exception():
    registry = {}

    async def boom():
        raise ValueError("nope")

    async def main():
        return await asyncio.gather(
            *[single_flight(registry, "k", boom) for _ in range(3)], return_exceptions=True
        )

    results = asyncio.run(main())
    assert all(isinstance(r, ValueError) for r in results)
    assert registry == {}           # entry removed even on failure


# ── full trajectory: problem -> code(success) -> answer ──────────────────────────

def test_full_trajectory_success():
    engine = FakeEngine([PROBLEM, CODE, ANSWER], mm={"pixel_values": "PV"})
    tool = FakeTool([{"success": True, "output_image_path": "/processed.png"}])
    solver = MATSolver(engine, tool, max_steps=5)

    out = asyncio.run(solver.solve("what is it?", "/corrupted.png"))

    # three training turns, each carrying the per-turn multimodal inputs
    assert len(out.turns) == 3
    assert all(t["multimodal_train_inputs"] == {"pixel_values": "PV"} for t in out.turns)
    # metadata for the rewards
    assert [s["type"] for s in out.planner_steps] == ["problem", "code", "answer"]
    assert out.code_exec_oks == [True]
    assert out.final_output == "42"
    # tool ran on the corrupted image
    assert tool.calls[0]["in"] == "/corrupted.png"
    # image feedback: the answer turn saw the *processed* image
    assert _img_of(engine.calls[0]) == "/corrupted.png"
    assert _img_of(engine.calls[2]) == "/processed.png"
    # tip for 'dark' was injected into the context before the code turn
    assert "identified the issue" in str(engine.calls[1])


# ── none trajectory: problem(none) -> answer (no code) ───────────────────────────

def test_none_trajectory():
    engine = FakeEngine([PROBLEM_NONE, ANSWER])
    tool = FakeTool([])
    out = asyncio.run(MATSolver(engine, tool).solve("q", "/clean.png"))
    assert [s["type"] for s in out.planner_steps] == ["problem", "answer"]
    assert out.code_exec_oks == []            # no code executed
    assert len(tool.calls) == 0
    assert out.final_output == "42"
    assert _TIP_NONE[:20] in str(engine.calls[1])   # 'none' tip injected


# ── code failure: image must NOT advance ─────────────────────────────────────────

def test_code_failure_keeps_image():
    engine = FakeEngine([PROBLEM, CODE, ANSWER])
    tool = FakeTool([{"success": False, "output_image_path": None, "error": "boom"}])
    out = asyncio.run(MATSolver(engine, tool).solve("q", "/corrupted.png"))
    assert out.code_exec_oks == [False]
    # answer turn still sees the original corrupted image (no successful update)
    assert _img_of(engine.calls[2]) == "/corrupted.png"


def test_tool_timeout_passed_through():
    engine = FakeEngine([PROBLEM, CODE, ANSWER])
    tool = FakeTool([{"success": True, "output_image_path": "/p.png"}])
    asyncio.run(MATSolver(engine, tool, tool_timeout=7).solve("q", "/corrupted.png"))
    assert tool.calls[0]["timeout"] == 7   # configurable timeout reaches the tool


# ── no-op code: ran but changed nothing -> code_exec_ok is False ─────────────────

def test_noop_code_not_counted():
    engine = FakeEngine([PROBLEM, CODE, ANSWER])
    # success but the tool reports the image was unchanged (e.g. a read->write copy)
    tool = FakeTool([{"success": True, "output_image_path": "/processed.png", "changed": False}])
    out = asyncio.run(MATSolver(engine, tool).solve("q", "/corrupted.png"))
    assert out.code_exec_oks == [False]            # no-op earns no code-exec credit
    # the (identical) image is still fed forward; only the reward signal differs
    assert _img_of(engine.calls[2]) == "/processed.png"


# ── termination by max_steps ─────────────────────────────────────────────────────

def test_max_steps_termination():
    # model never answers (keeps diagnosing) -> stops after max_steps
    engine = FakeEngine([PROBLEM] * 10)
    tool = FakeTool([])
    out = asyncio.run(MATSolver(engine, tool, max_steps=3).solve("q", "/img.png"))
    assert len(out.turns) == 3
    assert len(out.planner_steps) == 3


# ── malformed step stops the loop ────────────────────────────────────────────────

def test_malformed_step_stops():
    engine = FakeEngine([f"{THINK} garbage no tags", ANSWER])
    tool = FakeTool([])
    out = asyncio.run(MATSolver(engine, tool).solve("q", "/img.png"))
    assert len(out.turns) == 1                # stopped after the malformed step
    assert out.planner_steps[0]["type"] is None


# ── integration: real OpenCV tool + real cv2, only the LLM faked ─────────────────

def test_integration_real_cv2_tool():
    try:
        import cv2
        import numpy as np
        import tempfile
        import os
    except Exception as e:
        print(f"  (skipped integration: {e})")
        return

    from core.image_tool import run_opencv_code  # noqa: F401  (ensure module imports)

    # Build a real 90°-CW-rotated synthetic image.
    work = tempfile.mkdtemp(prefix="mat_solver_int_")
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:20, :, 1] = 255
    corrupted = os.path.join(work, "rot_in.png")
    cv2.imwrite(corrupted, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    problem = f"{THINK}\n<problem> {{'rotation90'}} </problem>"
    code = (f"{THINK}\n<code>\n```python\nimport cv2\n"
            "img=cv2.imread('path_to_input_image.jpg')\n"
            "fixed=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)\n"
            "cv2.imwrite('path_to_output_image.jpg', fixed)\n```\n</code>")
    engine = FakeEngine([problem, code, ANSWER])

    # Real tool loaded the same way rollout_mat does.
    import importlib.util
    tool_file = Path(__file__).resolve().parents[1] / "tools" / "opencv_editor" / "tool.py"
    spec = importlib.util.spec_from_file_location("_tool_opencv_editor_int", tool_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tool = mod.OpenCV_Editor_Tool()

    # cleanup_temp=False so we can inspect the processed image after solve returns.
    out = asyncio.run(MATSolver(engine, tool, work_dir=work, cleanup_temp=False).solve("q", corrupted))

    assert out.code_exec_oks == [True]
    processed = _img_of(engine.calls[2])
    assert processed != corrupted and os.path.exists(processed)
    # rotating back restores the original 40x60 shape
    assert cv2.imread(processed).shape == (40, 60, 3)


def test_temp_images_cleaned_up():
    """With default cleanup_temp, processed-image temp files are removed after solve (R5)."""
    try:
        import cv2
        import numpy as np
        import tempfile
        import os
    except Exception as e:
        print(f"  (skipped cleanup test: {e})")
        return

    work = tempfile.mkdtemp(prefix="mat_solver_cleanup_")
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    img[:20, :, 1] = 255
    corrupted = os.path.join(work, "rot_in.png")
    cv2.imwrite(corrupted, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    problem = f"{THINK}\n<problem> {{'rotation90'}} </problem>"
    code = (f"{THINK}\n<code>\n```python\nimport cv2\n"
            "img=cv2.imread('path_to_input_image.jpg')\n"
            "fixed=cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)\n"
            "cv2.imwrite('path_to_output_image.jpg', fixed)\n```\n</code>")
    engine = FakeEngine([problem, code, ANSWER])

    import importlib.util
    tool_file = Path(__file__).resolve().parents[1] / "tools" / "opencv_editor" / "tool.py"
    spec = importlib.util.spec_from_file_location("_tool_opencv_editor_clean", tool_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tool = mod.OpenCV_Editor_Tool()

    out = asyncio.run(MATSolver(engine, tool, work_dir=work).solve("q", corrupted))  # default cleanup
    assert out.code_exec_oks == [True]
    # the original corrupted image is untouched; the processed temp file is gone
    assert os.path.exists(corrupted)
    leftovers = [f for f in os.listdir(work) if f.startswith("mat_step_")]
    assert leftovers == []


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
