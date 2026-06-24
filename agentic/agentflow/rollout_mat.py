"""Slime rollout hooks for MAT-Coding visual agentic RL (plan M3 wiring).

Mirrors ``rollout.py`` (math AgentFlow) but drives the multimodal ``MATSolver``
and the rule-based ``mat_reward_func``. Wire it in the training launch script:

    --custom-generate-function-path  rollout_mat.generate
    --custom-rm-path                 core.mat_rewards.mat_reward_func
    --custom-eval-rollout-log-function-path  rollout_mat.eval_log   # optional

The dataset is produced by ``prepare_mat_data.py``; each sample carries
``metadata.input_image_path`` (the corrupted image the cv2 tool reads) and
``metadata.corruption_gt`` (the diagnosis GT).

The online loop length is capped by ``MAT_MAX_STEPS`` (env var, default 5); the
training launcher exports it so it's configurable without a slime argparser flag.

Runtime-only: imports slime; not exercised by the offline test-suite (the loop
logic itself is covered by tests/test_mat_solver.py with fakes). Validate
end-to-end on the GPU box with a real Qwen3-VL processor + SGLang.
"""

from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

from core.llm_engine import SGLangEngine
from core.mat_solver import MATSolver, baseline_answer, single_flight

_OPENCV_TOOL_TIMEOUT = 30
_DEFAULT_MAX_STEPS = 5
_DEFAULT_MAX_NEW_TOKENS = 1024  # MAT steps (problem/code/answer) are short; 2048 wasted budget

# In-flight registry so the N identical greedy baselines of one GRPO group collapse
# to a single computation (R3). Keyed by (question, image_path); cleared per call.
_baseline_registry: dict = {}


def _correction_enabled() -> bool:
    """A4 correction reward needs a per-rollout no-tool baseline. On by default."""
    return os.environ.get("MAT_CORRECTION_REWARD", "1").lower() in ("1", "true", "yes")


def _max_new_tokens() -> int:
    return int(os.environ.get("MAT_MAX_NEW_TOKENS", _DEFAULT_MAX_NEW_TOKENS))


def _greedy_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Deterministic sampling so the baseline is identical across a GRPO group."""
    p = dict(sampling_params or {})
    p.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
    return p


_TOOL = None  # the OpenCV tool is stateless across calls -> instantiate once, share it


def _load_opencv_tool():
    """Instantiate the OpenCV editor tool once (cached).

    The tool is stateless (``execute`` takes everything as args), so one shared
    instance is safe across concurrent rollouts. Re-running ``exec_module`` per
    sample — as this did before — was pure overhead (1200×8×epoch module re-execs).
    """
    global _TOOL
    if _TOOL is None:
        import importlib.util
        from pathlib import Path

        tool_file = Path(__file__).parent / "tools" / "opencv_editor" / "tool.py"
        spec = importlib.util.spec_from_file_location("_tool_opencv_editor", tool_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TOOL = module.OpenCV_Editor_Tool()
    return _TOOL


def _question_and_image(sample: Sample) -> tuple[str, str | None]:
    """Recover the query text and the corrupted-image file path for the solver.

    The image path is read straight from ``sample.prompt``'s image block — the same
    field slime built from the dataset's ``image_path`` (verified that
    ``process_vision_info`` leaves the path string in place). That makes ``image_path``
    the single source of truth (B4); ``metadata['input_image_path']`` is only a
    legacy fallback for datasets built before this change.
    """
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    prompt = sample.prompt
    if isinstance(prompt, str):
        return prompt, meta.get("input_image_path")

    # conversation form: last user message; pull text blocks + the image path.
    for msg in reversed(prompt):
        if msg.get("role") != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            return content, meta.get("input_image_path")
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        images = [b.get("image") for b in content if b.get("type") == "image" and isinstance(b.get("image"), str)]
        image_path = images[0] if images else meta.get("input_image_path")
        return "\n".join(texts), image_path
    return str(prompt), meta.get("input_image_path")


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """Online MAT-Coding rollout: problem -> code -> answer with real cv2 execution."""
    assert not getattr(args, "partial_rollout", False), "Partial rollout unsupported for MAT generation."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    processor = getattr(state, "processor", None)
    engine = SGLangEngine(
        url=url, tokenizer=state.tokenizer, sampling_params=sampling_params,
        max_new_tokens=_max_new_tokens(), enable_thinking=False, processor=processor,
    )

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    question, image_path = _question_and_image(sample)
    sample.metadata["original_question"] = question

    try:
        if not image_path:
            raise ValueError("no corrupted-image path on the sample (prompt image block / "
                             "metadata['input_image_path']); check prepare_mat_data output")

        tool = _load_opencv_tool()
        max_steps = int(os.environ.get("MAT_MAX_STEPS", _DEFAULT_MAX_STEPS))
        solver = MATSolver(engine, tool, max_steps=max_steps)

        # Run the tool trajectory and the (independent) no-tool baseline concurrently;
        # both hit the same SGLang server, which batches them (R2). A4 baseline is
        # greedy so all GRPO samples of one prompt share an identical reference; off
        # via env to save compute — the reward then degrades to non-correction.
        solve_task = asyncio.ensure_future(solver.solve(question, image_path))
        base_task = (
            asyncio.ensure_future(
                single_flight(
                    _baseline_registry, (question, image_path),
                    lambda: baseline_answer(
                        engine, question, image_path, sampling_params=_greedy_params(sampling_params)),
                )
            )
            if _correction_enabled() else None
        )
        tasks = [solve_task] + ([base_task] if base_task is not None else [])
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = results[0]
        if isinstance(out, BaseException):
            raise out

        sample.prompt = out.prompt_text
        sample.response = out.response
        sample.tokens = out.prompt_token_ids + out.token_ids
        sample.response_length = len(out.token_ids)
        sample.loss_mask = out.loss_mask if out.loss_mask is not None else [1] * len(out.token_ids)
        sample.rollout_log_probs = out.log_probs
        sample.status = Sample.Status.TRUNCATED if out.finish_reason == "length" else Sample.Status.COMPLETED

        # Metadata the §4.0 rewards consume (corruption_gt already set by the dataset).
        sample.metadata["final_output"] = out.final_output or ""
        sample.metadata["planner_steps"] = out.planner_steps or []
        sample.metadata["code_exec_oks"] = out.code_exec_oks or []

        if base_task is not None:
            base = results[1]
            if isinstance(base, BaseException):
                traceback.print_exception(base)  # best-effort; don't fail the rollout
            else:
                sample.metadata["baseline_output"] = base

        if out.turns:
            sample.train_metadata = {"turns": out.turns}

    except Exception:
        traceback.print_exc()
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED

    return sample


def eval_log(rollout_id, args, data, extra_metrics) -> bool:
    """Defer to the framework's default eval logging."""
    return False
