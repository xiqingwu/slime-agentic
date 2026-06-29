"""
AgentFlow rollout for single-GPU — all roles use the slime-managed SGLang engine.

Based on rollout.py with VLM processor support. Diff from rollout.py: hardcoded
ports (30000/30001) replaced with the slime router for single-GPU colocation.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.rollout.rm_hub.math_dapo_utils import last_boxed_only_string, remove_boxed, normalize_final_answer
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_rollout_step
from core.llm_engine import SGLangEngine
from core.solver import Solver
from core.rewarder import Rewarder

TOOLS_DIR = Path(__file__).parent / "tools"

# ── Patch Solver for 0.5B model: reduce max tokens and max steps ───────────────
from core import solver as _solver_mod
_solver_mod.Solver.MAX_TOTAL_TOKENS = 8192   # fit in 16K context window
_SOLVER_MAX_STEPS = 3  # fewer steps for smaller model


# ── Generate ──────────────────────────────────────────────────────────────────


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False) -> Sample:
    """Single-GPU AgentFlow: all models use the colocated slime router."""

    assert not getattr(args, "partial_rollout", False), \
        "Partial rollout is not supported for AgentFlow."

    state = GenerateState(args)
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # processor is None for text-only models; for VLMs it enables multimodal path
    processor = getattr(state, "processor", None)
    shared_engine = SGLangEngine(
        url=router_url,
        tokenizer=state.tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=4096,
        processor=processor,
    )

    question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata["original_question"] = question

    try:
        engine_map = {
            "default":       shared_engine,
            "planner":       shared_engine,
            "executor":      shared_engine,
            "verifier":      shared_engine,
            "base_generator": shared_engine,
            "python_coder":  shared_engine,   # no separate coder on 1 GPU
            "final_output":  shared_engine,
        }
        solver = Solver(engine_map=engine_map, tools_dir=str(TOOLS_DIR), max_steps=_SOLVER_MAX_STEPS)
        label = str(sample.label) if sample.label is not None else None
        out = await solver.solve(question, label=label)
        if out is None:
            sample.status = Sample.Status.ABORTED
            sample.rollout_log_probs = []
            return sample

        sample.prompt          = out.prompt_text
        sample.response        = out.response
        sample.tokens          = out.prompt_token_ids + out.token_ids
        sample.response_length = len(out.token_ids)
        sample.loss_mask       = out.loss_mask if out.loss_mask is not None else [1] * len(out.token_ids)
        sample.rollout_log_probs = out.log_probs
        sample.status = Sample.Status.TRUNCATED if out.finish_reason == "length" else Sample.Status.COMPLETED
        sample.metadata["final_output"] = out.final_output or ""

        if out.turns:
            sample.train_metadata = {"turns": out.turns}

    except Exception:
        traceback.print_exc()
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED

    return sample


# ── Reward function ───────────────────────────────────────────────────────────

def _extract_final_answer(response: str) -> str:
    boxed = last_boxed_only_string(response)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))
    lines = response.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        match = re.search(r'(?:answer|result|is|=)\s*[:\s]*\$?\\?boxed\{([^}]+)\}', line, re.IGNORECASE)
        if match:
            return normalize_final_answer(match.group(1))
        match_num = re.search(r'(?:answer|result|is|=)\s*[:\s]*\$?\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)', line, re.IGNORECASE)
        if match_num:
            return match_num.group(1).strip()
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line[:200]
    return ""


async def reward_func(args: Any, sample: Sample, **kwargs) -> dict:
    state = GenerateState(args)
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    engine = SGLangEngine(
        url=router_url,
        tokenizer=state.tokenizer,
        sampling_params={},
        max_new_tokens=2048,
    )
    rewarder = Rewarder(llm_engine=engine)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = metadata.get("original_question", "")
    if not question:
        question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]
    label = str(sample.label) if sample.label is not None else ""
    final_output = metadata.get("final_output", "") or sample.response or ""
    boxed = last_boxed_only_string(final_output)
    pred = normalize_final_answer(remove_boxed(boxed)) if boxed else _extract_final_answer(final_output)

    if pred and label and pred == label:
        score = 1.0
    else:
        score = await rewarder.compute_reward(
            question=question,
            model_response=final_output,
            groundtruth=label,
        )
    return {"score": score, "acc": score == 1.0, "pred": pred, "gt": label}


# ── Eval log ──────────────────────────────────────────────────────────────────

EVAL_SCORES_FILE = Path(__file__).parent / "eval_scores.json"
_eval_initialized = False


def _ensure_eval_file():
    global _eval_initialized
    if not _eval_initialized:
        if EVAL_SCORES_FILE.exists():
            EVAL_SCORES_FILE.unlink()
        EVAL_SCORES_FILE.write_text("[]")
        _eval_initialized = True


def eval_log(rollout_id, args, data, extra_metrics) -> bool:
    _ensure_eval_file()
    step = compute_rollout_step(args, rollout_id)
    all_entries = []
    for dataset_name, dataset_data in data.items():
        rewards = dataset_data.get("rewards", [])
        samples = dataset_data.get("samples", [])
        details = []
        for i, sample in enumerate(samples):
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            question = metadata.get("original_question", "")
            if not question:
                question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]
            label = str(sample.label) if sample.label is not None else ""
            final_out = metadata.get("final_output", "") or sample.response or ""
            boxed = last_boxed_only_string(final_out)
            pred = normalize_final_answer(remove_boxed(boxed)) if boxed else _extract_final_answer(final_out)
            score = rewards[i] if i < len(rewards) else None
            details.append({
                "question": question, "pred": pred, "label": label,
                "score": score, "final_output": final_out,
            })
        mean_score = sum(rewards) / len(rewards) if rewards else 0.0
        all_entries.append({
            "dataset": dataset_name, "step": step, "rollout_id": rollout_id,
            "mean_score": mean_score, "num_samples": len(samples), "details": details,
        })
    try:
        existing = json.loads(EVAL_SCORES_FILE.read_text())
    except Exception:
        existing = []
    existing.extend(all_entries)
    EVAL_SCORES_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return False
