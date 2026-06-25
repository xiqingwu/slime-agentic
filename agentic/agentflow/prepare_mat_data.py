"""Convert Visual-ARFT MAT-Coding data into AgentFlow online-rollout start points.

Plan reference: `docs/algorithms/plan_agentflow_visual.md` §5.1.

Visual-ARFT's `rft_agent_code_1_2k.json` (HuggingFace `laolao77/MAT`,
`MAT-Training/rft_agent_code_1_2k.json`) is **offline pre-split**: each trajectory
is exploded into 2-3 rows (`pre_problem` -> [`pre_code`] -> `pre_answer`), with a
teacher-written per-step `solution` and `context`.

AgentFlow runs the model **online**, so we only keep **one start point per
trajectory** and throw away the teacher steps/context. We do extract two
trajectory-level, path-independent ground truths the §4.0 rewards need:
  - ``corruption_gt``  — from the `pre_problem` row's ``<problem>{...}</problem>``
                          (for ``diagnosis_reward``)
  - ``gt``             — the final answer from the `pre_answer` row's ``<answer>``
                          (for ``outcome_reward``)

Output is JSONL (the only text format slime's ``read_file`` accepts besides
parquet), with fields matching the training launch flags:
    --input-key problem  --label-key gt  --multimodal-keys '{"image":"image_path"}'
``image_path`` is a **list** and ``problem`` carries the ``<image>`` placeholder,
both required by slime's multimodal dataset loader (slime/utils/data.py).

Schema of an input row (observed in the real data)::

    {"type": "pre_problem", "image_path": "overexposure_60_proc.png",
     "problem": "<query> ... </query>",
     "solution": "<problem> {'overexposure'} </problem>",
     "gt": "<think> ... </think>\\n<problem> {'overexposure'} </problem>"}
    {"type": "pre_code",   "image_path": "overexposure_60_proc.png", "context": "...",
     "solution": "<code>\\n```python ... ```\\n</code>", "gt": "..."}
    {"type": "pre_answer", "image_path": "overexposure_60_ori.png",  "context": "...",
     "solution": "<answer> 81 </answer>", "gt": "..."}

Note the input image differs per step: ``*_proc`` (corrupted, the agent's start)
vs ``*_ori`` (clean, used by the teacher's answer step). Trajectories are grouped
by the shared base id (filename with the ``_proc`` / ``_ori`` suffix stripped).

Usage::

    python agentic/agentflow/prepare_mat_data.py \\
        --input rft_agent_code_1_2k.json \\
        --output mat_coding_agentflow.jsonl \\
        --image-root /data/MAT/MAT-Training/images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

# Reuse the verbatim Visual-ARFT primitive so corruption parsing matches exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.mat_rewards import extract_problems  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_PLACEHOLDER = "<image>"
# Strip a trailing _proc / _ori (with optional extension) to get the trajectory id.
_SUFFIX_RE = re.compile(r"_(proc|ori)(\.[A-Za-z0-9]+)?$")


def trajectory_id(image_path: str) -> str:
    """`overexposure_60_proc.png` -> `overexposure_60` (groups steps of one traj)."""
    stem = os.path.basename(image_path)
    stripped = _SUFFIX_RE.sub("", stem)
    if stripped == stem:  # no _proc/_ori suffix -> fall back to extension-less name
        stripped = os.path.splitext(stem)[0]
    return stripped


def extract_answer(text: str) -> str:
    """Pull the answer out of ``<answer>...</answer>``; else strip & return as-is."""
    if not text:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return (m.group(1).strip() if m else text).strip()


def _resolve_image(name: str, image_root: str) -> str:
    return os.path.join(image_root, name) if image_root else name


def _group_by_trajectory(rows: list[dict]) -> "OrderedDict[str, dict]":
    """Group pre-split rows by trajectory id -> {type: row} (order preserved)."""
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for row in rows:
        tid = trajectory_id(row.get("image_path", ""))
        groups.setdefault(tid, {})[row.get("type")] = row
    return groups


def clean_bases_from_mat(rows: list[dict], image_root: str = "") -> list[dict]:
    """Extract clean (image, question, answer) bases from MAT-Training rows.

    Used by ``synthesize_mat_data.py`` to mint new corruptions from the *clean*
    ``_ori`` images (zero extra data — multiplies the 1200 clean bases). Returns
    ``[{clean_image_path, question, answers, trajectory_id}]``.
    """
    bases = []
    for tid, steps in _group_by_trajectory(rows).items():
        pre_problem = steps.get("pre_problem")
        pre_answer = steps.get("pre_answer")
        if pre_problem is None or pre_answer is None:
            continue
        question = pre_problem.get("problem", "")
        answer = extract_answer(pre_answer.get("solution", "") or pre_answer.get("gt", ""))
        if not question or not answer:
            continue
        bases.append({
            "clean_image_path": _resolve_image(pre_answer["image_path"], image_root),  # the _ori
            "question": question,
            "answers": [answer],
            "trajectory_id": tid,
        })
    return bases


def convert(
    rows: list[dict],
    image_root: str = "",
    add_placeholder: bool = True,
) -> list[dict]:
    """Group pre-split rows into trajectories and emit AgentFlow start points."""
    groups = _group_by_trajectory(rows)

    out = []
    skipped = 0
    for tid, steps in groups.items():
        pre_problem = steps.get("pre_problem")
        pre_answer = steps.get("pre_answer")
        if pre_problem is None or pre_answer is None:
            skipped += 1
            logger.warning("traj %s missing pre_problem or pre_answer; skipping", tid)
            continue

        # corruption label(s): diagnosis GT, path-independent
        corruption_gt = extract_problems(pre_problem.get("solution", ""))
        # final answer: outcome GT
        gt = extract_answer(pre_answer.get("solution", "") or pre_answer.get("gt", ""))

        query = pre_problem.get("problem", "")
        problem = f"{IMAGE_PLACEHOLDER}\n{query}" if add_placeholder else query

        # Agent starts from the corrupted image (pre_problem's *_proc).
        corrupted_img = _resolve_image(pre_problem["image_path"], image_root)
        clean_img = _resolve_image(pre_answer["image_path"], image_root)

        out.append({
            "problem": problem,
            "image_path": [corrupted_img],          # slime requires a list; single source
                                                    # of truth for the corrupted-image path
                                                    # (the cv2 tool reads it via the prompt
                                                    # block at rollout time — B4)
            "gt": gt,
            "metadata": {
                "corruption_gt": corruption_gt,
                "trajectory_id": tid,
                "num_steps": len(steps),            # 3 normal, 2 for 'none'
                "clean_image_path": clean_img,      # reference (not fed to the model)
            },
        })

    if skipped:
        logger.warning("skipped %d incomplete trajectories", skipped)
    return out


def convert_benchmark(
    rows: list[dict],
    image_root: str = "",
    add_placeholder: bool = True,
) -> list[dict]:
    """Convert MAT-Benchmark rows (MAT-Coding.json) to the same start-point format.

    Benchmark schema differs from training: one row per item with
    ``{id, question, answers(list), processed_image_path, ori_image_path,
    type(list), split}``. ``processed_image_path`` is the corrupted input.
    ``gt`` = first answer (full list kept in metadata for max-over-answers scoring).
    """
    out = []
    for row in rows:
        answers = row.get("answers") or []
        gt = str(answers[0]) if answers else ""
        query = row.get("question", "")
        problem = f"{IMAGE_PLACEHOLDER}\n{query}" if add_placeholder else query
        corrupted = _resolve_image(row["processed_image_path"], image_root)
        clean = _resolve_image(row.get("ori_image_path", ""), image_root) if row.get("ori_image_path") else ""
        out.append({
            "problem": problem,
            "image_path": [corrupted],              # single source of truth for the path (B4)
            "gt": gt,
            "metadata": {
                "corruption_gt": sorted(row.get("type") or []),
                "id": row.get("id"),
                "split": row.get("split"),
                "answers": answers,                 # all valid answers (max-F1/EM scoring)
                "clean_image_path": clean,
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="rft_agent_code_1_2k.json (train) or MAT-Coding.json (--benchmark)")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--image-root", default="", help="prefix prepended to every image filename")
    ap.add_argument("--no-image-placeholder", action="store_true",
                    help="do not prepend '<image>' to the problem text")
    ap.add_argument("--benchmark", action="store_true",
                    help="input is MAT-Benchmark/MAT-Coding.json (one row per item) instead of training data")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open(args.input, encoding="utf-8") as f:
        rows = json.load(f)
    logger.info("loaded %d rows from %s", len(rows), args.input)

    if args.benchmark:
        samples = convert_benchmark(rows, image_root=args.image_root, add_placeholder=not args.no_image_placeholder)
    else:
        samples = convert(rows, image_root=args.image_root, add_placeholder=not args.no_image_placeholder)

    with open(args.output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("wrote %d trajectories to %s", len(samples), args.output)


if __name__ == "__main__":
    main()
