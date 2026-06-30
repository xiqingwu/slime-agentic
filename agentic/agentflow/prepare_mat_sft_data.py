"""Convert MAT teacher trajectories into multimodal SFT examples.

Each original ``pre_problem`` / ``pre_code`` / ``pre_answer`` row becomes one
independent user-assistant training example.  This mirrors the online solver's
per-turn training layout while preserving the teacher's code demonstrations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

from core.mat_solver import SYSTEM_PROMPT_AGENT_CODE
from prepare_mat_data import _group_by_trajectory

logger = logging.getLogger(__name__)

STEP_TAGS = {
    "pre_problem": "problem",
    "pre_code": "code",
    "pre_answer": "answer",
}


def _resolve_image(name: str, image_root: str) -> str:
    return os.path.join(image_root, name) if image_root else name


def _teacher_target(row: dict, tag: str) -> str:
    """Prefer the reasoning-rich target when it contains the expected step tag."""
    for key in ("gt", "solution"):
        value = str(row.get(key) or "").strip()
        if re.search(rf"<{tag}>.*?</{tag}>", value, re.DOTALL):
            return value
    return ""


def convert_sft(rows: list[dict], image_root: str = "") -> list[dict]:
    """Convert pre-split MAT rows to one multimodal SFT example per teacher step."""
    output = []
    skipped = 0

    for trajectory_id, steps in _group_by_trajectory(rows).items():
        first = steps.get("pre_problem")
        if first is None or not first.get("problem"):
            skipped += len(steps)
            continue
        question = first["problem"]

        for step_type in ("pre_problem", "pre_code", "pre_answer"):
            row = steps.get(step_type)
            if row is None:
                continue
            tag = STEP_TAGS[step_type]
            target = _teacher_target(row, tag)
            image_name = row.get("image_path")
            if not target or not image_name:
                skipped += 1
                logger.warning("skipping %s/%s: missing image or <%s> target", trajectory_id, step_type, tag)
                continue

            context = str(row.get("context") or "").strip()
            user_text = f"{SYSTEM_PROMPT_AGENT_CODE}\n{question}"
            if context:
                user_text += f"\n{context}"

            output.append({
                "messages": [
                    {"role": "user", "content": f"<image>\n{user_text}"},
                    {"role": "assistant", "content": target},
                ],
                "image_path": [_resolve_image(image_name, image_root)],
                "metadata": {
                    "trajectory_id": trajectory_id,
                    "step_type": tag,
                },
            })

    if skipped:
        logger.warning("skipped %d malformed teacher steps", skipped)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="MAT rft_agent_code_1_2k.json")
    parser.add_argument("--output", required=True, help="output JSONL")
    parser.add_argument("--image-root", default="", help="prefix for MAT image paths")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with open(args.input, encoding="utf-8") as handle:
        rows = json.load(handle)
    samples = convert_sft(rows, image_root=args.image_root)
    with open(args.output, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    counts = {tag: sum(s["metadata"]["step_type"] == tag for s in samples) for tag in STEP_TAGS.values()}
    logger.info("wrote %d SFT steps to %s: %s", len(samples), args.output, counts)


if __name__ == "__main__":
    main()
