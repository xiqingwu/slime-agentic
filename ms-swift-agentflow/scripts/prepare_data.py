#!/usr/bin/env python3
"""Prepare ms-swift SFT and GRPO JSONL from raw MAT teacher data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mat_agentflow.protocol import SYSTEM_PROMPT, extract_answer, extract_problems  # noqa: E402


def trajectory_id(path: str) -> str:
    name = os.path.basename(path)
    return re.sub(r"_(?:proc|ori)(?:\.[^.]+)?$", "", name)


def group_rows(rows):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault(trajectory_id(row.get("image_path", "")), {})[row.get("type")] = row
    return groups


def target(row, tag):
    for key in ("gt", "solution"):
        text = str(row.get(key) or "").strip()
        if re.search(rf"<{tag}>.*?</{tag}>", text, re.DOTALL):
            return text
    return ""


def image_path(root, name):
    return os.path.abspath(os.path.join(root, name) if root else name)


def prepare_raw(rows, image_root):
    sft, rl = [], []
    for tid, steps in group_rows(rows).items():
        first, final = steps.get("pre_problem"), steps.get("pre_answer")
        if not first or not final:
            continue
        question = first.get("problem", "")
        answer = extract_answer(final.get("solution") or final.get("gt") or "")
        corruption = extract_problems(first.get("solution", ""))
        corrupted = image_path(image_root, first["image_path"])
        rl.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
            ],
            "images": [corrupted],
            "solution": answer,
            "answers": [answer],
            "corruption_gt": corruption,
            "trajectory_id": tid,
            "input_image": corrupted,
        })
        for row_type, tag in (("pre_problem", "problem"), ("pre_code", "code"), ("pre_answer", "answer")):
            row = steps.get(row_type)
            if not row:
                continue
            assistant = target(row, tag)
            if not assistant:
                continue
            context = str(row.get("context") or "").strip()
            prompt = question + (f"\n{context}" if context else "")
            sft.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{prompt}"},
                    {"role": "assistant", "content": assistant},
                ],
                "images": [image_path(image_root, row["image_path"])],
                "trajectory_id": tid,
                "step_type": tag,
            })
    return sft, rl


def convert_agentflow(rows):
    output = []
    for idx, row in enumerate(rows):
        meta = row.get("metadata") or {}
        images = row.get("image_path") or []
        images = [os.path.abspath(x) for x in images]
        answer = str(row.get("gt") or "")
        output.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.get("problem", "")},
            ],
            "images": images,
            "solution": answer,
            "answers": meta.get("answers") or [answer],
            "corruption_gt": meta.get("corruption_gt") or [],
            "trajectory_id": meta.get("trajectory_id") or f"converted_{idx}",
            "input_image": images[0] if images else "",
        })
    return output


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    raw = sub.add_parser("raw-mat")
    raw.add_argument("--input", required=True)
    raw.add_argument("--image-root", required=True)
    raw.add_argument("--sft-output", required=True)
    raw.add_argument("--rl-output", required=True)
    converted = sub.add_parser("agentflow-jsonl")
    converted.add_argument("--input", required=True)
    converted.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "raw-mat":
        with open(args.input, encoding="utf-8") as handle:
            rows = json.load(handle)
        sft, rl = prepare_raw(rows, args.image_root)
        write_jsonl(args.sft_output, sft)
        write_jsonl(args.rl_output, rl)
        print(json.dumps({"sft": len(sft), "rl": len(rl)}, indent=2))
    else:
        rows = read_jsonl(args.input)
        output = convert_agentflow(rows)
        write_jsonl(args.output, output)
        print(json.dumps({"rl": len(output)}, indent=2))


if __name__ == "__main__":
    main()
