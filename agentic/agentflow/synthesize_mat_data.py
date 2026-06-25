"""Programmatically synthesize MAT-Coding trajectory start points (plan §3.6 / D1).

Method C needs no teacher CoT, and the corruptions are deterministic cv2 ops, so we
can mint as many trajectory start points as we want from clean (image, question,
answer) bases — both the diagnosis GT (the corruption we apply) and the outcome GT
(the inherited answer) are known by construction. This is the lever for the "small
data (1200)" risk (plan §7.6): scale to 2k–5k.

Output schema is identical to ``prepare_mat_data.py`` (``problem`` with ``<image>`` /
``image_path`` list / ``gt`` / ``metadata.corruption_gt``), so it drops straight into
the training launcher. Corrupted images are written to ``--out-image-dir``.

Covered corruptions: rotation90/180, dark, overexposure, blur, noise, none.
**crop is excluded** (needs answer-region GT) — mix this output with the real
MAT-Training set (``prepare_mat_data.py``) for crop coverage:
    cat mat_coding_agentflow.jsonl mat_synth.jsonl > mat_train_all.jsonl

Usage (re-corrupt MAT's own clean images — zero extra data):
    python synthesize_mat_data.py \
        --from-mat /data/MAT/MAT-Training/rft_agent_code_1_2k.json \
        --image-root /data/MAT/MAT-Training/images \
        --num 4000 \
        --out-image-dir /data/MAT/synth_images \
        --output /data/MAT/mat_synth.jsonl

Or from a generic clean-VQA JSONL ({image_path|image, question, answer|answers}):
    python synthesize_mat_data.py --clean-data clean_vqa.jsonl --image-root /imgs ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.corruptions import apply_corruptions, SINGLE_CORRUPTIONS, ROTATIONS, DEGRADATIONS  # noqa: E402
from prepare_mat_data import IMAGE_PLACEHOLDER, clean_bases_from_mat, _resolve_image  # noqa: E402

logger = logging.getLogger(__name__)


def plan_corruptions(n: int, double_frac: float = 0.33, none_frac: float = 0.08, rng=None) -> list[list[str]]:
    """Assign a corruption-name list to each of ``n`` trajectories (pure, testable).

    Mirrors the MAT mix: ~8% none, ~33% double (rotation + one degradation, like the
    real set where doubles always include a rotation), the rest single corruptions
    spread evenly across SINGLE_CORRUPTIONS. Returns a shuffled list of name lists.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_none = round(n * none_frac)
    n_double = round(n * double_frac)
    n_single = max(0, n - n_none - n_double)

    plans: list[list[str]] = []
    for i in range(n_single):
        plans.append([SINGLE_CORRUPTIONS[i % len(SINGLE_CORRUPTIONS)]])
    for _ in range(n_double):
        rot = str(rng.choice(ROTATIONS))
        deg = str(rng.choice(DEGRADATIONS))
        plans.append(sorted([rot, deg]))   # sorted to match corruption_gt convention
    for _ in range(n_none):
        plans.append(["none"])

    rng.shuffle(plans)
    return plans


def build_row(question: str, answers, corrupted_path: str, corruption_names, *,
              clean_path: str | None = None, traj_id: str | None = None) -> dict:
    """Build one start-point row matching prepare_mat_data's schema (pure, testable)."""
    answers = [str(a) for a in (answers or []) if a not in (None, "")]
    return {
        "problem": f"{IMAGE_PLACEHOLDER}\n{question}",
        "image_path": [corrupted_path],
        "gt": answers[0] if answers else "",
        "metadata": {
            "corruption_gt": sorted(corruption_names),
            "answers": answers,
            "synthesized": True,
            "trajectory_id": traj_id,
            "clean_image_path": clean_path,
        },
    }


def _load_generic_bases(path: str, image_root: str) -> list[dict]:
    """Clean-VQA JSONL -> bases. Each line: {image_path|image, question, answer|answers}."""
    bases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            img = obj.get("image_path") or obj.get("image")
            question = obj.get("question") or obj.get("problem") or ""
            answers = obj.get("answers")
            if answers is None and obj.get("answer") is not None:
                answers = [obj["answer"]]
            if not img or not question or not answers:
                continue
            img = img[0] if isinstance(img, list) else img
            bases.append({
                "clean_image_path": _resolve_image(img, image_root),
                "question": question,
                "answers": list(answers),
                "trajectory_id": obj.get("id"),
            })
    return bases


def synthesize(bases: list[dict], num: int, out_image_dir: str, *,
               double_frac: float = 0.33, none_frac: float = 0.08, seed: int = 0) -> list[dict]:
    """Generate ``num`` synthesized rows, writing corrupted images to ``out_image_dir``."""
    import cv2

    if not bases:
        raise ValueError("no clean bases to synthesize from")
    os.makedirs(out_image_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    plans = plan_corruptions(num, double_frac=double_frac, none_frac=none_frac, rng=rng)

    rows, skipped = [], 0
    for i, names in enumerate(plans):
        base = bases[i % len(bases)]            # cycle bases -> each reused with varied corruptions
        clean_path = base["clean_image_path"]
        img = cv2.imread(clean_path)
        if img is None:
            skipped += 1
            continue
        corrupted = apply_corruptions(img, names, rng)
        out_name = f"{'+'.join(names)}_{i}_proc.png"
        out_path = os.path.join(out_image_dir, out_name)
        cv2.imwrite(out_path, corrupted)
        rows.append(build_row(
            base["question"], base["answers"], out_path, names,
            clean_path=clean_path, traj_id=f"synth_{i}",
        ))

    if skipped:
        logger.warning("skipped %d trajectories (unreadable clean image)", skipped)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-mat", help="MAT-Training rft_agent_code_1_2k.json (re-corrupt clean _ori images)")
    src.add_argument("--clean-data", help="generic clean-VQA JSONL ({image_path|image, question, answer|answers})")
    ap.add_argument("--image-root", default="", help="prefix prepended to clean image paths")
    ap.add_argument("--out-image-dir", required=True, help="dir to write synthesized corrupted images")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--num", type=int, required=True, help="number of trajectories to synthesize (e.g. 4000)")
    ap.add_argument("--double-frac", type=float, default=0.33, help="fraction of double corruptions")
    ap.add_argument("--none-frac", type=float, default=0.08, help="fraction of 'none' (no corruption)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.from_mat:
        with open(args.from_mat, encoding="utf-8") as f:
            mat_rows = json.load(f)
        bases = clean_bases_from_mat(mat_rows, image_root=args.image_root)
    else:
        bases = _load_generic_bases(args.clean_data, args.image_root)
    logger.info("loaded %d clean bases", len(bases))

    rows = synthesize(
        bases, args.num, args.out_image_dir,
        double_frac=args.double_frac, none_frac=args.none_frac, seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d synthesized trajectories to %s (images in %s)",
                len(rows), args.output, args.out_image_dir)


if __name__ == "__main__":
    main()
