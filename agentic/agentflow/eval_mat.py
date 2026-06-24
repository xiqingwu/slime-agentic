#!/usr/bin/env python3
"""MAT-Coding standalone evaluation (plan §8). Forward inference only.

Runs the online MAT loop (MATSolver + OpenCV tool) over MAT-Bench and reports
F1/EM overall and per split (simple/hard). With ``--baseline`` it also runs a
no-tool single-turn pass on the corrupted image and computes Call Gain / Call Harm
(does the tool *selectively* help — plan §1.3 / §8).

Input is the JSONL produced by:
    python prepare_mat_data.py --benchmark \\
        --input /data/MAT/MAT-Benchmark/MAT-Coding.json \\
        --output /data/MAT/mat_bench_agentflow.jsonl \\
        --image-root /data/MAT/MAT-Benchmark/MAT-Coding-image

Typical usage (server already running):
    python eval_mat.py --model /data/AgentFlow_Qwen3VL_MAT/ \\
        --eval-data /data/MAT/mat_bench_agentflow.jsonl \\
        --baseline --output mat_eval_results.json

GPU-only (real Qwen3-VL processor + SGLang). The loop/metric logic is covered
offline by tests/test_mat_solver.py and tests/test_mat_eval.py.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from core.llm_engine import SGLangEngine, messages_have_images  # noqa: E402
from core.mat_solver import MATSolver, SYSTEM_PROMPT_AGENT_CODE, baseline_answer  # noqa: E402
from core.mat_eval_metrics import (  # noqa: E402
    score_one, aggregate, aggregate_by_type, call_gain_harm, diagnosis_accuracy,
)
from core.mat_rewards import diagnosis_reward, mechanism_means  # noqa: E402
from core.tool_loader import load_opencv_tool, tool_timeout_from_env  # noqa: E402
import slime.utils.http_utils as _http_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_mat")


def _init_http_client(concurrency: int = 256) -> None:
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )


def load_benchmark(path: str) -> list[dict]:
    """Load the converted benchmark JSONL into eval items."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            meta = obj.get("metadata", {}) or {}
            question = re.sub(r"^<image>\s*", "", obj.get("problem", ""))
            # image_path (list) is the single source of truth for the path (B4);
            # fall back to the legacy metadata field for older eval JSONLs.
            image_path = (obj.get("image_path") or [None])[0] or meta.get("input_image_path")
            items.append({
                "question": question,
                "image_path": image_path,
                "answers": meta.get("answers") or ([obj["gt"]] if obj.get("gt") else []),
                "split": meta.get("split"),
                "id": meta.get("id"),
                "corruption_gt": meta.get("corruption_gt") or [],   # for type breakdown + diagnosis
            })
    return items


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return (m.group(1).strip() if m else text).strip()


async def _eval_one(engine, tool, item, max_steps, sem, idx, total, run_baseline):
    async with sem:
        question, image_path, answers = item["question"], item["image_path"], item["answers"]
        corruption_gt = item.get("corruption_gt") or []
        type_key = "+".join(corruption_gt) if corruption_gt else "unknown"
        rec = {"id": item.get("id"), "split": item.get("split"), "type": type_key,
               "question": question[:80]}

        # tool run
        planner_steps, code_exec_oks = [], []
        try:
            solver = MATSolver(engine, tool, max_steps=max_steps, tool_timeout=tool_timeout_from_env())
            out = await solver.solve(question, image_path)
            tool_pred = _extract_answer(out.final_output or "")
            planner_steps = out.planner_steps or []
            code_exec_oks = out.code_exec_oks or []
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%d/%d] solver error: %s", idx + 1, total, exc)
            tool_pred = ""
        s = score_one(tool_pred, answers)
        rec.update({"pred": tool_pred, "f1": s["f1"], "em": s["em"]})

        # mechanism signals (selective-tool-use story)
        problem_steps = [st for st in planner_steps if st.get("type") == "problem"]
        diag_correct = bool(problem_steps) and (
            diagnosis_reward(problem_steps[0].get("content", ""), corruption_gt) == 1.0
        )
        rec.update({
            "diagnosis_correct": diag_correct,
            "n_steps": len(planner_steps),
            "code_exec_oks": code_exec_oks,
            "tool_called": bool(code_exec_oks),
        })

        # optional no-tool baseline (shared definition with training A4 — B5)
        if run_baseline:
            try:
                base_pred = await baseline_answer(engine, question, image_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%d/%d] baseline error: %s", idx + 1, total, exc)
                base_pred = ""
            bs = score_one(base_pred, answers)
            rec.update({"baseline_pred": base_pred, "baseline_f1": bs["f1"], "baseline_em": bs["em"]})

        logger.info("[%d/%d] %s em=%d f1=%.2f", idx + 1, total, rec["split"], rec["em"], rec["f1"])
        return rec


async def run_eval(items, url, tokenizer, processor, sampling_params, concurrency, max_steps, run_baseline):
    _init_http_client(concurrency=concurrency * 4)
    engine = SGLangEngine(
        url=url, tokenizer=tokenizer, processor=processor,
        sampling_params=sampling_params, max_new_tokens=sampling_params.get("max_new_tokens", 2048),
    )
    tool = load_opencv_tool()
    sem = asyncio.Semaphore(concurrency)
    total = len(items)
    tasks = [_eval_one(engine, tool, it, max_steps, sem, i, total, run_baseline) for i, it in enumerate(items)]
    return await asyncio.gather(*tasks)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model path (for tokenizer + processor)")
    p.add_argument("--url", default="http://127.0.0.1:30000/generate", help="SGLang /generate URL")
    p.add_argument("--eval-data", required=True, help="converted MAT-Bench JSONL (prepare_mat_data --benchmark)")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--baseline", action="store_true", help="also run no-tool baseline + Call Gain/Harm")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=5)
    p.add_argument("--output", default="mat_eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from slime.utils.processing_utils import load_processor

    logger.info("loading tokenizer + processor: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = load_processor(args.model, trust_remote_code=True)
    if processor is None:
        logger.error("no processor loaded — is %s a VLM checkpoint?", args.model)
        sys.exit(1)

    items = load_benchmark(args.eval_data)
    if args.num_samples:
        items = items[:args.num_samples]
    logger.info("loaded %d benchmark items", len(items))

    sampling_params = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    t0 = time.time()
    records = asyncio.run(run_eval(
        items, args.url, tokenizer, processor, sampling_params,
        args.concurrency, args.max_steps, args.baseline,
    ))
    elapsed = time.time() - t0

    # mechanism stats: avg steps / tool-call rate / code-exec success + diagnosis acc
    metas = [{"planner_steps": [None] * r.get("n_steps", 0), "code_exec_oks": r.get("code_exec_oks") or []}
             for r in records]
    mechanism = {**mechanism_means(metas), "diagnosis_accuracy": diagnosis_accuracy(records)}

    summary = {
        "tool": aggregate(records),
        "tool_by_type": aggregate_by_type(records),
        "mechanism": mechanism,
        "elapsed_seconds": round(elapsed, 2),
    }
    if args.baseline:
        base_records = [{"f1": r.get("baseline_f1", 0.0), "em": r.get("baseline_em", 0), "split": r["split"]}
                        for r in records]
        summary["baseline"] = aggregate(base_records)
        pairs = [(bool(r.get("baseline_em")), bool(r.get("em"))) for r in records]
        summary["call_gain_harm"] = call_gain_harm(pairs)
        # per-type gain/harm: does the tool selectively help some corruptions, hurt others?
        summary["call_gain_harm_by_type"] = {
            t: call_gain_harm([(bool(r.get("baseline_em")), bool(r.get("em")))
                               for r in records if r.get("type") == t])
            for t in sorted({r.get("type") for r in records})
        }

    out = {"summary": summary, "details": records}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("results saved to %s", args.output)

    # console summary
    print("\n" + "=" * 60)
    for split, blk in summary["tool"].items():
        print(f"  tool/{split:8s}  EM={blk['em']:.3f}  F1={blk['f1']:.3f}  (n={blk['n']})")
    print("  " + "-" * 56)
    for t, blk in summary["tool_by_type"].items():
        print(f"  type/{t:14s}  EM={blk['em']:.3f}  F1={blk['f1']:.3f}  (n={blk['n']})")
    m = summary["mechanism"]
    print("  " + "-" * 56)
    print(f"  mechanism: diag_acc={m.get('diagnosis_accuracy', 0):.3f}  "
          f"tool_call_rate={m.get('tool_call_rate', 0):.3f}  "
          f"code_exec_ok={m.get('code_exec_success_rate', 0):.3f}  "
          f"avg_steps={m.get('avg_steps', 0):.2f}")
    if args.baseline:
        print("  " + "-" * 56)
        for split, blk in summary["baseline"].items():
            print(f"  base/{split:8s}  EM={blk['em']:.3f}  F1={blk['f1']:.3f}  (n={blk['n']})")
        g = summary["call_gain_harm"]
        print(f"  Call Gain={g['gain']} (rate {g['gain_rate']:.2f})  "
              f"Call Harm={g['harm']} (rate {g['harm_rate']:.2f})  net={g['net']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
