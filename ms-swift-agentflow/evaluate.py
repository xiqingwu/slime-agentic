#!/usr/bin/env python3
"""Run standalone multi-turn MAT evaluation with ms-swift's VllmEngine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mat_agentflow.rewards import aggregate_reward  # noqa: E402


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


async def evaluate(args, rows):
    from swift.infer_engine import InferRequest, RequestConfig, VllmEngine
    from swift.infer_engine.protocol import RolloutInferRequest
    from mat_agentflow.plugin import MATMultiTurnScheduler

    engine = VllmEngine(
        args.model,
        adapters=[args.adapter] if args.adapter else None,
        use_async_engine=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_lora_rank=args.max_lora_rank,
        limit_mm_per_prompt={"image": args.max_turns + 1},
    )
    scheduler = MATMultiTurnScheduler(infer_engine=engine, max_turns=args.max_turns)
    requests = [RolloutInferRequest(
        messages=row["messages"], images=row.get("images") or [],
        data_dict={k: v for k, v in row.items() if k not in ("messages", "images")},
        uuid=str(row.get("trajectory_id", idx)),
    ) for idx, row in enumerate(rows)]
    config = RequestConfig(
        max_tokens=args.max_tokens, temperature=args.temperature, top_p=1.0,
        return_details=True, logprobs=True,
    )
    outputs = await scheduler.async_infer(requests, config, use_tqdm=True)
    baseline_responses = [None] * len(rows)
    if args.baseline:
        baseline_requests = []
        for row in rows:
            question = next((m["content"] for m in reversed(row["messages"]) if m["role"] == "user"), "")
            baseline_requests.append(InferRequest(
                messages=[{"role": "user", "content": (
                    "<image>Answer the visual question directly without tools. " + question.replace("<image>", "")
                )}],
                images=row.get("images") or [],
            ))
        baseline_responses = await asyncio.gather(*[
            engine.infer_async(req, RequestConfig(max_tokens=256, temperature=0.0))
            for req in baseline_requests
        ])
    records = []
    for row, output, baseline_response in zip(rows, outputs, baseline_responses, strict=True):
        info = output.rollout_infos or {}
        baseline_text = baseline_response.choices[0].message.content if baseline_response is not None else None
        score = aggregate_reward(
            info.get("final_output") or output.response.choices[0].message.content,
            info.get("planner_steps") or [], row.get("corruption_gt") or [],
            info.get("code_exec_oks") or [], row.get("answers") or [row.get("solution", "")],
            baseline_output=baseline_text,
        )
        record = {
            "trajectory_id": row.get("trajectory_id"), "score": score,
            "rollout_infos": {k: v for k, v in info.items() if k != "images"},
            "messages": output.messages,
        }
        if baseline_response is not None:
            baseline_score = aggregate_reward(
                baseline_text, [], row.get("corruption_gt") or [], [],
                row.get("answers") or [row.get("solution", "")],
            )
            record["baseline"] = {"prediction": baseline_text, **baseline_score}
        records.append(record)
    return records


def summarize(records):
    keys = ("score", "outcome", "em", "format", "diagnosis", "code_exec", "correction")
    result = {key: sum(r["score"][key] for r in records) / max(len(records), 1) for key in keys}
    infos = [r["rollout_infos"] for r in records]
    result["tool_call_rate"] = sum(bool(x.get("code_exec_oks")) for x in infos) / max(len(infos), 1)
    attempts = [ok for x in infos for ok in (x.get("code_exec_oks") or [])]
    result["code_exec_success_rate"] = sum(bool(x) for x in attempts) / max(len(attempts), 1)
    result["avg_turns"] = sum(len(x.get("planner_steps") or []) for x in infos) / max(len(infos), 1)
    paired = [r for r in records if "baseline" in r]
    if paired:
        gain = sum(r["baseline"]["em"] == 0 and r["score"]["em"] == 1 for r in paired)
        harm = sum(r["baseline"]["em"] == 1 and r["score"]["em"] == 0 for r in paired)
        result.update({
            "baseline_em": sum(r["baseline"]["em"] for r in paired) / len(paired),
            "call_gain": gain,
            "call_harm": harm,
            "call_net": gain - harm,
        })
    result["n"] = len(records)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base Qwen3-VL model")
    parser.add_argument("--adapter", help="optional SFT/GRPO LoRA adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="mat_eval_results.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-lora-rank", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--baseline", action="store_true", help="also run direct no-tool VQA")
    args = parser.parse_args()
    rows = read_jsonl(args.data, args.limit)
    records = asyncio.run(evaluate(args, rows))
    payload = {"summary": summarize(records), "records": records}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
