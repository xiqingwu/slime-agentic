"""ms-swift plugin registering MAT multi-turn rollout and reward."""

from __future__ import annotations

import asyncio
import os

from swift.infer_engine import InferRequest, RequestConfig
from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from .image_executor import execute_opencv
from .protocol import (STEP_ANSWER, STEP_CODE, STEP_PROBLEM, build_tip, classify_step,
                       extract_answer, extract_problems)
from .rewards import aggregate_reward


class MATMultiTurnScheduler(MultiTurnScheduler):
    """Online diagnose -> OpenCV repair -> answer loop with dynamic images."""

    state_key = "_mat_agentflow_state"

    def _state(self, request):
        if request.data_dict is None:
            request.data_dict = {}
        return request.data_dict.setdefault(self.state_key, {
            "planner_steps": [],
            "code_exec_oks": [],
            "final_output": "",
            "tool_messages": [],
            "baseline_output": None,
        })

    def _infos(self, request):
        state = self._state(request)
        return {
            "planner_steps": list(state["planner_steps"]),
            "code_exec_oks": list(state["code_exec_oks"]),
            "final_output": state["final_output"],
            "tool_messages": list(state["tool_messages"]),
            "baseline_output": state["baseline_output"],
            "images": request.images,
        }

    async def on_trajectory_start(self, requests):
        for request in requests:
            self._state(request)
        if os.environ.get("MAT_CORRECTION_REWARD", "1").lower() not in ("1", "true", "yes"):
            return
        baseline_requests = []
        for request in requests:
            question = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
            baseline_requests.append(InferRequest(
                messages=[{"role": "user", "content": (
                    "<image>Answer this visual question directly without tools: " + question.replace("<image>", "")
                )}],
                images=request.images[:1],
            ))
        config = RequestConfig(max_tokens=256, temperature=0.0, top_p=1.0)
        responses = await asyncio.gather(*[
            self.infer_engine.infer_async(request, config) for request in baseline_requests
        ])
        for request, response in zip(requests, responses, strict=True):
            self._state(request)["baseline_output"] = response.choices[0].message.content

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        step_type = classify_step(completion)
        state = self._state(infer_request)
        state["planner_steps"].append({"type": step_type, "content": completion})
        if step_type == STEP_ANSWER:
            state["final_output"] = extract_answer(completion)
        done = step_type in (STEP_ANSWER, None)
        return {"done": done, "rollout_infos": self._infos(infer_request)}

    def check_finished(self, infer_request, response_choice, current_turn):
        if super().check_finished(infer_request, response_choice, current_turn):
            return True
        return classify_step(response_choice.message.content or "") in (STEP_ANSWER, None)

    def step(self, infer_request, response_choice, current_turn):
        completion = response_choice.message.content or ""
        step_type = classify_step(completion)
        state = self._state(infer_request)

        if step_type == STEP_PROBLEM:
            observation = build_tip(extract_problems(completion))
            infer_request.messages.append({"role": "user", "content": observation})
        elif step_type == STEP_CODE:
            image_value = infer_request.images[-1] if infer_request.images else None
            result = execute_opencv(
                completion, image_value, timeout=int(os.environ.get("MAT_TOOL_TIMEOUT", "30"))
            )
            state["code_exec_oks"].append(bool(result.success and result.changed))
            if result.success:
                infer_request.images.append(result.image)
                status = "image updated" if result.changed else "image unchanged"
                observation = f"<tool_response>code executed; {status}.<image></tool_response>"
            else:
                observation = f"<tool_response>code failed: {result.error}</tool_response>"
            state["tool_messages"].append(observation)
            infer_request.messages.append({"role": "user", "content": observation})
        else:
            infer_request.messages.append({"role": "user", "content": "Invalid action; follow the MAT protocol."})

        token_ids = list(response_choice.token_ids or [])
        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": [1] * len(token_ids),
            "rollout_infos": self._infos(infer_request),
        }


class MATReward(ORM):
    """Rule reward consuming dataset labels and scheduler rollout metadata."""

    def __call__(self, completions, solution=None, answers=None, corruption_gt=None,
                 rollout_infos=None, **kwargs):
        count = len(completions)
        solutions = solution or [""] * count
        answer_lists = answers or [[x] for x in solutions]
        corruptions = corruption_gt or [[] for _ in range(count)]
        infos = rollout_infos or [{} for _ in range(count)]
        rewards = []
        for completion, sol, valid_answers, expected, info in zip(
                completions, solutions, answer_lists, corruptions, infos, strict=False):
            if not valid_answers:
                valid_answers = [sol]
            final = info.get("final_output") or completion
            result = aggregate_reward(
                final,
                info.get("planner_steps") or [],
                expected,
                info.get("code_exec_oks") or [],
                valid_answers,
                baseline_output=info.get("baseline_output"),
            )
            rewards.append(result["score"])
        return rewards


multi_turns["mat_agentflow"] = MATMultiTurnScheduler
orms["mat_agentflow_reward"] = MATReward
