import asyncio
import sys
import types
from types import SimpleNamespace

from PIL import Image

# Test the plugin contract without requiring the full ms-swift/vLLM environment.
swift_rewards = types.ModuleType("swift.rewards")
swift_rewards.orms = {}
swift_rewards.ORM = type("ORM", (), {"__init__": lambda self, *args, **kwargs: None})
swift_infer = types.ModuleType("swift.infer_engine")
swift_infer.InferRequest = lambda **kwargs: SimpleNamespace(**kwargs)
swift_infer.RequestConfig = lambda **kwargs: SimpleNamespace(**kwargs)
swift_multi_turn = types.ModuleType("swift.rollout.multi_turn")
swift_multi_turn.multi_turns = {}


class StubScheduler:
    def __init__(self, infer_engine=None, max_turns=None, *args, **kwargs):
        self.infer_engine = infer_engine
        self.max_turns = max_turns

    def check_finished(self, infer_request, response_choice, current_turn):
        return response_choice.finish_reason == "length" or bool(self.max_turns and current_turn >= self.max_turns)


swift_multi_turn.MultiTurnScheduler = StubScheduler
sys.modules["swift.rewards"] = swift_rewards
sys.modules["swift.infer_engine"] = swift_infer
sys.modules["swift.rollout.multi_turn"] = swift_multi_turn

from mat_agentflow.plugin import MATMultiTurnScheduler


def choice(content, token_ids=(1, 2)):
    return SimpleNamespace(
        message=SimpleNamespace(content=content), token_ids=list(token_ids),
        finish_reason="stop", logprobs=None,
    )


def request(image):
    return SimpleNamespace(messages=[{"role": "user", "content": "<image>q"}], images=[image], data_dict={})


def test_problem_turn_adds_tip_and_tracks_tokens():
    scheduler = MATMultiTurnScheduler(max_turns=5)
    req = request(Image.new("RGB", (8, 8)))
    out = choice("<think>x</think><problem>{'blur'}</problem>")
    asyncio.run(scheduler.on_turn_end(req, out, 1))
    result = scheduler.step(req, out, 1)
    assert result["response_loss_mask"] == [1, 1]
    assert req.messages[-1]["role"] == "user"
    assert "Repair" in req.messages[-1]["content"]


def test_answer_turn_finishes_and_records_output():
    scheduler = MATMultiTurnScheduler(max_turns=5)
    req = request(Image.new("RGB", (8, 8)))
    out = choice("<think>x</think><answer>42</answer>")
    result = asyncio.run(scheduler.on_turn_end(req, out, 2))
    assert result["done"] is True
    assert result["rollout_infos"]["final_output"] == "42"
