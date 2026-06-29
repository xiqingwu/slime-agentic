"""Offline tests for custom_convert's multimodal_train_inputs alignment (plan §7.4 / M2).

Verifies the per-sequence multimodal list stays aligned 1:1 with tokens through
turn-expansion and trimming, and is only emitted when an image is present.

Run standalone:  python agentic/agentflow/tests/test_custom_convert.py
Or with pytest:  pytest agentic/agentflow/tests/test_custom_convert.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_convert import custom_convert  # noqa: E402


class _Status:
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"


class FakeSample:
    Status = _Status

    def __init__(self, index, reward, *, tokens=None, response_length=1, loss_mask=None,
                 train_metadata=None, multimodal_train_inputs=None, rollout_log_probs=None,
                 status=_Status.COMPLETED, remove_sample=False):
        self.index = index
        self._reward = reward
        self.tokens = tokens if tokens is not None else [1] * response_length
        self.response_length = response_length
        self.loss_mask = loss_mask
        self.train_metadata = train_metadata
        self.multimodal_train_inputs = multimodal_train_inputs
        self.rollout_log_probs = rollout_log_probs
        self.status = status
        self.remove_sample = remove_sample

    def get_reward_value(self, args):
        return self._reward


def _args(global_batch_size=1):
    # rewards_normalization=False -> normalized == raw, keeps assertions simple.
    return SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=False,
        grpo_std_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
        global_batch_size=global_batch_size,
    )


def test_non_turn_samples_emit_multimodal_when_present():
    samples = [
        FakeSample(0, 1.0, multimodal_train_inputs={"pixel_values": "A"}),
        FakeSample(1, 0.0, multimodal_train_inputs=None),
    ]
    out = custom_convert(_args(), samples)
    assert "multimodal_train_inputs" in out
    mm = out["multimodal_train_inputs"]
    assert len(mm) == len(out["tokens"]) == 2
    assert mm[0] == {"pixel_values": "A"} and mm[1] is None


def test_text_only_omits_multimodal_key():
    samples = [FakeSample(0, 1.0), FakeSample(1, 0.0)]
    out = custom_convert(_args(), samples)
    assert "multimodal_train_inputs" not in out


def test_turn_expansion_aligns_multimodal_per_turn():
    # One trajectory, 3 turns: image only on turn 0 (e.g. first turn sees the image).
    turns = [
        {"tokens": [1, 2], "response_length": 1, "loss_mask": [1],
         "multimodal_train_inputs": {"pixel_values": "img0"}},
        {"tokens": [3, 4], "response_length": 1, "loss_mask": [1],
         "multimodal_train_inputs": None},
        {"tokens": [5, 6], "response_length": 1, "loss_mask": [1]},  # key absent -> None
    ]
    samples = [FakeSample(0, 1.0, train_metadata={"turns": turns})]
    out = custom_convert(_args(), samples)
    mm = out["multimodal_train_inputs"]
    assert len(mm) == len(out["tokens"]) == 3
    assert mm[0] == {"pixel_values": "img0"}
    assert mm[1] is None and mm[2] is None


def test_no_trim_keeps_all_sequences():
    # v0.3.0: no sample-count trimming. build_dp_schedule packs by rollout, so all
    # 3 sequences are kept even though global_batch_size=2 (was: trimmed to 2).
    samples = [
        FakeSample(0, 1.0, multimodal_train_inputs={"pixel_values": "A"}),
        FakeSample(1, 1.0, multimodal_train_inputs={"pixel_values": "B"}),
        FakeSample(2, 1.0, multimodal_train_inputs={"pixel_values": "C"}),
    ]
    out = custom_convert(_args(global_batch_size=2), samples)
    assert len(out["tokens"]) == 3
    mm = out["multimodal_train_inputs"]
    assert mm == [{"pixel_values": "A"}, {"pixel_values": "B"}, {"pixel_values": "C"}]


def test_rollout_ids_group_turns_of_a_trajectory():
    # All turns of one trajectory share a rollout_id (= its sample.index) so
    # v0.3.0's scheduler keeps them in one rollout group; turns are contiguous.
    turns = [
        {"tokens": [1, 2], "response_length": 1, "loss_mask": [1]},
        {"tokens": [3, 4], "response_length": 1, "loss_mask": [1]},
    ]
    samples = [
        FakeSample(7, 1.0, train_metadata={"turns": turns}),
        FakeSample(9, 0.0),
    ]
    out = custom_convert(_args(), samples)
    assert out["rollout_ids"] == out["sample_indices"] == [7, 7, 9]


def test_mixed_turn_and_nonturn_alignment():
    turns = [
        {"tokens": [1, 2], "response_length": 1, "loss_mask": [1],
         "multimodal_train_inputs": {"pixel_values": "t0"}},
        {"tokens": [3, 4], "response_length": 1, "loss_mask": [1]},
    ]
    samples = [
        FakeSample(0, 1.0, train_metadata={"turns": turns}),
        FakeSample(1, 0.0, multimodal_train_inputs={"pixel_values": "single"}),
    ]
    out = custom_convert(_args(), samples)
    mm = out["multimodal_train_inputs"]
    # 2 turns expanded + 1 non-turn = 3 sequences
    assert len(mm) == len(out["tokens"]) == 3
    assert mm[0] == {"pixel_values": "t0"}
    assert mm[1] is None
    assert mm[2] == {"pixel_values": "single"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
