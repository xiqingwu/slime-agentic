import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sft_mat import _encode_sample


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is False
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "pixel_values": torch.ones((4, 8)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0, 0]]),
        }


class FakeMaskGenerator:
    def get_loss_mask_with_multimodal_alignment(self, messages, input_ids):
        assert messages[-1]["role"] == "assistant"
        return input_ids, [0, 0, 0, 0, 1, 1]

    def get_response_lengths(self, masks):
        return [2]


def test_encode_sample_keeps_visual_tensors_and_masks_assistant():
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "/x.png"}, {"type": "text", "text": "q"}]},
        {"role": "assistant", "content": "<code>x</code>"},
    ]
    tokens, mask, response_length, multimodal = _encode_sample(messages, FakeProcessor(), FakeMaskGenerator())
    assert tokens == [10, 11, 12, 13, 14, 15]
    assert mask == [1, 1]
    assert response_length == 2
    assert set(multimodal) == {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
