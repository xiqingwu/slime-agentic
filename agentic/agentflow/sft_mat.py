"""Multimodal supervised rollout for MAT-Coding teacher demonstrations."""

from __future__ import annotations

import copy
import os

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.processing_utils import load_processor, load_tokenizer

TOKENIZER = None
PROCESSOR = None
MASK_GENERATOR = None


def _pixel_limited(messages: list[dict]) -> list[dict]:
    raw = os.environ.get("MAT_MAX_PIXELS")
    if not raw:
        return messages
    max_pixels = int(raw)
    limited = copy.deepcopy(messages)
    for message in limited:
        if not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") == "image":
                block["max_pixels"] = max_pixels
    return limited


def _encode_sample(messages, processor, mask_generator):
    messages = _pixel_limited(messages)
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"][0].tolist()
    input_ids, full_mask = mask_generator.get_loss_mask_with_multimodal_alignment(messages, input_ids)
    response_length = mask_generator.get_response_lengths([full_mask])[0]
    if response_length == 0:
        raise ValueError("MAT SFT example has no supervised assistant tokens")
    multimodal = {
        key: value for key, value in encoded.items() if key not in ("input_ids", "attention_mask")
    } or None
    return input_ids, full_mask[-response_length:], response_length, multimodal


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Build teacher-forced VLM samples without starting an SGLang server."""
    assert not evaluation
    assert args.rollout_global_dataset

    global TOKENIZER, PROCESSOR, MASK_GENERATOR
    if TOKENIZER is None:
        TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    if PROCESSOR is None:
        raise RuntimeError(f"failed to load multimodal processor from {args.hf_checkpoint}")
    if MASK_GENERATOR is None:
        MASK_GENERATOR = MultiTurnLossMaskGenerator(TOKENIZER, tokenizer_type=args.loss_mask_type)

    grouped_samples = data_buffer.get_samples(args.rollout_batch_size)
    samples = []
    for group in grouped_samples:
        (sample,) = group
        tokens, loss_mask, response_length, multimodal = _encode_sample(
            sample.prompt, PROCESSOR, MASK_GENERATOR
        )
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = loss_mask
        sample.multimodal_train_inputs = multimodal
        sample.reward = 0
        samples.append(sample)
    return samples
