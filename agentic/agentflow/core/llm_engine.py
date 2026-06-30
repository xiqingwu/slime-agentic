"""SGLang HTTP engine for AgentFlow, with optional multimodal (VLM) support.

Plan reference: `docs/algorithms/plan_agentflow_visual.md` §5.2 (M2).

Text-only behaviour is unchanged (backward compatible). When a ``processor`` is
supplied *and* the chat messages carry image content, the engine takes slime's
multimodal path instead (mirrors slime/rollout/sglang_rollout.py:120-151):

- build ``input_ids`` + ``multimodal_train_inputs`` via the HF processor,
- send ``input_ids`` plus base64 ``image_data`` to SGLang (not raw ``text``),
- return ``multimodal_train_inputs`` on the GenerationOutput so the training-side
  custom_convert can emit them aligned with each turn (M3).

slime imports are lazy (inside ``_post`` / ``_process_vision_info`` / ... ) so this
module imports even where slime isn't installed, and the pure helpers
(``build_generate_payload`` / ``parse_generate_output`` / ``messages_have_images``)
are unit-testable offline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationOutput:
    prompt_text: str
    prompt_token_ids: list[int]
    response: str
    token_ids: list[int]
    log_probs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"
    # Filled by the solver in multi-turn mode: same length as token_ids;
    # 1 = model-generated tokens that participate in loss, 0 = injected prompt tokens that do not
    loss_mask: list[int] | None = None
    # The planner's final consolidated answer, used for reward scoring
    final_output: str | None = None
    # Multi-turn split: each turn is an independent training sequence;
    # custom_convert uses this field to unroll turns
    turns: list[dict] | None = None
    # Processor outputs for VLM training (pixel_values, image_grid_thw, ...);
    # None for text-only turns. Aligns with prompt_token_ids' expanded vision tokens.
    multimodal_train_inputs: dict | None = None
    # MAT-Coding online loop bookkeeping (set by MATSolver, consumed by mat_reward_func):
    # planner_steps = [{"type": problem|code|answer, "content": <raw step text>}, ...]
    planner_steps: list[dict] | None = None
    # code_exec_oks = per <code> step exec-success flags (for code_exec_reward)
    code_exec_oks: list[bool] | None = None


# ── Lazy slime seams (kept as module functions so tests can monkeypatch them) ─────

async def _post(url: str, payload: dict, headers: dict | None = None):
    from slime.utils.http_utils import post

    return await post(url, payload, headers=headers)


def _process_vision_info(messages: list, processor) -> dict:
    from slime.utils.processing_utils import process_vision_info

    return process_vision_info(messages, processor)


def _with_image_limits(messages: list[dict], max_pixels: int | None) -> list[dict]:
    """Return messages with a per-image pixel cap, without mutating the caller."""
    if max_pixels is None:
        return messages
    limited = copy.deepcopy(messages)
    for msg in limited:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                block["max_pixels"] = max_pixels
    return limited


def _encode_image(image) -> str:
    from slime.utils.processing_utils import encode_image_for_rollout_engine
    from PIL import Image

    if isinstance(image, str):
        image = Image.open(image)
    return encode_image_for_rollout_engine(image)


# ── Pure helpers (offline-testable) ──────────────────────────────────────────────

def messages_have_images(messages: list[dict]) -> bool:
    """True if any message carries an image/video content block (Qwen chat format)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and str(block.get("type", "")).startswith(("image", "video")):
                    return True
    return False


def build_generate_payload(
    sampling_params: dict,
    *,
    text: str | None = None,
    input_ids: list[int] | None = None,
    image_data: list[str] | None = None,
) -> dict:
    """Build the SGLang /generate payload. Exactly one of text / input_ids is set."""
    if (text is None) == (input_ids is None):
        raise ValueError("provide exactly one of text or input_ids")
    payload: dict[str, Any] = {"sampling_params": sampling_params, "return_logprob": True}
    if text is not None:
        payload["text"] = text
    else:
        payload["input_ids"] = input_ids
    if image_data:
        payload["image_data"] = image_data
    return payload


def parse_generate_output(out: dict) -> tuple[str, list[int], list[float], str]:
    """Extract (response_text, token_ids, log_probs, finish_reason) from SGLang output."""
    meta = out["meta_info"]
    finish_reason = meta["finish_reason"]["type"]
    response = out["text"]
    token_logprobs = meta.get("output_token_logprobs", [])
    token_ids = [item[1] for item in token_logprobs]
    log_probs = [item[0] for item in token_logprobs]
    return response, token_ids, log_probs, finish_reason


class SGLangEngine:
    """Lightweight wrapper around the SGLang HTTP /generate endpoint.

    Pass ``processor`` to enable the multimodal path for image-bearing messages.
    """

    def __init__(
        self,
        url: str,
        tokenizer: Any,
        sampling_params: dict,
        max_new_tokens: int | None = None,
        enable_thinking: bool = False,
        processor: Any = None,
        max_pixels: int | None = None,
    ):
        self.url = url
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_pixels = max_pixels
        self.sampling_params = dict(sampling_params)
        self.enable_thinking = enable_thinking
        if max_new_tokens is not None:
            self.sampling_params["max_new_tokens"] = max_new_tokens

    def _encode_multimodal(self, messages: list[dict]):
        """Return (prompt_text, input_ids, multimodal_train_inputs, images)."""
        limited_messages = _with_image_limits(messages, self.max_pixels)

        # Render once for logging, then let the processor tokenize the original
        # structured messages atomically.  In particular, do not feed a rendered
        # prompt and a separately collected image list back into Qwen3-VL: newer
        # transformers releases can expand their image markers differently in
        # those two phases, leaving input_ids and image_grid_thw out of sync.
        prompt_text = self.processor.apply_chat_template(
            limited_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        proc_out = self.processor.apply_chat_template(
            limited_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        )

        input_ids = proc_out["input_ids"][0].tolist()
        multimodal_train_inputs = {
            k: v for k, v in proc_out.items() if k not in ("input_ids", "attention_mask")
        } or None
        images = (_process_vision_info(limited_messages, self.processor).get("images") or [])
        return prompt_text, input_ids, multimodal_train_inputs, images

    def _encode_text(self, messages: list[dict]):
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    async def generate(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict | None = None,
    ) -> GenerationOutput:
        """Accept standard chat messages and return a GenerationOutput.

        Uses the multimodal path when a processor is set and messages carry images;
        otherwise the original text path. ``sampling_params`` overrides the defaults.
        """
        params = sampling_params if sampling_params is not None else self.sampling_params

        multimodal_train_inputs = None
        if self.processor is not None and messages_have_images(messages):
            prompt_text, prompt_token_ids, multimodal_train_inputs, images = self._encode_multimodal(messages)
            image_data = [_encode_image(img) for img in images] if images else None
            payload = build_generate_payload(params, input_ids=prompt_token_ids, image_data=image_data)
        else:
            prompt_text, prompt_token_ids = self._encode_text(messages)
            payload = build_generate_payload(params, text=prompt_text)

        out = await _post(self.url, payload)
        response, token_ids, log_probs, finish_reason = parse_generate_output(out)

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
            multimodal_train_inputs=multimodal_train_inputs,
        )
