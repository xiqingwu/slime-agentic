"""Offline tests for the multimodal SGLang engine (plan M2 / §5.2).

The slime HTTP/vision seams are monkeypatched, so the multimodal path is exercised
without a real VLM/SGLang server. One test uses the *real* slime
``encode_image_for_rollout_engine`` (needs slime + PIL on path; run from repo root).

Run standalone:  python agentic/agentflow/tests/test_llm_engine.py
Or with pytest:  pytest agentic/agentflow/tests/test_llm_engine.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.llm_engine as eng  # noqa: E402
from core.llm_engine import (  # noqa: E402
    messages_have_images,
    build_generate_payload,
    parse_generate_output,
    SGLangEngine,
    GenerationOutput,
)

CANNED_OUT = {
    "text": " the answer",
    "meta_info": {
        "finish_reason": {"type": "stop"},
        "output_token_logprobs": [[-0.1, 100], [-0.2, 101], [-0.3, 102]],
    },
}


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "TEXT_PROMPT"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2, 3]}


class FakeProcessor:
    def __init__(self):
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            return_dict=False, return_tensors=None, **kwargs):
        self.chat_template_kwargs = kwargs   # capture enable_thinking etc.
        self.template_messages = messages
        if tokenize:
            assert return_dict is True
            assert return_tensors == "pt"
            return FakeBatch({
                "input_ids": FakeTensor([[10, 11, 12, 13]]),
                "attention_mask": FakeTensor([[1, 1, 1, 1]]),
                "pixel_values": "PV",
                "image_grid_thw": "THW",
            })
        return "MM_PROMPT"

    def __call__(self, text=None, **kwargs):
        # mimic HF processor output: input_ids (batched) + modality tensors
        return {"input_ids": [[10, 11, 12, 13]], "pixel_values": "PV", "image_grid_thw": "THW"}


class FakeRow(list):
    def tolist(self):
        return list(self)


class FakeTensor(list):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return FakeRow(value) if isinstance(value, list) else value


class FakeBatch(dict):
    pass


def _text_msgs():
    return [{"role": "user", "content": "hello"}]


def _image_msgs():
    return [{"role": "user", "content": [
        {"type": "image", "image": "/x/in.png"},
        {"type": "text", "text": "what is this?"},
    ]}]


# ── messages_have_images ─────────────────────────────────────────────────────────

def test_messages_have_images():
    assert messages_have_images(_image_msgs()) is True
    assert messages_have_images(_text_msgs()) is False
    assert messages_have_images([{"role": "user", "content": [{"type": "text", "text": "x"}]}]) is False


# ── build_generate_payload ───────────────────────────────────────────────────────

def test_payload_text():
    p = build_generate_payload({"max_new_tokens": 8}, text="hi")
    assert p["text"] == "hi" and "input_ids" not in p
    assert p["return_logprob"] is True and "image_data" not in p


def test_payload_input_ids_with_images():
    p = build_generate_payload({}, input_ids=[1, 2], image_data=["data:img"])
    assert p["input_ids"] == [1, 2] and p["image_data"] == ["data:img"]
    assert "text" not in p


def test_payload_requires_exactly_one_source():
    for kwargs in ({}, {"text": "a", "input_ids": [1]}):
        try:
            build_generate_payload({}, **kwargs)
            assert False, "should have raised"
        except ValueError:
            pass


# ── parse_generate_output ────────────────────────────────────────────────────────

def test_parse_output():
    resp, tokens, logps, finish = parse_generate_output(CANNED_OUT)
    assert resp == " the answer"
    assert tokens == [100, 101, 102]
    assert logps == [-0.1, -0.2, -0.3]
    assert finish == "stop"


def test_parse_output_empty_logprobs():
    out = {"text": "", "meta_info": {"finish_reason": {"type": "length"}}}
    resp, tokens, logps, finish = parse_generate_output(out)
    assert tokens == [] and logps == [] and finish == "length"


# ── full generate: text path (processor None) ────────────────────────────────────

def test_generate_text_path():
    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return CANNED_OUT

    old = eng._post
    eng._post = fake_post
    try:
        engine = SGLangEngine(url="http://x/generate", tokenizer=FakeTokenizer(),
                              sampling_params={}, max_new_tokens=16)
        out = asyncio.run(engine.generate(_text_msgs()))
    finally:
        eng._post = old

    assert "text" in captured["payload"] and "image_data" not in captured["payload"]
    assert isinstance(out, GenerationOutput)
    assert out.prompt_token_ids == [1, 2, 3]
    assert out.token_ids == [100, 101, 102]
    assert out.multimodal_train_inputs is None


# ── full generate: multimodal path (processor + images) ──────────────────────────

def test_generate_multimodal_path():
    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return CANNED_OUT

    saved = (eng._post, eng._process_vision_info, eng._encode_image)
    eng._post = fake_post
    eng._process_vision_info = lambda messages, processor: {"images": ["PIL_OBJ"], "videos": []}
    eng._encode_image = lambda image: f"data:enc:{image}"
    proc = FakeProcessor()
    try:
        engine = SGLangEngine(url="http://x/generate", tokenizer=FakeTokenizer(),
                              sampling_params={}, max_new_tokens=16, processor=proc,
                              enable_thinking=False)
        out = asyncio.run(engine.generate(_image_msgs()))
    finally:
        eng._post, eng._process_vision_info, eng._encode_image = saved

    payload = captured["payload"]
    # SGLang receives text so it expands image placeholders once; the locally
    # processed IDs are retained separately for training-side alignment.
    assert payload["text"] == "MM_PROMPT"
    assert payload["image_data"] == ["data:enc:PIL_OBJ"]
    assert "input_ids" not in payload
    # GenerationOutput carries processor tensors for training-side alignment
    assert out.prompt_token_ids == [10, 11, 12, 13]
    assert out.multimodal_train_inputs == {"pixel_values": "PV", "image_grid_thw": "THW"}
    # enable_thinking is forwarded to the chat template (parity with the text path)
    assert proc.chat_template_kwargs.get("enable_thinking") is False


def test_generate_multimodal_max_pixels_injected():
    """max_pixels is forwarded into images_kwargs when set on the engine."""
    proc = FakeProcessor()

    saved = (eng._post, eng._process_vision_info, eng._encode_image)
    eng._post = lambda url, payload, headers=None: asyncio.coroutine(lambda: CANNED_OUT)()
    eng._process_vision_info = lambda messages, processor: {"images": ["PIL_OBJ"], "videos": []}
    eng._encode_image = lambda image: "data:enc"

    async def fake_post(url, payload, headers=None):
        return CANNED_OUT

    eng._post = fake_post
    try:
        engine = SGLangEngine(url="http://x/generate", tokenizer=FakeTokenizer(),
                              sampling_params={}, processor=proc, max_pixels=401408)
        asyncio.run(engine.generate(_image_msgs()))
    finally:
        eng._post, eng._process_vision_info, eng._encode_image = saved

    image_block = proc.template_messages[0]["content"][0]
    assert image_block["max_pixels"] == 401408
    assert "max_pixels" not in _image_msgs()[0]["content"][0]


def test_generate_processor_but_no_images_uses_text_path():
    """Processor present but a text-only turn must still use the text path."""
    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return CANNED_OUT

    old = eng._post
    eng._post = fake_post
    try:
        engine = SGLangEngine(url="http://x/generate", tokenizer=FakeTokenizer(),
                              sampling_params={}, max_new_tokens=16, processor=FakeProcessor())
        out = asyncio.run(engine.generate(_text_msgs()))
    finally:
        eng._post = old

    assert "text" in captured["payload"]
    assert out.multimodal_train_inputs is None


# ── real slime image encoding (sanity) ───────────────────────────────────────────

def test_real_encode_image_roundtrip():
    try:
        from slime.utils.processing_utils import encode_image_for_rollout_engine
        from PIL import Image
    except Exception as e:  # slime/PIL not importable from this cwd
        print(f"  (skipped real encode: {e})")
        return
    img = Image.new("RGB", (8, 8), (123, 50, 200))
    enc = encode_image_for_rollout_engine(img)
    assert enc.startswith("data:image/png;base64,")
    assert len(enc) > 30


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
