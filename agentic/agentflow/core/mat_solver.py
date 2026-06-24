"""Online multi-turn solver for the MAT-Coding visual agentic task (plan M3 / §4.1).

Implements the Visual-ARFT think/problem/code/answer protocol as a real online
loop (vs Visual-ARFT's offline pre-split steps):

    turn 0: model sees the corrupted image + query  -> <problem> (diagnosis)
            system injects a <tips> hint (crop/none/other), per Visual-ARFT eval
    turn 1: model emits <code> (OpenCV fix) -> tool executes it, the *processed*
            image is fed back as the input for the next turn
    ...     repeat until the model emits <answer> (or max_steps)

Every model generation is one training turn (loss_mask = 1), carrying the
``multimodal_train_inputs`` for the image it actually saw that turn. The reward
metadata the §4.0 rewards need (``planner_steps`` for format/diagnosis,
``code_exec_oks`` for code-exec) is assembled here.

I/O is injected: ``engine`` (an ``SGLangEngine``) and ``tool`` (the OpenCV editor)
are passed in, so the whole loop is unit-testable offline with fakes. The pure
helpers (``build_tip``, ``parse_step``, ``extract_answer``) are tested directly.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid

from .llm_engine import GenerationOutput
from .mat_rewards import classify_step, extract_problems, STEP_PROBLEM, STEP_CODE, STEP_ANSWER
from .image_tool import extract_code

logger = logging.getLogger(__name__)

# Verbatim from Visual-ARFT grpo_agent_code.py (keeps the protocol comparable).
SYSTEM_PROMPT_AGENT_CODE = """# Role
You are a step-by-step image processing assistant.
Your task is to solve an image-based task by applying OpenCV operations one step at a time, optionally using a reasoning chain.

# Output Format
At each step, output **only one** of the following, preceded by a <think> tag:
1. <problem> Describe the image issue from {'rotation90', 'rotation180', 'dark', 'overexposure', 'blur', 'noise', 'crop', 'none'} </problem>
2. <code> OpenCV code to process and save the image </code>
3. <answer> Final answer based on the processed image </answer>

# Image Processing Rules
- Always read from `'path_to_input_image.jpg'` and write to `'path_to_output_image.jpg'`.

# Output Format (strict):
Always begin with <think>. Then, depending on current reasoning chain, output one of the following:

## 1. If this is the first step and only the query is given, output in the following format:
<think> Initial analysis of the image issue. </think>
<problem> {'problem1', ...} </problem>

## 2. If <problem> is given, continue with image operations:
<think> Explain what to fix next. </think>
<code>
```python
One Python code block using OpenCV to perform the operation, and save the processed images.
```
</code>

## 3. If ready to conclude:
<think> Summarize the processing steps and provide the result or outcome </think>
<answer> Final answer, as briefly as possible</answer>

# Current reasoning chain:
"""

# <tips> templates copied from Visual-ARFT eval (evaluation_mat_coding_visual_arft.py).
_TIP_CROP = ("<tips> We now need to crop the image. Please provide the Python code. "
             "Use [x_min, y_min, x_max, y_max] to represent the bounding box coordinates. </tips>")
_TIP_NONE = ("<tips> The image has no issues, so no code is needed in the next step. "
             "You can directly provide the answer. </tips>")


def build_tip(problems: list[str]) -> str:
    """Build the hint injected after a <problem> step, per Visual-ARFT eval."""
    if not problems:
        return _TIP_NONE
    if problems[0] == "crop":
        return _TIP_CROP
    if problems[0] == "none":
        return _TIP_NONE
    return (f"<tips> Now that we have identified the issue in the image: {problems}, "
            f"please proceed to address it by outputting the python code. </tips>")


def extract_answer(text: str) -> str:
    """Pull text inside <answer>...</answer>, else the stripped text."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return (m.group(1).strip() if m else text).strip()


# ── Shared no-tool baseline (used by BOTH training A4 and eval B5) ────────────────
# The single definition of "what would the model answer WITHOUT the tool" — used to
# compute the correction-targeted reward (plan §4.0 optional → now core) and the eval
# Call Gain/Harm (plan §1.3/§8). Defining it once keeps the training reward signal and
# the eval metric measuring the *same* counterfactual. It deliberately mirrors the
# tool path's task framing (same corrupted image, same question) and differs only in
# that no tool is offered, so the comparison isolates the tool's marginal effect as
# cleanly as this protocol allows.
BASELINE_PROMPT = (
    "You are a visual question answering assistant. The image may be degraded "
    "(rotated, too dark/bright, blurred, noisy, or cropped). Without using any "
    "tools or code, look at the image and answer the question as briefly as "
    "possible. Put the final answer inside <answer> </answer>."
)


def build_baseline_messages(question: str, image_path: str) -> list[dict]:
    """One-shot, no-tool message: the corrupted image + question + direct-answer ask."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": f"{BASELINE_PROMPT}\n\n{question}"},
        ],
    }]


async def baseline_answer(engine, question: str, image_path: str, sampling_params: dict | None = None) -> str:
    """Generate the no-tool answer and return just the extracted answer text.

    Pass a greedy ``sampling_params`` (temperature 0) at train time so all GRPO
    samples of one prompt share an identical baseline — then the correction reward
    reflects each rollout's tool use against a fixed reference, not sampling noise.
    """
    gen = await engine.generate(build_baseline_messages(question, image_path), sampling_params=sampling_params)
    return extract_answer(gen.response)


def parse_step(response: str):
    """Classify a model step and extract its payload.

    Returns ``(step_type, payload)`` where payload is:
      - problem -> sorted list of corruption labels
      - code    -> the extracted python snippet (or None)
      - answer  -> the answer text
      - None    -> (None, None) when the step is malformed/ambiguous
    """
    step = classify_step(response)
    if step == STEP_PROBLEM:
        return STEP_PROBLEM, extract_problems(response)
    if step == STEP_CODE:
        return STEP_CODE, extract_code(response)
    if step == STEP_ANSWER:
        return STEP_ANSWER, extract_answer(response)
    return None, None


def _user_image_message(image_path: str, text: str) -> list[dict]:
    """A single Qwen-format user message with the current image + accumulated text."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text},
        ],
    }]


class MATSolver:
    def __init__(self, engine, tool, max_steps: int = 5, work_dir: str | None = None,
                 cleanup_temp: bool = True):
        self.engine = engine            # SGLangEngine (multimodal)
        self.tool = tool                # OpenCV_Editor_Tool (or anything with async execute)
        self.max_steps = max_steps
        self.work_dir = work_dir or tempfile.gettempdir()
        self.cleanup_temp = cleanup_temp  # delete processed-image temp files after solve (R5)

    def _next_output_path(self, input_path: str) -> str:
        ext = os.path.splitext(input_path)[1] or ".png"
        return os.path.join(self.work_dir, f"mat_step_{uuid.uuid4().hex}{ext}")

    @staticmethod
    def _cleanup(paths: list[str]) -> None:
        """Delete the processed-image temp files this trajectory created (R5).

        Their pixel tensors are already captured into each turn's
        ``multimodal_train_inputs`` at generation time, so the files on disk are
        dead weight once ``solve`` returns — left unchecked they fill /tmp over a run.
        """
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

    async def solve(self, question: str, image_path: str) -> GenerationOutput:
        created_images: list[str] = []
        try:
            return await self._solve(question, image_path, created_images)
        finally:
            if self.cleanup_temp:
                self._cleanup(created_images)

    async def _solve(self, question: str, image_path: str, created_images: list[str]) -> GenerationOutput:
        current_image = image_path
        context = ""               # accumulated steps/tips/results (text side)
        turns: list[dict] = []
        planner_steps: list[dict] = []
        code_exec_oks: list[bool] = []
        final_answer = ""
        finish_reason = "stop"
        last_problems: list[str] = []

        for step_idx in range(self.max_steps):
            text = f"{SYSTEM_PROMPT_AGENT_CODE}\n{question}\n{context}".rstrip()
            messages = _user_image_message(current_image, text)

            gen = await self.engine.generate(messages)
            finish_reason = gen.finish_reason

            # Record this generation as one training turn.
            turns.append({
                "tokens": list(gen.prompt_token_ids) + list(gen.token_ids),
                "response_length": len(gen.token_ids),
                "loss_mask": [1] * len(gen.token_ids),
                "rollout_log_probs": list(gen.log_probs),
                "multimodal_train_inputs": gen.multimodal_train_inputs,
            })

            step_type, payload = parse_step(gen.response)
            planner_steps.append({"type": step_type, "content": gen.response})

            if step_type == STEP_ANSWER:
                final_answer = payload
                context += f"\n{gen.response}"
                break

            if step_type == STEP_PROBLEM:
                last_problems = payload or []
                tip = build_tip(last_problems)
                context += f"\n{gen.response}\n{tip}"
                continue

            if step_type == STEP_CODE:
                code = payload
                if not code:
                    code_exec_oks.append(False)
                    context += f"\n{gen.response}\n<result> no code extracted </result>"
                    continue
                out_path = self._next_output_path(current_image)
                created_images.append(out_path)   # track for cleanup (R5)
                result = await self.tool.execute(
                    code=gen.response, input_image_path=current_image, output_image_path=out_path,
                )
                success = bool(result.get("success"))
                changed = bool(result.get("changed", True))
                # Only count it as a real tool use when the code ran AND actually
                # edited the image: a no-op copy shouldn't earn the code-exec reward.
                code_exec_oks.append(success and changed)
                if success:
                    current_image = result["output_image_path"]   # feed processed image forward
                    note = "image updated" if changed else "image unchanged (no-op)"
                    context += f"\n{gen.response}\n<result> code executed; {note} </result>"
                else:
                    context += f"\n{gen.response}\n<result> code failed: {result.get('error')} </result>"
                continue

            # malformed/ambiguous step: record and stop to avoid looping.
            logger.warning("[mat step %d] unparseable step; stopping", step_idx)
            context += f"\n{gen.response}"
            break

        return self._assemble(turns, planner_steps, code_exec_oks, final_answer, context, finish_reason)

    @staticmethod
    def _assemble(turns, planner_steps, code_exec_oks, final_answer, context, finish_reason) -> GenerationOutput:
        """Concatenate turns into one sequence for framework compatibility.

        Real training data is unrolled per-turn by custom_convert; this concatenated
        view is only for status/response/eval display (mirrors the math solver).
        """
        if not turns:
            return GenerationOutput(
                prompt_text="", prompt_token_ids=[], response="", token_ids=[], log_probs=[],
                finish_reason=finish_reason, loss_mask=[], final_output=final_answer,
                turns=[], planner_steps=planner_steps, code_exec_oks=code_exec_oks,
            )

        first = turns[0]
        first_prompt_len = len(first["tokens"]) - first["response_length"]
        prompt_token_ids = first["tokens"][:first_prompt_len]

        cat_token_ids = list(first["tokens"][first_prompt_len:])
        cat_loss_mask = list(first["loss_mask"])
        cat_log_probs = list(first["rollout_log_probs"])
        for t in turns[1:]:
            t_prompt_len = len(t["tokens"]) - t["response_length"]
            cat_token_ids += t["tokens"]
            cat_loss_mask += [0] * t_prompt_len + t["loss_mask"]
            cat_log_probs += [0.0] * t_prompt_len + t["rollout_log_probs"]

        return GenerationOutput(
            prompt_text="",
            prompt_token_ids=prompt_token_ids,
            response=context.strip(),
            token_ids=cat_token_ids,
            log_probs=cat_log_probs,
            finish_reason=finish_reason,
            loss_mask=cat_loss_mask,
            final_output=final_answer,
            turns=turns,
            planner_steps=planner_steps,
            code_exec_oks=code_exec_oks,
        )
