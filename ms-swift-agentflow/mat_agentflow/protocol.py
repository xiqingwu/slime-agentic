"""Protocol primitives shared by rollout, reward, data preparation, and eval."""

from __future__ import annotations

import ast
import re

STEP_PROBLEM = "problem"
STEP_CODE = "code"
STEP_ANSWER = "answer"
VALID_PROBLEMS = {
    "rotation90", "rotation180", "dark", "overexposure", "blur", "noise", "crop", "none"
}

SYSTEM_PROMPT = """# Role
You are a step-by-step image processing assistant. Diagnose image corruption,
repair it with OpenCV when needed, and answer the visual question.

# Strict protocol
Output exactly one action per turn, always preceded by <think>...</think>:
1. First turn: <problem> {'blur', ...} </problem>
2. After a non-none problem: <code> one Python/OpenCV code block </code>
3. When the image is ready: <answer> brief final answer </answer>

OpenCV code must read 'path_to_input_image.jpg' and write
'path_to_output_image.jpg'. Do not output an answer before diagnosing the image.
"""


def classify_step(text: str) -> str | None:
    found = [name for name in (STEP_PROBLEM, STEP_CODE, STEP_ANSWER)
             if re.search(rf"<{name}>.*?</{name}>", text or "", re.DOTALL)]
    return found[0] if len(found) == 1 else None


def extract_problems(text: str) -> list[str]:
    match = re.search(r"<problem>(.*?)</problem>", text or "", re.DOTALL)
    if not match:
        return []
    body = match.group(1).strip()
    try:
        value = ast.literal_eval(body)
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (set, list, tuple)):
            values = [str(x) for x in value]
        else:
            values = []
    except (SyntaxError, ValueError):
        values = re.findall(r"[A-Za-z][A-Za-z0-9_]*", body)
    return sorted({x.lower() for x in values if x.lower() in VALID_PROBLEMS})


def extract_code(text: str) -> str | None:
    match = re.search(r"<code>(.*?)</code>", text or "", re.DOTALL)
    if not match:
        return None
    body = match.group(1).strip()
    fenced = re.search(r"```(?:python)?\s*(.*?)```", body, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else body).strip() or None


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    return (match.group(1) if match else text or "").strip()


def build_tip(problems: list[str]) -> str:
    if not problems or problems == ["none"]:
        return "<tips>The image needs no repair. Answer the question now.</tips>"
    if "crop" in problems:
        return ("<tips>Crop to the relevant answer region. Output OpenCV code using "
                "[x_min, y_min, x_max, y_max].</tips>")
    return f"<tips>Repair the diagnosed corruption {problems}. Output OpenCV code now.</tips>"
