"""OpenCV image-editing tool for MAT-Coding (plan M3 / §5.3).

Unlike ``python_coder`` (which asks an LLM to *write* code from a natural-language
query), this tool **executes the model's own ``<code>`` step** against the current
image: it path-rewrites the fixed placeholders, runs the snippet in a sandboxed
subprocess, and returns the new image path plus an exec-success flag.

``execute`` returns a dict so the solver can both (a) feed ``output_image_path`` to
the next turn and (b) collect ``success`` into ``sample.metadata['code_exec_oks']``
for ``core.mat_rewards.code_exec_reward``.
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tools.base import BaseTool

# Import the execution engine. Works whether 'core' is importable as a package
# (rollout cwd = agentflow/) or only as a sibling directory (tool loaded in isolation).
try:
    from core.image_tool import run_opencv_code
except ImportError:  # pragma: no cover - import-path fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.image_tool import run_opencv_code

TOOL_NAME = "OpenCV_Image_Editor_Tool"

TOOL_DESCRIPTION = """
Executes a model-written OpenCV (cv2) Python snippet to repair/transform an image.
The snippet must read from 'path_to_input_image.jpg' and write to
'path_to_output_image.jpg'; these placeholders are rewritten to the real files.
Returns whether execution succeeded and the path of the produced image.
"""

TOOL_DEMO_COMMANDS = [
    {
        "command": "execution = tool.execute(code=\"<code>```python\\nimport cv2\\nimg=cv2.imread('path_to_input_image.jpg')\\ncv2.imwrite('path_to_output_image.jpg', cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))\\n```</code>\", input_image_path='/x/in.png')",
        "description": "Rotate a 90°-rotated image back and save the result.",
    },
]


class OpenCV_Editor_Tool(BaseTool):
    def __init__(self, llm_engine=None):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS,
        )
        # No LLM needed: the model's <code> is executed directly.
        self.llm_engine = llm_engine

    async def execute(
        self,
        code: str,
        input_image_path: str,
        output_image_path: str | None = None,
        timeout: int = 30,
    ) -> dict:
        """Run the OpenCV snippet. Never raises — returns a structured dict.

        Uses the warm forkserver executor when ``MAT_CV2_FORKSERVER=1`` (B3, avoids
        re-importing cv2 per call), else the per-call resource-limited subprocess
        (B2). Both honour the same ImageExecResult contract; forkserver falls back to
        the subprocess if it can't be imported/started.
        """
        runner = run_opencv_code
        if os.environ.get("MAT_CV2_FORKSERVER", "0").lower() in ("1", "true", "yes"):
            try:
                from core.image_worker import run_in_forkserver
                runner = run_in_forkserver
            except Exception:
                runner = run_opencv_code

        result = await runner(
            code_text=code,
            input_image_path=input_image_path,
            output_image_path=output_image_path,
            timeout=timeout,
        )
        return result.as_dict()
