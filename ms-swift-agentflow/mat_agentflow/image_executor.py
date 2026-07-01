"""Resource-limited OpenCV execution for model-generated repair snippets."""

from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .protocol import extract_code


@dataclass
class ImageExecResult:
    success: bool
    image: Image.Image | None = None
    changed: bool = False
    error: str = ""
    stdout: str = ""


def load_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path") or value.get("image")
    if isinstance(value, str) and value.startswith("data:"):
        value = value.split(",", 1)[1]
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return Image.open(path).convert("RGB")
        try:
            return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        except Exception as exc:
            raise ValueError("unsupported image string") from exc
    raise TypeError(f"unsupported image type: {type(value)!r}")


def _sanitize(code: str) -> str:
    forbidden = re.compile(
        r"\b(?:subprocess|socket|requests|urllib|shutil|pathlib|glob|system|popen|eval|exec|compile|__import__)\b"
    )
    if forbidden.search(code):
        raise ValueError("code contains a forbidden operation")
    return code


def _limit_resources():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
        # cv2/torch-linked runtimes reserve several GiB of virtual address space
        # during import even when resident memory is small.
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    except Exception:
        pass


def execute_opencv(code_text: str, image_value, timeout: int = 30) -> ImageExecResult:
    code = extract_code(code_text) or code_text.strip()
    if not code:
        return ImageExecResult(False, error="no code extracted")
    try:
        code = _sanitize(code)
        source = load_image(image_value)
    except Exception as exc:
        return ImageExecResult(False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="mat_swift_") as tmp:
        input_path = Path(tmp) / "input.png"
        output_path = Path(tmp) / "output.png"
        source.save(input_path)
        code = code.replace("path_to_input_image.jpg", str(input_path))
        code = code.replace("path_to_output_image.jpg", str(output_path))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], cwd=tmp, env=env, text=True,
                capture_output=True, timeout=timeout, preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return ImageExecResult(False, error=f"code timed out after {timeout}s")
        if proc.returncode != 0:
            error = proc.stderr or f"execution failed with exit code {proc.returncode}"
            return ImageExecResult(False, error=error[-2000:], stdout=proc.stdout[-2000:])
        if not output_path.is_file():
            return ImageExecResult(False, error="code produced no output image", stdout=proc.stdout[-2000:])
        result = cv2.imread(str(output_path))
        original = cv2.imread(str(input_path))
        if result is None:
            return ImageExecResult(False, error="output is not a readable image")
        changed = original.shape != result.shape or not np.array_equal(original, result)
        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return ImageExecResult(True, Image.fromarray(rgb), changed=changed, stdout=proc.stdout[-2000:])
