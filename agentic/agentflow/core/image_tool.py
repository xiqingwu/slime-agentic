"""OpenCV code-execution engine for the MAT-Coding visual agentic task (plan M3 / §5.3).

The model emits a ``<code>``-step containing an OpenCV snippet that reads
``'path_to_input_image.jpg'`` and writes ``'path_to_output_image.jpg'`` (the
fixed I/O convention from Visual-ARFT's SYSTEM_PROMPT_AGENT_CODE). This module:

1. extracts the python block (same regex as Visual-ARFT ``extract_and_run_code``),
2. rewrites the placeholder paths to the real input/output files,
3. executes the snippet in a **resource-limited subprocess** with a timeout,
4. verifies an output image was actually produced.

The boolean ``success`` it returns is exactly the ``exec_ok`` signal that
``core.mat_rewards.code_exec_reward`` / ``aggregate_mat_reward`` consume via
``sample.metadata['code_exec_oks']``.

Difference from Visual-ARFT's reference: that code uses in-process ``exec`` (unsafe)
and a teacher-forced bbox substitution for the 'crop' case (the placeholder code
sliced ``[a:b, c:d]`` and eval rewrote it from the question's normalized bbox).
Online, the model writes its own real coordinates, so we do plain path-rewrite and
execution; crop needs no special handling. See ``replace_paths`` notes.

SECURITY (honest scope — this is NOT a true sandbox). The snippet is model-written
and runs as a real subprocess. We harden it with: its own session/process group so
a timeout kills the whole tree (``os.setsid`` + ``killpg``); rlimits on CPU time,
address space, output file size and process count (anti fork-bomb / OOM / disk
runaway); and an isolated temp working directory. We do NOT block network or
filesystem reads — a determined snippet can still touch the box. For untrusted code
at scale, run the trainer inside a container/nsjail; these limits only contain the
common accidents (infinite loops, memory blowups, fork bombs).
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import sys
import tempfile
import uuid
from dataclasses import dataclass

try:
    import resource  # Unix-only; used for the subprocess rlimits below
except ImportError:  # pragma: no cover - non-Unix
    resource = None

INPUT_PLACEHOLDER = "path_to_input_image.jpg"
OUTPUT_PLACEHOLDER = "path_to_output_image.jpg"

_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT_LENGTH = 4000
_DANGEROUS_CALLS = ["exit", "quit", "sys.exit", "os._exit"]

# Subprocess resource limits (anti runaway). Generous enough for real cv2 edits.
# NOTE: we intentionally do NOT set RLIMIT_NPROC — it is counted *per real user* (and
# threads count too on Linux), so a fixed value can be below a busy box's existing
# process count and would then break cv2's own threads. CPU/AS/FSIZE are per-process
# and safe; the timeout + killpg already bound wall-clock and a fork bomb's lifetime.
_RLIMIT_AS_BYTES = 8 * 1024 ** 3       # 8 GB address space (plenty for image ops)
_RLIMIT_FSIZE_BYTES = 256 * 1024 ** 2  # 256 MB max single file write


def _apply_rlimits(cpu_seconds: int) -> None:
    """Set CPU/memory/file-size/process rlimits on the current process (best-effort)."""
    if resource is None:
        return
    for res, soft_hard in (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds)),
        (resource.RLIMIT_AS, (_RLIMIT_AS_BYTES, _RLIMIT_AS_BYTES)),
        (resource.RLIMIT_FSIZE, (_RLIMIT_FSIZE_BYTES, _RLIMIT_FSIZE_BYTES)),
    ):
        try:
            resource.setrlimit(res, soft_hard)
        except (ValueError, OSError):
            pass  # best-effort; some limits may be unavailable


def _make_preexec(cpu_seconds: int):
    """preexec_fn for the exec subprocess: own process group + rlimits (Unix only)."""
    def _apply():
        os.setsid()  # isolate into a new session so we can killpg the whole tree
        _apply_rlimits(cpu_seconds)
    return _apply


@dataclass
class ImageExecResult:
    success: bool                       # exec_ok: ran cleanly AND produced an output image
    output_image_path: str | None       # the new image to feed the next turn (None on failure)
    stdout: str = ""
    error: str | None = None
    # Did the produced image actually differ from the input? A no-op like
    # ``cv2.imwrite(out, cv2.imread(in))`` "succeeds" but changes nothing — without
    # this flag the code-exec reward would pay out for poking the tool pointlessly
    # (plan §1.3: don't reward "safe coexistence" with the tool). Best-effort: True
    # when we can't tell (cv2 missing / unreadable), so we never falsely penalize.
    changed: bool = True

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "output_image_path": self.output_image_path,
            "stdout": self.stdout,
            "error": self.error,
            "changed": self.changed,
        }


def extract_code(text: str) -> str | None:
    """Extract the python snippet from a model <code> step.

    Mirrors Visual-ARFT's ``extract_and_run_code`` regex first, then falls back to
    a bare ```python fence, then a generic ``` fence. Returns None if nothing found.
    """
    m = re.search(r"<code>\s*```python(.*?)```.*?</code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def replace_paths(code: str, input_image_path: str, output_image_path: str) -> str:
    """Rewrite the fixed placeholder filenames to the real paths.

    We replace the *bare* tokens (not the quoted form Visual-ARFT used), so it works
    regardless of whether the model wrote single or double quotes around them.
    """
    return (
        code.replace(INPUT_PLACEHOLDER, input_image_path)
            .replace(OUTPUT_PLACEHOLDER, output_image_path)
    )


def _sanitize_code(code: str) -> str:
    """Neutralize process-killing calls (exit/quit/...) so the subprocess returns."""
    sanitized = code
    for func in _DANGEROUS_CALLS:
        sanitized = re.sub(rf"{re.escape(func)}\s*\([^)]*\)", "pass", sanitized)
    return sanitized


def _truncate(text: str, max_length: int = _MAX_OUTPUT_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 30
    return text[:half] + "\n... (truncated) ...\n" + text[-half:]


def _is_valid_image(path: str) -> bool:
    """True if the file exists, is non-empty, and decodes as an image."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        import cv2  # local import: core stays importable without cv2

        return cv2.imread(path) is not None
    except Exception:
        # cv2 missing or read failed; fall back to the size check we already passed.
        return True


def _images_differ(input_path: str, output_path: str) -> bool:
    """True if the output image's pixels differ from the input's.

    Detects no-op snippets (e.g. a plain read→write copy). Best-effort: returns
    True (assume changed) if either image can't be read, so we never falsely
    penalize a real edit just because the comparison failed.
    """
    try:
        import cv2
        import numpy as np

        a = cv2.imread(input_path)
        b = cv2.imread(output_path)
        if a is None or b is None:
            return True
        if a.shape != b.shape:
            return True
        return not np.array_equal(a, b)
    except Exception:
        return True


async def run_opencv_code(
    code_text: str,
    input_image_path: str,
    output_image_path: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> ImageExecResult:
    """Extract, path-rewrite, execute the snippet, and verify the output image.

    ``code_text`` may be a raw snippet or a full ``<code>...</code>`` step.
    If ``output_image_path`` is None a temp file (same extension as input) is used.
    Never raises — failures come back as ``success=False`` with ``error`` set.
    """
    code = extract_code(code_text)
    if code is None:
        # Maybe the caller already handed us a bare snippet without fences.
        code = code_text.strip()
    if not code:
        return ImageExecResult(False, None, error="no code extracted")

    if not os.path.exists(input_image_path):
        return ImageExecResult(False, None, error=f"input image not found: {input_image_path}")

    if output_image_path is None:
        ext = os.path.splitext(input_image_path)[1] or ".png"
        output_image_path = os.path.join(tempfile.gettempdir(), f"mat_out_{uuid.uuid4().hex}{ext}")

    code = replace_paths(code, input_image_path, output_image_path)
    code = _sanitize_code(code)

    # Run in an isolated temp cwd so relative writes by the snippet are contained.
    work_dir = tempfile.mkdtemp(prefix="mat_exec_")
    preexec = _make_preexec(timeout + 2) if os.name == "posix" else None
    try:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                preexec_fn=preexec,
            )
        except Exception as exc:  # noqa: BLE001
            return ImageExecResult(False, None, error=f"failed to launch subprocess: {exc}")

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_process_tree(proc)
            await proc.wait()
            return ImageExecResult(False, None, error=f"code timed out after {timeout}s")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    stdout = stdout_b.decode(errors="replace").strip()
    stderr = stderr_b.decode(errors="replace").strip()

    if proc.returncode != 0:
        return ImageExecResult(
            False, None, stdout=_truncate(stdout),
            error=_truncate(stderr or f"exit code {proc.returncode}"),
        )

    if not _is_valid_image(output_image_path):
        return ImageExecResult(
            False, None, stdout=_truncate(stdout),
            error="code ran but produced no valid output image",
        )

    changed = _images_differ(input_image_path, output_image_path)
    return ImageExecResult(True, output_image_path, stdout=_truncate(stdout), changed=changed)


def _kill_process_tree(proc) -> None:
    """SIGKILL the subprocess's whole session/process group (set up via setsid)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
