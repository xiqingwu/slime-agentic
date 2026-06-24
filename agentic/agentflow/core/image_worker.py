"""Warm forkserver executor for OpenCV snippets (plan B3 — exec-speed).

``image_tool.run_opencv_code`` spawns a fresh ``python -c`` per ``<code>`` step,
which re-imports cv2/numpy every time (~hundreds of ms). Across 1200×8×(turns)
rollouts that adds up. This module amortizes the import cost using a
``multiprocessing`` *forkserver*: a server process imports cv2/numpy once at
startup (``set_forkserver_preload``), and every exec is a fresh child forked from
that server — so the heavy import is already paid, while each snippet still runs in
its own short-lived process (no state leaks between snippets, same isolation the
subprocess path gives, plus the same rlimits as ``image_tool`` — see B2).

It returns the exact same ``ImageExecResult`` contract as ``run_opencv_code`` (incl.
the ``changed`` flag), so the tool can swap between the two transparently.

OPT-IN: enabled via ``MAT_CV2_FORKSERVER=1`` (default off). It needs validation on
the real Ray/CUDA training box before becoming the default — forkserver interacts
with the parent process's state, and that can't be exercised by the offline tests
(which do cover the executor's behaviour with real cv2).
"""

from __future__ import annotations

import asyncio
import io
import multiprocessing as mp
import os
import tempfile
import uuid

from .image_tool import (
    ImageExecResult,
    extract_code,
    replace_paths,
    _sanitize_code,
    _is_valid_image,
    _images_differ,
    _truncate,
    _apply_rlimits,
)

_ctx = None  # cached forkserver context (created lazily, preloads cv2/numpy)


def _get_ctx():
    global _ctx
    if _ctx is None:
        ctx = mp.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(["cv2", "numpy"])
        except Exception:  # pragma: no cover - preload is best-effort
            pass
        _ctx = ctx
    return _ctx


def _child_exec(code: str, input_path: str, output_path: str, cpu_seconds: int, conn) -> None:
    """Runs in a forkserver child: apply rlimits, exec the snippet, report back.

    cv2/numpy are already imported (forkserver preload), so this is fast. The result
    dict mirrors the fields ImageExecResult needs.
    """
    # Reuse the same rlimit policy as the subprocess path (B2); no setsid needed
    # since this child is already its own process.
    _apply_rlimits(cpu_seconds)

    result = {"success": False, "error": None, "changed": False, "stdout": ""}
    buf = io.StringIO()
    try:
        import contextlib

        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<model_code>", "exec"), {"__name__": "__main__"})
    except BaseException as exc:  # noqa: BLE001 - report any failure, don't crash silently
        result["error"] = _truncate(f"{type(exc).__name__}: {exc}")
        result["stdout"] = _truncate(buf.getvalue().strip())
        _send(conn, result)
        return

    result["stdout"] = _truncate(buf.getvalue().strip())
    if not _is_valid_image(output_path):
        result["error"] = "code ran but produced no valid output image"
    else:
        result["success"] = True
        result["changed"] = _images_differ(input_path, output_path)
    _send(conn, result)


def _send(conn, obj) -> None:
    try:
        conn.send(obj)
        conn.close()
    except Exception:  # pragma: no cover - parent may have gone away
        pass


def _recv_blocking(conn, timeout: float):
    """Block until a result arrives or ``timeout`` elapses; '<timeout>' on expiry."""
    if conn.poll(timeout):
        try:
            return conn.recv()
        except EOFError:
            return None  # child died without sending
    return "<timeout>"


async def run_in_forkserver(
    code_text: str,
    input_image_path: str,
    output_image_path: str | None = None,
    timeout: int = 30,
) -> ImageExecResult:
    """forkserver-backed twin of ``image_tool.run_opencv_code`` (same contract)."""
    code = extract_code(code_text)
    if code is None:
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

    ctx = _get_ctx()
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_exec,
        args=(code, input_image_path, output_image_path, timeout + 2, send_conn),
        daemon=True,
    )
    proc.start()
    send_conn.close()  # parent drops its copy so recv sees EOF if the child dies

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _recv_blocking, recv_conn, float(timeout))
    recv_conn.close()

    if data == "<timeout>":
        proc.kill()
        await loop.run_in_executor(None, proc.join, 2)
        return ImageExecResult(False, None, error=f"code timed out after {timeout}s")

    await loop.run_in_executor(None, proc.join, 2)
    if data is None:
        return ImageExecResult(False, None, error=f"worker died (exit code {proc.exitcode})")

    if not data["success"]:
        return ImageExecResult(False, None, stdout=data.get("stdout", ""), error=data.get("error"))
    return ImageExecResult(True, output_image_path, stdout=data.get("stdout", ""), changed=data["changed"])
