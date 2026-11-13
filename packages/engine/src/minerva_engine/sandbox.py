"""Process isolation and resource limits for symbolic computation.

Every operation runs in a *separate child process* started from a `forkserver` that has
SymPy pre-imported (so start-up is cheap and the API process never imports user-driven
state). The child applies OS limits (CPU seconds and address space, where the platform
enforces them), computes, and sends back a JSON-compatible dict through a pipe. The parent
enforces a wall-clock timeout and kills the child if it is exceeded or if it dies
(for instance out of memory). A bounded semaphore caps concurrent children.

This is defence in depth, not a security boundary against arbitrary code: user input is
never executed as code (see `textparse`/`latexparse`). The container/Kubernetes layer is
expected to add its own CPU, memory and network restrictions.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import resource
import threading
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

from .models import Limits

_PRELOAD = ["sympy", "minerva_engine.ops"]
_ctx_lock = threading.Lock()
_ctx: Any = None


def _context() -> Any:
    global _ctx
    with _ctx_lock:
        if _ctx is None:
            ctx = multiprocessing.get_context("forkserver")
            ctx.set_forkserver_preload(_PRELOAD)
            _ctx = ctx
        return _ctx


class Outcome(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    KILLED = "killed"  # child died without answering (typically memory limit)
    ERROR = "error"  # child raised an unexpected exception
    BUSY = "busy"  # concurrency cap reached
    UNSUPPORTED = "unsupported"  # refused before running: the operation does not take this input


@dataclass
class SandboxResult:
    outcome: Outcome
    value: dict[str, Any] | None = None


def _child_main(conn: Connection, op: str, payload: dict[str, Any], cpu: int, mem: int) -> None:
    try:
        for which, limit in (
            (resource.RLIMIT_CPU, cpu),
            (resource.RLIMIT_AS, mem),
            (resource.RLIMIT_CORE, 0),
        ):
            # Not enforceable everywhere (e.g. RLIMIT_AS on macOS); the timeout still applies.
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(which, (limit, limit))
        from .ops import OPERATIONS

        result = OPERATIONS[op](payload)
        conn.send(("ok", result))
    except BaseException as exc:
        with contextlib.suppress(Exception):
            conn.send(("error", type(exc).__name__))
    finally:
        conn.close()


class Sandbox:
    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()
        self._slots = threading.BoundedSemaphore(self.limits.max_concurrency)

    def run(self, op: str, payload: dict[str, Any], timeout: float | None = None) -> SandboxResult:
        """`timeout` replaces the default wall-clock limit for this call (long, bounded searches)."""
        seconds = timeout if timeout is not None else self.limits.timeout_seconds
        if not self._slots.acquire(timeout=self.limits.timeout_seconds):
            return SandboxResult(Outcome.BUSY)
        try:
            return self._run(op, payload, seconds)
        finally:
            self._slots.release()

    def _run(self, op: str, payload: dict[str, Any], seconds: float) -> SandboxResult:
        ctx = _context()
        parent, child = ctx.Pipe(duplex=False)
        cpu = max(1, int(seconds) + 2)
        mem = self.limits.memory_mb * 1024 * 1024
        proc = ctx.Process(target=_child_main, args=(child, op, payload, cpu, mem), daemon=True)
        proc.start()
        child.close()
        try:
            if not parent.poll(seconds):
                return SandboxResult(Outcome.TIMEOUT)
            try:
                kind, value = parent.recv()
            except (EOFError, OSError):
                return SandboxResult(Outcome.KILLED)
            if kind == "ok":
                return SandboxResult(Outcome.OK, value)
            if value in ("MemoryError", "RecursionError"):
                return SandboxResult(Outcome.KILLED)  # hit a resource limit, not a bug
            return SandboxResult(Outcome.ERROR)
        finally:
            parent.close()
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
