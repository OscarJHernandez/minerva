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
import contextvars
import multiprocessing
import os
import resource
import threading
import time
import traceback
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from typing import Any

from .models import Limits

_PRELOAD = ["sympy", "minerva_engine.ops"]

# NumPy's BLAS starts one worker thread per core and allocates a buffer for each. Under RLIMIT_AS that
# allocation can fail ("OpenBLAS error: Memory allocation still failed") on machines with many cores, and
# every ODE run then dies as `resource_limit`. The operations here are small: one thread is enough.
_SINGLE_THREAD = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_ctx_lock = threading.Lock()
_ctx: Any = None


def _context() -> Any:
    global _ctx
    with _ctx_lock:
        if _ctx is None:
            for name in _SINGLE_THREAD:  # inherited by the forkserver and its children
                os.environ.setdefault(name, "1")
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
    # For `Outcome.ERROR` only: the class of the exception and where our code raised it (see `_site`).
    error_type: str | None = None
    error_module: str | None = None
    error_line: int | None = None
    # Where the time went (milliseconds, measured by the sandbox itself, no extra work): waiting for a
    # free slot, then running the child. `ms` is their sum. Set for every outcome, `busy` and a call the
    # caller cancelled included.
    wait_ms: int = 0
    run_ms: int = 0

    @property
    def ms(self) -> int:
        return self.wait_ms + self.run_ms


# Classes whose names are fixed by code we ship or depend on; any other name is reported as `other`.
_KNOWN_ERROR_ROOTS = frozenset(
    {"builtins", "sympy", "mpmath", "numpy", "scipy", "minerva_engine", "minerva_knowledge"}
)


def _site(exc: BaseException) -> tuple[str, str | None, int | None]:
    """Where an exception was raised, as the child reports it: class name, module and line of the
    innermost frame of our own code. Never the message, which can hold the user's expression."""
    cls = type(exc)
    name = cls.__name__ if cls.__module__.split(".")[0] in _KNOWN_ERROR_ROOTS else "other"
    innermost: tuple[str, int] | None = None
    for frame, line in traceback.walk_tb(exc.__traceback__):
        module = str(frame.f_globals.get("__name__", ""))
        if module.startswith("minerva_"):
            innermost = (module, line)
    return (name, *innermost) if innermost else (name, None, None)


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
            conn.send(("error", _site(exc)))
    finally:
        conn.close()


# A caller that gives up on a call (the agent's time limit, a client that left) cannot stop a Python thread.
# It sets this event instead: the sandbox looks at it while it waits, kills its child and frees its slot, and
# starts no further child for the same caller. The event travels with the context a worker thread copies.
cancel_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "minerva_sandbox_cancel", default=None
)
_SLICE = 0.05  # how often a waiting call looks at its cancel event, in seconds


def _cancelled() -> bool:
    event = cancel_event.get()
    return event is not None and event.is_set()


class Sandbox:
    def __init__(self, limits: Limits | None = None) -> None:
        self.limits = limits or Limits()
        self._slots = threading.BoundedSemaphore(self.limits.max_concurrency)

    def run(self, op: str, payload: dict[str, Any], timeout: float | None = None) -> SandboxResult:
        """`timeout` replaces the default wall-clock limit for this call (long, bounded searches).

        The result says how long the call waited for a slot (`wait_ms`) and how long its child ran
        (`run_ms`)."""
        seconds = timeout if timeout is not None else self.limits.timeout_seconds
        if _cancelled():
            return SandboxResult(Outcome.KILLED)  # the caller has left: start nothing
        acquired, wait_ms = self._acquire()
        if not acquired:
            outcome = Outcome.BUSY if not _cancelled() else Outcome.KILLED
            return SandboxResult(outcome, wait_ms=wait_ms)
        try:
            started = time.monotonic()
            result = self._run(op, payload, seconds)
            result.wait_ms = wait_ms
            result.run_ms = int((time.monotonic() - started) * 1000)
            return result
        finally:
            self._slots.release()

    def _acquire(self) -> tuple[bool, int]:
        """Wait for a slot; returns whether it was taken and how long the wait took (ms).

        The wait is bounded by the sandbox's default timeout, not the per-call one: a call with a longer
        limit is not queued longer."""
        started = time.monotonic()
        deadline = started + self.limits.timeout_seconds

        def waited() -> int:
            return int((time.monotonic() - started) * 1000)

        while not _cancelled():
            left = deadline - time.monotonic()
            if self._slots.acquire(timeout=max(0.0, min(_SLICE, left))):
                return True, waited()
            if left <= 0:
                return False, waited()
        return False, waited()

    def _run(self, op: str, payload: dict[str, Any], seconds: float) -> SandboxResult:
        ctx = _context()
        parent, child = ctx.Pipe(duplex=False)
        cpu = max(1, int(seconds) + 2)
        mem = self.limits.memory_mb * 1024 * 1024
        proc = ctx.Process(target=_child_main, args=(child, op, payload, cpu, mem), daemon=True)
        proc.start()
        child.close()
        try:
            deadline = time.monotonic() + seconds
            while not parent.poll(max(0.0, min(_SLICE, deadline - time.monotonic()))):
                if _cancelled():
                    return SandboxResult(Outcome.KILLED)  # the finally below kills the child
                if time.monotonic() >= deadline:
                    return SandboxResult(Outcome.TIMEOUT)
            try:
                kind, value = parent.recv()
            except (EOFError, OSError):
                return SandboxResult(Outcome.KILLED)
            if kind == "ok":
                return SandboxResult(Outcome.OK, value)
            name, module, line = value
            if name in ("MemoryError", "RecursionError"):
                return SandboxResult(Outcome.KILLED)  # hit a resource limit, not a bug
            return SandboxResult(Outcome.ERROR, None, name, module, line)
        finally:
            parent.close()
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=5)
