"""Optional observation of sandbox calls, so a caller can correlate engine work with its own trace.

A caller binds a trace id with `trace_scope`; the engine reports every sandbox call (which operation,
how it ended, how long it took) to its `observer`, together with the id bound at that moment. The record
holds names, an outcome and a duration only: never an expression, a payload or a result. The trace id
travels in a context variable, so it follows work handed to a thread pool without any change to the
operations themselves. An observer that raises is ignored: observing must never change a computation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_trace_id: ContextVar[str | None] = ContextVar("minerva_trace_id", default=None)


@dataclass(frozen=True)
class SandboxCall:
    """One operation run in the sandbox."""

    operation: str  # the operation's name, e.g. "simplify"
    outcome: str  # ok | timeout | killed | error | busy (the sandbox's own words)
    ms: int  # wall clock including the wait for a free sandbox slot
    trace_id: str | None


Observer = Callable[[SandboxCall], None]


def current_trace_id() -> str | None:
    return _trace_id.get()


@contextmanager
def trace_scope(trace_id: str | None) -> Iterator[None]:
    """Bind a trace id to the engine calls made inside the block (and in threads started from it)."""
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        try:
            _trace_id.reset(token)
        except ValueError:
            # Closed from another context (an async generator finalised late): clear, do not fail.
            _trace_id.set(None)
