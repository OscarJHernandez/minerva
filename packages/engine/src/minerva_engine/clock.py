"""Interrupt a block of SymPy work at a deadline, so a stage can be abandoned and the rest reported.

The sandbox's wall-clock limit kills the whole child and the user gets nothing; an analysis whose first
stages are done can stop the stage in flight instead and return what is established. Unix main thread only:
elsewhere the block runs to completion and the sandbox limit still applies.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Iterator


class TimeUp(Exception):
    """The deadline passed inside the block."""


@contextlib.contextmanager
def stop_at(deadline: float | None) -> Iterator[None]:
    """Raise `TimeUp` inside the block once `time.monotonic()` reaches `deadline` (None: no limit)."""
    if deadline is None:
        yield
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeUp
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def expire(signum: int, frame: object) -> None:
        raise TimeUp

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, remaining))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
