"""Turning the engine's distribution diagnostics into allow-listed log lines.

The sandboxed child cannot log, so the operation returns its stage events as data. Here each event
becomes one structured line carrying the request id as trace id, the operation, the stage, its
status, the duration and a reason from a fixed list. Expressions, results and messages never go
into a log line (a test enforces it).
"""

from __future__ import annotations

from minerva_engine import DistributionResult

from .logging import get_logger

log = get_logger()


def log_distribution(result: DistributionResult, user_id: str) -> None:
    """One line per stage event; `request_id` is bound by the request middleware."""
    for event in result.diagnostics.events:
        log.info(
            "distribution_event",
            operation="distribution",
            stage=event.stage,
            status=event.status,
            duration_ms=event.duration_ms,
            reason=event.reason,
            user_id=user_id,
        )
