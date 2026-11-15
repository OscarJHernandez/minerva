"""Turning the engine's distribution diagnostics into allow-listed log lines.

The sandboxed child cannot log, so the operation returns its stage events as data. Here each event
becomes one structured line carrying the request id as trace id, the operation, the stage, its
status, the duration and a reason from a fixed list. Expressions, results and messages never go
into a log line (a test enforces it).
"""

from __future__ import annotations

from minerva_engine import DistributionResult, StatMechResult

from .logging import get_logger

log = get_logger()


def log_distribution(result: DistributionResult, user_id: str | None = None) -> None:
    """One line per stage event; `request_id` is bound by the request middleware."""
    for event in result.diagnostics.events:
        log.info(
            "distribution_event",
            operation="distribution",
            stage=event.stage,
            status=event.status,
            duration_ms=event.duration_ms,
            reason=event.reason,
            **({"user_id": user_id} if user_id else {}),
        )


def log_statmech(result: StatMechResult, system_kind: str, user_id: str | None = None) -> None:
    """Stage events of a statistical-mechanics computation: names, counts and enumerated reasons."""
    for event in result.diagnostics.events:
        log.info(
            "statmech_event",
            operation="statmech",
            stage=event.stage,
            status=event.status,
            duration_ms=event.duration_ms,
            reason=event.reason,
            system_kind=system_kind,
            levels=result.diagnostics.levels,
            parameters=result.diagnostics.parameters,
            points_checked=result.diagnostics.points_checked,
            points_agreed=result.diagnostics.points_agreed,
            **({"user_id": user_id} if user_id else {}),
        )
