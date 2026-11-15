"""Structured log event for the vector calculus operation.

One `vector_calculus` event per request: the status, a fixed-vocabulary reason, the coordinate system,
the action and counters. Never a field name, an expression or a component: those are the user's
derivation, and the allow-list in `logging.py` drops anything else.
"""

from __future__ import annotations

from minerva_engine import VectorResult

from .logging import get_logger

log = get_logger()


def log_vector(result: VectorResult) -> None:
    d = result.diagnostics
    log.info(
        "vector_calculus",
        operation="vector_calculus",
        status=result.status.value,
        reason=d.reason,
        coordinate_system=d.coordinates,
        vector_action=d.action,
        operator_depth=d.operator_depth,
        points=d.points,
        duration_ms=result.duration_ms,
    )
