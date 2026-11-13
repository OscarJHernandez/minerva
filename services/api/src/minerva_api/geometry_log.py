"""Structured log events for the geometry operation.

One `geometry` event per request and one `geometry_stage` event per stage, so that the stage that was slow,
that grew the expression, or that failed can be found from the logs. Only enumerated names (the stage
and quantity names, statuses, failure categories) and counters (milliseconds, operation counts, the
dimension) are logged. Never a coordinate, a metric component, a function name or a tensor component: those
are the user's derivation, and the allow-list in `logging.py` drops anything else.
"""

from __future__ import annotations

from minerva_engine import GeometryResult

from .logging import get_logger

log = get_logger()


def log_geometry(result: GeometryResult) -> None:
    log.info(
        "geometry",
        operation="geometry",
        status=result.status.value,
        dimension=result.dimension,
        quantities=[q.kind for q in result.quantities],
        duration_ms=result.duration_ms,
    )
    for stage in result.diagnostics:
        log.info(
            "geometry_stage",
            operation="geometry",
            stage=stage.stage,
            status=stage.status,
            failure=stage.failure,
            duration_ms=stage.duration_ms,
            ops=stage.ops,
            dimension=result.dimension,
        )
