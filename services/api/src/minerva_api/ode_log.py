"""Structured log events for the differential-equation operations.

One `ode` event per request and one `ode_stage` event per stage. Only counters, durations and fixed
vocabulary (the method and class names, the stage names, a failure category) are logged: never an
equation, an unknown or parameter name, a condition or a value. The allow-list in `logging.py` drops
anything else, and the engine's diagnostics hold nothing else to begin with.
"""

from __future__ import annotations

from minerva_engine.ode import OdeDiagnostics

from .logging import get_logger

log = get_logger()


def log_ode(operation: str, status: str, diagnostics: OdeDiagnostics, duration_ms: int) -> None:
    fields = {
        "ode_method": diagnostics.method,
        "methods_tried": diagnostics.methods_tried or None,
        "order": diagnostics.order,
        "dimension": diagnostics.state_dimension,
        "classes": diagnostics.classes or None,
        "candidates": diagnostics.candidates,
        "verified": diagnostics.verified,
        "dropped": diagnostics.dropped,
        "solution_kind": diagnostics.solution_kind,
        "steps": diagnostics.steps,
        "evaluations": diagnostics.evaluations,
        "max_residual": diagnostics.max_residual,
        "reason": diagnostics.failure_reason,
        "trace_id": diagnostics.trace_id,
    }
    log.info(
        "ode",
        operation=operation,
        status=status,
        duration_ms=duration_ms,
        **{k: v for k, v in fields.items() if v is not None},
    )
    for stage in diagnostics.stages:
        log.info("ode_stage", operation=operation, stage=stage.name, duration_ms=int(stage.ms))
