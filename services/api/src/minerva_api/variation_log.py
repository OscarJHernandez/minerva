"""Structured log events for the variation operation: the action, the status and counters.

Never a functional, an expression, a path, a parameter value or a boundary condition: those are the
user's work. The allow-list in `logging.py` drops anything else.
"""

from __future__ import annotations

from minerva_engine import VariationReport

from .logging import get_logger

log = get_logger()


def log_variation(report: VariationReport) -> None:
    log.info(
        "variation",
        operation="variation",
        action=report.action,
        status=report.status.value,
        functions=report.diagnostics.functions,
        highest_order=report.diagnostics.highest_order,
        stages=report.diagnostics.stages,
        duration_ms=report.duration_ms,
    )
