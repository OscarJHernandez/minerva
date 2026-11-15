"""Structured log events for the approximation operation: the method, status and counters.

Never an expression, a point, an interval, a tolerance or a coefficient: those are the user's work.
The allow-list in `logging.py` drops anything else.
"""

from __future__ import annotations

from minerva_engine import ApproximationReport

from .logging import get_logger

log = get_logger()


def log_approximation(report: ApproximationReport) -> None:
    log.info(
        "approximation",
        operation="approximation",
        approx_method=report.method,
        status=report.status.value,
        reason=report.reason,
        checks=report.diagnostics.checks,
        checks_passed=report.diagnostics.checks_passed,
        warnings=report.diagnostics.warnings,
        candidates=report.diagnostics.candidates,
        duration_ms=report.duration_ms,
    )
