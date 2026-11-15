"""Structured log event for creative telescoping: status, outcome and counters.

Never the integrand, a bound, a symbol name, an operator or a certificate: those are the user's
work and stay in their own card. The allow-list in `logging.py` drops anything else.
"""

from __future__ import annotations

from minerva_engine import TelescopingResult

from .logging import get_logger

log = get_logger()


def log_telescoping(result: TelescopingResult) -> None:
    d = result.diagnostics
    log.info(
        "telescoping",
        operation="creative_telescoping",
        status=result.status.value,
        reason=result.reason,
        highest_order=d.order,
        stages=[f"{s.name}:{s.status}" for s in d.stages],
        duration_ms=result.duration_ms,
    )
