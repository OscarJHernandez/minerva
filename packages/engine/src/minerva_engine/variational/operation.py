"""Entry point of the `variation` operation: one call, one action, explicit statuses."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..errors import LimitError, MinervaMathError, ParseError, UnsupportedInput
from ..models import Limits, Status
from .analysis import analyze
from .models import Diagnostics, VariationInput, VariationReport
from .setup import Problem, read_problem

Handler = Callable[[Problem, VariationInput, float], VariationReport]


def _analyze(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    analysis = analyze(problem)
    return VariationReport(
        status=Status.OK,
        action="analyze",
        analysis=analysis,
        message="Structure of the functional (no variation has been computed yet).",
    )


HANDLERS: dict[str, Handler] = {"analyze": _analyze}


def variation(data: VariationInput, limits: Limits) -> VariationReport:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    report: VariationReport
    try:
        problem = read_problem(data, limits)
        handler = HANDLERS.get(data.action)
        if handler is None:
            raise UnsupportedInput(f"the action {data.action!r} is not available yet")
        report = handler(problem, data, deadline)
        report.diagnostics = Diagnostics(
            functions=problem.jets.count,
            highest_order=max(problem.jets.orders),
        )
    except ParseError as exc:
        report = VariationReport(status=Status.PARSE_ERROR, action=data.action, message=str(exc))
    except UnsupportedInput as exc:
        report = VariationReport(status=Status.UNSUPPORTED, action=data.action, message=str(exc))
    except LimitError as exc:
        report = VariationReport(status=Status.RESOURCE_LIMIT, action=data.action, message=str(exc))
    except MinervaMathError as exc:  # any other user-facing error
        report = VariationReport(status=Status.NOT_VERIFIED, action=data.action, message=str(exc))
    report.duration_ms = (time.monotonic() - started) * 1000
    report.diagnostics.duration_ms = report.duration_ms
    return report
