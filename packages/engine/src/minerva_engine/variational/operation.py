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


def _first(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from . import report as rep
    from .firstvar import first_variation
    from .verify import (
        VariationForm,
        check_el_pointwise,
        check_first_variation,
        check_ibp_step,
        sympy_euler,
    )

    derived = first_variation(problem)
    form = VariationForm.of(derived)
    check = check_first_variation(problem, form, deadline=deadline)
    pointwise = check_el_pointwise(problem, form.interior)
    stage_checks = [
        check_ibp_step(problem, derived.stages[k], derived.stages[k + 1])
        for k in range(len(derived.stages) - 1)
    ]
    crosscheck = sympy_euler(problem, form.interior)
    if not (
        check.passed
        and pointwise.passed
        and all(c.passed for c in stage_checks)
        and crosscheck is not False
    ):
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="first",
            message="The derived first variation did not pass its independent numerical checks, "
            "so it is not shown.",
        )
    first = rep.build(problem, derived, check, pointwise, stage_checks, crosscheck)
    return VariationReport(
        status=Status.OK,
        action="first",
        analysis=analyze(problem),
        first=first,
        message="First variation derived; every integration by parts is listed and nothing is dropped.",
        diagnostics=Diagnostics(stages=len(derived.stages)),
    )


HANDLERS: dict[str, Handler] = {"analyze": _analyze, "first": _first}


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
        report.diagnostics.functions = problem.jets.count
        report.diagnostics.highest_order = max(problem.jets.orders)
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
