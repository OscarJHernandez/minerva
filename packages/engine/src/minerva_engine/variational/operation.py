"""Entry point of the `variation` operation: one call, one action, explicit statuses."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..errors import LimitError, MinervaMathError, ParseError, UnsupportedInput
from ..models import Limits, Status
from .analysis import analyze
from .models import Diagnostics, VariationInput, VariationReport
from .numeric_setup import MissingValues
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
    contradicted = (
        check.verdict == "failed"
        or not pointwise.passed
        or any(not c.passed for c in stage_checks)
        or crosscheck is False
    )
    if contradicted:
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="first",
            message="The derived first variation did not pass its independent numerical checks, "
            "so it is not shown.",
        )
    if not check.passed:  # inconclusive: nothing disagreed, but nothing could be decided either
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="first",
            message="The first variation could not be checked: the finite-difference comparison "
            "with the action was inconclusive on every interval tried (the integrand may be singular, "
            "not real or too large there). Nothing was contradicted and nothing is shown. Give bounds "
            "(for example 1 and 2) or assumptions such as 0 < a < b, and ask again.",
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


def _evaluate(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from .action import evaluate

    return evaluate(problem, data, deadline)


def _verify(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from .userverify import verify

    return verify(problem, data, deadline)


def _solve(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from .stationary import solve

    return solve(problem, data, deadline)


def _second(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from .secondrun import second

    return second(problem, data, deadline)


HANDLERS: dict[str, Handler] = {
    "analyze": _analyze,
    "first": _first,
    "evaluate": _evaluate,
    "verify": _verify,
    "second": _second,
    "solve": _solve,
}


def variation(data: VariationInput, limits: Limits) -> VariationReport:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    report: VariationReport
    problem: Problem | None = None
    try:
        problem = read_problem(data, limits)
        handler = HANDLERS.get(data.action)
        if handler is None:
            raise UnsupportedInput(f"the action {data.action!r} is not available yet")
        report = handler(problem, data, deadline)
        if report.analysis is None:
            report.analysis = analyze(problem)  # structure and next steps accompany every result
        report.diagnostics.functions = problem.jets.count
        report.diagnostics.highest_order = max(problem.jets.orders)
    except MissingValues as exc:
        report = VariationReport(
            status=Status.CONDITIONS_NOT_SATISFIED, action=data.action, message=str(exc)
        )
    except ParseError as exc:
        report = VariationReport(status=Status.PARSE_ERROR, action=data.action, message=str(exc))
    except UnsupportedInput as exc:
        report = VariationReport(status=Status.UNSUPPORTED, action=data.action, message=str(exc))
    except LimitError as exc:
        report = VariationReport(status=Status.RESOURCE_LIMIT, action=data.action, message=str(exc))
    except MinervaMathError as exc:  # any other user-facing error
        report = VariationReport(status=Status.NOT_VERIFIED, action=data.action, message=str(exc))
    if problem is not None:
        try:
            report.input_latex = problem.latex(problem.lagrangian)
        except Exception:  # printing only; the result does not depend on it
            report.input_latex = None
    report.duration_ms = (time.monotonic() - started) * 1000
    report.diagnostics.duration_ms = report.duration_ms
    return report
