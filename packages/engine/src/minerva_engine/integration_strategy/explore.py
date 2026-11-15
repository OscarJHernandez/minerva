"""Bounded exploration: try the promising techniques, verify every candidate, keep the first.

The order comes from `detect` (specific techniques first). At most `MAX_TECHNIQUES` are tried unless
one is requested by name; each candidate must pass the independent checkers against the *original*
integrand or it is discarded and counted. A candidate that was blocked (a condition of the technique
was not established) is reported as such and never applied. Nothing here calls a language model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import sympy as sp

from ..integrate import (
    IntegrateVerification,
    QuadratureVerification,
    check_antiderivative,
    check_definite,
)
from ..latexprint import to_latex
from ..models import Limits
from . import detect, numeric, transforms
from .models import Attempt, Step, StrategyReport, Suggestion, Technique, Verification
from .transforms import Ctx, Found

MAX_TECHNIQUES = 4
RUNNERS = {
    "frullani": transforms.frullani,
    "parity": transforms.parity,
    "reflection": transforms.reflection,
    "schwinger": transforms.schwinger,
    "partial_fractions": transforms.partial_fractions,
    "substitution": transforms.substitution,
    "parts": transforms.parts,
    "trigonometric": transforms.trigonometric,
    "parameter_differentiation": transforms.parameter_differentiation,
}


@dataclass
class Explored:
    report: StrategyReport
    expr: sp.Expr | None = None  # the antiderivative or the value; None if nothing was verified
    definite: bool = False
    antiderivative: IntegrateVerification | None = None
    quadrature: QuadratureVerification | None = None
    estimate: numeric.Estimate | None = None


def suggest(
    f: sp.Expr, x: sp.Symbol, lower: sp.Expr | None, upper: sp.Expr | None
) -> list[Suggestion]:
    try:
        return detect.detect(f, x, lower, upper)
    except Exception:  # a detection failure must never reach the mathematics
        return []


def _limit(F: sp.Expr, x: sp.Symbol, point: sp.Expr, side: str) -> sp.Expr | None:
    try:
        value = sp.limit(F, x, point, side) if point.is_finite else sp.limit(F, x, point)
    except Exception:
        return None
    bad = (sp.nan, sp.zoo, sp.oo, -sp.oo, sp.Limit, sp.AccumBounds)
    return None if value.has(*bad) else value


def _definite_from(F: sp.Expr, x: sp.Symbol, lo: sp.Expr, hi: sp.Expr) -> sp.Expr | None:
    upper, lower = _limit(F, x, hi, "-"), _limit(F, x, lo, "+")
    return None if upper is None or lower is None else sp.simplify(upper - lower)


def _tidy(f: sp.Expr, x: sp.Symbol, F: sp.Expr) -> sp.Expr:
    """The simplest equivalent form that still passes the derivative check ("simplest useful result")."""
    if sp.count_ops(F) > 150:
        return F
    for tidy in (sp.simplify(F), sp.trigsimp(F)):
        if (
            tidy != F
            and sp.count_ops(tidy) < sp.count_ops(F)
            and check_antiderivative(f, x, tidy).passed
        ):
            F = tidy
    return F


def _verified(
    f: sp.Expr, x: sp.Symbol, found: Found, ctx: Ctx
) -> tuple[sp.Expr, IntegrateVerification | None, QuadratureVerification | None] | None:
    """The result to show, if the independent checks pass."""
    if not ctx.definite or ctx.lower is None or ctx.upper is None:
        if found.kind != "antiderivative":
            return None
        check = check_antiderivative(f, x, found.expr)
        if not check.passed:
            return None
        found.expr = _tidy(f, x, found.expr)
        if sp.count_ops(found.expr) <= 150:
            try:
                check.symbolic_zero = bool(sp.simplify(sp.diff(found.expr, x) - f) == 0)
            except Exception:
                check.symbolic_zero = False
        return found.expr, check, None
    value = (
        found.expr if found.kind == "value" else _definite_from(found.expr, x, ctx.lower, ctx.upper)
    )
    if value is None:
        return None
    quad = check_definite(f, x, ctx.lower, ctx.upper, value, deadline=ctx.deadline)
    return (value, None, quad) if quad.passed else None


def explore(
    f: sp.Expr,
    x: sp.Symbol,
    lower: sp.Expr | None,
    upper: sp.Expr | None,
    *,
    limits: Limits,
    deadline: float,
    assumptions: dict[str, list[str]],
    only: Technique | None = None,
) -> Explored:
    ctx = Ctx(x, lower, upper, limits, deadline, assumptions)
    suggestions = suggest(f, x, lower, upper)
    report = StrategyReport(suggestions=suggestions)
    if only is not None:
        order: list[Technique] = [only]
    else:
        order = []
        for s in suggestions:
            if s.technique not in order and s.technique != "numerical":
                order.append(s.technique)
        order = order[:MAX_TECHNIQUES]
    why = {s.technique: s.why for s in suggestions}
    for technique in order:
        if time.monotonic() > deadline:
            report.attempts.append(Attempt(technique=technique, outcome="deadline"))
            break
        runner = RUNNERS.get(technique)
        if runner is None:
            if technique == "numerical":
                continue
            report.attempts.append(Attempt(technique=technique, outcome="not_applicable"))
            continue
        try:
            found = runner(f, ctx)
        except Exception:  # a technique failing must never reach the mathematics
            found = None
        if found is None:
            outcome = "no_progress" if technique in why else "not_applicable"
            report.attempts.append(Attempt(technique=technique, outcome=outcome))
            continue
        if found.blocked:
            report.attempts.append(
                Attempt(technique=technique, outcome="blocked", detail=found.blocked)
            )
            continue
        verified = _verified(f, x, found, ctx)
        if verified is None:
            report.candidates_rejected += 1
            report.attempts.append(Attempt(technique=technique, outcome="failed_verification"))
            continue
        value, anti, quad = verified
        report.attempts.append(Attempt(technique=technique, outcome="verified"))
        report.technique = technique
        report.why = why.get(technique)
        report.steps = _with_result(found.steps, value, ctx)
        report.conditions_latex = found.conditions
        level: Verification = (
            "proved" if anti is not None and anti.symbolic_zero else "numerically_checked"
        )
        report.verification = level
        return Explored(report, value, ctx.definite, anti, quad)
    if (
        ctx.definite
        and lower is not None
        and upper is not None
        and (only in (None, "numerical"))
        and not detect.parameters(f, x)
    ):
        estimate = numeric.estimate(f, x, lower, upper)
        if isinstance(estimate, numeric.Estimate):
            report.technique, report.why = "numerical", "no_closed_form_for_a_number"
            report.verification = "numeric_estimate"
            report.attempts.append(Attempt(technique="numerical", outcome="verified"))
            report.steps = [
                Step(what="numerical value", latex=estimate.value),
                Step(what="estimated error (the two rules' disagreement)", latex=estimate.error),
            ]
            report.note = (
                "A numerical estimate only: no closed form was found. It is not a formula and not "
                "a proof; two independent quadrature rules agreed to the stated error."
            )
            return Explored(report, None, True, estimate=estimate)
        report.attempts.append(
            Attempt(technique="numerical", outcome="failed_verification", detail=estimate)
        )
    elif only == "numerical":  # a free symbol, or an indefinite integral: no number to give
        report.attempts.append(Attempt(technique="numerical", outcome="not_applicable"))
    return Explored(report)


def _with_result(steps: list[Step], value: sp.Expr, ctx: Ctx) -> list[Step]:
    label = "value of the integral" if ctx.definite else "antiderivative"
    return [*steps, Step(what=label, latex=to_latex(value))]
