"""Compare approximations of one function on one interval, and search for the simplest that meets a tolerance.

Every candidate is built and verified by its own method and then measured on the same two grids.
The one chosen is the simplest that meets the tolerance (fewest coefficients, then smaller error);
without a tolerance it is the most accurate at the stated budget, and the table shows the cost.
Nothing is a bound: errors are sampled.
"""

from __future__ import annotations

from typing import Any

import mpmath
import sympy as sp

from ..models import Status
from .chebyshev import LADDER, build, candidate_of
from .checks import compile_mp
from .common import Context, Refused, parse_text
from .expand import taylor
from .interval import Measured, Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Candidate, Complexity
from .pade import pade_core

DEFAULT_METHODS = ("taylor", "pade", "chebyshev")
CAP = 30


def _split(budget: int) -> tuple[int, int]:
    """[m/n] with m + n = budget, as balanced as possible (denominator the smaller half)."""
    n = budget // 2
    return budget - n, n


def _taylor_candidate(
    ctx: Context, ref: Reference, x0: sp.Expr, degree: int
) -> tuple[Candidate, Measured | None]:
    data = ctx.data.model_copy(update={"method": "taylor", "point": str(x0), "order": degree})
    sub = Context(data, ctx.limits, ctx.started, ctx.expr, ctx.var)
    try:
        report = taylor(sub)
    except Refused as refused:
        return Candidate(
            method="taylor",
            label=f"Taylor, degree {degree}",
            status=refused.status,
            reason=refused.reason,
        ), None
    poly = parse_text(report.result_text or "0", ctx.data, ctx.limits)
    measured = measure(ref, compile_mp(poly, ctx.var))
    report.complexity = Complexity(degree=degree, coefficients=degree + 1)
    report.error = measured.summary
    c = candidate_of(report, measured, f"Taylor, degree {degree}")
    return c, measured


def _pade_candidate(
    ctx: Context, ref: Reference, x0: sp.Expr, budget: int
) -> tuple[Candidate, Measured | None]:
    m, n = _split(budget)
    try:
        report, approx, _ = pade_core(ctx, m, n, x0, ref)
    except Refused as refused:
        return Candidate(
            method="pade", label=f"Padé [{m}/{n}]", status=refused.status, reason=refused.reason
        ), None
    measured = measure(ref, compile_mp(approx, ctx.var))
    report.error = measured.summary
    if measured.undefined_at is not None and "pole_in_interval" not in report.warnings:
        report.warnings.append("pole_in_interval")
    return candidate_of(report, measured, f"Padé [{m}/{n}]"), measured


def _chebyshev_candidate(
    ctx: Context, ref: Reference, degree: int
) -> tuple[Candidate, Measured | None]:
    report, measured = build(ref, degree, ctx.var.name)
    return candidate_of(report, measured, f"Chebyshev, degree {degree}"), measured


def _fit(candidate: Candidate, measured: Measured | None, tolerance: Any, kind: str | None) -> None:
    if candidate.status is Status.OK and measured is not None:
        candidate.tolerance_met = tolerance_met(measured, tolerance, kind)
        if "pole_in_interval" in candidate.warnings:
            candidate.tolerance_met = False


def _max_abs(c: Candidate) -> float:
    return float(c.error.max_abs) if c.error and c.error.max_abs else float("inf")


def compare(ctx: Context) -> ApproximationReport:
    data = ctx.data
    ref = build_reference(ctx)
    tolerance, kind = parse_tolerance(ctx)
    methods = [m for m in (data.methods or list(DEFAULT_METHODS)) if m in DEFAULT_METHODS]
    if not methods:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            f"Methods to compare: {', '.join(DEFAULT_METHODS)}.",
        )
    if data.degree is None and tolerance is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the largest degree to allow, or an accuracy target.",
            ["What is the largest degree, or the accuracy you need?"],
        )
    cap = min(data.degree if data.degree is not None else CAP, CAP)
    notes: list[str] = []
    if data.point is not None:
        x0 = parse_text(data.point, data, ctx.limits)
    else:
        x0 = sp.nsimplify(mpmath.nstr((ref.a + ref.b) / 2, 15), rational=True)
        notes.append(
            "No expansion point was given, so Taylor and Padé are expanded about the midpoint of the interval."
        )
    budgets = [cap] if tolerance is None else [d for d in LADDER if d <= cap] or [cap]
    chosen_pool: list[tuple[Candidate, Measured | None]] = []
    history: list[Candidate] = []
    for budget in budgets:
        if ctx.expired():
            notes.append("The time budget ended the search before the largest degree.")
            break
        round_: list[tuple[Candidate, Measured | None]] = []
        if "taylor" in methods:
            round_.append(_taylor_candidate(ctx, ref, x0, budget))
        if "pade" in methods and budget >= 2:
            round_.append(_pade_candidate(ctx, ref, x0, budget))
        if "chebyshev" in methods:
            round_.append(_chebyshev_candidate(ctx, ref, budget))
        for cand, measured in round_:
            _fit(cand, measured, tolerance, kind)
        history += [c for c, _ in round_]
        chosen_pool = round_
        if tolerance is not None and any(c.tolerance_met for c, _ in round_):
            break
    usable = [
        (c, m)
        for c, m in chosen_pool
        if c.status is Status.OK and "pole_in_interval" not in c.warnings
    ]
    if tolerance is not None:
        met = [(c, m) for c, m in usable if c.tolerance_met]
        pick = min(
            met, key=lambda cm: (cm[0].complexity.coefficients or 0, _max_abs(cm[0])), default=None
        )
    else:
        pick = min(usable, key=lambda cm: _max_abs(cm[0]), default=None)
    if pick is None:
        pick = min(usable, key=lambda cm: _max_abs(cm[0]), default=None)
    if pick is None:
        raise Refused(
            Status.NOT_VERIFIED, "verification_failed", "No candidate could be built and verified."
        )
    chosen, measured = pick
    chosen.chosen = True
    report = ApproximationReport(
        status=Status.OK,
        method="compare",
        variable=ctx.var.name,
        domain_latex=ref.domain_latex,
        result_latex=chosen.result_latex,
        candidates=history if tolerance is not None else [c for c, _ in chosen_pool],
        complexity=chosen.complexity,
        error=chosen.error,
        error_curve=list(chosen.error_curve),
        warnings=list(dict.fromkeys([*ref.warnings, *chosen.warnings])),
        notes=[*notes, *ref.notes],
        evidence="numeric_estimate",
    )
    met_any = tolerance is None or any(c.tolerance_met for c, _ in usable)
    report.tolerance_met = None if tolerance is None else bool(chosen.tolerance_met)
    if tolerance is None:
        report.message = (
            f"Compared on the interval at a budget of degree {cap}. {chosen.label} has the smallest sampled error "
            f"({chosen.error.max_abs if chosen.error else '?'}); the table shows what each costs. "
            "Errors are measured on sampled points, not bounds."
        )
    elif met_any:
        report.message = (
            f"{chosen.label} is the simplest candidate that meets the tolerance on the sampled points "
            f"({chosen.complexity.coefficients} coefficients). This is numerical support, not a guarantee."
        )
    else:
        report.message = (
            f"No candidate met the tolerance within degree {cap} (on the sampled points). {chosen.label} "
            "is the most accurate found; it is shown but does NOT satisfy the request."
        )
    return report
