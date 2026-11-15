"""Chebyshev approximation of a function on a finite interval, kept in the Chebyshev basis.

The coefficients are computed in mpmath from the function's values at the Chebyshev nodes and are
cross-checked against `numpy.polynomial.Chebyshev.interpolate`, an independent implementation. The
printed coefficients (12 significant digits) are the ones that are measured, so what is shown is what
was tested. The error is measured on two grids that share no point with each other or with the nodes.
"""

from __future__ import annotations

from typing import Any

import mpmath
import numpy as np
from numpy.polynomial import Chebyshev

from ..models import Status
from .common import Context, Refused, finish
from .interval import Measured, Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Candidate, Check, Complexity, OrderTerm

DIGITS = 12
MAX_DEGREE = 64
AGREE = mpmath.mpf(
    "1e-9"
)  # mpmath and numpy coefficients must agree to this share of the largest one
SLOW = mpmath.mpf("1e-4")  # last coefficient above this share of the largest: not converged


def _round(value: Any) -> Any:
    return mpmath.mpf(mpmath.nstr(value, DIGITS, min_fixed=-mpmath.inf, max_fixed=mpmath.inf))


def nodes(n: int) -> list[Any]:
    """Chebyshev points of the first kind on [-1, 1]."""
    return [mpmath.cos(mpmath.pi * (2 * j + 1) / (2 * n)) for j in range(n)]


def coefficients(ref: Reference, degree: int) -> list[Any]:
    """Interpolation at degree+1 Chebyshev nodes: a_k = (2/n) sum_j f(x_j) T_k(t_j), a_0 halved."""
    n = degree + 1
    with mpmath.workdps(40):
        ts = nodes(n)
        values = []
        for t in ts:
            x = ((ref.b - ref.a) * t + (ref.a + ref.b)) / 2
            y = ref.f(x)
            values.append(mpmath.re(y))
        out = []
        for k in range(n):
            total = sum(
                (
                    v * mpmath.cos(k * mpmath.pi * (2 * j + 1) / (2 * n))
                    for j, v in enumerate(values)
                ),
                mpmath.mpf(0),
            )
            out.append(total * 2 / n)
        out[0] /= 2
        largest = max(abs(c) for c in out)
        # coefficients at rounding level are zeros of the series, not information
        return [
            _round(c) if abs(c) > mpmath.mpf(10) ** -30 * largest else mpmath.mpf(0) for c in out
        ]


def evaluator(ref: Reference, coeffs: list[Any]) -> Any:
    """Clenshaw's recurrence in the Chebyshev basis (stable; no monomial conversion)."""

    def at(x: Any) -> Any:
        t = ref.t(x)
        b1 = b2 = mpmath.mpf(0)
        for c in reversed(coeffs[1:]):
            b1, b2 = 2 * t * b1 - b2 + c, b1
        return t * b1 - b2 + coeffs[0]

    return at


def numpy_coefficients(ref: Reference, degree: int) -> list[float] | None:
    def func(xs: Any) -> Any:
        return np.array([float(mpmath.re(ref.f(mpmath.mpf(float(x))))) for x in xs])

    try:
        return [
            float(c)
            for c in Chebyshev.interpolate(func, degree, domain=[float(ref.a), float(ref.b)]).coef
        ]
    except (ValueError, TypeError, ArithmeticError, OverflowError):
        return None


def latex_series(ref: Reference, coeffs: list[Any]) -> str:
    text = ""
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        piece = mpmath.nstr(abs(c), DIGITS)
        term = rf"{piece}\,T_{{{k}}}(t)"
        sign = "-" if c < 0 else "+"
        text = (f"-{term}" if c < 0 else term) if not text else f"{text} {sign} {term}"
    return text


def mapping_latex(ref: Reference, variable: str) -> str:
    a, b = mpmath.nstr(ref.a, 12), mpmath.nstr(ref.b, 12)
    return rf"t = \frac{{2{variable} - ({a} + {b})}}{{{b} - ({a})}}"


def build(
    ref: Reference,
    degree: int,
    variable: str,
    tolerance: Any | None = None,
    kind: str | None = None,
) -> tuple[ApproximationReport, Measured | None]:
    if not 0 <= degree <= MAX_DEGREE:
        report = ApproximationReport(
            status=Status.UNSUPPORTED,
            reason="order_out_of_range",
            method="chebyshev",
            message=f"Give a degree between 0 and {MAX_DEGREE}.",
        )
        return report, None
    coeffs = coefficients(ref, degree)
    approx = evaluator(ref, coeffs)
    measured = measure(ref, approx)
    ours = [float(c) for c in coeffs]
    theirs = numpy_coefficients(ref, degree)
    largest = max(abs(c) for c in ours) or 1.0
    if theirs is not None and not all(np.isfinite(theirs)):
        theirs = None
    if theirs is None:
        coefficient_check = Check(
            name="coefficients against numpy.polynomial.Chebyshev.interpolate",
            passed=None,
            detail="not run: numpy could not evaluate the function in double precision (values out of its range)",
        )
    else:
        gap = max(abs(a - b) for a, b in zip(ours, theirs, strict=True))
        coefficient_check = Check(
            name="coefficients against numpy.polynomial.Chebyshev.interpolate",
            passed=bool(gap <= float(AGREE) * largest),
            points=degree + 1,
            detail=f"largest difference {gap:.2e} (numpy works in double precision)",
        )
    stable = Check(
        name="error measured on two disjoint grids",
        passed=not measured.unstable,
        points=measured.summary.samples,
        detail="the two grids give the same maximum error within a factor 1.5"
        if not measured.unstable
        else "the two grids disagree; the error estimate is not reliable",
    )
    report = ApproximationReport(
        status=Status.OK,
        method="chebyshev",
        variable=variable,
        domain_latex=ref.domain_latex,
        result_latex=latex_series(ref, coeffs),
        statement_latex=mapping_latex(ref, variable),
        checks=[coefficient_check, stable],
        error=measured.summary,
        error_curve=measured.curve,
        complexity=Complexity(degree=degree, coefficients=degree + 1),
        warnings=list(ref.warnings),
        notes=list(ref.notes),
    )
    for k, c in enumerate(coeffs):
        if c == 0:
            continue
        report.orders.append(
            OrderTerm(
                order=k,
                coefficient_latex=mpmath.nstr(c, DIGITS),
                term_latex=rf"{mpmath.nstr(c, DIGITS)}\,T_{{{k}}}(t)",
            )
        )
    if coefficient_check.passed is False or measured.unstable or measured.undefined_at is not None:
        report.status = Status.NOT_VERIFIED
        report.reason = "verification_failed"
        report.result_latex = None
        report.message = "A Chebyshev approximation was computed but its checks did not agree, so it is not shown."
        if measured.unstable:
            report.warnings.append("error_estimate_unstable")
        return report, measured
    if abs(coeffs[-1]) > SLOW * max(abs(c) for c in coeffs) and degree >= 3:
        report.warnings.append("slow_coefficient_decay")
        report.notes.append(
            "The last coefficients are not small: the function is not well resolved at this degree "
            "(a kink, a jump, or too low a degree)."
        )
    met = tolerance_met(measured, tolerance, kind)
    report.tolerance_met = met
    report.message = (
        f"Chebyshev interpolation of degree {degree} on the interval, kept in the Chebyshev basis. "
        f"The largest error on {measured.summary.samples} sampled points is {measured.summary.max_abs}; "
        "this is a sampled estimate, not a bound."
    )
    report.evidence = "numeric_estimate"
    return report, measured


LADDER = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64)


def candidate_of(report: ApproximationReport, measured: Measured | None, label: str) -> Candidate:
    return Candidate(
        method=report.method,
        label=label,
        status=report.status,
        reason=report.reason,
        result_latex=report.result_latex,
        complexity=report.complexity,
        error=report.error,
        tolerance_met=report.tolerance_met,
        warnings=list(report.warnings),
        error_curve=list(measured.curve) if measured is not None else [],
    )


def chebyshev(ctx: Context) -> ApproximationReport:
    ref = build_reference(ctx)
    tolerance, kind = parse_tolerance(ctx)
    data = ctx.data
    variable = ctx.var.name
    if data.degree is not None:
        report, _ = build(ref, data.degree, variable, tolerance, kind)
        return finish(report, evidence=report.evidence, warnings=[])
    if tolerance is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give a degree, or an accuracy target to search for the lowest degree that meets it.",
            ["Which degree, or which accuracy?"],
        )
    cap = min(MAX_DEGREE, (data.max_terms - 1) if data.max_terms else MAX_DEGREE)
    tried: list[Candidate] = []
    last: ApproximationReport | None = None
    for degree in (d for d in LADDER if d <= cap):
        if ctx.expired():
            break
        report, measured = build(ref, degree, variable, tolerance, kind)
        tried.append(candidate_of(report, measured, f"degree {degree}"))
        last = report
        if report.status is Status.OK and report.tolerance_met:
            tried[-1].chosen = True
            report.candidates = tried
            report.message += (
                " It is the lowest degree tried that meets the tolerance on the sampled points."
            )
            return report
    if last is None:
        raise Refused(
            Status.RESOURCE_LIMIT,
            "resource_limit",
            "The time budget was used up before a degree could be tried.",
        )
    last.candidates = tried
    last.tolerance_met = False
    last.reason = None
    last.message += (
        f" The requested tolerance was NOT met by any degree up to {tried[-1].complexity.degree}"
        " (on the sampled points); the last degree tried is shown."
    )
    return last
