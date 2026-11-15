"""Legendre expansion on a finite interval (least squares, with the affine map to [-1, 1]).

a_n = (2n + 1)/2 * integral of f(x(t)) P_n(t) dt, by composite Gauss-Legendre in mpmath with two panel
counts that must agree, and independently by numpy's `leggauss` in double precision. The error of the
truncation is measured on two disjoint grids; no spherical harmonics or PDE solving is attempted.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath
import numpy as np
from numpy.polynomial import legendre as npl

from ..models import Status
from .chebyshev import candidate_of
from .common import Context, Refused, finish
from .fourier import jump_points
from .interval import Measured, Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Check, Complexity, OrderTerm
from .quadnodes import DPS, breakpoints, composite, usable

MAX_DEGREE = 40
LADDER = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40)
AGREE = mpmath.mpf("1e-10")
NUMPY_AGREE = 1e-8
DIGITS = 12


def _legendre_values(t: Any, degree: int) -> list[Any]:
    values = [mpmath.mpf(1), t]
    for n in range(1, degree):
        values.append(((2 * n + 1) * t * values[n] - n * values[n - 1]) / (n + 1))
    return values[: degree + 1]


def _project(xs: list[Any], ws: list[Any], ys: list[Any], ref: Reference, degree: int) -> list[Any]:
    sums = [mpmath.mpf(0)] * (degree + 1)
    scale = (ref.b - ref.a) / 2  # dx = scale dt
    for x, w, y in zip(xs, ws, ys, strict=True):
        p = _legendre_values(ref.t(x), degree)
        for n in range(degree + 1):
            sums[n] += w * y * p[n]
    return [(2 * n + 1) / (2 * scale) * s for n, s in enumerate(sums)]


def coefficients(ctx: Context, ref: Reference, degree: int) -> tuple[list[Any], Any, list[Any]]:
    with mpmath.workdps(DPS):
        breaks = sorted(
            {*breakpoints(ctx.expr, ctx.var, ref.a, ref.b), *jump_points(ref)}, key=float
        )
        points = [ref.a, *[b for b in breaks if ref.a < b < ref.b], ref.b]
        cycles = degree / float(ref.b - ref.a)
        coarse = composite(points, cycles / 2, ref.f, 1)
        fine = composite(points, cycles / 2, ref.f, 2)
        if not (usable(coarse) and usable(fine)):
            raise Refused(
                Status.UNSUPPORTED,
                "domain_invalid",
                "The function could not be evaluated for the integrals.",
            )
        c1 = _project(coarse.xs, coarse.ws, coarse.ys, ref, degree)
        c2 = _project(fine.xs, fine.ws, fine.ys, ref, degree)
        gap = max(abs(p - q) for p, q in zip(c1, c2, strict=True))
        largest = max(abs(c) for c in c2) or mpmath.mpf(1)
        tidy = [
            mpmath.mpf(mpmath.nstr(c, DIGITS, min_fixed=-mpmath.inf, max_fixed=mpmath.inf))
            if abs(c) > mpmath.mpf(10) ** -30 * largest
            else mpmath.mpf(0)
            for c in c2
        ]
        return tidy, gap, breaks


def numpy_coefficients(
    ref: Reference, degree: int, breaks: list[Any] | None = None
) -> list[float] | None:
    """The same coefficients by numpy's Gauss-Legendre rule in double precision, piece by piece."""
    points = [float(ref.a), *[float(b) for b in (breaks or [])], float(ref.b)]
    base, weights = npl.leggauss(max(100, 4 * (degree + 1)))
    xs: list[float] = []
    ws: list[float] = []
    for lo, hi in itertools.pairwise(points):
        xs += [(hi - lo) / 2 * t + (hi + lo) / 2 for t in base]
        ws += [(hi - lo) / 2 * w for w in weights]
    try:
        values = np.array([float(mpmath.re(ref.f(mpmath.mpf(x)))) for x in xs])
    except (ValueError, TypeError, ArithmeticError, OverflowError):
        return None
    if not np.all(np.isfinite(values)):
        return None
    ts = (2 * np.array(xs) - (float(ref.a) + float(ref.b))) / (float(ref.b) - float(ref.a))
    scale = (float(ref.b) - float(ref.a)) / 2
    out = []
    for n in range(degree + 1):
        pn = npl.legval(ts, [0] * n + [1])
        out.append((2 * n + 1) / (2 * scale) * float(np.sum(np.array(ws) * values * pn)))
    return out


def evaluator(ref: Reference, coeffs: list[Any]) -> Any:
    def at(x: Any) -> Any:
        t = ref.t(x)
        p = _legendre_values(t, len(coeffs) - 1)
        return sum((c * v for c, v in zip(coeffs, p, strict=True)), mpmath.mpf(0))

    return at


def build(
    ctx: Context, ref: Reference, degree: int, tolerance: Any | None, kind: str | None
) -> tuple[ApproximationReport, Measured | None]:
    coeffs, gap, breaks = coefficients(ctx, ref, degree)
    largest = max(abs(c) for c in coeffs) or mpmath.mpf(1)
    approx = evaluator(ref, coeffs)
    measured = measure(ref, approx)
    theirs = numpy_coefficients(ref, degree, breaks)
    if theirs is None:
        numpy_check = Check(
            name="coefficients against numpy leggauss",
            passed=None,
            detail="not run: outside double precision",
        )
    else:
        diff = max(abs(float(c) - t) for c, t in zip(coeffs, theirs, strict=True))
        numpy_check = Check(
            name="coefficients against numpy leggauss (independent nodes, double precision)",
            passed=bool(diff <= NUMPY_AGREE * float(largest)),
            points=degree + 1,
            detail=f"largest difference {diff:.2e}",
        )
    checks = [
        Check(
            name="coefficients: two quadrature rules agree",
            passed=bool(gap <= AGREE * largest),
            points=degree + 1,
            detail=f"largest difference {mpmath.nstr(gap, 3)} (panel count doubled)",
        ),
        numpy_check,
        Check(
            name="truncation error measured on two disjoint grids",
            passed=not measured.unstable,
            points=measured.summary.samples,
        ),
    ]
    mapping = rf"t = \frac{{2{ctx.var.name} - ({mpmath.nstr(ref.a, 12)} + {mpmath.nstr(ref.b, 12)})}}{{{mpmath.nstr(ref.b, 12)} - ({mpmath.nstr(ref.a, 12)})}}"
    text = ""
    for n, c in enumerate(coeffs):
        if c == 0:
            continue
        piece = rf"{mpmath.nstr(abs(c), DIGITS)}\,P_{{{n}}}(t)"
        text = (
            (f"-{piece}" if c < 0 else piece)
            if not text
            else f"{text} {'-' if c < 0 else '+'} {piece}"
        )
    report = ApproximationReport(
        status=Status.OK,
        method="legendre",
        variable=ctx.var.name,
        domain_latex=ref.domain_latex,
        result_latex=text or "0",
        statement_latex=mapping,
        checks=checks,
        error=measured.summary,
        error_curve=measured.curve,
        complexity=Complexity(degree=degree, coefficients=sum(1 for c in coeffs if c != 0)),
        warnings=list(ref.warnings),
        notes=list(ref.notes),
    )
    for n, c in enumerate(coeffs):
        if c != 0:
            report.orders.append(
                OrderTerm(
                    order=n,
                    coefficient_latex=mpmath.nstr(c, DIGITS),
                    term_latex=rf"{mpmath.nstr(c, DIGITS)}\,P_{{{n}}}(t)",
                )
            )
    if any(c.passed is False for c in checks):
        report.status = Status.NOT_VERIFIED
        report.reason = "verification_failed"
        report.result_latex = None
        report.message = (
            "A Legendre expansion was computed but its checks did not agree, so it is not shown."
        )
        return report, measured
    report.tolerance_met = tolerance_met(measured, tolerance, kind)
    report.evidence = "numeric_estimate"
    report.message = (
        f"Legendre expansion of degree {degree}: the least-squares polynomial on the interval. The largest "
        f"error on {measured.summary.samples} sampled points is {measured.summary.max_abs} "
        "(a sampled estimate, not a bound)."
    )
    return report, measured


def legendre(ctx: Context) -> ApproximationReport:
    ref = build_reference(ctx)
    tolerance, kind = parse_tolerance(ctx)
    data = ctx.data
    if data.degree is not None:
        if not 0 <= data.degree <= MAX_DEGREE:
            raise Refused(
                Status.UNSUPPORTED,
                "order_out_of_range",
                f"Give a degree between 0 and {MAX_DEGREE}.",
            )
        report, _ = build(ctx, ref, data.degree, tolerance, kind)
        return finish(report, evidence=report.evidence)
    if tolerance is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give a degree, or an accuracy target.",
            ["Which degree, or which accuracy?"],
        )
    cap = min(MAX_DEGREE, (data.max_terms - 1) if data.max_terms else MAX_DEGREE)
    history = []
    last: ApproximationReport | None = None
    for degree in (d for d in LADDER if d <= cap):
        if ctx.expired():
            break
        report, measured = build(ctx, ref, degree, tolerance, kind)
        history.append(candidate_of(report, measured, f"degree {degree}"))
        last = report
        if report.status is Status.OK and report.tolerance_met:
            history[-1].chosen = True
            report.candidates = history
            report.message += (
                " It is the lowest degree tried that meets the tolerance on the sampled points."
            )
            return report
    if last is None:
        raise Refused(
            Status.RESOURCE_LIMIT,
            "resource_limit",
            "The time budget ended before any degree was tried.",
        )
    last.candidates = history
    last.tolerance_met = False
    last.message += f" The tolerance was NOT met by any degree up to {history[-1].complexity.degree} (on the sampled points)."
    return last
