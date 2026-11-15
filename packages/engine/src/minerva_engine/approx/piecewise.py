"""Piecewise polynomial approximation on an interval: linear, PCHIP or cubic spline (SciPy).

Knots carry exact function values; the number of segments is the user's, or is found by a bounded
adaptive subdivision that splits the segments whose midpoint error exceeds the tolerance. Outside
the interval the approximation is undefined (no extrapolation). The error is measured on two
disjoint grids, which share no point with the knots.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from ..models import Status
from .common import Context, Refused, finish
from .interval import Measured, Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Check, Complexity, OrderTerm

KINDS = ("linear", "pchip", "cubic")
DEFAULT_KIND = "pchip"
MAX_SEGMENTS = 512
START_SEGMENTS = 4
SHOWN_KNOTS = 60


def _interpolant(kind: str, xs: np.ndarray, ys: np.ndarray) -> Any:
    if kind == "linear":
        return lambda t: np.interp(t, xs, ys, left=np.nan, right=np.nan)
    if kind == "pchip":
        return PchipInterpolator(xs, ys, extrapolate=False)
    return CubicSpline(xs, ys, bc_type="not-a-knot", extrapolate=False)


def _wrap(interpolant: Any) -> Any:
    def at(x: Any) -> Any:
        value = float(interpolant(float(x)))
        return None if np.isnan(value) else mpmath.mpf(value)

    return at


def _knots(ref: Reference, count: int) -> tuple[np.ndarray, np.ndarray]:
    xs = [ref.a + (ref.b - ref.a) * i / count for i in range(count + 1)]
    return np.array([float(x) for x in xs]), np.array([float(mpmath.re(ref.f(x))) for x in xs])


def _refined(ref: Reference, knots: list[Any], kind: str, tolerance: Any, cap: int) -> list[Any]:
    """Split the segments whose midpoint (and quarter points) differ from f by more than the tolerance."""
    while len(knots) - 1 < cap:
        xs = np.array([float(k) for k in knots])
        ys = np.array([float(mpmath.re(ref.f(k))) for k in knots])
        fn = _interpolant(kind, xs, ys)
        new = []
        for lo, hi in itertools.pairwise(knots):
            worst = mpmath.mpf(0)
            for frac in (mpmath.mpf(1) / 4, mpmath.mpf(1) / 2, mpmath.mpf(3) / 4):
                x = lo + (hi - lo) * frac
                worst = max(worst, abs(mpmath.re(ref.f(x)) - mpmath.mpf(float(fn(float(x))))))
            if worst > tolerance:
                new.append((lo + hi) / 2)
        if not new:
            return knots
        knots = sorted([*knots, *new[: cap - (len(knots) - 1)]], key=float)
    return knots


def build_piecewise(
    ctx: Context,
    ref: Reference,
    knots: list[Any],
    kind: str,
    tolerance: Any | None,
    tol_kind: str | None,
    notes: list[str],
) -> tuple[ApproximationReport, Measured]:
    xs = np.array([float(k) for k in knots])
    ys = np.array([float(mpmath.re(ref.f(k))) for k in knots])
    fn = _interpolant(kind, xs, ys)
    approx = _wrap(fn)
    measured: Measured = measure(ref, approx)
    segments = len(knots) - 1
    at_knots = max(abs(float(fn(float(k))) - float(mpmath.re(ref.f(k)))) for k in knots)
    checks = [
        Check(
            name="reproduces the function at the knots",
            passed=bool(at_knots <= 1e-12 * (1 + float(ref.scale))),
            points=len(knots),
            detail=f"largest difference {at_knots:.2e}",
        ),
        Check(
            name="error measured on two disjoint grids (none is a knot)",
            passed=not measured.unstable,
            points=measured.summary.samples,
        ),
    ]
    coefficients = {"linear": 2, "pchip": 4, "cubic": 4}[kind] * segments
    report = ApproximationReport(
        status=Status.OK,
        method="piecewise",
        variable=ctx.var.name,
        domain_latex=ref.domain_latex,
        statement_latex=rf"\text{{{kind} through {segments + 1} knots on }}{ref.domain_latex}",
        checks=checks,
        error=measured.summary,
        error_curve=measured.curve,
        complexity=Complexity(
            segments=segments, coefficients=coefficients, degree=1 if kind == "linear" else 3
        ),
        warnings=list(ref.warnings),
        notes=[*notes, *ref.notes],
    )
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        if i >= SHOWN_KNOTS:
            report.notes.append(f"Only the first {SHOWN_KNOTS} of {len(knots)} knots are listed.")
            break
        report.orders.append(
            OrderTerm(
                order=i,
                coefficient_latex=f"{y:.12g}",
                term_latex=rf"x = {x:.12g},\; f = {y:.12g}",
            )
        )
    report.warnings.append("no_extrapolation")
    report.notes.append(
        "Outside the interval the approximation is undefined: it never extrapolates."
    )
    if not all(c.passed for c in checks):
        report.status = Status.NOT_VERIFIED
        report.reason = "verification_failed"
        report.message = (
            "A piecewise approximation was built but its checks did not agree, so it is not shown."
        )
        return report, measured
    report.tolerance_met = tolerance_met(measured, tolerance, tol_kind)
    report.evidence = "numeric_estimate"
    if kind != "linear" and tolerance is not None and tolerance < 1e-13:
        report.notes.append(
            "SciPy evaluates in double precision, so errors below about 1e-15 cannot be resolved."
        )
    report.message = (
        f"{kind.capitalize()} approximation with {segments} segments. The largest error on "
        f"{measured.summary.samples} sampled points is "
        f"{measured.summary.max_abs} (a sampled estimate, not a bound)."
    )
    return report, measured


def uniform_knots(ref: Reference, count: int) -> list[Any]:
    return [ref.a + (ref.b - ref.a) * i / count for i in range(count + 1)]


def piecewise(ctx: Context) -> ApproximationReport:
    ref = build_reference(ctx)
    data = ctx.data
    kind = data.kind or DEFAULT_KIND
    if kind not in KINDS:
        raise Refused(Status.UNSUPPORTED, "unsupported_input", f"Kinds: {', '.join(KINDS)}.")
    tolerance, tol_kind = parse_tolerance(ctx)
    notes = (
        []
        if data.kind
        else [f"No kind was given, so {DEFAULT_KIND} (shape-preserving cubic Hermite) was used."]
    )
    if data.segments is not None:
        knots = uniform_knots(ref, data.segments)
    elif tolerance is not None and tol_kind == "absolute":
        cap = min(MAX_SEGMENTS, data.max_terms or MAX_SEGMENTS)
        knots = _refined(ref, uniform_knots(ref, START_SEGMENTS), kind, tolerance, cap)
        notes.append(
            f"Segments were split where the midpoint error exceeded the tolerance (at most {cap})."
        )
    else:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the number of segments, or an absolute accuracy target for adaptive subdivision.",
            ["How many segments, or which absolute accuracy?"],
        )
    report, _ = build_piecewise(ctx, ref, knots, kind, tolerance, tol_kind, notes)
    return finish(report, evidence=report.evidence)
