"""Padé approximants [m/n] about a point, from exact Taylor coefficients.

SymPy gives the Taylor coefficients and the exact linear system for the denominator. The result is
cross-checked against `mpmath.pade` (a separate implementation of the same definition) and is shown
only if the residual f - P/Q falls like h^(m+n+1), measured directly on f at high precision. A Padé
approximant is never assumed to beat the Taylor polynomial of the same order: both are measured on
the interval when one is given.
"""

from __future__ import annotations

from typing import Any

import mpmath
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .checks import DPS, compile_mp, local_residual_check
from .common import (
    Context,
    Refused,
    finish,
    finite_real_point,
    join_terms,
    refuse_non_analytic,
    sympy_failed,
)
from .expand import laurent_coefficients
from .interval import Reference, build_reference, measure, parse_tolerance, tolerance_met
from .models import ApproximationReport, Candidate, Check, Complexity, ErrorSummary, Labeled

MAX_TOTAL = 24  # m + n
POLE_REGULAR = mpmath.mpf("1e-6")


def taylor_coefficients(expr: sp.Expr, var: sp.Symbol, x0: sp.Expr, order: int) -> list[sp.Expr]:
    refuse_non_analytic(expr, var)
    try:
        candidate = sp.series(expr, var, x0, order + 1).removeO()
    except (NotImplementedError, ValueError, TypeError, RecursionError, AttributeError) as exc:
        raise sympy_failed(exc) from exc
    if candidate.has(sp.Order, sp.zoo, sp.oo, sp.nan):
        raise Refused(
            Status.UNSUPPORTED, "not_analytic", "There is no Taylor series at this point."
        )
    coeffs = laurent_coefficients(candidate, var, x0)
    if coeffs is None or any(k < 0 for k in coeffs):
        raise Refused(
            Status.UNSUPPORTED,
            "not_analytic",
            "The function is not analytic at this point (a pole, branch point or logarithm), so a "
            "Padé approximant built from its Taylor coefficients is not defined.",
        )
    return [coeffs.get(k, sp.Integer(0)) for k in range(order + 1)]


def solve_pade(c: list[sp.Expr], m: int, n: int) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """Numerator and denominator coefficients (ascending in h), denominator constant term 1."""

    def coef(k: int) -> sp.Expr:
        return c[k] if 0 <= k < len(c) else sp.Integer(0)

    if n == 0:
        return [coef(k) for k in range(m + 1)], [sp.Integer(1)]
    matrix = sp.Matrix(n, n, lambda i, j: coef(m + 1 + i - (j + 1)))
    rhs = sp.Matrix(n, 1, lambda i, _: -coef(m + 1 + i))
    if sp.simplify(matrix.det()) == 0:
        raise Refused(
            Status.UNSUPPORTED,
            "degenerate_system",
            f"The [{m}/{n}] Padé system is singular (the Taylor coefficients make this "
            "approximant degenerate or non-unique); try other degrees.",
        )
    q = [sp.Integer(1), *[sp.simplify(v) for v in matrix.LUsolve(rhs)]]
    p = [
        sp.simplify(sum((q[j] * coef(k - j) for j in range(min(k, n) + 1)), sp.Integer(0)))
        for k in range(m + 1)
    ]
    return p, q


def _ascending(coeffs: list[sp.Expr], var: sp.Symbol, x0: sp.Expr) -> str:
    """A polynomial in (var - x0) written in ascending powers, term by term."""
    return join_terms(
        [c * (var - x0) ** k for k, c in enumerate(coeffs) if c != 0] or [sp.Integer(0)]
    )


def _poly(coeffs: list[sp.Expr], var: sp.Symbol, x0: sp.Expr) -> sp.Expr:
    return sum((c * (var - x0) ** k for k, c in enumerate(coeffs)), sp.Integer(0))


def _analyse_poles(
    ref_expr: sp.Expr,
    var: sp.Symbol,
    q_poly: sp.Expr,
    p_poly: sp.Expr,
    x0: sp.Expr,
    interval: tuple[Any, Any] | None,
) -> tuple[list[Labeled], list[str], list[str]]:
    """Poles of the approximant: where, whether f is regular there, whether one is in the interval."""
    extras: list[Labeled] = []
    warnings: list[str] = []
    notes: list[str] = []
    if (q_poly.free_symbols | p_poly.free_symbols) - {var}:
        notes.append(
            "The poles of the approximant depend on the symbolic parameters and were not located."
        )
        return extras, warnings, notes
    q = sp.Poly(sp.expand(q_poly), var)
    if q.degree() < 1:
        return extras, warnings, notes
    with mpmath.workdps(DPS):
        coeffs = [mpmath.mpmathify(sp.N(c, DPS)) for c in q.all_coeffs()]
        try:
            roots = mpmath.polyroots(coeffs, maxsteps=500, extraprec=200)
        except (mpmath.libmp.NoConvergence, ZeroDivisionError):
            notes.append("The poles of the approximant could not be located numerically.")
            return extras, warnings, notes
        f = compile_mp(ref_expr, var)
        zeros = []
        pn = sp.Poly(sp.expand(p_poly), var)
        if pn.degree() >= 1:
            try:
                zeros = mpmath.polyroots(
                    [mpmath.mpmathify(sp.N(c, DPS)) for c in pn.all_coeffs()],
                    maxsteps=500,
                    extraprec=200,
                )
            except (mpmath.libmp.NoConvergence, ZeroDivisionError):
                zeros = []
        for r in roots:
            where = mpmath.nstr(r, 6)
            value = f(r)
            near = f(r + POLE_REGULAR)
            regular = (
                value is not None
                and near is not None
                and abs(value) < 1e12
                and abs(near - value) < 1e3 * POLE_REGULAR * (1 + abs(value))
            )
            doublet = any(abs(r - z) < mpmath.mpf("1e-6") * (1 + abs(r)) for z in zeros)
            real_here = abs(mpmath.im(r)) < mpmath.mpf("1e-12") * (1 + abs(r))
            inside = bool(interval and real_here and interval[0] <= mpmath.re(r) <= interval[1])
            label = "pole of the approximant"
            if regular or doublet:
                label += ": the original function has no singularity there"
                if "spurious_pole" not in warnings:
                    warnings.append("spurious_pole")
                notes.append(
                    f"The approximant has a pole at {var.name} = {where}, although the original function does not."
                )
            if inside:
                label += " (inside the requested interval)"
                if "pole_in_interval" not in warnings:
                    warnings.append("pole_in_interval")
            extras.append(Labeled(label=label, latex=f"{to_latex(var)} = {where}"))
    return extras, warnings, notes


def _mp_agree(p: list[sp.Expr], q: list[sp.Expr], c: list[sp.Expr], m: int, n: int) -> Check:
    if any(s.free_symbols for s in [*p, *q]):
        return Check(
            name="coefficients against mpmath.pade",
            passed=None,
            detail="symbolic parameters: not run",
        )
    with mpmath.workdps(DPS):
        cm = [mpmath.mpmathify(sp.N(x, DPS)) for x in c]
        try:
            pm, qm = mpmath.pade(cm, m, n)
        except (ZeroDivisionError, ValueError, mpmath.libmp.NoConvergence):
            return Check(
                name="coefficients against mpmath.pade",
                passed=None,
                detail="mpmath.pade could not solve it",
            )
        ours = [mpmath.mpmathify(sp.N(x, DPS)) for x in [*p, *q]]
        theirs = [*pm, *qm]
        scale = max(abs(x) for x in ours) or 1
        gap = max(abs(a - b) for a, b in zip(ours, theirs, strict=False))
        return Check(
            name="coefficients against mpmath.pade",
            passed=bool(gap < mpmath.mpf(10) ** -30 * scale),
            points=len(ours),
            detail="an independent solution of the same linear system",
        )


def pade_core(
    ctx: Context, m: int, n: int, x0: sp.Expr, reference: Reference | None
) -> tuple[ApproximationReport, sp.Expr, list[sp.Expr]]:
    """The verified [m/n] approximant about x0 (raises Refused otherwise), its expression and the Taylor coefficients."""
    var = ctx.var
    if m + n > MAX_TOTAL:
        raise Refused(
            Status.UNSUPPORTED,
            "order_out_of_range",
            f"The total degree m + n is limited to {MAX_TOTAL}.",
        )
    c = taylor_coefficients(ctx.expr, var, x0, m + n)
    p, q = solve_pade(c, m, n)
    p_poly, q_poly = _poly(p, var, x0), _poly(q, var, x0)
    approx = sp.cancel(p_poly / q_poly)
    num, den = sp.fraction(approx)
    reduced_m = sp.Poly(num, var).degree() if num.has(var) else 0
    reduced_n = sp.Poly(den, var).degree() if den.has(var) else 0
    claimed = m + n + 1
    verdict, points, detail = local_residual_check(ctx.expr, approx, var, x0, claimed)
    checks = [
        Check(name="f - P/Q falls like h^(m+n+1)", passed=verdict, points=points, detail=detail),
        _mp_agree(p, q, c, m, n),
    ]
    if verdict is not True or any(k.passed is False for k in checks):
        raise Refused(
            Status.NOT_VERIFIED
            if verdict is False or any(k.passed is False for k in checks)
            else Status.UNSUPPORTED,
            "verification_failed",
            "A Padé approximant was computed but its contact with the function could not be "
            "confirmed, so it is not shown.",
        )
    interval = (reference.a, reference.b) if reference is not None else None
    extras, warnings, notes = _analyse_poles(ctx.expr, var, q_poly, p_poly, x0, interval)
    reduced = (reduced_m, reduced_n) != (m, n)
    if reduced:
        shown = to_latex(approx)
    else:
        shown = rf"\frac{{{_ascending(p, var, x0)}}}{{{_ascending(q, var, x0)}}}"
    report = ApproximationReport(
        status=Status.OK,
        method="pade",
        input_latex=to_latex(ctx.expr),
        variable=var.name,
        domain_latex=rf"{to_latex(var)} \to {to_latex(x0)}",
        result_latex=shown,
        result_text=str(approx),
        remainder_latex=rf"O\left(\left({to_latex(var - x0)}\right)^{{{claimed}}}\right)",
        checks=checks,
        extras=extras,
        notes=notes,
        complexity=Complexity(
            numerator_degree=reduced_m,
            denominator_degree=reduced_n,
            coefficients=reduced_m + reduced_n + 1,
        ),
        message=(
            f"The [{m}/{n}] Padé approximant about the point; it matches the Taylor coefficients through "
            f"degree {m + n} (checked on the function itself)."
        ),
    )
    if (reduced_m, reduced_n) != (m, n):
        report.notes.append(f"The fraction reduces to degrees [{reduced_m}/{reduced_n}].")
    return finish(report, evidence="series_checked", warnings=warnings), approx, c  # type: ignore[arg-type]


def pade(ctx: Context) -> ApproximationReport:
    data = ctx.data
    if data.degree is None or data.denominator_degree is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The degrees of the numerator and denominator are yours to choose.",
            ["Which numerator and denominator degrees [m/n]?"],
        )
    x0 = finite_real_point(ctx)
    reference = build_reference(ctx) if data.lower is not None and data.upper is not None else None
    report, approx, c = pade_core(ctx, data.degree, data.denominator_degree, x0, reference)
    if reference is not None:
        report.warnings += [w for w in reference.warnings if w not in report.warnings]
        _compare_with_taylor(
            report, ctx, reference, approx, c, x0, data.degree + data.denominator_degree
        )
    return report


def _compare_with_taylor(
    report: ApproximationReport,
    ctx: Context,
    ref: Reference,
    approx: sp.Expr,
    c: list[sp.Expr],
    x0: sp.Expr,
    total: int,
) -> None:
    var = ctx.var
    taylor_expr = _poly(c, var, x0)
    pade_fn = compile_mp(approx, var)
    taylor_fn = compile_mp(taylor_expr, var)
    results: list[tuple[str, Any, ErrorSummary, Complexity]] = []
    for label, fn, cx in (
        ("Padé", pade_fn, report.complexity),
        ("Taylor (same order)", taylor_fn, Complexity(degree=total, coefficients=total + 1)),
    ):
        measured = measure(ref, fn)
        met = tolerance_met(measured, *parse_tolerance(ctx))
        report.candidates.append(
            Candidate(
                method=label.split(" ")[0].lower(),
                label=label,
                status=Status.OK,
                complexity=cx,
                error=measured.summary,
                tolerance_met=met,
            )
        )
        results.append((label, measured.max_abs, measured.summary, cx))
        if label == "Padé":
            report.error = measured.summary
            report.error_curve = measured.curve
            report.tolerance_met = met
            if measured.undefined_at is not None and "pole_in_interval" not in report.warnings:
                report.warnings.append("pole_in_interval")
    better = results[0][1] < results[1][1]
    report.notes.append(
        "On the interval the largest sampled error is "
        f"{results[0][2].max_abs} for Padé and {results[1][2].max_abs} for the Taylor polynomial of the same "
        f"order: Padé is {'more' if better else 'not more'} accurate here (sampled, not a bound)."
    )
