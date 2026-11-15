"""Expansions that compose other operations: expand an integral, a root, or an inverse function.

integral_expansion  expand the integrand in a small parameter, integrate term by term with the verified
                    integrator, and check the result against direct quadrature of the ORIGINAL integral.
root_expansion      the root of F(x, eps) = 0 that is known at eps = 0, order by order; checked against
                    the residual of F and against a separately computed root.
reversion           the inverse of y = f(x) near a point where f' is not zero (a root expansion of
                    f(x) - y = 0).
"""

from __future__ import annotations

from typing import Any

import mpmath
import sympy as sp

from ..integrate import integrate
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..series import series
from .checks import falls_at_rate, local_residual_check, parameter_sets
from .common import (
    Context,
    Refused,
    checks_from_series,
    finish,
    join_terms,
    parse_text,
    remainder,
    require_order,
)
from .expand import laurent_coefficients
from .models import ApproximationReport, Check, Complexity, Labeled, OrderTerm

MAX_ORDER = 8
DPS = 50
EPS_START = mpmath.mpf(2) ** -6  # integral expansion: eps = 2^-6, 2^-7, 2^-8, 2^-9


def _small_symbol(ctx: Context) -> sp.Symbol:
    if not ctx.data.small:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            "Name the small parameter (a single symbol).",
            ["Which symbol is the small parameter?"],
        )
    value = parse_text(ctx.data.small, ctx.data, ctx.limits)
    if not isinstance(value, sp.Symbol):
        raise Refused(
            Status.UNSUPPORTED,
            "small_parameter_not_isolable",
            "The small parameter must be a single symbol here.",
        )
    return value


def _numeric_bound(ctx: Context, text: str | None, which: str) -> sp.Expr:
    if text is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            f"The {which} limit of the integral is yours to choose.",
            ["What are the limits of the integral?"],
        )
    value = parse_text(text, ctx.data, ctx.limits)
    if value.free_symbols or value.is_finite is not True or value.is_real is not True:
        raise Refused(
            Status.UNSUPPORTED, "point_invalid", "The limits must be finite real numbers."
        )
    return value


# --- integral expansion ------------------------------------------------------------------------
def _quad(fn: Any, lo: Any, hi: Any) -> tuple[Any, Any]:
    value, err = mpmath.quad(fn, mpmath.linspace(lo, hi, 5), error=True, maxdegree=8)
    return value, err


def integral_expansion(ctx: Context) -> ApproximationReport:
    n = require_order(ctx.data, 1, MAX_ORDER)
    eps = _small_symbol(ctx)
    lo = _numeric_bound(ctx, ctx.data.lower, "lower")
    hi = _numeric_bound(ctx, ctx.data.upper, "upper")
    x = ctx.var
    g = ctx.expr
    if eps not in g.free_symbols or x not in g.free_symbols:
        raise Refused(
            Status.UNSUPPORTED,
            "variable_missing",
            "The integrand must depend on both the variable and the small parameter.",
        )
    taylor = series(
        ExpressionInput(input=to_text(g), format="text", assumptions=ctx.data.assumptions),
        eps.name,
        "0",
        n,
        ctx.limits,
    )
    if taylor.status is not Status.OK or not taylor.result_text:
        raise Refused(
            taylor.status,
            "verification_failed" if taylor.status is Status.NOT_VERIFIED else "unsupported_input",
            "The integrand has no verified Taylor expansion in the small parameter: "
            + taylor.message,
        )
    series_expr = parse_text(taylor.result_text, ctx.data, ctx.limits)
    coeffs = laurent_coefficients(series_expr, eps, sp.Integer(0))
    if coeffs is None or any(k < 0 for k in coeffs):
        raise Refused(
            Status.UNSUPPORTED,
            "not_analytic",
            "The expansion in the small parameter is not a power series.",
        )
    integrals: dict[int, sp.Expr] = {}
    inner = Limits(
        **{
            **ctx.limits.model_dump(),
            "timeout_seconds": max(1.0, ctx.limits.timeout_seconds * 0.25),
        }
    )
    for k, c in sorted(coeffs.items()):
        if ctx.expired():
            raise Refused(
                Status.RESOURCE_LIMIT,
                "resource_limit",
                "The time budget ended while integrating the terms.",
            )
        piece = integrate(
            ExpressionInput(input=to_text(c), format="text", assumptions=ctx.data.assumptions),
            x.name,
            inner,
            lower=str(lo),
            upper=str(hi),
        )
        if piece.status is not Status.OK or not piece.result_text:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                f"The integral of the order-{k} term could not be evaluated (it may diverge, or have no verified "
                "closed form); the expansion under the integral may not be uniform on the interval.",
            )
        integrals[k] = sp.simplify(parse_text(piece.result_text, ctx.data, ctx.limits))
    total = sum((v * eps**k for k, v in integrals.items()), sp.Integer(0))
    # decisive check: the ORIGINAL integral by quadrature, at eps = 2^-6 ... 2^-9
    decided = 0
    verdict: bool | None = None
    detail = "no parameter set could be evaluated"
    with mpmath.workdps(DPS):
        for values in (
            parameter_sets(g.subs(eps, 1), x, 3, seed=20260907)
            if g.free_symbols - {x, eps}
            else [{}]
        ):
            residuals: list[Any] = []
            fn_original = sp.lambdify((x, eps), g.subs(values), modules="mpmath")
            approx_fn = sp.lambdify(eps, total.subs(values), modules="mpmath")
            for j in range(4):
                e = EPS_START / 2**j
                value, err = _quad(
                    lambda t, e=e, fo=fn_original: fo(t, e),
                    mpmath.mpf(str(sp.N(lo, DPS))),
                    mpmath.mpf(str(sp.N(hi, DPS))),
                )
                if err > mpmath.mpf(10) ** -30 * (1 + abs(value)):
                    residuals = []
                    break
                residuals.append(value - mpmath.mpmathify(approx_fn(e)))
            if len(residuals) < 4:
                continue
            res = falls_at_rate(residuals, n + 1, exact_below=mpmath.mpf(10) ** -35)
            if res is False:
                verdict, detail = (
                    False,
                    "the error of the truncated expansion does not fall at the claimed order",
                )
                break
            if res:
                decided += 1
                verdict, detail = (
                    True,
                    f"error falls like eps^{n + 1} when eps is halved from 2^-6, against quadrature of the original integral",
                )
    if verdict is not True:
        raise Refused(
            Status.NOT_VERIFIED if verdict is False else Status.UNSUPPORTED,
            "verification_failed",
            "The term-by-term result did not match direct quadrature of the original integral at the claimed order "
            "(the expansion under the integral is probably not uniform), so it is not shown.",
        )
    report = ApproximationReport(
        status=Status.OK,
        method="integral_expansion",
        input_latex=rf"\int_{{{to_latex(lo)}}}^{{{to_latex(hi)}}} {to_latex(g)}\,d{to_latex(x)}",
        variable=eps.name,
        domain_latex=rf"{to_latex(eps)} \to 0",
        result_latex=join_terms([v * eps**k for k, v in sorted(integrals.items()) if v != 0]),
        result_text=str(total),
        remainder_latex=remainder(to_latex(eps), n + 1),
        checks=[
            Check(
                name="against quadrature of the original integral",
                passed=True,
                points=decided,
                detail=detail,
            ),
            *(checks_from_series(taylor.verification) if taylor.verification else []),
        ],
        complexity=Complexity(degree=n, coefficients=len(integrals)),
        message=(
            "The integrand was expanded in the small parameter, each coefficient integrated with the verified "
            "integrator, and the sum compared with quadrature of the original integral."
        ),
    )
    for k, v in sorted(integrals.items()):
        report.orders.append(
            OrderTerm(order=k, coefficient_latex=to_latex(v), term_latex=to_latex(v * eps**k))
        )
    report.notes.append(
        "Integrating term by term is justified only when the expansion is uniform on the interval; the "
        "comparison with the original integral is what was checked, at the sampled values of the parameter."
    )
    return finish(report, evidence="numeric_estimate", warnings=["local_only"])


# --- root and inverse-function expansions -----------------------------------------------------
def solve_implicit(
    F: sp.Expr, x: sp.Symbol, eps: sp.Symbol, x0: sp.Expr, order: int
) -> list[sp.Expr]:
    """Coefficients c_0 = x0, c_1, ..., c_order of the root x(eps) of F(x, eps) = 0 with x(0) = x0."""
    if sp.simplify(F.subs({x: x0, eps: 0})) != 0:
        raise Refused(
            Status.UNSUPPORTED,
            "point_invalid",
            "The given value is not a root of the equation at eps = 0.",
        )
    fx = sp.simplify(sp.diff(F, x).subs({x: x0, eps: 0}))
    if fx == 0:
        raise Refused(
            Status.UNSUPPORTED,
            "degenerate_system",
            "The root is not simple (dF/dx = 0 there): the correction is not a power series in eps, and no "
            "rescaling is attempted.",
        )
    coeffs = [x0]
    for k in range(1, order + 1):
        c = sp.Dummy("c")
        trial = sum((ci * eps**i for i, ci in enumerate(coeffs)), sp.Integer(0)) + c * eps**k
        residual = sp.series(F.subs(x, trial), eps, 0, k + 1).removeO()
        equation = sp.expand(residual).coeff(eps, k)
        solved = sp.solve(equation, c)
        if len(solved) != 1:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                "The coefficients could not be solved order by order.",
            )
        coeffs.append(sp.simplify(solved[0]))
    return coeffs


def _root_report(
    ctx: Context,
    F: sp.Expr,
    x: sp.Symbol,
    eps: sp.Symbol,
    x0: sp.Expr,
    n: int,
    method: str,
    label: sp.Expr,
    eps_to_y: sp.Expr,
) -> ApproximationReport:
    coeffs = solve_implicit(F, x, eps, x0, n)
    approx = sum((c * eps**i for i, c in enumerate(coeffs)), sp.Integer(0))
    residual_expr = F.subs(x, approx)
    claimed = n + 1
    ok1, pts1, det1 = local_residual_check(
        residual_expr, sp.Integer(0), eps, sp.Integer(0), claimed
    )
    # a separately computed root: Newton from the series value, at eps = 1e-4 * 2^-j
    ok2: bool | None = None
    det2 = "no parameter set could be evaluated"
    used = 0
    with mpmath.workdps(12 * claimed + 60):
        for values in (
            parameter_sets(F, x, 3, seed=20260908) if (F.free_symbols - {x, eps}) else [{}]
        ):
            Fn = sp.lambdify((x, eps), F.subs(values), modules="mpmath")
            sn = sp.lambdify(eps, approx.subs(values), modules="mpmath")
            diffs: list[Any] = []
            for j in range(5):
                e = mpmath.mpf(10) ** -4 / 2**j
                try:
                    root = mpmath.findroot(
                        lambda t, e=e, Fm=Fn: Fm(t, e),
                        sn(e),
                        tol=mpmath.mpf(10) ** -(12 * claimed + 40),
                        maxsteps=100,
                    )
                except (ValueError, ZeroDivisionError, mpmath.libmp.NoConvergence):
                    diffs = []
                    break
                diffs.append(root - sn(e))
            if len(diffs) < 5:
                continue
            verdict = falls_at_rate(
                diffs, claimed, exact_below=mpmath.mpf(10) ** -(12 * claimed + 20)
            )
            if verdict is False:
                ok2, det2 = (
                    False,
                    "the series does not approach the computed root at the claimed order",
                )
                break
            if verdict:
                used += 1
                ok2, det2 = True, f"the difference from a Newton root falls like eps^{claimed}"
    checks = [
        Check(
            name="residual of the equation falls like eps^(n+1)",
            passed=ok1,
            points=pts1,
            detail=det1,
        ),
        Check(name="against a separately computed root", passed=ok2, points=used, detail=det2),
    ]
    if ok1 is not True or ok2 is False:
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            "The expansion was derived but its checks did not agree, so it is not shown.",
        )
    report = ApproximationReport(
        status=Status.OK,
        method=method,
        input_latex=to_latex(ctx.expr),
        variable=x.name,
        checks=checks,
        complexity=Complexity(degree=n, coefficients=n + 1),
    )
    shown = sum((c * label**i for i, c in enumerate(coeffs)), sp.Integer(0))
    report.result_latex = join_terms([c * label**i for i, c in enumerate(coeffs) if c != 0])
    report.result_text = str(sum((c * eps**i for i, c in enumerate(coeffs)), sp.Integer(0)))
    for i, c in enumerate(coeffs):
        report.orders.append(
            OrderTerm(order=i, coefficient_latex=to_latex(c), term_latex=to_latex(c * label**i))
        )
    del shown, eps_to_y
    return report


def root_expansion(ctx: Context) -> ApproximationReport:
    n = require_order(ctx.data, 1, MAX_ORDER)
    eps = _small_symbol(ctx)
    if ctx.data.point is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The root at eps = 0 that should be followed is yours to give.",
            ["Which root at eps = 0?"],
        )
    x0 = parse_text(ctx.data.point, ctx.data, ctx.limits)
    x = ctx.var
    if x not in ctx.expr.free_symbols or eps not in ctx.expr.free_symbols:
        raise Refused(
            Status.UNSUPPORTED,
            "variable_missing",
            "The equation must contain both the unknown and the small parameter.",
        )
    report = _root_report(ctx, ctx.expr, x, eps, x0, n, "root_expansion", eps, eps)
    report.domain_latex = rf"{to_latex(eps)} \to 0"
    report.remainder_latex = remainder(to_latex(eps), n + 1)
    report.message = (
        f"The root of F = 0 that equals {to_latex(x0)} at eps = 0, as a series in {to_latex(eps)}; checked "
        "against the residual of the equation and a separately computed root."
    )
    report.notes.append(
        "Only this root is followed; other roots of the equation are not described by it."
    )
    return finish(report, evidence="series_checked", warnings=["local_only"])


def reversion(ctx: Context) -> ApproximationReport:
    n = require_order(ctx.data, 1, MAX_ORDER)
    if ctx.data.point is None or not ctx.data.small:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the point x0 and the name of the new variable y (the inverse is x as a function of y).",
            ["About which x0, and what should the new variable be called?"],
        )
    x = ctx.var
    x0 = parse_text(ctx.data.point, ctx.data, ctx.limits)
    y = sp.Symbol(ctx.data.small.strip())
    if y in ctx.expr.free_symbols or y == x:
        raise Refused(
            Status.PARSE_ERROR, "variable_missing", "Choose a new name for the inverse variable."
        )
    eps = sp.Symbol("eps_y", real=True)
    y0 = sp.simplify(ctx.expr.subs(x, x0))
    F = ctx.expr - y0 - eps
    if sp.simplify(sp.diff(ctx.expr, x).subs(x, x0)) == 0:
        raise Refused(
            Status.UNSUPPORTED,
            "degenerate_system",
            "f'(x0) = 0: the function is not locally invertible there (no power-series inverse).",
        )
    dist = sp.Symbol(rf"\left({to_latex(y - y0)}\right)" if y0 != 0 else to_latex(y))
    report = _root_report(ctx, F, x, eps, x0, n, "reversion", dist, eps)
    report.domain_latex = rf"{to_latex(y)} \to {to_latex(y0)}"
    report.remainder_latex = remainder(to_latex(dist), n + 1)
    report.message = (
        f"The inverse of y = f(x) near x = {to_latex(x0)}, where y = {to_latex(y0)}, as a series in "
        f"{to_latex(y - y0)}; checked by substituting it back and against a separately computed inverse."
    )
    report.extras.append(Labeled(label="Value of y at the point", latex=to_latex(y0)))
    report.notes.append(
        "A local inverse: it describes the branch of the inverse through this point only."
    )
    return finish(report, evidence="series_checked", warnings=["local_only"])
