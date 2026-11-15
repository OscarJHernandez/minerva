"""Laplace's method for an integral g(x) exp(M f(x)) over [lower, upper] with M large.

Only an interior maximum of f is supported (one, nondegenerate, strictly the global maximum on the
interval). The leading term and corrections come from expanding f and g about the maximum and
integrating the Gaussian moments exactly. The result is shown only if the ratio to quadrature of the
original integral approaches 1 at the claimed rate when M is doubled. Endpoint maxima, several equal
maxima and stationary phase are refused.
"""

from __future__ import annotations

from typing import Any

import mpmath
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .checks import falls_at_rate
from .common import Context, Refused, finish, parse_text, remainder, require_order
from .models import ApproximationReport, Check, Complexity, Labeled, OrderTerm

MAX_ORDER = 3
DPS = 40
M_START = 5000
GRID = 2001
MARGIN = mpmath.mpf("1e-9")


def split_integrand(expr: sp.Expr, M: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    """expr = g * exp(M f): returns (g, f), both free of M, or raises."""
    exponent = sp.Integer(0)
    rest = sp.Integer(1)
    for factor in sp.Mul.make_args(expr):
        if isinstance(factor, sp.exp) and M in factor.free_symbols:
            exponent += factor.args[0]
        elif factor.is_Pow and isinstance(factor.base, sp.exp) and M in factor.free_symbols:
            exponent += factor.base.args[0] * factor.exp
        elif M in factor.free_symbols:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                "The integrand must be g(x) times exp(M f(x)), with g free of M.",
            )
        else:
            rest *= factor
    if exponent == 0:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "The integrand must contain a factor exp(M f(x)).",
        )
    f = sp.simplify(exponent / M)
    if M in f.free_symbols:
        raise Refused(Status.UNSUPPORTED, "unsupported_input", "The exponent must be linear in M.")
    return rest, f


def maximum(f: sp.Expr, x: sp.Symbol, lo: Any, hi: Any) -> sp.Expr:
    """The one interior, nondegenerate global maximum of f on (lo, hi), else a refusal."""
    critical = sp.solveset(sp.diff(f, x), x, sp.Interval.open(sp.nsimplify(lo), sp.nsimplify(hi)))
    if not isinstance(critical, sp.FiniteSet) or len(critical) != 1:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "Laplace's method here needs exactly one stationary point of f inside the interval; "
            "endpoint maxima and several maxima are not supported.",
        )
    star = next(iter(critical))
    with mpmath.workdps(DPS):
        second = mpmath.mpmathify(sp.N(sp.diff(f, x, 2).subs(x, star), DPS))
        if not second < 0:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                "The stationary point is not a nondegenerate maximum (f'' must be negative there).",
            )
        fn = sp.lambdify(x, f, modules="mpmath")
        top = mpmath.mpmathify(sp.N(f.subs(x, star), DPS))
        a, b = mpmath.mpf(str(lo)), mpmath.mpf(str(hi))
        s0 = mpmath.mpmathify(sp.N(star, DPS))
        for i in range(GRID):
            t = a + (b - a) * i / (GRID - 1)
            if abs(t - s0) < (b - a) / 40:
                continue
            v = fn(t)
            if v is None or not mpmath.isfinite(v) or mpmath.re(v) >= top - MARGIN:
                raise Refused(
                    Status.UNSUPPORTED,
                    "unsupported_input",
                    "The stationary point is not the strict global maximum of f on the interval "
                    "(another point is as high, or the endpoint is), which this method does not cover.",
                )
    return star


def corrections(
    g: sp.Expr, f: sp.Expr, x: sp.Symbol, star: sp.Expr, order: int
) -> tuple[list[sp.Expr], sp.Expr]:
    """a_0..a_order with I ~ exp(M f0) sqrt(2 pi / (M A)) sum a_k M^-k, and A = -f''(x*)."""
    u, t = sp.symbols("u t", positive=True)
    A = sp.simplify(-sp.diff(f, x, 2).subs(x, star))
    s = u * t / sp.sqrt(A)
    top = 2 * order + 2  # f is needed through s^top, g through s^(2 order)
    remainder_f = sum(
        (sp.diff(f, x, j).subs(x, star) / sp.factorial(j) * s**j for j in range(3, top + 1)),
        sp.Integer(0),
    )
    G = sum(
        (sp.diff(g, x, j).subs(x, star) / sp.factorial(j) * s**j for j in range(0, 2 * order + 1)),
        sp.Integer(0),
    )
    exponent = sp.expand(remainder_f / u**2)  # M * R(s), with M = 1 / u^2
    expansion = sp.series(sp.exp(exponent), u, 0, 2 * order + 1).removeO()
    poly = sp.expand(sp.series(sp.expand(G * expansion), u, 0, 2 * order + 1).removeO())
    moments = {j: sp.factorial2(j - 1) if j else sp.Integer(1) for j in range(0, 4 * order + 8, 2)}
    out: list[sp.Expr] = []
    for k in range(order + 1):
        coefficient = sp.expand(poly.coeff(u, 2 * k))
        total = sp.Integer(0)
        for (power,), c in sp.Poly(coefficient, t).terms() if coefficient != 0 else []:
            total += c * (moments.get(int(power), sp.Integer(0)) if int(power) % 2 == 0 else 0)
        out.append(sp.simplify(total))
    return out, A


def laplace(ctx: Context) -> ApproximationReport:
    m = require_order(ctx.data, 0, MAX_ORDER) if ctx.data.order is not None else None
    if m is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The number of correction terms is yours to choose (0 for the leading term only).",
            ["How many corrections beyond the leading term?"],
        )
    if not ctx.data.small:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            "Name the large parameter M.",
            ["Which symbol is the large parameter?"],
        )
    M = parse_text(ctx.data.small, ctx.data, ctx.limits)
    if not isinstance(M, sp.Symbol):
        raise Refused(
            Status.UNSUPPORTED,
            "small_parameter_not_isolable",
            "The large parameter must be a single symbol.",
        )
    if ctx.data.lower is None or ctx.data.upper is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The limits of the integral are yours to choose.",
            ["What are the limits?"],
        )
    lo = parse_text(ctx.data.lower, ctx.data, ctx.limits)
    hi = parse_text(ctx.data.upper, ctx.data, ctx.limits)
    for v in (lo, hi):
        if v.free_symbols or v.is_finite is not True or v.is_real is not True:
            raise Refused(
                Status.UNSUPPORTED, "point_invalid", "The limits must be finite real numbers."
            )
    x = ctx.var
    if ctx.expr.free_symbols - {x, M}:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            "Give numerical values for the other symbols.",
            ["What values do the other symbols take?"],
        )
    g, f = split_integrand(ctx.expr, M)
    with mpmath.workdps(DPS):
        lo_f, hi_f = mpmath.mpf(str(sp.N(lo, DPS))), mpmath.mpf(str(sp.N(hi, DPS)))
    star = maximum(f, x, lo_f, hi_f)
    g0 = sp.simplify(g.subs(x, star))
    if g0 == 0:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "g vanishes at the maximum, so the leading term is of higher order; not supported.",
        )
    a, A = corrections(g, f, x, star, m)
    f0 = sp.simplify(f.subs(x, star))
    series_in_M = sum((c / M**k for k, c in enumerate(a)), sp.Integer(0))
    lead = sp.exp(M * f0) * sp.sqrt(2 * sp.pi / (M * A))
    approx = lead * series_in_M
    # check: the ratio to quadrature of the original integral approaches 1 at the rate M^-(m+1)
    with mpmath.workdps(DPS):
        gfn = sp.lambdify(x, g, modules="mpmath")
        ffn = sp.lambdify(x, f, modules="mpmath")
        f0n = mpmath.mpmathify(sp.N(f0, DPS))
        sn = mpmath.mpmathify(sp.N(star, DPS))
        An = mpmath.mpmathify(sp.N(A, DPS))
        coeffs_n = [mpmath.mpmathify(sp.N(c, DPS)) for c in a]
        ratios: list[Any] = []
        for j in range(4):
            Mj = M_START * 2**j
            width = 12 / mpmath.sqrt(Mj * An)
            pts = sorted({lo_f, max(lo_f, sn - width), sn, min(hi_f, sn + width), hi_f})
            value, err = mpmath.quad(
                lambda s, Mj=Mj: gfn(s) * mpmath.exp(Mj * (ffn(s) - f0n)),
                pts,
                error=True,
                maxdegree=8,
            )
            if err > mpmath.mpf(10) ** -25 * abs(value):
                ratios = []
                break
            predicted = mpmath.sqrt(2 * mpmath.pi / (Mj * An)) * sum(
                (c / Mj**k for k, c in enumerate(coeffs_n)), mpmath.mpf(0)
            )
            ratios.append(value / predicted - 1)
        verdict = (
            falls_at_rate(ratios, m + 1, exact_below=mpmath.mpf(10) ** -30)
            if len(ratios) == 4
            else None
        )
    if verdict is not True:
        raise Refused(
            Status.NOT_VERIFIED if verdict is False else Status.UNSUPPORTED,
            "verification_failed",
            "The asymptotic result did not match quadrature of the original integral at the claimed rate, so it is not shown.",
        )
    report = ApproximationReport(
        status=Status.OK,
        method="laplace",
        input_latex=rf"\int_{{{to_latex(lo)}}}^{{{to_latex(hi)}}} {to_latex(ctx.expr)}\,d{to_latex(x)}",
        variable=M.name,
        domain_latex=rf"{to_latex(M)} \to \infty",
        result_latex=to_latex(approx),
        result_text=str(approx),
        remainder_latex=rf"\text{{relative error }} {remainder(to_latex(M), -(m + 1))}",
        checks=[
            Check(
                name="ratio to quadrature of the original integral",
                passed=True,
                points=4,
                detail=f"the relative error falls like M^-{m + 1} when M is doubled from {M_START}",
            ),
        ],
        complexity=Complexity(coefficients=m + 1),
        message=(
            "Laplace's method about the interior maximum: the leading term and "
            f"{m} correction{'s' if m != 1 else ''}, compared with quadrature of the original integral."
        ),
    )
    report.extras.append(Labeled(label="Location of the maximum", latex=to_latex(star)))
    report.extras.append(Labeled(label="Leading term", latex=to_latex(lead * g0)))
    for k, c in enumerate(a):
        report.orders.append(
            OrderTerm(order=k, coefficient_latex=to_latex(c), term_latex=to_latex(c / M**k))
        )
    report.notes.append(
        "An asymptotic expansion in the large parameter M: it is not a convergent series, and only an "
        "interior, nondegenerate global maximum is covered."
    )
    return finish(report, evidence="numeric_estimate", warnings=["asymptotic_not_convergent"])
