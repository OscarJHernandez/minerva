"""Local symbolic expansions: Taylor, perturbation, Laurent and asymptotic series.

SymPy proposes every series. A series is shown only if `series.check_series` (Cauchy-integral
estimates of the coefficients and the decay of the remainder, neither of which uses `sympy.series`)
passes, or, for expansions with logarithms or fractional powers, if the residual falls at the claimed
rate along a doubling sequence of the large variable.
"""

from __future__ import annotations

import mpmath
import sympy as sp

from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Status
from ..series import SeriesVerification, check_series, series
from .checks import compile_mp, falls_at_rate, parameter_sets
from .common import (
    LOCAL_NOTE,
    Context,
    Refused,
    checks_from_series,
    distance_symbol,
    finish,
    finite_real_point,
    join_terms,
    parse_text,
    powers_latex,
    refuse_non_analytic,
    remainder,
    require_order,
    sympy_failed,
)
from .models import ApproximationReport, Check, Evidence, Labeled, OrderTerm

MAX_POLE = 8  # the highest pole order recognised
MAX_TOTAL = 24  # order + pole order handed to the Cauchy check


def laurent_coefficients(
    expr: sp.Expr, var: sp.Symbol, x0: sp.Expr, max_pole: int = MAX_POLE
) -> dict[int, sp.Expr] | None:
    """{k: c_k} for a finite sum of c_k (var - x0)^k with integer k, or None if it is not one."""
    h = sp.Dummy("h")
    shifted = sp.expand(expr.subs(var, x0 + h) * h**max_pole)
    try:
        poly = sp.Poly(shifted, h)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return None
    out: dict[int, sp.Expr] = {}
    for (k,), c in poly.terms():
        c = sp.simplify(c)
        if c != 0:
            out[int(k) - max_pole] = c
    return out


# --- Taylor -----------------------------------------------------------------------------------
def taylor(ctx: Context) -> ApproximationReport:
    order = require_order(ctx.data)
    finite_real_point(ctx)  # the same reading as the series operation, for a clear refusal
    result = series(
        ExpressionInput(
            input=ctx.data.input, format=ctx.data.format, assumptions=ctx.data.assumptions
        ),
        ctx.data.variable,
        str(ctx.data.point),
        order,
        ctx.limits,
    )
    report = ApproximationReport(
        status=result.status,
        method="taylor",
        message=result.message,
        input_latex=result.input_latex,
        variable=result.variable or ctx.data.variable,
        domain_latex=f"{result.variable} \\to {result.point_latex}" if result.point_latex else None,
    )
    if result.status is not Status.OK or result.verification is None:
        report.reason = (
            "verification_failed" if result.status is Status.NOT_VERIFIED else ("unsupported_input")
        )
        if result.verification is not None:
            report.checks = checks_from_series(result.verification)
        if "negative or fractional" in result.message or "pole" in result.message:
            report.notes.append("The laurent method handles poles; asymptotic handles infinity.")
        return report
    x0 = finite_real_point(ctx)
    report.result_latex = result.result_latex
    report.result_text = result.result_text
    report.remainder_latex = result.remainder_latex
    report.measured_errors = result.measured_errors
    report.checks = checks_from_series(result.verification)
    report.notes = [*result.notes, LOCAL_NOTE]
    report.complexity.degree = order
    report.complexity.coefficients = len(result.terms)
    dist = distance_symbol(ctx.var, x0)
    for term in result.terms:
        report.orders.append(
            OrderTerm(
                order=term.power,
                coefficient_latex=term.coefficient_latex,
                term_latex=term.coefficient_latex
                + ("" if term.power == 0 else rf"\,{to_latex(dist)}^{{{term.power}}}"),
            )
        )
    return finish(report, evidence="series_checked", warnings=["local_only"])


# --- Laurent ----------------------------------------------------------------------------------
def laurent_core(
    expr: sp.Expr, var: sp.Symbol, x0: sp.Expr, order: int
) -> tuple[dict[int, sp.Expr], int, SeriesVerification]:
    """Verified Laurent coefficients through (var - x0)^order, and the pole order m (0: regular)."""
    refuse_non_analytic(expr, var)
    try:
        candidate = sp.series(expr, var, x0, order + 1).removeO()
    except (NotImplementedError, ValueError, TypeError, RecursionError, AttributeError) as exc:
        raise sympy_failed(exc) from exc
    if candidate.has(sp.Order, sp.zoo, sp.oo, sp.nan):
        raise Refused(
            Status.UNSUPPORTED,
            "essential_singularity",
            "The expansion at this point is not a Laurent series.",
        )
    coeffs = laurent_coefficients(candidate, var, x0)
    if coeffs is None:
        raise Refused(
            Status.UNSUPPORTED,
            "not_laurent",
            "The expansion has logarithms or fractional powers here (a branch point or an "
            "essential singularity), so it is not a Laurent series.",
        )
    if not coeffs:
        raise Refused(
            Status.UNSUPPORTED, "unsupported_input", "The expression is identically zero here."
        )
    pole = max(0, -min(coeffs))
    if order + pole > MAX_TOTAL:
        raise Refused(
            Status.RESOURCE_LIMIT,
            "resource_limit",
            "The order plus the pole order is too large to check.",
        )
    shifted = sum((c * (var - x0) ** (k + pole) for k, c in coeffs.items()), sp.Integer(0))
    check = check_series(expr * (var - x0) ** pole, var, x0, shifted, order + pole)
    if not check.passed:
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            f"A Laurent series was produced but could not be verified, so it is not shown. {check.note}.",
        )
    return coeffs, pole, check


def laurent(ctx: Context) -> ApproximationReport:
    order = require_order(ctx.data)
    x0 = finite_real_point(ctx)
    coeffs, pole, check = laurent_core(ctx.expr, ctx.var, x0, order)
    dist = distance_symbol(ctx.var, x0)
    report = ApproximationReport(
        status=Status.OK,
        method="laurent",
        input_latex=to_latex(ctx.expr),
        variable=ctx.var.name,
        domain_latex=rf"{to_latex(ctx.var)} \to {to_latex(x0)}",
        result_latex=powers_latex(coeffs, dist),
        remainder_latex=remainder(to_latex(dist), order + 1),
        message=(
            "Expanded with SymPy; the coefficients (all the negative powers included) and the "
            "remainder order were checked by estimates that do not use sympy.series."
        ),
        checks=[
            Check(
                name=f"{c.name} (after multiplying by the pole factor)",
                passed=c.passed,
                points=c.points,
                detail=c.detail,
            )
            for c in checks_from_series(check)
        ],
    )
    report.complexity.coefficients = len(coeffs)
    for k, c in sorted(coeffs.items()):
        report.orders.append(
            OrderTerm(order=k, coefficient_latex=to_latex(c), term_latex=to_latex(c * dist**k))
        )
    if pole:
        principal = {k: c for k, c in coeffs.items() if k < 0}
        report.extras.append(Labeled(label="Pole order", latex=str(pole)))
        report.extras.append(Labeled(label="Principal part", latex=powers_latex(principal, dist)))
        report.extras.append(
            Labeled(label="Residue", latex=to_latex(coeffs.get(-1, sp.Integer(0))))
        )
        report.notes.append(
            f"A pole of order {pole}; the residue is the coefficient of 1/({to_latex(ctx.var - x0)})."
        )
    else:
        report.extras.append(Labeled(label="Pole order", latex="0"))
        report.extras.append(Labeled(label="Residue", latex="0"))
        report.notes.append(
            "There is no pole at this point: the Laurent series is a Taylor series."
        )
        value = ctx.expr.subs(ctx.var, x0)
        if value.has(sp.nan, sp.zoo, sp.oo, sp.I * sp.oo):
            report.warnings.append("removable_singularity")
            report.notes.append(
                "The expression is undefined at the point but the singularity is removable."
            )
    report.notes.append(LOCAL_NOTE)
    return finish(report, evidence="series_checked", warnings=["local_only"])


# --- Perturbation -----------------------------------------------------------------------------
def _fresh(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "q"
    return name


def _eliminate(small: sp.Expr, q: sp.Symbol) -> tuple[sp.Symbol, sp.Expr]:
    """Solve `small = q` for a symbol that goes to 0 with q, so that expanding in q is a Taylor series."""
    for s in sorted(small.free_symbols, key=lambda s: s.name):
        try:
            solutions = sp.solve(small - q, s)
        except (NotImplementedError, ValueError):
            continue
        if len(solutions) != 1:
            continue
        at_zero = solutions[0].subs(q, 0)
        if at_zero == 0:
            return s, solutions[0]
    raise Refused(
        Status.UNSUPPORTED,
        "small_parameter_not_isolable",
        "The small parameter must be a single symbol, or a combination (such as v/c) that is zero "
        "when one of its symbols is zero.",
    )


def perturbation(ctx: Context) -> ApproximationReport:
    order = require_order(ctx.data)
    if not ctx.data.small:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            "Name the small parameter (a symbol such as epsilon, or a combination such as v/c).",
            ["Which quantity is small?"],
        )
    small = parse_text(ctx.data.small, ctx.data, ctx.limits)
    if not small.free_symbols:
        raise Refused(
            Status.UNSUPPORTED, "parameter_missing", "The small parameter must contain a symbol."
        )
    taken = {s.name for s in ctx.expr.free_symbols | small.free_symbols}
    q = sp.Symbol(_fresh("q", taken), real=small.is_real is True or None)
    eliminated, solution = _eliminate(small, q)
    in_q = ctx.expr.subs(eliminated, solution)
    assumptions = {k: v for k, v in ctx.data.assumptions.items() if k != eliminated.name}
    if small.is_real is True:
        assumptions[q.name] = ["real"]
    result = series(
        ExpressionInput(input=to_text(in_q), format="text", assumptions=assumptions),
        q.name,
        "0",
        order,
        ctx.limits,
    )
    if result.status is not Status.OK or result.verification is None:
        raise Refused(
            result.status,
            "verification_failed" if result.status is Status.NOT_VERIFIED else "unsupported_input",
            result.message,
        )
    in_q_series = parse_text_with(result.result_text or "0", assumptions, ctx)
    coeffs = laurent_coefficients(in_q_series, q, sp.Integer(0))
    if coeffs is None:
        raise Refused(
            Status.NOT_VERIFIED, "verification_failed", "The expansion could not be read back."
        )
    back = {k: sp.expand(c) for k, c in coeffs.items()}
    total = sp.Integer(0)
    report = ApproximationReport(
        status=Status.OK,
        method="perturbation",
        input_latex=to_latex(ctx.expr),
        variable=to_latex(small),
        domain_latex=rf"{to_latex(small)} \to 0",
    )
    small_symbol = sp.Symbol(
        rf"\left({to_latex(small)}\right)" if not isinstance(small, sp.Symbol) else to_latex(small)
    )
    for k in sorted(back):
        term = sp.expand(back[k] * small**k)
        total += term
        report.orders.append(
            OrderTerm(
                order=k,
                coefficient_latex=to_latex(back[k]),
                term_latex=to_latex(term),
            )
        )
    report.result_latex = join_terms([sp.expand(back[k] * small**k) for k in sorted(back)])
    report.result_text = str(total)
    report.remainder_latex = remainder(to_latex(small_symbol), order + 1)
    report.checks = checks_from_series(result.verification)
    report.complexity.degree = order
    report.complexity.coefficients = len(back)
    report.assumptions_stated = [
        f"{to_latex(eliminated)} was replaced by {to_latex(solution)} so that {to_latex(small)} = {to_latex(q)}; "
        "the result is written back in the original symbols."
    ]
    report.message = (
        "Expanded in the small parameter; the coefficients and the remainder order were checked "
        "numerically (the other symbols sampled), and the result is written in the original symbols."
    )
    report.notes.append(
        "Each order is listed on its own: the first-order correction is the term of order 1, the "
        "second-order correction the term of order 2, and so on."
    )
    return finish(report, evidence="series_checked", warnings=["local_only"])


def parse_text_with(text: str, assumptions: dict[str, list[str]], ctx: Context) -> sp.Expr:
    return parse_text(text, ExpressionInput(input="0", assumptions=assumptions), ctx.limits)


# --- Asymptotic -------------------------------------------------------------------------------
def _infinity_sign(ctx: Context) -> int:
    text = ctx.data.point
    if text is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Say which way the variable goes: oo or -oo.",
            ["Does the variable go to +infinity or -infinity?"],
        )
    value = parse_text(text, ctx.data, ctx.limits)
    if value == sp.oo:
        return 1
    if value == -sp.oo:
        return -1
    raise Refused(
        Status.UNSUPPORTED,
        "point_invalid",
        "The point must be oo or -oo for an asymptotic expansion.",
    )


DIGITS_PER_POWER = 12  # the residual is evaluated at x = 10^12 ... : the next term is 1e-12 smaller


def _residual_check(
    expr: sp.Expr, approx: sp.Expr, var: sp.Symbol, sign: int, claimed: int
) -> Check:
    """Along x = +-10^12 * 2^j the residual must fall by 2^-claimed per doubling, at every parameter set.

    Run at a precision that resolves a residual of size x^-claimed, so that a coefficient wrong by
    one part in 1e9 is seen (the next true term is only 1e-12 of the last one kept).
    """
    sets = parameter_sets(expr, var, 4, seed=20260901)
    decided = 0
    digits = DIGITS_PER_POWER * claimed + 60
    exact_below = mpmath.mpf(10) ** -(DIGITS_PER_POWER * claimed + 40)
    with mpmath.workdps(digits):
        for values in sets:
            f = compile_mp(expr.subs(values), var)
            a = compile_mp(approx.subs(values), var)
            residuals = []
            for j in range(5):
                x = sign * mpmath.mpf(10) ** DIGITS_PER_POWER * 2**j
                fv, av = f(x), a(x)
                residuals.append(None if fv is None or av is None else fv - av)
            verdict = falls_at_rate(residuals, claimed, exact_below=exact_below)
            if verdict is False:
                return Check(
                    name="residual",
                    passed=False,
                    points=len(sets),
                    detail="the residual does not fall at the claimed rate",
                )
            if verdict:
                decided += 1
    if decided == 0:
        return Check(
            name="residual", passed=None, points=len(sets), detail="no point could be evaluated"
        )
    return Check(
        name="residual",
        passed=True,
        points=decided,
        detail=f"the residual falls like x^-{claimed} when the variable is doubled four times from 10^12",
    )


def asymptotic(ctx: Context) -> ApproximationReport:
    order = require_order(ctx.data)
    sign = _infinity_sign(ctx)
    refuse_non_analytic(ctx.expr, ctx.var)
    taken = {s.name for s in ctx.expr.free_symbols}
    u = sp.Symbol(_fresh("u", taken), positive=True)
    expr_u = ctx.expr.subs(ctx.var, sign / u)
    arrow = r"\infty" if sign > 0 else r"-\infty"
    report = ApproximationReport(
        status=Status.OK,
        method="asymptotic",
        input_latex=to_latex(ctx.expr),
        variable=ctx.var.name,
        domain_latex=rf"{to_latex(ctx.var)} \to {arrow}",
        remainder_latex=remainder(to_latex(ctx.var), -(order + 1)),
    )
    proposal = _with_logs(expr_u, u, ctx.var, sign, order)
    leftover = [
        f for f in proposal.atoms(sp.Function) if f.func is not sp.log and ctx.var in f.free_symbols
    ]
    if leftover or proposal.has(sp.Order):
        raise Refused(
            Status.UNSUPPORTED,
            "essential_singularity",
            "SymPy could not turn this into powers of 1/x (it is exponentially large or small, or "
            "oscillates); no asymptotic power expansion is given.",
        )
    if proposal == 0 and ctx.expr != 0:
        raise Refused(
            Status.UNSUPPORTED,
            "essential_singularity",
            "The function is smaller than every power of 1/x (exponentially small), so its power "
            "expansion is zero and says nothing; the leading method gives its dominant behaviour.",
        )
    residual = _residual_check(ctx.expr, proposal, ctx.var, sign, order + 1)
    if residual.passed is not True:
        raise Refused(
            Status.NOT_VERIFIED if residual.passed is False else Status.UNSUPPORTED,
            "verification_failed" if residual.passed is False else "unsupported_input",
            "No expansion in powers of 1/x (with logarithms) could be verified for this expression, "
            "so none is shown.",
        )
    report.checks = [residual]
    evidence: Evidence = "numeric_estimate"
    try:
        # A second, independent leg where it applies: Cauchy estimates in u = 1/x. It needs the
        # function to continue analytically around u = 0, which a branch of a square root may not.
        _coeffs, pole, check = laurent_core(expr_u, u, sp.Integer(0), order)
    except Refused:
        report.notes.append(
            "The coefficients could not also be checked by Cauchy estimates in 1/x (the expression "
            "has logarithms, a branch, or no analytic continuation there); the residual check decided."
        )
    else:
        report.checks += [
            Check(name=f"{c.name} (in u = 1/x)", passed=c.passed, points=c.points, detail=c.detail)
            for c in checks_from_series(check)
        ]
        evidence = "series_checked"
        if pole:
            report.extras.append(
                Labeled(label="Leading power", latex=rf"{to_latex(ctx.var)}^{{{pole}}}")
            )
    powers = _powers_in(proposal, ctx.var)
    if powers is not None:
        report.result_latex = powers_latex(powers, ctx.var, descending=True)
        for p, c in sorted(powers.items(), reverse=True):
            report.orders.append(
                OrderTerm(
                    order=p, coefficient_latex=to_latex(c), term_latex=to_latex(c * ctx.var**p)
                )
            )
        report.complexity.coefficients = len(powers)
    else:
        report.result_latex = to_latex(proposal)
        report.notes.append("The expansion contains logarithms or fractional powers of x.")
    report.result_text = str(proposal)
    report.complexity.degree = order
    report.message = (
        "An asymptotic expansion, not a convergent power series: it describes the function as the "
        "variable grows, and adding terms need not improve the approximation at a fixed x."
    )
    report.notes.append("Direction matters: the expansion for +oo and for -oo can differ.")
    return finish(report, evidence=evidence, warnings=["asymptotic_not_convergent"])


def _powers_in(expr: sp.Expr, var: sp.Symbol) -> dict[int, sp.Expr] | None:
    """{p: c_p} for expr = sum c_p var^p with integer p (negative allowed), else None."""
    return laurent_coefficients(expr, var, sp.Integer(0))


def _with_logs(expr_u: sp.Expr, u: sp.Symbol, var: sp.Symbol, sign: int, order: int) -> sp.Expr:
    try:
        candidate = sp.series(expr_u, u, 0, order + 1).removeO()
    except (NotImplementedError, ValueError, TypeError, RecursionError, AttributeError) as exc:
        raise sympy_failed(exc) from exc
    if candidate.has(sp.Order, sp.zoo, sp.oo, sp.nan):
        raise Refused(
            Status.UNSUPPORTED, "essential_singularity", "The expansion is not in powers of 1/x."
        )
    back = sp.expand(sp.expand_log(candidate.subs(u, sign / var), force=True))
    return back
