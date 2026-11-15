"""Euler-Maclaurin approximation of a finite sum, and acceleration of a slowly convergent series.

sum_{k=a}^{b} f(k) = integral_a^b f + (f(a) + f(b))/2 + sum_{j=1}^{p} B_2j/(2j)! (f^(2j-1)(b) - f^(2j-1)(a)) + R_p

The integral is the verified definite integral where SymPy finds a closed form and two-rule
quadrature otherwise. The formula is decided by two checks that do not use its derivation: direct
summation (the error must be consistent with the first neglected term) and an exactness test on
polynomials, which the formula reproduces exactly up to degree 2p + 1. The expansion is asymptotic:
adding corrections need not keep helping, and no bound on the remainder is claimed.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath
import sympy as sp

from ..integrate import integrate
from ..latexprint import to_latex
from ..models import ExpressionInput, Limits, Status
from .common import Context, Refused, finish, parse_text, require_order
from .interval import build_reference
from .models import ApproximationReport, Candidate, Check, Complexity, ErrorSummary, Labeled
from .quadnodes import time_limit

MAX_ORDER = 8
DIRECT_CAP = 100_000  # terms summed directly
SLACK = 3  # the error may be this many times the first neglected term
DPS = 40
SIMPLIFY_SECONDS = 2.0
ESTIMATE_DROP = 6  # terms dropped to estimate an accelerator's own error


def _bernoulli_coefficient(j: int) -> sp.Rational:
    return sp.bernoulli(2 * j) / sp.factorial(2 * j)


def correction(f: sp.Expr, k: sp.Symbol, a: sp.Expr, b: sp.Expr, j: int) -> sp.Expr:
    """The j-th Euler-Maclaurin term B_2j/(2j)! (f^(2j-1)(b) - f^(2j-1)(a))."""
    d = sp.diff(f, k, 2 * j - 1)
    return _bernoulli_coefficient(j) * (d.subs(k, b) - d.subs(k, a))


def _instance(f: sp.Expr, k: sp.Symbol, a: int, b: int, p: int) -> tuple[Any, list[Any], Any]:
    """(direct sum, the approximations with 0..p corrections, the first neglected term) for one b."""
    fn = sp.lambdify(k, f, modules="mpmath")
    with mpmath.workdps(DPS):
        total = sum((mpmath.mpf(fn(mpmath.mpf(i))) for i in range(a, b + 1)), mpmath.mpf(0))
        pieces = [a + (b - a) * mpmath.mpf(t) / 8 for t in range(9)]
        integral, _ = mpmath.quad(lambda x: fn(x), pieces, error=True)
        base = integral + (fn(mpmath.mpf(a)) + fn(mpmath.mpf(b))) / 2
        approx = [base]
        for j in range(1, p + 2):
            term = mpmath.mpmathify(sp.N(correction(f, k, sp.Integer(a), sp.Integer(b), j), DPS))
            approx.append(approx[-1] + term)
        return total, approx[: p + 1], approx[p + 1] - approx[p]


def polynomial_self_test(k: sp.Symbol, p: int) -> Check:
    """The formula with p corrections is exact for k^m, m <= 2p + 1: compare with the exact power sum."""
    a, b = 2, 9
    for m in range(0, 2 * p + 2):
        f = k**m
        exact = sum(i**m for i in range(a, b + 1))
        approx = sp.integrate(f, (k, a, b)) + (f.subs(k, a) + f.subs(k, b)) / 2
        approx += sum(
            (correction(f, k, sp.Integer(a), sp.Integer(b), j) for j in range(1, p + 1)),
            sp.Integer(0),
        )
        if sp.simplify(approx - exact) != 0:
            return Check(
                name="exact for polynomials up to degree 2p + 1",
                passed=False,
                points=m + 1,
                detail=f"fails for k^{m}",
            )
    return Check(
        name="exact for polynomials up to degree 2p + 1",
        passed=True,
        points=2 * p + 2,
        detail="compared with exact power sums",
    )


def _formula_latex(k: sp.Symbol, p: int) -> str:
    terms = [r"\int_a^b f(x)\,dx + \frac{f(a)+f(b)}{2}"]
    for j in range(1, p + 1):
        c = _bernoulli_coefficient(j)
        order = "'" * (2 * j - 1) if 2 * j - 1 <= 3 else f"^{{({2 * j - 1})}}"
        sign = "-" if c < 0 else "+"
        terms.append(rf"{sign} \frac{{{abs(c.p)}}}{{{c.q}}}\left(f{order}(b)-f{order}(a)\right)")
    return " ".join(terms)


def euler_maclaurin(ctx: Context) -> ApproximationReport:
    data = ctx.data
    p = require_order(data, 1, MAX_ORDER)
    if data.lower is None or data.upper is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The summation limits are yours to choose.",
            ["What are the lower and upper limits of the sum?"],
        )
    k = ctx.var
    lower = parse_text(data.lower, data, ctx.limits)
    upper = parse_text(data.upper, data, ctx.limits)
    if not lower.is_Integer:
        raise Refused(Status.UNSUPPORTED, "point_invalid", "The lower limit must be an integer.")
    a = int(lower)
    symbols = upper.free_symbols
    others = ctx.expr.free_symbols - {k} - symbols
    if others:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "parameter_missing",
            "Give values for the other symbols in the summand; only the upper limit may be symbolic.",
            ["What values do the other symbols take?"],
        )
    if len(symbols) > 1:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "At most one symbol (the upper limit, e.g. N) is supported.",
        )
    f = ctx.expr
    if f.atoms(sp.Piecewise, sp.floor, sp.ceiling, sp.Abs, sp.sign) and any(
        k in x.free_symbols for x in f.atoms(sp.Piecewise, sp.floor, sp.ceiling, sp.Abs, sp.sign)
    ):
        raise Refused(
            Status.UNSUPPORTED,
            "not_analytic",
            "The summand must be smooth: Euler-Maclaurin needs derivatives of every order used.",
        )
    symbolic = bool(symbols)
    if symbolic:
        instances = [a + 20, a + 40, a + 80]
        b_requested: sp.Expr = upper
    else:
        if not upper.is_Integer:
            raise Refused(
                Status.UNSUPPORTED,
                "point_invalid",
                "The upper limit must be an integer or a symbol.",
            )
        b = int(upper)
        if b <= a:
            raise Refused(
                Status.UNSUPPORTED, "point_invalid", "The upper limit must exceed the lower one."
            )
        instances = [b if b - a + 1 <= DIRECT_CAP else a + DIRECT_CAP - 1]
        b_requested = upper
    ref_ctx = ctx
    build_reference(
        ref_ctx, (mpmath.mpf(a), mpmath.mpf(max(instances) if not symbolic else instances[-1]))
    )
    rows: list[tuple[int, Any, list[Any], Any]] = []
    for b in instances:
        total, approx, neglected = _instance(f, k, a, b, p)
        rows.append((b, total, approx, neglected))
    self_test = polynomial_self_test(k, p)
    within = True
    improves = True
    with mpmath.workdps(DPS):
        for _b, total, approx, neglected in rows:
            err = abs(total - approx[p])
            exact_enough = err <= mpmath.mpf(10) ** -25 * (1 + abs(total))
            if not (exact_enough or err <= SLACK * abs(neglected)):
                within = False
            if not (exact_enough or err <= abs(total - approx[0])):
                improves = False
    checks = [
        self_test,
        Check(
            name="the corrections improve on the integral plus half the end terms",
            passed=improves,
            points=len(rows),
            detail="the error with all corrections is not larger than with none",
        ),
        Check(
            name="error against direct summation is consistent with the first neglected term",
            passed=within,
            points=len(rows),
            detail="instances: "
            + ", ".join(
                f"{label}={b}"
                for label, (b, *_rest) in zip(itertools.repeat("b"), rows, strict=False)
            ),
        ),
    ]
    if any(c.passed is False for c in checks):
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            "The Euler-Maclaurin approximation did not improve on, or agree with, direct summation (the "
            "summand may oscillate, have large derivatives at the ends, or too many corrections were "
            "asked for); try fewer corrections or a larger lower limit. It is not shown.",
        )
    b_show, total_show, approx_show, neglected_show = rows[-1]
    report = ApproximationReport(
        status=Status.OK,
        method="euler_maclaurin",
        input_latex=to_latex(f),
        variable=k.name,
        domain_latex=rf"k = {a},\ldots,{to_latex(b_requested)}",
        statement_latex=_formula_latex(k, p),
        checks=checks,
        complexity=Complexity(coefficients=p + 2),
        evidence="numeric_estimate",
    )
    # the approximation itself
    if symbolic:
        result = integrate(
            ExpressionInput(input=str(f), format="text", assumptions=ctx.data.assumptions),
            k.name,
            Limits(
                **{
                    **ctx.limits.model_dump(),
                    "timeout_seconds": max(1.0, ctx.limits.timeout_seconds * 0.3),
                }
            ),
        )
        if result.status is not Status.OK or not result.result_text:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                "No verified antiderivative was found, so a formula in the symbolic upper limit cannot be given. "
                "Use a numeric upper limit for a numerical value.",
            )
        antiderivative = parse_text(result.result_text, ctx.data, ctx.limits)
        integral = antiderivative.subs(k, upper) - antiderivative.subs(k, a)
        value = integral + (f.subs(k, a) + f.subs(k, upper)) / 2
        value += sum(
            (correction(f, k, sp.Integer(a), upper, j) for j in range(1, p + 1)), sp.Integer(0)
        )
        n_sym = next(iter(symbols))
        for b in instances:  # the closed form must reproduce the instance values
            with mpmath.workdps(DPS):
                got = mpmath.mpmathify(sp.N(value.subs(n_sym, b), DPS))
                want = next(r[2][p] for r in rows if r[0] == b)
                if abs(got - want) > mpmath.mpf(10) ** -15 * (1 + abs(want)):
                    raise Refused(
                        Status.NOT_VERIFIED,
                        "verification_failed",
                        "The closed form disagreed with the numerical evaluation, so it is not shown.",
                    )
        report.checks.append(
            Check(
                name="closed form against numerical evaluation",
                passed=True,
                points=len(instances),
                detail="N = " + ", ".join(map(str, instances)),
            )
        )
        shown = sp.expand(value)
        with time_limit(SIMPLIFY_SECONDS):  # a slow simplification keeps the (equal) expanded form
            shown = sp.simplify(value)
        report.result_latex = to_latex(shown)
        report.result_text = str(shown)
        report.notes.append(
            "The formula in the symbolic limit was checked for N = "
            + ", ".join(map(str, instances))
            + " (not proved for every N)."
        )
    else:
        report.result_latex = mpmath.nstr(approx_show[p], 20)
        report.result_text = mpmath.nstr(approx_show[p], 20)
    # error table: what each number of correction terms achieves, measured against the direct sum
    for j in range(p + 1):
        err = abs(total_show - approx_show[j])
        report.candidates.append(
            Candidate(
                method="euler_maclaurin",
                label=f"{j} correction term{'s' if j != 1 else ''}",
                status=Status.OK,
                complexity=Complexity(coefficients=j + 2),
                error=ErrorSummary(samples=1, max_abs=mpmath.nstr(err, 4)),
                chosen=j == p,
            )
        )
    report.error = ErrorSummary(
        samples=len(rows),
        max_abs=mpmath.nstr(abs(total_show - approx_show[p]), 4),
        at=f"b = {b_show}",
    )
    report.extras.append(
        Labeled(
            label="First neglected term (an estimate of the error)",
            latex=mpmath.nstr(neglected_show, 4),
        )
    )
    if not symbolic and b_show != int(upper):
        report.notes.append(
            f"The upper limit is larger than direct summation allows ({DIRECT_CAP} terms); the formula was checked "
            f"for b = {b_show} and its value at the requested limit is the same formula."
        )
        with mpmath.workdps(DPS):
            fn = sp.lambdify(k, f, modules="mpmath")
            b_req = int(upper)
            integral = mpmath.quad(
                lambda x: fn(x), [a + (b_req - a) * mpmath.mpf(t) / 8 for t in range(9)]
            )
            val = integral + (fn(mpmath.mpf(a)) + fn(mpmath.mpf(b_req))) / 2
            for j in range(1, p + 1):
                val += mpmath.mpmathify(
                    sp.N(correction(f, k, sp.Integer(a), sp.Integer(b_req), j), DPS)
                )
            report.result_latex = mpmath.nstr(val, 20)
    report.message = (
        f"Euler-Maclaurin with {p} Bernoulli correction term{'s' if p != 1 else ''}. At b = {b_show} the error "
        f"against direct summation is {mpmath.nstr(abs(total_show - approx_show[p]), 4)}; the first neglected term "
        f"is {mpmath.nstr(abs(neglected_show), 4)}."
    )
    report.notes.append(
        "Assumptions: the summand is smooth on [a, b]. The expansion is asymptotic: more corrections need "
        "not improve the result, and no bound on the remainder is claimed. The error shown is a measurement."
    )
    return finish(report, evidence="numeric_estimate")


# --- sequence acceleration --------------------------------------------------------------------
def accelerate(ctx: Context) -> ApproximationReport:
    data = ctx.data
    if data.max_terms is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The number of terms to use is yours to choose.",
            ["How many terms of the series may be used?"],
        )
    if data.lower is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the first index of the series.",
            ["Where does the sum start?"],
        )
    k = ctx.var
    first = parse_text(data.lower, data, ctx.limits)
    if not first.is_Integer or ctx.expr.free_symbols - {k}:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "A series in one index with an integer start is expected.",
        )
    n_terms = data.max_terms
    fn = sp.lambdify(k, ctx.expr, modules="mpmath")
    with mpmath.workdps(DPS):
        partial: list[Any] = []
        running = mpmath.mpf(0)
        for i in range(int(first), int(first) + n_terms):
            v = fn(mpmath.mpf(i))
            if not mpmath.isfinite(v):
                raise Refused(Status.UNSUPPORTED, "domain_invalid", "A term is not finite.")
            running += v
            partial.append(running)
        last_terms = [
            abs(fn(mpmath.mpf(i))) for i in range(int(first) + n_terms - 3, int(first) + n_terms)
        ]
        first_term = abs(fn(mpmath.mpf(int(first))))
        shrinking = (
            last_terms[0] >= last_terms[1] >= last_terms[2] and last_terms[-1] < first_term / 10
        )
        if not shrinking:
            raise Refused(
                Status.UNSUPPORTED,
                "domain_invalid",
                "The terms are not shrinking to zero, so the series is not shown to converge.",
            )

        def richardson(seq: list[Any]) -> Any:
            return mpmath.richardson(seq)[0]

        def shanks(seq: list[Any]) -> Any:
            return mpmath.shanks(seq)[-1][-1]

        results: dict[str, Any] = {}
        for name, method in (("Richardson", richardson), ("Shanks", shanks)):
            try:
                full = method(partial)
                shorter = method(partial[:-ESTIMATE_DROP])
            except (ValueError, ZeroDivisionError, TypeError, IndexError):
                continue
            # the method's own estimate of its error: how far it moves when the last terms are dropped
            results[name] = (full, abs(full - shorter))
        try:
            default = mpmath.nsum(fn, [int(first), mpmath.inf])
            other = mpmath.nsum(fn, [int(first), mpmath.inf], method="euler-maclaurin")
            results["mpmath.nsum"] = (default, abs(default - other))
        except (ValueError, ZeroDivisionError, TypeError, mpmath.libmp.NoConvergence):
            pass
    # an accelerator counts only if its own error estimate is small; at least two must, and agree
    scale_all = max(1, max(abs(v) for v, _ in results.values()))
    reliable = {n: (v, e) for n, (v, e) in results.items() if e <= mpmath.mpf("1e-6") * scale_all}
    if len(reliable) < 2:
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            "Fewer than two accelerations reached a small estimated error, so no value is shown.",
        )
    values = [v for v, _ in reliable.values()]
    spread = max(values) - min(values)
    scale = max(1, max(abs(v) for v in values))
    agree = spread <= mpmath.mpf("1e-8") * scale
    report = ApproximationReport(
        status=Status.OK if agree else Status.NOT_VERIFIED,
        method="accelerate",
        input_latex=to_latex(ctx.expr),
        variable=k.name,
        evidence="numeric_estimate",
        checks=[
            Check(
                name="independent accelerations agree",
                passed=bool(agree),
                points=len(results),
                detail=f"spread {mpmath.nstr(spread, 3)} between "
                + ", ".join(reliable)
                + (
                    ""
                    if len(reliable) == len(results)
                    else "; set aside (large own error estimate): "
                    + ", ".join(set(results) - set(reliable))
                ),
            )
        ],
    )
    if not agree:
        report.reason = "verification_failed"
        report.message = "The accelerations disagree, so no value is shown."
        return report
    best = reliable.get("Richardson") or next(iter(reliable.values()))
    report.result_latex = mpmath.nstr(best[0], 15)
    report.result_text = mpmath.nstr(best[0], 15)
    for name, (v, e) in results.items():
        report.candidates.append(
            Candidate(
                method="accelerate",
                label=name,
                status=Status.OK,
                result_latex=mpmath.nstr(v, 15),
                error=ErrorSummary(samples=1, max_abs=mpmath.nstr(e, 3)),
            )
        )
    report.candidates.append(
        Candidate(
            method="accelerate",
            label=f"plain partial sum of {n_terms} terms",
            status=Status.OK,
            result_latex=mpmath.nstr(partial[-1], 15),
        )
    )
    report.message = (
        "The accelerated values agree; this is a numerical estimate of the sum, not an identity."
    )
    report.notes.append(
        "Convergence of the series is not proved here, and the estimate is only as good as the terms' regular behaviour."
    )
    return finish(report, evidence="numeric_estimate")
