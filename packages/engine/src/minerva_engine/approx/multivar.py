"""Taylor expansion of a function of several variables about a point, to a total degree.

SymPy derivatives build the polynomial. It is shown only if, along several lines through the point
(directions chosen here, not by the model), the restriction of the function has the restricted
polynomial as its Taylor polynomial, as decided by `series.check_series`.
"""

from __future__ import annotations

import itertools
import random

import sympy as sp

from ..build import Builder
from ..latexprint import to_latex
from ..models import Status
from ..series import check_series
from .common import (
    Context,
    Refused,
    finish,
    join_terms,
    parse_text,
    refuse_non_analytic,
    remainder,
    require_order,
)
from .models import ApproximationReport, Check, Labeled, OrderTerm

MAX_VARIABLES = 4
MAX_DEGREE = 6
LINES = 6


def _multi_indices(n: int, degree: int) -> list[tuple[int, ...]]:
    out = []
    for total in range(degree + 1):
        for combo in itertools.product(range(total + 1), repeat=n):
            if sum(combo) == total:
                out.append(combo)
    return out


def multivariable(ctx: Context) -> ApproximationReport:
    data = ctx.data
    order = require_order(data, 1, MAX_DEGREE)
    if not data.variables or len(data.variables) != len(data.points):
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the variables and one expansion point for each.",
            ["Which variables, and about which point?"],
        )
    if len(set(data.variables)) != len(data.variables):
        raise Refused(Status.PARSE_ERROR, "variable_missing", "A variable is listed twice.")
    builder = Builder(data.assumptions, ctx.limits)
    symbols = [builder.symbol(v) for v in data.variables]
    centre = [parse_text(p, data, ctx.limits) for p in data.points]
    for c in centre:
        if c.free_symbols or c.is_finite is not True or c.is_real is not True:
            raise Refused(
                Status.UNSUPPORTED,
                "point_invalid",
                "Each point coordinate must be a finite real number.",
            )
    f = ctx.expr
    for s in symbols:
        refuse_non_analytic(f, s)
    if not set(symbols) & f.free_symbols:
        raise Refused(
            Status.UNSUPPORTED,
            "variable_missing",
            "The expression does not depend on these variables.",
        )
    at = dict(zip(symbols, centre, strict=True))
    shifts = [s - c for s, c in zip(symbols, centre, strict=True)]
    total = sp.Integer(0)
    report = ApproximationReport(
        status=Status.OK,
        method="multivariable",
        input_latex=to_latex(f),
        variable=", ".join(data.variables),
    )
    by_degree: dict[int, sp.Expr] = {}
    for index in _multi_indices(len(symbols), order):
        derivative = f
        for s, k in zip(symbols, index, strict=True):
            if k:
                derivative = sp.diff(derivative, s, k)
        value = derivative.subs(at)
        if value.has(sp.nan, sp.zoo, sp.oo):
            raise Refused(
                Status.UNSUPPORTED,
                "not_analytic",
                "A derivative is undefined at the point (a pole or branch point): there is no "
                "Taylor polynomial here.",
            )
        weight = sp.Integer(1)
        for s_, k in zip(shifts, index, strict=True):
            weight *= s_**k / sp.factorial(k)
        term = value * weight
        total += term
        by_degree[sum(index)] = by_degree.get(sum(index), sp.Integer(0)) + term
    total = sp.expand(total)
    # independent check along lines through the point
    t = sp.Symbol("t_line", real=True)
    rng = random.Random(20260905)  # noqa: S311 - deterministic sampling, not cryptographic
    passed = 0
    note = ""
    for _ in range(LINES):
        direction = [sp.Rational(rng.randint(-5, 5), rng.randint(1, 4)) for _ in symbols]
        if all(d == 0 for d in direction):
            continue
        line = {s: c + t * d for s, c, d in zip(symbols, centre, direction, strict=True)}
        restricted = f.subs(line, simultaneous=True)
        candidate = sp.expand(total.subs(line, simultaneous=True))
        check = check_series(restricted, t, sp.Integer(0), candidate, order)
        if not check.passed:
            note = check.note
            raise Refused(
                Status.NOT_VERIFIED,
                "verification_failed",
                f"A polynomial was produced but failed the check along a line through the point, so it is not shown. {note}",
            )
        passed += 1
    report.checks = [
        Check(
            name="restriction to lines through the point",
            passed=True,
            points=passed,
            detail="on each line the polynomial restricted to it passed the coefficient and remainder checks",
        )
    ]
    displacement = ", ".join(to_latex(s_) for s_ in shifts)
    norm = rf"\left\|\left({displacement}\right)\right\|"
    report.result_latex = join_terms(
        [sp.expand(by_degree[d]) for d in sorted(by_degree) if sp.expand(by_degree[d]) != 0]
    )
    report.result_text = str(total)
    report.remainder_latex = remainder(norm, order + 1)
    report.domain_latex = rf"\left({', '.join(to_latex(s) for s in symbols)}\right) \to \left({', '.join(to_latex(c) for c in centre)}\right)"
    for degree in sorted(by_degree):
        term = sp.expand(by_degree[degree])
        if term != 0:
            report.orders.append(
                OrderTerm(order=degree, coefficient_latex=to_latex(term), term_latex=to_latex(term))
            )
    report.complexity.degree = order
    report.complexity.coefficients = len(sp.Poly(total, *symbols).terms()) if total != 0 else 0
    gradient = sp.Matrix([sp.diff(f, s).subs(at) for s in symbols])
    report.extras.append(Labeled(label="Gradient at the point", latex=to_latex(gradient)))
    if order >= 2:
        hessian = sp.hessian(f, symbols).subs(at)
        report.extras.append(Labeled(label="Hessian at the point", latex=to_latex(hessian)))
        if all(g == 0 for g in gradient):
            report.notes.append(
                "The gradient vanishes at the point, so it is a stationary point and the quadratic "
                "term is the Hessian form; the Hessian alone does not say whether it is a minimum."
            )
    report.assumptions_stated = [
        "The remainder is measured in the Euclidean norm of the displacement from the point.",
        "It holds where the function has continuous partial derivatives of order "
        f"{order + 1} near the point.",
    ]
    report.message = (
        "A Taylor polynomial of total degree "
        f"{order}; it was checked along {passed} lines through the point by estimates that do not "
        "use sympy.series."
    )
    return finish(report, evidence="series_checked", warnings=["local_only"])
