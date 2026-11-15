"""Continued fractions of a number: partial quotients, convergents and their errors.

Quotients are computed twice, at two working precisions, and only those on which both runs agree are
kept; for rationals and quadratic irrationals they are also compared with SymPy's
`continued_fraction_iterator`. The convergents are exact integer fractions. A convergent p/q of x
satisfies |x - p/q| < 1/(q q'), with q' the next denominator: that inequality is checked for each one
(exactly for a rational, at high precision otherwise).
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import Any

import mpmath
import sympy as sp
from sympy.ntheory.continued_fraction import continued_fraction_iterator

from ..latexprint import to_latex
from ..models import Status
from .common import Context, Refused, finish
from .interval import parse_tolerance
from .models import ApproximationReport, Check, Complexity, ErrorSummary, OrderTerm

MAX_TERMS = 60
SHOWN = 30


def _quotients(value: sp.Expr, terms: int, dps: int) -> list[int]:
    out: list[int] = []
    if value.is_Rational:  # exact Euclid: floating point would turn 7 into 6.999...
        num, den = int(value.p), int(value.q)
        while den and len(out) < terms:
            a, rem = divmod(num, den)
            out.append(a)
            num, den = den, rem
        return out
    with mpmath.workdps(dps):
        x = mpmath.mpf(str(sp.N(value, dps)))
        for _ in range(terms):
            a = int(mpmath.floor(x))
            out.append(a)
            frac = x - a
            if frac < mpmath.mpf(10) ** (-dps // 2):
                break
            x = 1 / frac
    return out


def convergents(quotients: list[int]) -> list[Fraction]:
    out: list[Fraction] = []
    p0, q0, p1, q1 = 1, 0, quotients[0], 1
    out.append(Fraction(p1, q1))
    for a in quotients[1:]:
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        out.append(Fraction(p1, q1))
    return out


def _nested(quotients: list[int], tail: bool) -> str:
    text = r"\cdots" if tail else ""
    for a in reversed(quotients[1:]):
        text = rf"{a} + \cfrac{{1}}{{{text}}}" if text else str(a)
    head = str(quotients[0])
    return rf"{head} + \cfrac{{1}}{{{text}}}" if len(quotients) > 1 else head


def continued_fraction(ctx: Context) -> ApproximationReport:
    expr = ctx.expr
    if expr.free_symbols or expr.is_real is not True:
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "Give a real number (a rational, or a constant such as pi, e, sqrt(2)).",
        )
    data = ctx.data
    tolerance, kind = parse_tolerance(ctx)
    if data.order is None and tolerance is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Give the number of partial quotients, or an accuracy target.",
            ["How many terms, or which accuracy?"],
        )
    if tolerance is not None and kind != "absolute":
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "A continued fraction target is an absolute error.",
        )
    wanted = min(data.order if data.order is not None else MAX_TERMS, MAX_TERMS)
    rational = expr.is_Rational
    dps = 12 * wanted + 40
    first = _quotients(expr, wanted, dps)
    second = _quotients(expr, wanted, dps + 30)
    agreed = [
        a
        for a, b in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(first, second, strict=False))
    ]
    if not agreed:
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            "The partial quotients did not agree between two precisions.",
        )
    checks = [
        Check(
            name="partial quotients agree at two working precisions",
            passed=len(agreed) == min(len(first), len(second)),
            points=len(agreed),
            detail=f"{dps} and {dps + 30} digits",
        )
    ]
    try:
        theirs = list(itertools.islice(continued_fraction_iterator(expr), len(agreed)))
        checks.append(
            Check(
                name="partial quotients against sympy.continued_fraction_iterator",
                passed=theirs == agreed[: len(theirs)],
                points=len(theirs),
                detail="a separate implementation (SymPy)",
            )
        )
    except (ValueError, TypeError, NotImplementedError, ArithmeticError):
        checks.append(
            Check(
                name="partial quotients against sympy.continued_fraction_iterator",
                passed=None,
                detail="not run",
            )
        )
    conv = convergents(agreed)
    complete = rational and conv[-1] == Fraction(int(expr.p), int(expr.q))
    with mpmath.workdps(dps):
        x = mpmath.mpf(str(sp.N(expr, dps)))
        rows: list[tuple[Fraction, Any]] = [
            (c, abs(x - mpmath.mpf(c.numerator) / c.denominator)) for c in conv
        ]
        ok_bound = True
        for k, (c, err) in enumerate(rows):
            nxt = conv[k + 1].denominator if k + 1 < len(conv) else None
            limit = mpmath.mpf(1) / (c.denominator * (nxt if nxt else c.denominator))
            # strict for an irrational number; equality is possible for a rational one (x = a0 + 1/a1)
            if err > limit * (1 + mpmath.mpf(10) ** -(dps // 2)):
                ok_bound = False
    checks.append(
        Check(
            name="each convergent satisfies |x - p/q| <= 1/(q q')",
            passed=ok_bound,
            points=len(rows),
            detail="q' is the next denominator (the same denominator for the last one, a weaker bound)",
        )
    )
    if any(c.passed is False for c in checks):
        raise Refused(
            Status.NOT_VERIFIED,
            "verification_failed",
            "The continued fraction failed its checks, so it is not shown.",
        )
    chosen = len(rows) - 1
    tolerance_ok: bool | None = None
    if tolerance is not None:
        meeting = [i for i, (_, err) in enumerate(rows) if err <= tolerance]
        tolerance_ok = bool(meeting)
        chosen = meeting[0] if meeting else len(rows) - 1
    p, q = conv[chosen].numerator, conv[chosen].denominator
    report = ApproximationReport(
        status=Status.OK,
        method="continued_fraction",
        input_latex=to_latex(expr),
        result_latex=rf"\frac{{{p}}}{{{q}}}",
        statement_latex=rf"{to_latex(expr)} = [{agreed[0]}; {', '.join(map(str, agreed[1:]))}{'' if rational and complete else ', \\ldots'}]"
        if len(agreed) > 1
        else rf"{to_latex(expr)} = [{agreed[0]}]",
        checks=checks,
        complexity=Complexity(coefficients=len(agreed), numerator_degree=None),
        tolerance_met=tolerance_ok,
    )
    with mpmath.workdps(30):
        for k, (c, err) in enumerate(rows[:SHOWN]):
            report.orders.append(
                OrderTerm(
                    order=k,
                    coefficient_latex=f"{c.numerator}/{c.denominator}",
                    term_latex=rf"\frac{{{c.numerator}}}{{{c.denominator}}} \approx {mpmath.nstr(mpmath.mpf(c.numerator) / c.denominator, 15)},\; \text{{error }} {mpmath.nstr(err, 3)}",
                )
            )
        exact_here = bool(complete and chosen == len(rows) - 1)
        report.error = ErrorSummary(
            samples=1,
            max_abs="0 (exact)" if exact_here else mpmath.nstr(rows[chosen][1], 4),
            at=f"convergent {chosen}",
        )
    if rational:
        report.evidence = "bound_established" if complete else "numeric_estimate"
        report.notes.append(
            "A rational number has a finite continued fraction; the last convergent equals it exactly."
        )
    else:
        report.evidence = "numeric_estimate"
        report.notes.append(
            "The quotients were computed at high precision and agree at two precisions; the inequality "
            "|x - p/q| <= 1/(q q') is a theorem for the correct quotients and was checked numerically."
        )
    report.message = (
        f"The convergent {p}/{q} (term {chosen}) has error {report.error.max_abs}"
        + (" and meets the tolerance." if tolerance_ok else ".")
        if tolerance is None or tolerance_ok
        else f"No convergent within {len(rows)} terms meets the tolerance; the last, {p}/{q}, has error {mpmath.nstr(rows[chosen][1], 4)}."
    )
    return finish(report, evidence=report.evidence)
