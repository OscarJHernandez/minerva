"""Duhamel integrals: integral of e^{(a s + b) A} M e^{(-a s + d) A} ds, with one matrix A.

The two exponentials involve the same matrix and their exponents in the integration variable cancel
(a + c = 0), which is exactly the structure of the variation-of-constants formula
∫₀ᵗ e^{(t-s)A} B e^{sA} ds. The Sylvester operator of this form, X ↦ -aAX + aXA, is always
singular, so the reduction to a Sylvester equation never determines the integral; this method names
the structure and chooses among three honestly different outcomes:

* M commutes with A (declared, or visible in the entries): the integrand does not depend on s,
  and the integral is (upper - lower) M e^{(b+d)A}: an evaluated closed form;
* explicit entries: the block-matrix exponential, evaluated when SymPy finds it in time, else
  computed numerically when every bound and entry is a number (a numerical approximation);
* otherwise: the exact block-matrix representation, never presented as an evaluated integral.

Every outcome is checked against quadrature of the full integrand at sampled parameters.
"""

from __future__ import annotations

import random

import mpmath as mp
import sympy as sp

from ..latexprint import to_latex
from .expressions import Exp, Factor, Sym, Term, render_terms
from .methods import (
    _SERIES,
    Attempt,
    Outcome,
    Problem,
    _all_explicit,
    _definite,
    _full_candidate,
    _judge_full,
    _merge,
    _reduction,
    _x0_block,
    match_template,
    matrices_in,
    run_block_exponential,
)
from .models import MatrixStep
from .verify import (
    check_integral,
    close,
    make_env,
    poly_equal,
    poly_integrate,
    series_of_terms,
)

DIGITS_SHOWN = 20


def _same_matrix_cancelling(p: Problem) -> tuple[str, sp.Expr, sp.Expr] | None:
    """(A, beta, delta) when the integrand is L e^{(a s+b)A} M e^{(-a s+d)A} R."""
    if p.var is None or len(p.terms) != 1 or not _definite(p):
        return None
    tmpl = match_template(p.terms[0], p.var)
    if isinstance(tmpl, str) or len(tmpl.exps) != 2:
        return None
    (_, name1, a1, b1), (_, name2, a2, b2) = tmpl.exps
    if name1 != name2 or sp.simplify(a1 + a2) != 0:
        return None
    return name1, b1, b2


def _commutes_with(p: Problem, name: str, factors: tuple[Factor, ...]) -> list[str] | None:
    """Commutation statements needed for the factors to commute with `name`, or None."""
    needs: list[str] = []
    for f in factors:
        names = [f.name] if isinstance(f, Sym) else list(f.names)
        if not names:
            return None
        for other in names:
            if other == name:
                continue
            if not p.comm.commutes(name, other):
                return None
            a, b = sorted((name, other))
            needs.append(f"{a}{b} = {b}{a}")
    return needs


def run_duhamel(p: Problem) -> Attempt:
    found = _same_matrix_cancelling(p)
    if found is None:
        return Attempt()
    assert p.var is not None and p.lower is not None and p.upper is not None  # noqa: S101
    name, beta, delta = found
    tmpl = match_template(p.terms[0], p.var)
    assert not isinstance(tmpl, str)  # noqa: S101 - checked by _same_matrix_cancelling
    p.scratch["structure"] = (
        f"Duhamel form: e^((as+b){name}) M e^((-as+d){name}) with one matrix {name} and "
        "exponents in the integration variable that cancel; the Sylvester operator of this form is "
        "singular, so the equation alone never determines the integral."
    )
    needs = _commutes_with(p, name, tmpl.mid)
    if needs is not None:
        return _commuting_closed_form(p, name, beta + delta, tmpl.left, tmpl.mid, tmpl.right, needs)
    return _block_form(p, name)


def _commuting_closed_form(
    p: Problem,
    name: str,
    shift: sp.Expr,
    left: tuple[Factor, ...],
    mid: tuple[Factor, ...],
    right: tuple[Factor, ...],
    needs: list[str],
) -> Attempt:
    assert p.var is not None and p.lower is not None and p.upper is not None  # noqa: S101
    var, lower, upper = p.var, p.lower, p.upper
    exp_part: tuple[Factor, ...] = (
        (Exp(((name, sp.expand(shift)),)),) if sp.expand(shift) != 0 else ()
    )
    result_terms = (
        Term(sp.expand(p.terms[0].coeff * (upper - lower)), (*left, *mid, *exp_part, *right)),
    )
    count = check_integral(
        p.setup,
        p.ev,
        var,
        p.integrand,
        lower,
        upper,
        p.shape,
        lambda env, x: close(x, p.ev.terms(result_terms, env, p.shape[0])),
        deadline=p.deadline,
    )
    try:
        symbolic: bool | None = poly_equal(
            poly_integrate(series_of_terms(p.terms), var, lower, upper),
            series_of_terms(result_terms),
            p.comm,
        )
    except ValueError:
        symbolic = None
    latex = render_terms(result_terms)
    return Attempt(
        Outcome(
            "expression",
            latex,
            count,
            "adaptive quadrature (mpmath) of the integrand at sampled parameters against the "
            "closed form, random complex matrices that respect the declared commutation",
            steps=[
                MatrixStep(
                    method="duhamel",
                    title="Duhamel integral with a middle factor that commutes with the exponent",
                    before_latex=f"\\int_{{{to_latex(lower)}}}^{{{to_latex(upper)}}} "
                    f"{render_terms(p.terms)}\\,d{to_latex(var)}",
                    after_latex=latex,
                    note="the integrand does not depend on the variable: the exponentials merge "
                    "because the middle factor commutes with the exponent",
                )
            ],
            assumptions=list(dict.fromkeys(needs)),
            symbolic=symbolic,
            symbolic_method=_SERIES + " (the series of the integrand is integrated term by term)",
        )
    )


def _block_form(p: Problem, name: str) -> Attempt:
    red = _reduction(p)
    if isinstance(red, str):
        return Attempt()
    attempt = run_block_exponential(p)
    outcome = attempt.outcome
    if outcome is None:
        return attempt
    steps = [
        s.model_copy(update={"method": "duhamel"}) if s.method == "block_exponential" else s
        for s in outcome.steps
    ]
    outcome.steps = steps
    outcome.notes = [
        "The Sylvester equation of a Duhamel integral has a singular operator (X ↦ aXA - aAX): "
        "it does not determine the integral, so the result rests on the block-matrix exponential "
        "and the quadrature check, not on solving the equation",
        *outcome.notes,
    ]
    _maybe_numeric(p, outcome, name)
    return Attempt(outcome=outcome, blocked=attempt.blocked)


def _maybe_numeric(p: Problem, outcome: Outcome, name: str) -> None:
    """With explicit matrices and numeric bounds but no closed form, report a numerical value."""
    if outcome.kind != "representation":
        return
    assert p.lower is not None and p.upper is not None  # noqa: S101
    names = matrices_in(p.terms)
    scalars = [s for s in p.ev.symbols if s != p.var]
    if scalars or not _all_explicit(names, p.explicit):
        return
    if not (p.lower.is_number and p.upper.is_number):
        return
    red = p.scratch.get("reduction")
    if red is None or isinstance(red, str):
        return
    candidate = _full_candidate(p, red, _x0_block(p, red))
    env = make_env(p.setup, p.ev, random.Random(20260101))  # noqa: S311 - test point
    with mp.workdps(30):
        value = candidate(env)
        count = _judge_full(p, red, lambda e: _x0_block(p, red)(e))
    rows = [
        [mp.nstr(value[i, j], DIGITS_SHOWN, strip_zeros=True) for j in range(value.cols)]
        for i in range(value.rows)
    ]
    outcome.kind = "numeric"
    outcome.entries = rows
    outcome.result_latex = (
        "\\begin{pmatrix} "
        + " \\\\ ".join(" & ".join(_decimal_latex(x) for x in row) for row in rows)
        + " \\end{pmatrix}"
    )
    outcome.evaluated = False
    outcome.count = _merge(outcome.count, count)
    outcome.notes.append(
        f"no closed form was found in time: the value is a numerical approximation to "
        f"{DIGITS_SHOWN} digits from the block-matrix exponential, checked against quadrature"
    )
    outcome.steps.append(
        MatrixStep(
            method="duhamel",
            title="Numerical value from the block-matrix exponential",
            before_latex=outcome.result_latex,
            after_latex=None,
            note="a numerical approximation, not a closed form",
        )
    )


def _decimal_latex(text: str) -> str:
    return text.replace("e+", "\\cdot 10^{").replace("e-", "\\cdot 10^{-") + (
        "}" if "e" in text else ""
    )


__all__ = ["run_duhamel"]
