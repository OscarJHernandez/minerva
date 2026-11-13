"""Derivative of a matrix exponential whose exponent does not commute with its derivative.

d/dt exp(X(t)) is X'(t) exp(X(t)) only if X and X' commute. In general it is the Frechet derivative

    integral from 0 to 1 of  exp((1-s) X) X' exp(s X) ds,

which equals the upper right block of exp([[X, X'], [0, X]]). Nothing here assumes commutativity:
the familiar form is reported as `blocked` with the condition it needs. Verification never uses the
route that produced the result:

* the central difference of exp(X(t)) (mpmath.expm on random complex matrices; time-dependent
  matrices are sampled as quadratic polynomials in the variable, honouring the declared commutation),
* the integral computed by quadrature against the block exponential,
* the exact leg: both sides as formal power series in the letters of the polynomial coefficients of
  the matrices (the Frechet series comes from the Beta integral, not from differentiating a series).
"""

from __future__ import annotations

import sympy as sp

from ..latexprint import to_latex
from .expressions import Factor, Frechet, Sym, Term, Terms, render_terms
from .methods import (
    Attempt,
    Outcome,
    Problem,
    SymbolicTimeout,
    _entries,
    _merge,
    _numeric_candidate,
    _simplified,
    _terms_eval,
    derivative_series_leg,
    derive_terms,
    explicit_value,
    matrices_in,
    symbolic_budget,
)
from .models import MatrixBlocked, MatrixStep
from .verify import (
    QUAD_DIGITS,
    QUAD_ERROR,
    Count,
    Env,
    check_derivative,
    close,
    numeric_points,
)

QUADRATURE_POINTS = 4


def _frechet_factors(terms: Terms) -> list[Frechet]:
    return [f for t in terms for f in t.factors if isinstance(f, Frechet)]


def _familiar_form(terms: Terms) -> str:
    """The derivative as X' exp(X): what is NOT applied, because it needs X X' = X' X."""
    out: list[Term] = []
    for t in terms:
        pieces: list[tuple[sp.Expr, tuple[Factor, ...]]] = [(t.coeff, ())]
        for f in t.factors:
            if isinstance(f, Frechet):
                options: list[tuple[sp.Expr, tuple[Factor, ...]]] = [
                    (c, (Sym(name, 1, d), f.base)) for name, d, c in f.direction
                ]
            else:
                options = [(sp.Integer(1), (f,))]
            pieces = [(c0 * c1, (*fs, *g)) for c0, fs in pieces for c1, g in options]
        out += [Term(c, fs) for c, fs in pieces]
    return render_terms(tuple(out))


def _quadrature_count(p: Problem, factors: list[Frechet]) -> Count:
    """The block exponential against the integral computed from its definition."""

    def check(env: Env) -> bool | None:
        for f in factors:
            integral, err = p.ev.frechet_quadrature(f, env)
            if err > QUAD_ERROR:
                return None  # the quadrature itself did not converge: not a usable point
            if not close(integral, p.ev.frechet_block(f, env)):
                return False
        return True

    return numeric_points(
        p.setup,
        p.ev,
        check,
        deadline=p.deadline,
        max_points=QUADRATURE_POINTS,
        digits=QUAD_DIGITS,
    )


def _explicit_derivative(p: Problem) -> sp.Matrix | None:
    """d/dt of the explicit value, when SymPy can find the exponential in closed form."""
    assert p.var is not None  # noqa: S101
    if not matrices_in(p.terms) <= set(p.explicit):
        return None
    try:
        with symbolic_budget(p):
            value = explicit_value(p.terms, p.explicit, p.shape[0])
            return _simplified(value.diff(p.var))
    except (NotImplementedError, ValueError, sp.matrices.exceptions.MatrixError, SymbolicTimeout):
        return None  # no closed form in time: the verified representation is reported instead


def run_frechet_derivative(p: Problem) -> Attempt:
    assert p.var is not None  # noqa: S101
    var = p.var
    derived = derive_terms(p.terms, var, p.comm, p.setup.dependent, frechet=True)
    if isinstance(derived, str):
        return Attempt(unsupported=derived)
    factors = _frechet_factors(derived.terms)
    if not factors:
        return Attempt()
    formula = render_terms(derived.terms)
    blocked = [
        MatrixBlocked(
            method="exp_derivative",
            title="Use d/dt exp(X) = X' exp(X)",
            needs="X X' = X' X for the exponent X (not established)",
            latex=_familiar_form(derived.terms),
        )
    ]
    count, directions = check_derivative(
        p.setup, p.ev, var, p.integrand, _terms_eval(p, derived.terms), deadline=p.deadline
    )
    count = _merge(count, _quadrature_count(p, factors))
    symbolic = derivative_series_leg(p, derived.terms)
    numeric_method = (
        f"central difference (step 1e-12, 25 digits; {' and '.join(directions)} direction) of "
        "exp(X(t)) by mpmath.expm on random complex matrices, against the Frechet derivative as "
        "the upper right block of the exponential of [[X, X'], [0, X]]; the same quantity from "
        "quadrature of its integral definition at 16 digits"
    )
    steps = [
        MatrixStep(
            method="exp_frechet_derivative",
            title="Frechet derivative of the exponential",
            before_latex=render_terms(p.terms),
            after_latex=formula,
            note=(
                "no commutation is assumed: d/dt exp(X) = ∫₀¹ exp((1-s)X) X' exp(sX) ds, equal to "
                "the upper right block of exp([[X, X'], [0, X]])"
            ),
        )
    ]
    notes = [
        "The integral is an exact representation, not an evaluated matrix. It reduces to "
        "X' exp(X) only if X X' = X' X."
    ]
    assumptions = list(derived.assumptions)

    value = _explicit_derivative(p)
    if value is not None:
        explicit_count, _ = check_derivative(
            p.setup, p.ev, var, p.integrand, _numeric_candidate(p, value), deadline=p.deadline
        )
        steps.append(
            MatrixStep(
                method="exp_frechet_derivative",
                title="Evaluate with explicit entries",
                before_latex=formula,
                after_latex=str(to_latex(value)),
                note="differentiated entrywise from the closed form of the exponential",
            )
        )
        return Attempt(
            Outcome(
                "explicit",
                str(to_latex(value)),
                _merge(count, explicit_count),
                numeric_method,
                entries=_entries(value),
                steps=steps,
                assumptions=assumptions,
                # the exact leg checks the formula above, not these entries: it can only veto
                symbolic=False if symbolic is False else None,
                symbolic_method=_SERIES_FRECHET,
            ),
            blocked,
        )
    return Attempt(
        Outcome(
            "representation",
            formula,
            count,
            numeric_method,
            evaluated=False,
            steps=steps,
            assumptions=assumptions,
            symbolic=symbolic,
            symbolic_method=_SERIES_FRECHET,
            notes=notes,
        ),
        blocked,
    )


_SERIES_FRECHET = (
    "the matrices expanded as quadratic polynomials in the variable with constant matrix "
    "coefficients; exp(X) differentiated term by term as a formal power series, against the "
    "series of the integral obtained from the Beta integral (coefficient 1/(i+j+1)! on "
    "X^i X' X^j), compared through total degree 4 modulo the declared commutations; exact, but "
    "not a proof for all degrees"
)
