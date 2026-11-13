"""The method registry: which technique applies to which structure, and its own verification.

Each `Method` declares the pattern it recognises, what it requires, and its limits; `run` either
returns a verified-ready `Outcome`, reports why it is blocked, or says it does not apply. Methods
are tried in a fixed order and the language model has no part in the choice. Every method checks
its own result with `verify.py` before an `Outcome` leaves this module.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mpmath as mp
import sympy as sp

from ..latexprint import to_latex
from .expressions import (
    MAX_DERIV,
    Canonical,
    Commutation,
    Exp,
    Factor,
    Frechet,
    Sym,
    Term,
    Terms,
    canonicalize,
    collect,
    depends_on,
    name_latex,
    render_factors,
    render_terms,
    substitute,
)
from .models import Action, MatrixBlocked, MatrixStep, ResultKind
from .verify import (
    Count,
    Env,
    Evaluator,
    Poly,
    Setup,
    check_derivative,
    check_equal,
    check_integral,
    close,
    expand_dependent,
    letter,
    make_env,
    poly_add,
    poly_diff,
    poly_equal,
    poly_integrate,
    poly_mul,
    poly_subs,
    series_of_terms,
    series_terms_equal,
)

UNKNOWN = "__X__"  # the unknown matrix of a Sylvester equation
MAX_CLOSED_FORM_OPS = 3000  # larger SymPy closed forms are too slow to verify and unreadable


@dataclass
class Problem:
    action: Action
    terms: Terms
    shape: tuple[int, int]
    setup: Setup
    ev: Evaluator
    invertible: dict[str, str]  # matrix name -> where that is known from
    var: sp.Symbol | None
    lower: sp.Expr | None
    upper: sp.Expr | None
    deadline: float
    scratch: dict[str, Any] = field(default_factory=dict)

    @property
    def comm(self) -> Commutation:
        return self.setup.comm

    @property
    def explicit(self) -> dict[str, sp.Matrix]:
        return self.setup.explicit

    def integrand(self, env: Env) -> mp.matrix:
        return self.ev.terms(self.terms, env)


@dataclass
class Outcome:
    kind: ResultKind
    result_latex: str | None
    count: Count
    numeric_method: str
    evaluated: bool = True
    entries: list[list[str]] | None = None
    equation_latex: str | None = None
    steps: list[MatrixStep] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    symbolic: bool | None = None
    symbolic_method: str = ""
    notes: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Attempt:
    outcome: Outcome | None = None
    blocked: list[MatrixBlocked] = field(default_factory=list)
    unsupported: str | None = None


@dataclass(frozen=True)
class Method:
    id: str
    title: str
    actions: tuple[Action, ...]
    pattern: str
    requires: str
    limits: str
    run: Callable[[Problem], Attempt]


# -- shared helpers ---------------------------------------------------------------------------
class SymbolicTimeout(Exception):
    """A symbolic (SymPy) step used its share of the time budget."""


@contextmanager
def symbolic_budget(p: Problem, fraction: float = 0.35) -> Iterator[None]:
    """Bound a SymPy call, which cannot otherwise be interrupted, to part of what is left of the
    deadline so the numeric checks still have time. Works in the main thread (the sandbox child);
    elsewhere the call is unbounded and the sandbox timeout applies."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    seconds = max(0.5, (p.deadline - time.monotonic()) * fraction)

    def expired(signum: int, frame: Any) -> None:
        raise SymbolicTimeout

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def matrices_in(terms: Terms) -> set[str]:
    names: set[str] = set()
    for t in terms:
        for f in t.factors:
            names.update([f.name] if isinstance(f, Sym) else f.names)
    return names


def with_derivatives(p: Problem) -> dict[str, sp.Matrix]:
    """Explicit matrices plus the entries of the derivatives of the time-dependent ones, keyed A, A', A''."""
    out = dict(p.explicit)
    for name in p.setup.dependent:
        if name in p.explicit:
            for d in range(1, MAX_DERIV + 1):
                out[name + "'" * d] = p.setup.derivative_entries(name, d)
    return out


def _all_explicit(names: set[str], explicit: dict[str, sp.Matrix]) -> bool:
    return all(n in explicit for n in names)


def _entries(m: sp.Matrix) -> list[list[str]]:
    return [[str(m[i, j]) for j in range(m.cols)] for i in range(m.rows)]


def _simplified(m: sp.Matrix) -> sp.Matrix:
    return m.applyfunc(lambda e: sp.simplify(e) if sp.count_ops(e) < 200 else e)


def explicit_factor(f: Factor, explicit: dict[str, sp.Matrix]) -> sp.Matrix:
    if isinstance(f, Sym):
        return explicit[f.name + "'" * f.deriv] ** f.power
    if isinstance(f, Frechet):
        raise NotImplementedError("a Frechet factor has no direct SymPy value")
    total = sp.zeros(*explicit[f.arg[0][0]].shape)
    for name, c in f.arg:
        total += c * explicit[name]
    return total.exp()


def explicit_product(
    factors: tuple[Factor, ...], explicit: dict[str, sp.Matrix], size: int
) -> sp.Matrix:
    out = sp.eye(size)
    for f in factors:
        out = out * explicit_factor(f, explicit)
    return out


def explicit_value(
    terms: Terms, explicit: dict[str, sp.Matrix], size: int | None = None
) -> sp.Matrix:
    total: sp.Matrix | None = None
    for t in terms:
        product: sp.Matrix | None = None
        for f in t.factors:
            value = explicit_factor(f, explicit)
            product = value if product is None else product * value
        if product is None:  # an exponential that vanished: the identity
            assert size is not None  # noqa: S101 - callers pass the size when this can happen
            product = sp.eye(size)
        piece = t.coeff * product
        total = piece if total is None else total + piece
    assert total is not None  # noqa: S101
    return total


def _numeric_candidate(p: Problem, value: sp.Matrix) -> Callable[[Env], mp.matrix]:
    """The mpmath value of an explicit SymPy matrix whose entries may contain scalar symbols."""

    def at(env: Env) -> mp.matrix:
        out = mp.matrix(*value.shape)
        for i in range(value.rows):
            for j in range(value.cols):
                out[i, j] = p.ev.scalar(sp.sympify(value[i, j]), env.scalars)
        return out

    return at


def _terms_eval(p: Problem, terms: Terms) -> Callable[[Env], mp.matrix]:
    return lambda env: p.ev.terms(terms, env)


# -- evaluate ---------------------------------------------------------------------------------
def run_exp_explicit(p: Problem) -> Attempt:
    names = matrices_in(p.terms)
    if not _all_explicit(names, p.explicit):
        missing = sorted(n for n in names if n not in p.explicit)
        return Attempt(
            unsupported=f"to evaluate, give the entries of every matrix (missing: {', '.join(missing)})"
        )
    try:
        with symbolic_budget(p):
            value = _simplified(explicit_value(p.terms, p.explicit))
    except (NotImplementedError, ValueError, sp.matrices.exceptions.MatrixError):
        return Attempt(unsupported="SymPy could not find a closed form for this matrix exponential")
    except SymbolicTimeout:
        return Attempt(
            unsupported="SymPy did not find a closed form for this matrix exponential in time"
        )
    count = check_equal(
        p.setup, p.ev, p.integrand, _numeric_candidate(p, value), deadline=p.deadline
    )
    return Attempt(
        Outcome(
            "explicit",
            str(to_latex(value)),
            count,
            "exponentials by mpmath.expm at sampled parameter values (30 digits), independent of "
            "the Jordan-form route that produced the result",
            entries=_entries(value),
            steps=[
                MatrixStep(
                    method="exp_explicit",
                    title="Evaluate with explicit entries",
                    before_latex=render_terms(p.terms),
                    after_latex=str(to_latex(value)),
                    note="exponentials through the Jordan form; no diagonalisability assumed",
                )
            ],
        )
    )


# -- simplify ---------------------------------------------------------------------------------
def run_merge(p: Problem) -> Attempt:
    canon = canonicalize(p.terms, p.comm)
    blocked = [
        MatrixBlocked(method="exp_merge_commuting", title=t, needs=n, latex=x)
        for t, n, x in canon.blocked
    ]
    if canon.terms == p.terms:
        return Attempt(blocked=blocked)
    count = check_equal(
        p.setup, p.ev, p.integrand, _terms_eval(p, canon.terms), deadline=p.deadline
    )
    symbolic = series_terms_equal(p.terms, canon.terms, p.comm)
    steps = [
        MatrixStep(
            method="exp_merge_commuting",
            title=e.title,
            before_latex=e.before,
            after_latex=e.after,
            note=f"uses {e.assumption}" if e.assumption else "holds for any matrices",
        )
        for e in canon.events
    ]
    if not steps:
        steps = [
            MatrixStep(
                method="exp_merge_commuting",
                title="Order a matrix before the exponential of itself",
                before_latex=render_terms(p.terms),
                after_latex=render_terms(canon.terms),
                note="a function of M commutes with M",
            )
        ]
    return Attempt(
        Outcome(
            "expression",
            render_terms(canon.terms),
            count,
            "random complex matrices of the declared sizes, respecting the declared commutation",
            steps=steps,
            assumptions=canon.assumptions,
            symbolic=symbolic,
            symbolic_method=_SERIES,
        ),
        blocked,
    )


_SERIES = (
    "both sides expanded as formal power series in the matrices (free algebra modulo the declared "
    "commutations) and compared through total degree 4; exact, but not a proof for all degrees"
)


# -- differentiate ----------------------------------------------------------------------------
def derive_terms(
    terms: Terms,
    var: sp.Symbol,
    comm: Commutation,
    dependent: frozenset[str] = frozenset(),
    *,
    frechet: bool = False,
) -> Canonical | str:
    """Ordered product rule. A string is the reason the derivative is out of scope.

    d/dt exp(X) is X' exp(X) only if X commutes with X'; otherwise it is the Frechet derivative
    (an integral), which is produced only when `frechet` is set."""
    out: list[Term] = []
    assumptions: list[str] = []
    for t in terms:
        if t.coeff.has(var):
            out.append(Term(sp.diff(t.coeff, var), t.factors))
        for j, f in enumerate(t.factors):
            if not depends_on(f, var, dependent):
                continue
            before, after = t.factors[:j], t.factors[j + 1 :]
            if isinstance(f, Sym):
                if f.deriv + 1 > MAX_DERIV:
                    return (
                        "derivatives of a time-dependent matrix beyond the second are not supported"
                    )
                for i in range(f.power):  # (A^p)' = sum A^i A' A^(p-1-i)
                    mid = (
                        *(Sym(f.name, 1, f.deriv),) * i,
                        Sym(f.name, 1, f.deriv + 1),
                        *(Sym(f.name, 1, f.deriv),) * (f.power - 1 - i),
                    )
                    out.append(Term(t.coeff, (*before, *mid, *after)))
                continue
            if isinstance(f, Frechet):
                return "the derivative of an unevaluated integral is not supported"
            reason = _exp_commutes_with_derivative(f, var, comm, assumptions, dependent)
            if reason is None:
                for name, c in f.arg:
                    dc = sp.diff(c, var)
                    if dc != 0:
                        out.append(Term(t.coeff * dc, (*before, Sym(name), f, *after)))
                    if name in dependent:
                        out.append(Term(t.coeff * c, (*before, Sym(name, 1, 1), f, *after)))
                continue
            if not frechet:
                return reason
            direction: list[tuple[str, int, sp.Expr]] = []
            for name, c in f.arg:
                dc = sp.diff(c, var)
                if dc != 0:
                    direction.append((name, 0, dc))
                if name in dependent:
                    direction.append((name, 1, c))
            out.append(Term(t.coeff, (*before, Frechet(f, tuple(direction)), *after)))
    canon = canonicalize(_collect_terms(out), comm)
    canon.assumptions = list(dict.fromkeys([*assumptions, *canon.assumptions]))
    return canon


def _collect_terms(terms: list[Term]) -> Terms:
    return collect(tuple(terms))


def _exp_commutes_with_derivative(
    f: Exp,
    var: sp.Symbol,
    comm: Commutation,
    assumptions: list[str],
    dependent: frozenset[str] = frozenset(),
) -> str | None:
    """None when the exponent X commutes with X' (and says why); otherwise the reason it is not
    established. Declared commutation of two matrices holds at all times, so it covers their
    derivatives too."""
    items = f.arg
    for name, _ in items:
        if name in dependent:
            if name in comm.self_commuting:
                assumptions.append(f"{name}(t1) {name}(t2) = {name}(t2) {name}(t1) at all times")
            else:
                return (
                    f"{name} depends on the variable and {name} {name}' = {name}' {name} is not "
                    "established"
                )
    for i, (ni, ci) in enumerate(items):
        for nj, cj in items[i + 1 :]:
            if comm.commutes(ni, nj):
                a, b = sorted((ni, nj))
                assumptions.append(f"{a}{b} = {b}{a}")
            elif ni in dependent or nj in dependent:
                a, b = sorted((ni, nj))
                return f"{a}{b} = {b}{a} is not established for the time-dependent matrices"
            elif sp.simplify(ci * sp.diff(cj, var) - cj * sp.diff(ci, var)) == 0:
                assumptions.append(
                    f"the coefficients of {ni} and {nj} in the exponent are proportional, so the "
                    "exponent commutes with its derivative"
                )
            else:
                return (
                    f"the exponent mixes {ni} and {nj}, which do not commute, with coefficients "
                    "that are not proportional, so it does not commute with its derivative"
                )
    return None


def derivative_series_leg(p: Problem, derived: Terms) -> bool | None:
    """Exact leg: the series of the original, differentiated term by term, against the series of
    the derivative. Time-dependent matrices are expanded as polynomials in the variable first."""
    assert p.var is not None  # noqa: S101
    try:
        original, comm = expand_dependent(p.terms, p.var, p.setup.dependent, p.comm)
        candidate, _ = expand_dependent(derived, p.var, p.setup.dependent, p.comm)
        return poly_equal(
            poly_diff(series_of_terms(original), p.var), series_of_terms(candidate), comm
        )
    except ValueError:
        return None


def run_derivative(p: Problem) -> Attempt:
    assert p.var is not None  # noqa: S101 - calculus checks this before running
    var = p.var
    derived = derive_terms(p.terms, var, p.comm, p.setup.dependent)
    if isinstance(derived, str):
        return Attempt(unsupported=derived)
    candidate = _terms_eval(p, derived.terms)
    count, directions = check_derivative(
        p.setup, p.ev, var, p.integrand, candidate, deadline=p.deadline
    )
    symbolic = derivative_series_leg(p, derived.terms)
    return Attempt(
        Outcome(
            "expression",
            render_terms(derived.terms),
            count,
            f"central difference (step 1e-12, 25 digits; {' and '.join(directions)} direction) of "
            "the original expression, evaluated with mpmath.expm on random matrices",
            steps=[
                MatrixStep(
                    method="exp_derivative",
                    title="Ordered product rule; d/dt exp(X) = X' exp(X)",
                    before_latex=render_terms(p.terms),
                    after_latex=render_terms(derived.terms),
                    note="valid because the exponent commutes with its derivative",
                )
            ],
            assumptions=derived.assumptions,
            symbolic=symbolic,
            symbolic_method=_SERIES
            + " (the series of the original is differentiated term by term)",
        )
    )


# -- integrate: shared analysis ---------------------------------------------------------------
@dataclass
class Template:
    coeff: sp.Expr
    left: tuple[Factor, ...]
    exps: list[tuple[Exp, str, sp.Expr, sp.Expr]]  # (factor, matrix, alpha, beta) in order
    mid: tuple[Factor, ...]
    right: tuple[Factor, ...]

    @property
    def core(self) -> tuple[Factor, ...]:
        if len(self.exps) == 1:
            return (self.exps[0][0],)
        return (self.exps[0][0], *self.mid, self.exps[1][0])


def _affine(s: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
    try:
        poly = sp.Poly(sp.expand(s), var)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None
    a, b = poly.coeff_monomial(var), poly.coeff_monomial(1)
    if a.has(var) or b.has(var):
        return None
    return a, b


def match_template(term: Term, var: sp.Symbol) -> Template | str:
    dep = [i for i, f in enumerate(term.factors) if depends_on(f, var)]
    if term.coeff.has(var):
        return (
            "a scalar function of the variable multiplies the exponentials (that needs integration "
            "by parts, which is not supported)"
        )
    if not 1 <= len(dep) <= 2:
        return f"{len(dep)} exponentials depend on the variable; the supported structure has one or two"
    exps: list[tuple[Exp, str, sp.Expr, sp.Expr]] = []
    for i in dep:
        f = term.factors[i]
        assert isinstance(f, Exp)  # noqa: S101
        if len(f.arg) != 1:
            return (
                "an exponent that adds several matrices and depends on the variable is not supported "
                "(that is a time-ordered exponential)"
            )
        name, s = f.arg[0]
        ab = _affine(s, var)
        if ab is None:
            return "the exponent must be affine in the integration variable"
        exps.append((f, name, ab[0], ab[1]))
    left = term.factors[: dep[0]]
    right = term.factors[dep[-1] + 1 :]
    mid = term.factors[dep[0] + 1 : dep[-1]] if len(dep) == 2 else ()
    return Template(term.coeff, left, exps, mid, right)


def _bound_text(p: Problem) -> str:
    assert p.var is not None and p.lower is not None and p.upper is not None  # noqa: S101
    return f"\\int_{{{to_latex(p.lower)}}}^{{{to_latex(p.upper)}}} {render_terms(p.terms)}\\,d{to_latex(p.var)}"


def _integral_notation(p: Problem) -> str:
    if p.lower is None or p.upper is None:
        return f"\\int {render_terms(p.terms)}\\,d{to_latex(p.var)}"
    return _bound_text(p)


def _definite(p: Problem) -> bool:
    return p.lower is not None and p.upper is not None


# -- integrate: closed forms ------------------------------------------------------------------
def _closed_form(p: Problem, allow_exp: bool, method_id: str, title: str) -> Attempt:
    """Term-by-term antiderivatives: polynomial scalars, and one exponential of an invertible matrix."""
    assert p.var is not None  # noqa: S101
    var = p.var
    antiderivative: list[Term] = []
    blocked: list[MatrixBlocked] = []
    assumptions: list[str] = []
    for term in p.terms:
        dep = [f for f in term.factors if depends_on(f, var)]
        if not dep:
            if not term.coeff.is_polynomial(var):
                return Attempt(
                    unsupported="the scalar coefficient is not a polynomial in the variable"
                )
            antiderivative.append(Term(sp.integrate(sp.expand(term.coeff), var), term.factors))
            continue
        if not allow_exp or len(dep) != 1:
            return Attempt()
        tmpl = match_template(term, var)
        if isinstance(tmpl, str):
            return Attempt()
        _, name, alpha, _beta = tmpl.exps[0]
        needs = []
        if name not in p.invertible:
            needs.append(f"{name} is invertible")
        if alpha.is_zero is not False:
            needs.append(f"{to_latex(alpha)} \\neq 0")
        if needs:
            blocked.append(
                MatrixBlocked(
                    method=method_id,
                    title=title,
                    needs=" and ".join(needs) + " (not established)",
                    latex=render_factors([tmpl.exps[0][0]]),
                )
            )
            return Attempt(blocked=blocked)
        assumptions.append(f"{name} is invertible ({p.invertible[name]})")
        if not alpha.is_number:
            assumptions.append(f"{to_latex(alpha)} \\neq 0 (follows from the declared assumptions)")
        e1 = tmpl.exps[0][0]
        antiderivative.append(
            Term(
                term.coeff / alpha,
                (*tmpl.left, Sym(name, -1), e1, *tmpl.right),
            )
        )
    g = canonicalize(_collect_terms(antiderivative), p.comm)
    # the antiderivative is differentiated in the free algebra: exact, and independent of quadrature
    back = derive_terms(g.terms, var, p.comm)
    expected = canonicalize(p.terms, p.comm).terms
    symbolic = None if isinstance(back, str) else back.terms == expected
    symbolic_method = (
        "d/dt of the antiderivative, taken with the ordered product rule in the free algebra "
        "(exact), equals the integrand"
    )
    if _definite(p):
        assert p.lower is not None and p.upper is not None  # noqa: S101
        top = substitute(g.terms, var, p.upper)
        bottom = tuple(Term(-t.coeff, t.factors) for t in substitute(g.terms, var, p.lower))
        value = canonicalize(_collect_terms([*top, *bottom]), p.comm)
        cand = _terms_eval(p, value.terms)
        count = check_integral(
            p.setup,
            p.ev,
            var,
            p.integrand,
            p.lower,
            p.upper,
            p.shape,
            lambda env, x: close(x, cand(env)),
            deadline=p.deadline,
        )
        result_terms = value.terms
        method_note = "the antiderivative evaluated at the bounds"
        numeric_method = (
            "adaptive quadrature (mpmath Gauss-Legendre) of the integrand on random matrices"
        )
    else:
        result_terms = g.terms
        gcand = _terms_eval(p, g.terms)
        count, directions = check_derivative(
            p.setup, p.ev, var, gcand, p.integrand, deadline=p.deadline
        )
        method_note = "an antiderivative (a constant matrix may be added)"
        numeric_method = (
            f"central difference of the antiderivative against the integrand "
            f"({' and '.join(directions)} direction), random complex matrices"
        )
    return Attempt(
        Outcome(
            "expression",
            render_terms(result_terms),
            count,
            numeric_method,
            steps=[
                MatrixStep(
                    method=method_id,
                    title=title,
                    before_latex=_integral_notation(p),
                    after_latex=render_terms(result_terms),
                    note=method_note,
                )
            ],
            assumptions=list(dict.fromkeys([*assumptions, *g.assumptions])),
            symbolic=symbolic,
            symbolic_method=symbolic_method,
        ),
        blocked,
    )


def run_integrate_polynomial(p: Problem) -> Attempt:
    return _closed_form(
        p, False, "integrate_polynomial", "Integrate the scalar factor; the matrices are constant"
    )


def run_integral_invertible(p: Problem) -> Attempt:
    return _closed_form(
        p,
        True,
        "exp_integral_invertible",
        "Integrate e^{(at+b)A} using A invertible: (aA)^{-1} e^{(at+b)A}",
    )


# -- integrate: Sylvester reduction and block exponential ------------------------------------
@dataclass
class Reduction:
    tmpl: Template
    core: Terms
    p_name: str
    q_name: str | None
    alpha: sp.Expr
    gamma: sp.Expr
    equation_latex: str
    outcome: Outcome  # the verified equation, used when nothing can be evaluated


def _names_map(p: Problem) -> dict[str, str]:
    return {UNKNOWN: "\\mathcal{X}" if "X" in p.setup.shapes else "X"}


def _core_value_at(p: Problem, core: Terms, value: sp.Expr) -> Terms:
    assert p.var is not None  # noqa: S101
    return substitute(core, p.var, value)


def _lhs_eval(p: Problem, tmpl: Template) -> Callable[[Env, mp.matrix], mp.matrix]:
    _, pn, a1, _ = tmpl.exps[0]

    def lhs(env: Env, x: mp.matrix) -> mp.matrix:
        out = p.ev.scalar(a1, env.scalars) * env.mats[pn] * x
        if len(tmpl.exps) == 2:
            _, qn, a2, _ = tmpl.exps[1]
            out = out + p.ev.scalar(a2, env.scalars) * x * env.mats[qn]
        return out

    return lhs


def build_reduction(p: Problem) -> Reduction | str:
    assert p.var is not None and p.lower is not None and p.upper is not None  # noqa: S101
    if len(p.terms) != 1:
        return "a sum of several terms needs each term evaluated; the reduction handles one term"
    tmpl = match_template(p.terms[0], p.var)
    if isinstance(tmpl, str):
        return tmpl
    var = p.var
    core: Terms = (Term(sp.Integer(1), tmpl.core),)
    _, pn, a1, _b1 = tmpl.exps[0]
    qn = tmpl.exps[1][1] if len(tmpl.exps) == 2 else None
    a2 = tmpl.exps[1][2] if len(tmpl.exps) == 2 else sp.Integer(0)
    lhs_terms: list[Term] = [Term(a1, (Sym(pn), Sym(UNKNOWN)))]
    if qn is not None:
        lhs_terms.append(Term(a2, (Sym(UNKNOWN), Sym(qn))))
    top = _core_value_at(p, core, p.upper)
    bottom = tuple(Term(-t.coeff, t.factors) for t in _core_value_at(p, core, p.lower))
    names = _names_map(p)
    equation = f"{render_terms(tuple(lhs_terms), names)} = {render_terms((*top, *bottom))}"

    # numeric: the quadrature X of the core must satisfy the equation (substitution)
    lhs = _lhs_eval(p, tmpl)
    size = _core_shape(p, tmpl)[0]
    low = _core_value_at(p, core, p.lower)

    def rhs(env: Env) -> mp.matrix:
        return p.ev.terms(top, env, size) - p.ev.terms(low, env, size)

    def equation_holds(env: Env, x: mp.matrix) -> bool:
        return close(lhs(env, x), rhs(env))

    count = check_integral(
        p.setup,
        p.ev,
        var,
        _terms_eval(p, core),
        p.lower,
        p.upper,
        _core_shape(p, tmpl),
        equation_holds,
        deadline=p.deadline,
    )
    # exact (through degree 4): the series of X = ∫ core satisfies the equation
    symbolic: bool | None
    try:
        x_poly = poly_integrate(series_of_terms(core), var, p.lower, p.upper)
        lhs_poly: Poly = poly_mul(letter(pn, a1), x_poly)
        if qn is not None:
            lhs_poly = poly_add(lhs_poly, poly_mul(x_poly, letter(qn, a2)))
        core_series = series_of_terms(core)
        rhs_poly = poly_add(
            poly_subs(core_series, var, p.upper), poly_subs(core_series, var, p.lower), -1
        )
        symbolic = poly_equal(lhs_poly, rhs_poly, p.comm)
    except ValueError:
        symbolic = None
    outcome = Outcome(
        "equation",
        None,
        count,
        "the quadrature value X of the integral, put into the equation, on random matrices",
        evaluated=False,
        equation_latex=equation,
        steps=_reduction_steps(p, tmpl, equation, pn, qn),
        symbolic=symbolic,
        symbolic_method=_SERIES + " (the series of X is integrated term by term)",
        notes=["the integral is not evaluated: this is an equation that X satisfies"],
    )
    return Reduction(tmpl, core, pn, qn, a1, a2, equation, outcome)


def _core_shape(p: Problem, tmpl: Template) -> tuple[int, int]:
    n = p.setup.shapes[tmpl.exps[0][1]][0]
    if len(tmpl.exps) == 2:
        m = p.setup.shapes[tmpl.exps[1][1]][0]
        return n, m
    return n, n


def _reduction_steps(
    p: Problem, tmpl: Template, equation: str, pn: str, qn: str | None
) -> list[MatrixStep]:
    form = (
        f"{name_latex(pn)}, {name_latex(qn)} constant"
        if qn is not None
        else f"{name_latex(pn)} constant"
    )
    return [
        MatrixStep(
            method="sylvester_reduction",
            title="Recognise the form e^{(at+b)P} C e^{(ct+d)Q}",
            before_latex=_integral_notation(p),
            after_latex=None,
            note=f"{form}; F' = aPF + cFQ",
        ),
        MatrixStep(
            method="sylvester_reduction",
            title="Integrate F' = aPF + cFQ over the interval",
            before_latex=None,
            after_latex=equation,
            note="a Sylvester equation for X = ∫F; it holds without assuming invertibility",
        ),
    ]


def _unknown_free(p: Problem, red: Reduction) -> tuple[sp.Matrix, sp.Matrix] | None:
    """Kronecker form of the Sylvester operator and the right-hand side, for explicit matrices."""
    names = {red.p_name} | ({red.q_name} if red.q_name else set())
    if not _all_explicit(names | matrices_in(red.core), p.explicit):
        return None
    pm = p.explicit[red.p_name]
    n = pm.rows
    qm = p.explicit[red.q_name] if red.q_name else sp.zeros(n, n)
    m = qm.rows
    omega = red.alpha * sp.kronecker_product(sp.eye(m), pm) + red.gamma * sp.kronecker_product(
        qm.T, sp.eye(n)
    )
    assert p.lower is not None and p.upper is not None  # noqa: S101
    rhs = explicit_value(_core_value_at(p, red.core, p.upper), p.explicit, n) - explicit_value(
        _core_value_at(p, red.core, p.lower), p.explicit, n
    )
    return omega, rhs


def _full_from_core(p: Problem, red: Reduction, x0: sp.Matrix) -> sp.Matrix:
    tmpl = red.tmpl
    left = explicit_product(
        tmpl.left, p.explicit, p.shape[0] if not tmpl.left else _rows_of(p, tmpl.left)
    )
    right_size = x0.cols
    right = explicit_product(tmpl.right, p.explicit, right_size)
    return _simplified(tmpl.coeff * left * x0 * right)


def _rows_of(p: Problem, factors: tuple[Factor, ...]) -> int:
    f = factors[0]
    name = f.name if isinstance(f, Sym) else f.names[0]
    return p.setup.shapes[name][0]


def _full_candidate(
    p: Problem, red: Reduction, x0: Callable[[Env], mp.matrix]
) -> Callable[[Env], mp.matrix]:
    tmpl = red.tmpl

    def at(env: Env) -> mp.matrix:
        x = x0(env)
        if tmpl.left:
            x = p.ev.terms((Term(sp.Integer(1), tmpl.left),), env) * x
        if tmpl.right:
            x = x * p.ev.terms((Term(sp.Integer(1), tmpl.right),), env)
        return p.ev.scalar(tmpl.coeff, env.scalars) * x

    return at


def _judge_full(p: Problem, red: Reduction, x0: Callable[[Env], mp.matrix]) -> Count:
    assert p.var is not None and p.lower is not None and p.upper is not None  # noqa: S101
    cand = _full_candidate(p, red, x0)
    return check_integral(
        p.setup,
        p.ev,
        p.var,
        p.integrand,
        p.lower,
        p.upper,
        p.shape,
        lambda env, x: close(x, cand(env)),
        deadline=p.deadline,
    )


def _trivial_wrapper(tmpl: Template) -> bool:
    return tmpl.coeff == 1 and not tmpl.left and not tmpl.right


def _merge(a: Count, b: Count) -> Count:
    return Count(a.checked + b.checked, a.agreed + b.agreed)


def _need_definite(p: Problem) -> Attempt | None:
    if not _definite(p):
        return Attempt(
            unsupported=(
                "no closed antiderivative was found; give both bounds for a definite integral to "
                "use the Sylvester reduction"
            )
        )
    return None


def _reduction(p: Problem) -> Reduction | str:
    if "reduction" not in p.scratch:
        p.scratch["reduction"] = build_reduction(p)
    result: Reduction | str = p.scratch["reduction"]
    return result


def run_sylvester(p: Problem) -> Attempt:
    assert p.var is not None  # noqa: S101
    if not _definite(p):
        # an indefinite integral of this form has no Sylvester reduction; say why only if it matches
        if len(p.terms) == 1 and not isinstance(match_template(p.terms[0], p.var), str):
            return _need_definite(p) or Attempt()
        return Attempt()
    red = _reduction(p)
    if isinstance(red, str):
        return Attempt(unsupported=red)
    pair = red.p_name + (f" and Q = {red.q_name}" if red.q_name else "")
    p.scratch["structure"] = (
        f"Sylvester form: the integrand is F(t) = e^((at+b)P) C e^((ct+d)Q) with P = {pair}, "
        "all constant, so X = integral of F satisfies aPX + cXQ = F(upper) - F(lower)."
    )
    if red.outcome.count.passed is False or red.outcome.symbolic is False:
        return Attempt(outcome=red.outcome)  # reported as not verified by the caller
    unk = _unknown_free(p, red)
    if unk is None:
        return Attempt()
    omega, rhs = unk
    det = sp.simplify(omega.det())
    if det.is_zero is not False:
        red.outcome.notes.append(
            "the Sylvester operator is singular or its invertibility is not provable here, so the "
            "equation does not determine the integral by itself"
            if det == 0
            else "the invertibility of the Sylvester operator could not be decided for these entries"
        )
        return Attempt()
    rows, cols = rhs.shape
    vec = sp.Matrix([rhs[i, j] for j in range(cols) for i in range(rows)])  # column-major vec
    sol = omega.LUsolve(vec)
    x0 = sp.Matrix(rows, cols, lambda i, j: sol[j * rows + i])
    x0 = _simplified(x0)
    full = _full_from_core(p, red, x0)
    count = _judge_full(p, red, _numeric_candidate(p, x0))
    # exact: the explicit result satisfies the equation entry by entry
    residual = _simplified(
        red.alpha * p.explicit[red.p_name] * x0
        + (x0 * p.explicit[red.q_name] * red.gamma if red.q_name else sp.zeros(*x0.shape))
        - rhs
    )
    outcome = Outcome(
        "explicit",
        str(to_latex(full)),
        count,
        "adaptive quadrature (mpmath) of the integrand against the solved result, random parameter values",
        entries=_entries(full),
        equation_latex=red.equation_latex,
        steps=[
            *red.outcome.steps,
            MatrixStep(
                method="sylvester_reduction",
                title="Solve the Sylvester equation",
                before_latex=red.equation_latex,
                after_latex=str(to_latex(full)),
                note="the Kronecker operator has a nonzero determinant, so the solution is unique",
            ),
        ],
        symbolic=bool(residual == sp.zeros(*residual.shape)),
        symbolic_method=(
            "the explicit result substituted into the Sylvester equation: every entry of the "
            "residual simplifies to zero (exact)"
        ),
        assumptions=[
            "the Sylvester operator X -> aPX + cXQ is invertible (its Kronecker determinant is nonzero)"
        ],
    )
    return Attempt(outcome=outcome)


def _block_matrix(p: Problem, red: Reduction, env: Env) -> mp.matrix:
    pm = env.mats[red.p_name]
    n = pm.rows
    if red.q_name is not None:
        qm = env.mats[red.q_name]
        c = p.ev.terms((Term(sp.Integer(1), red.tmpl.mid),), env) if red.tmpl.mid else mp.eye(n)
    else:
        qm = mp.zeros(n, n)
        c = mp.eye(n)
    m = qm.rows
    k = mp.zeros(n + m, n + m)
    a1 = p.ev.scalar(red.alpha, env.scalars)
    a2 = p.ev.scalar(red.gamma, env.scalars) if red.q_name else mp.mpf(0)
    for i in range(n):
        for j in range(n):
            k[i, j] = -a1 * pm[i, j]
        for j in range(m):
            k[i, n + j] = c[i, j]
    for i in range(m):
        for j in range(m):
            k[n + i, n + j] = a2 * qm[i, j]
    return k


def _x0_block(p: Problem, red: Reduction) -> Callable[[Env], mp.matrix]:
    """X0 = e^{bP}( e^{a b P} [exp(bK)]_12 - e^{a a P} [exp(aK)]_12 ) e^{dQ}  (Van Loan)."""
    tmpl = red.tmpl

    def at(env: Env) -> mp.matrix:
        assert p.lower is not None and p.upper is not None  # noqa: S101
        k = _block_matrix(p, red, env)
        pm = env.mats[red.p_name]
        n = pm.rows
        a1 = p.ev.scalar(red.alpha, env.scalars)
        beta1 = p.ev.scalar(tmpl.exps[0][3], env.scalars)

        def i_of(tau: mp.mpc) -> mp.matrix:
            big = mp.expm(tau * k)
            top_right = big[0:n, n : k.cols]
            return mp.expm(a1 * tau * pm) * top_right

        b = p.ev.scalar(p.upper, env.scalars)
        a = p.ev.scalar(p.lower, env.scalars)
        x = mp.expm(beta1 * pm) * (i_of(b) - i_of(a))
        if len(tmpl.exps) == 2:
            qm = env.mats[red.q_name or ""]
            delta = p.ev.scalar(tmpl.exps[1][3], env.scalars)
            x = x * mp.expm(delta * qm)
        return x

    return at


def _block_latex(p: Problem, red: Reduction) -> str:
    tmpl = red.tmpl
    qn = name_latex(red.q_name) if red.q_name else None
    assert p.lower is not None and p.upper is not None  # noqa: S101

    def paren(e: sp.Expr) -> str:
        text = str(to_latex(e))
        return f"\\left({text}\\right)" if isinstance(e, sp.Add) else text

    neg_alpha = -red.alpha
    c = render_terms((Term(sp.Integer(1), tmpl.mid),)) if tmpl.mid else "I"
    if qn is not None:
        k = (
            "\\begin{pmatrix} "
            f"{render_terms((Term(neg_alpha, (Sym(red.p_name),)),))} & {c} \\\\ 0 & "
            f"{render_terms((Term(red.gamma, (Sym(red.q_name or ''),)),))} \\end{{pmatrix}}"
        )
    else:
        k = (
            "\\begin{pmatrix} "
            f"{render_terms((Term(neg_alpha, (Sym(red.p_name),)),))} & I \\\\ 0 & 0 \\end{{pmatrix}}"
        )

    def block(bound: sp.Expr) -> str:
        scaled = sp.expand(red.alpha * bound)
        e_front = render_factors([Exp(((red.p_name, scaled),))]) if scaled != 0 else ""
        arg = "" if bound == 1 else f"{paren(bound)}\\,"
        return f"{e_front}\\left[\\exp\\left({arg}{k}\\right)\\right]_{{12}}"

    beta1 = tmpl.exps[0][3]
    front = render_factors([Exp(((red.p_name, beta1),))]) + " " if beta1 != 0 else ""
    back = ""
    if len(tmpl.exps) == 2 and tmpl.exps[1][3] != 0:
        back = " " + render_factors([Exp(((red.q_name or "", tmpl.exps[1][3]),))])
    if p.lower == 0:  # the integral from 0 to tau of the block formula is zero at tau = 0
        inner = f"\\left({block(p.upper)}\\right)"
    elif p.upper == 0:
        inner = f"-\\left({block(p.lower)}\\right)"
    else:
        inner = f"\\left({block(p.upper)} - {block(p.lower)}\\right)"
    wrap_left = (render_factors(list(tmpl.left)) + " ") if tmpl.left else ""
    wrap_right = (" " + render_factors(list(tmpl.right))) if tmpl.right else ""
    coeff = "" if tmpl.coeff == 1 else f"{paren(tmpl.coeff)} "
    return f"{coeff}{wrap_left}{front}{inner}{back}{wrap_right}"


def run_block_exponential(p: Problem) -> Attempt:
    if "reduction" not in p.scratch:
        return Attempt()
    red = p.scratch["reduction"]
    if isinstance(red, str):
        return Attempt()
    if red.outcome.count.passed is False or red.outcome.symbolic is False:
        return Attempt()
    x0 = _x0_block(p, red)
    # the block formula must reproduce the quadrature of the full integrand ...
    count = _judge_full(p, red, x0)
    # ... and the equation of the core (already counted in the reduction)
    count = _merge(count, red.outcome.count)
    names = matrices_in(red.core) | {red.p_name} | ({red.q_name} if red.q_name else set())
    steps = list(red.outcome.steps)
    result_latex = _block_latex(p, red)
    kind: ResultKind = "representation"
    entries: list[list[str]] | None = None
    evaluated = False
    notes = [
        "X is expressed through one block-matrix exponential; it is an exact representation, "
        "valid without assuming invertibility or diagonalisability, but it is not an evaluated "
        "entrywise result"
    ]
    symbolic = red.outcome.symbolic
    symbolic_method = red.outcome.symbolic_method
    if _all_explicit(names | matrices_in(p.terms), p.explicit):
        try:
            with symbolic_budget(p):
                explicit = _evaluate_block_explicit(p, red)
        except (
            NotImplementedError,
            ValueError,
            sp.matrices.exceptions.MatrixError,
            SymbolicTimeout,
        ):
            explicit = None
        if explicit is not None:
            x0e, residual_zero = explicit
            full = _full_from_core(p, red, x0e)
            count = _merge(_judge_full(p, red, _numeric_candidate(p, x0e)), red.outcome.count)
            result_latex = str(to_latex(full))
            entries = _entries(full)
            kind, evaluated = "explicit", True
            notes = [
                "evaluated through the block-matrix exponential: no invertibility or "
                "diagonalisability of the matrices is needed"
            ]
            symbolic = residual_zero
            symbolic_method = (
                "the explicit result substituted into the Sylvester equation: every entry of the "
                "residual simplifies to zero (exact). The equation alone does not determine the "
                "integral when its operator is singular; the quadrature check does"
            )
    steps.append(
        MatrixStep(
            method="block_exponential",
            title="Block-matrix exponential (Van Loan)",
            before_latex=red.equation_latex,
            after_latex=result_latex,
            note="exp of [[-aP, C],[0, cQ]] contains the integral in its upper-right block",
        )
    )
    return Attempt(
        Outcome(
            kind,
            result_latex,
            count,
            "adaptive quadrature (mpmath) of the integrand against the block formula, and the "
            "quadrature value substituted into the Sylvester equation, random matrices",
            evaluated=evaluated,
            entries=entries,
            equation_latex=red.equation_latex,
            steps=steps,
            symbolic=symbolic,
            symbolic_method=symbolic_method,
            notes=notes,
        )
    )


def _evaluate_block_explicit(p: Problem, red: Reduction) -> tuple[sp.Matrix, bool | None] | None:
    assert p.lower is not None and p.upper is not None  # noqa: S101
    tmpl = red.tmpl
    pm = p.explicit[red.p_name]
    n = pm.rows
    if red.q_name:
        qm = p.explicit[red.q_name]
        c = explicit_product(tmpl.mid, p.explicit, n)
    else:
        qm, c = sp.zeros(n, n), sp.eye(n)
    m = qm.rows
    k = sp.zeros(n + m, n + m)
    k[:n, :n] = -red.alpha * pm
    k[:n, n:] = c
    k[n:, n:] = red.gamma * qm

    def i_of(tau: sp.Expr) -> sp.Matrix:
        big = (tau * k).exp()
        return (red.alpha * tau * pm).exp() * big[:n, n:]

    _, _, _, beta1 = tmpl.exps[0]
    x0 = (beta1 * pm).exp() * (i_of(p.upper) - i_of(p.lower))
    if len(tmpl.exps) == 2:
        x0 = x0 * (tmpl.exps[1][3] * qm).exp()
    if sum(sp.count_ops(e) for e in x0) > MAX_CLOSED_FORM_OPS:
        return None  # eigenvalues without a short closed form: keep the exact representation
    x0 = _simplified(x0)
    rhs = explicit_value(_core_value_at(p, red.core, p.upper), p.explicit, n) - explicit_value(
        _core_value_at(p, red.core, p.lower), p.explicit, n
    )
    lhs = red.alpha * pm * x0 + (x0 * qm * red.gamma if red.q_name else sp.zeros(*x0.shape))
    residual = _simplified(lhs - rhs)
    return x0, _residual_is_zero(p, residual)


def _residual_is_zero(p: Problem, residual: sp.Matrix) -> bool | None:
    """True when every entry is exactly zero; None when SymPy could not simplify an entry to zero
    but it vanishes numerically at sampled parameters (inconclusive, never a refutation); False
    when it is clearly not zero."""
    if residual == sp.zeros(*residual.shape):
        return True
    import random

    rng = random.Random(20260102)  # noqa: S311 - test points, not security
    with mp.workdps(30):
        for _ in range(3):
            env = make_env(p.setup, p.ev, rng)
            for e in residual:
                if e == 0:
                    continue
                try:
                    value = p.ev.scalar(sp.sympify(e), env.scalars)
                except (ArithmeticError, ValueError, ZeroDivisionError):
                    return False
                if abs(value) > 1e-15:
                    return False
    return None


def run_equation_only(p: Problem) -> Attempt:
    red = p.scratch.get("reduction")
    if red is None or isinstance(red, str):
        return Attempt()
    return Attempt(outcome=red.outcome)


# -- the registry -----------------------------------------------------------------------------
def _run_duhamel(p: Problem) -> Attempt:
    from .duhamel import run_duhamel  # duhamel imports this module

    return run_duhamel(p)


def _run_frechet(p: Problem) -> Attempt:
    from .frechet import run_frechet_derivative  # frechet imports this module

    return run_frechet_derivative(p)


METHODS: tuple[Method, ...] = (
    Method(
        "exp_explicit",
        "Evaluate with explicit entries",
        ("evaluate",),
        "ordered sums and products of explicit matrices and their exponentials",
        "every matrix given by its entries (entries may contain symbols)",
        "SymPy must find the closed form (Jordan form); symbolic entries can fail",
        run_exp_explicit,
    ),
    Method(
        "exp_merge_commuting",
        "Merge exponentials of commuting matrices",
        ("simplify",),
        "adjacent exponentials, or a matrix next to an exponential of itself",
        "the commutation of the matrices involved: declared, or visible in explicit entries",
        "never reorders matrices that are not known to commute; reports the blocked merge instead",
        run_merge,
    ),
    Method(
        "exp_derivative",
        "Ordered product rule for exponentials",
        ("differentiate",),
        "constant matrices times scalar functions of t inside exponentials",
        "each exponent commutes with its own derivative (one matrix, proportional coefficients, "
        "or declared commutation)",
        "when the exponent does not commute with its derivative the Frechet method applies",
        run_derivative,
    ),
    Method(
        "exp_frechet_derivative",
        "Frechet derivative of the exponential",
        ("differentiate",),
        "exp(X(t)) where the exponent X is a sum of scalar functions of t times matrices and "
        "possibly time-dependent matrices A(t)",
        "none: no commutation is assumed; X X' = X' X is reported as blocked",
        "an exact integral representation, evaluated only for explicit entries that SymPy can "
        "exponentiate; second and higher derivatives of the exponential are not supported",
        _run_frechet,
    ),
    Method(
        "integrate_polynomial",
        "Integrate the scalar factor of constant matrices",
        ("integrate",),
        "constant matrix products times a polynomial in the variable",
        "none",
        "the scalar coefficient must be a polynomial in the variable",
        run_integrate_polynomial,
    ),
    Method(
        "exp_integral_invertible",
        "Closed form for ∫ e^{(at+b)A} dt with A invertible",
        ("integrate",),
        "L e^{(at+b)A} R with L, R constant",
        "A declared invertible (or visibly invertible from its entries) and a provably nonzero",
        "one exponential of one matrix, affine in the variable",
        run_integral_invertible,
    ),
    Method(
        "duhamel",
        "Duhamel integral ∫ e^{(as+b)A} M e^{(-as+d)A} ds",
        ("integrate",),
        "one matrix A in both exponentials, exponents in the variable that cancel (a + c = 0)",
        "none for the representation; M commuting with A (declared or from entries) gives the "
        "closed form (upper - lower) M e^{(b+d)A}",
        "a definite integral of a single term with affine exponents; the Sylvester equation of "
        "this form is singular and is never solved; the representation is not an evaluated result",
        _run_duhamel,
    ),
    Method(
        "sylvester_reduction",
        "Sylvester equation for ∫ e^{(at+b)P} C e^{(ct+d)Q} dt",
        ("integrate",),
        "L e^{(at+b)P} C e^{(ct+d)Q} R with constant L, C, R; P, Q constant",
        "none for the equation; solving it needs a nonzero Kronecker determinant (explicit matrices)",
        "a definite integral of a single term; abstract matrices give the equation, not X",
        run_sylvester,
    ),
    Method(
        "block_exponential",
        "Block-matrix exponential representation",
        ("integrate",),
        "the same form, when the Sylvester operator does not determine X",
        "none: no invertibility, no diagonalisability",
        "abstract matrices give an exact representation, not entries",
        run_block_exponential,
    ),
    Method(
        "sylvester_equation",
        "Report the verified equation only",
        ("integrate",),
        "the same form, when nothing could be evaluated",
        "none",
        "the integral is not evaluated",
        run_equation_only,
    ),
)


def select(p: Problem) -> tuple[Method | None, Attempt, list[MatrixBlocked], str | None]:
    """Try the methods for this action in order. Returns (method, attempt, blocked, unsupported)."""
    blocked: list[MatrixBlocked] = []
    reason: str | None = None
    for method in METHODS:
        if p.action not in method.actions:
            continue
        attempt = method.run(p)
        blocked.extend(attempt.blocked)
        if attempt.outcome is not None:
            return method, attempt, blocked, reason
        if attempt.unsupported and reason is None:
            reason = attempt.unsupported
    return None, Attempt(), blocked, reason
