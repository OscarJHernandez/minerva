"""Antiderivatives, each checked by differentiating the answer numerically.

`sympy.integrate` proposes a candidate `F` for `integral of f dx`; it never vouches for itself. The
candidate is accepted only if an independent numerical derivative of `F` (the same 40-digit central
difference as `differentiate.check_derivative`, in the real and, for a variable that may be complex,
the imaginary direction) agrees with `f` at every sampled point. The symbolic test
`simplify(F' - f) == 0` is reported separately as extra evidence and never decides the outcome.

A `Piecewise` candidate (for example `x**n` with the case `n = -1`) is checked branch by branch: an
equality condition is substituted into both `f` and the branch, the generic branch is sampled as is. A
branch whose condition cannot be sampled (an inequality on a parameter or on the variable) makes the
whole result `not_verified`.

Not covered: definite integrals (the fundamental theorem fails for integrands with a singularity
inside the interval, so they need independent quadrature, not this check), and a missing closed form
is reported as `unsupported`: it does not prove that none exists.
"""

from __future__ import annotations

import time

import sympy as sp
from pydantic import BaseModel, Field

from .build import Builder
from .differentiate import (
    RESULT_NODE_FACTOR,
    TIDY_MAX_OPS,
    DerivativeVerification,
    _nodes,
    check_derivative,
)
from .errors import LimitError, MinervaMathError
from .models import ExpressionInput, Limits, Status

SPECIAL_FUNCTIONS: tuple[type[sp.Basic], ...] = (
    sp.erf,
    sp.erfc,
    sp.erfi,
    sp.Ei,
    sp.expint,
    sp.li,
    sp.Si,
    sp.Ci,
    sp.Shi,
    sp.Chi,
    sp.fresnels,
    sp.fresnelc,
    sp.polylog,
    sp.gamma,
    sp.uppergamma,
    sp.lowergamma,
    sp.besselj,
    sp.bessely,
    sp.besseli,
    sp.besselk,
    sp.hyper,
    sp.meijerg,
)


class IntegrateVerification(DerivativeVerification):
    method: str = "central finite difference of the antiderivative, 40 digits"
    symbolic_zero: bool | None = None  # simplify(F' - f) == 0, extra evidence only
    branches_checked: int = 1


class IntegrateResult(BaseModel):
    status: Status
    input_latex: str | None = None
    variable: str | None = None
    notation_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    verification: IntegrateVerification | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: object) -> IntegrateResult:
    return IntegrateResult(status=status, message=message, **extra)


def _branches(candidate: sp.Expr) -> list[tuple[sp.Expr, object]]:
    if isinstance(candidate, sp.Piecewise):
        return [(e, c) for e, c in candidate.args]
    return [(candidate, sp.true)]


def _fixing(a: sp.Expr, b: sp.Expr, var: sp.Symbol) -> dict[sp.Symbol, sp.Expr] | None:
    """The substitution that makes `a == b`, if one parameter determines it uniquely."""
    free = (a - b).free_symbols - {var}
    if len(free) != 1 or var in (a - b).free_symbols:
        return None
    (symbol,) = free
    try:
        solutions = sp.solve(sp.Eq(a, b), symbol)
    except Exception:
        return None
    if len(solutions) != 1 or not solutions[0].is_number:
        return None
    return {symbol: solutions[0]}


def check_antiderivative(f: sp.Expr, var: sp.Symbol, candidate: sp.Expr) -> IntegrateVerification:
    """Numerically differentiate `candidate` and compare with `f`, branch by branch."""
    checked = agreed = 0
    directions: set[str] = set()
    branches = _branches(candidate)
    notes: list[str] = []
    passed = True
    excluded: list[sp.Basic] = []  # earlier "a != b" conditions: later branches hold where a == b
    for expr, cond in branches:
        target, source = f, expr
        substitution: dict[sp.Symbol, sp.Expr] | None = None
        if isinstance(cond, sp.Equality):
            substitution = _fixing(cond.lhs, cond.rhs, var)
        elif isinstance(cond, sp.Unequality):
            excluded.append(cond)  # sampled generically: a random sample avoids a == b
        elif cond is sp.true or cond is True:
            if len(excluded) == 1:
                substitution = _fixing(excluded[0].lhs, excluded[0].rhs, var)
            elif excluded:
                substitution = None
                passed = False
        else:
            passed = False
        needs_substitution = isinstance(cond, sp.Equality) or (
            (cond is sp.true or cond is True) and len(excluded) == 1
        )
        if needs_substitution:
            if substitution is None:
                passed = False
            else:
                target, source = f.subs(substitution), expr.subs(substitution)
        if not passed:
            notes.append(f"the case '{sp.latex(cond)}' cannot be sampled")
            break
        verdict = check_derivative(source, var, target)
        checked += verdict.points_checked
        agreed += verdict.points_agreed
        directions |= set(verdict.directions)
        if not verdict.passed:
            passed = False
            notes.append(verdict.note)
            break
    if passed:
        note = "the numerical derivative of the antiderivative agreed with the integrand everywhere"
    else:
        note = "; ".join(notes) or "verification failed"
    return IntegrateVerification(
        directions=[d for d in ("real", "imaginary") if d in directions] or ["real"],
        steps_checked=len(branches),
        branches_checked=len(branches),
        points_checked=checked,
        points_agreed=agreed,
        passed=passed,
        note=note,
    )


def integrate(data: ExpressionInput, variable: str, limits: Limits) -> IntegrateResult:
    from .ops import parse_expression  # local import: ops imports this module

    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    try:
        f = parse_expression(data, limits)
        symbol = Builder(data.assumptions, limits).symbol(variable)
    except LimitError as exc:
        return _fail(Status.RESOURCE_LIMIT, str(exc), variable=variable)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc), variable=variable)

    base = {"input_latex": sp.latex(f), "variable": variable}
    if symbol.is_integer or symbol.is_rational:
        return _fail(
            Status.UNSUPPORTED,
            f"{variable} is declared integer or rational, so it is not a continuous variable. "
            "Remove that assumption or declare it real.",
            **base,
        )
    notation = str(sp.latex(sp.Integral(f, symbol)))

    try:
        candidate = sp.integrate(f, symbol)
    except Exception:  # SymPy raises many things for what it cannot do
        return _fail(Status.UNSUPPORTED, "Minerva could not integrate this expression.", **base)
    if candidate.has(sp.Integral):
        return _fail(
            Status.UNSUPPORTED,
            "No closed form was found. This does not prove that none exists.",
            notation_latex=notation,
            **base,
        )
    if _nodes(candidate) > limits.max_tree_nodes * RESULT_NODE_FACTOR:
        return _fail(Status.RESOURCE_LIMIT, "The antiderivative became too large to check.", **base)

    verification = check_antiderivative(f, symbol, candidate)
    if not verification.passed:
        return _fail(
            Status.NOT_VERIFIED,
            "An antiderivative was produced but could not be verified, so it is not shown. "
            f"{verification.note}.",
            verification=verification,
            notation_latex=notation,
            **base,
        )

    final = candidate
    if (
        not isinstance(candidate, sp.Piecewise)
        and _nodes(candidate) > 1
        and sp.count_ops(candidate) <= TIDY_MAX_OPS
        and time.monotonic() < deadline
    ):
        tidy = sp.simplify(candidate)
        if tidy != candidate and check_antiderivative(f, symbol, tidy).passed:
            final = tidy

    if sp.count_ops(final) <= TIDY_MAX_OPS and time.monotonic() < deadline:
        try:
            verification.symbolic_zero = bool(sp.simplify(sp.diff(final, symbol) - f) == 0)
        except Exception:
            verification.symbolic_zero = False

    notes = [
        "This is one antiderivative: add an arbitrary constant, which may differ on each "
        "connected piece of the domain."
    ]
    if final.has(*SPECIAL_FUNCTIONS):
        notes.append("The result uses special functions that are not elementary.")
    if symbol not in f.free_symbols:
        notes.append(f"The integrand does not depend on {variable}.")
    return IntegrateResult(
        status=Status.OK,
        notation_latex=notation,
        result_latex=sp.latex(final),
        result_text=str(final),
        verification=verification,
        notes=notes,
        message="Integrated with SymPy; the antiderivative was verified by a numerical derivative.",
        **base,
    )
