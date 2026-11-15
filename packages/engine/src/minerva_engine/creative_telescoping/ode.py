"""The ODE for I(a) and its solution, with the solution checked against the equation."""

from __future__ import annotations

import sympy as sp

from .operators import DifferentialOperator

_I = sp.Function("I")


def equation(operator: DifferentialOperator, rhs: sp.Expr) -> sp.Eq:
    a = operator.var
    lhs = sum(
        (c * sp.Derivative(_I(a), (a, k)) if k else c * _I(a))
        for k, c in enumerate(operator.coefficients)
        if c != 0
    )
    return sp.Eq(lhs, rhs)


def general_solution(operator: DifferentialOperator, rhs: sp.Expr) -> sp.Expr | None:
    """The general solution `I(a) = ...` with constants C1, ..., or None if SymPy finds none."""
    a = operator.var
    try:
        solved = sp.dsolve(equation(operator, rhs), _I(a))
    except Exception:  # NotImplementedError, and the many things SymPy raises for what it cannot do
        return None
    if isinstance(solved, list) or not isinstance(solved, sp.Eq):
        return None
    candidate = solved.rhs
    if candidate.has(sp.Integral, sp.Derivative, _I):
        return None
    return candidate


def satisfies(operator: DifferentialOperator, rhs: sp.Expr, candidate: sp.Expr) -> bool:
    """Exact: L[candidate] - rhs simplifies to zero. (Numerical legs are the caller's.)"""
    residual = operator.apply(candidate) - rhs
    try:
        return bool(sp.simplify(residual) == 0)
    except Exception:
        return False


def constants_of(expr: sp.Expr) -> list[sp.Symbol]:
    found = [s for s in expr.free_symbols if s.name.startswith("C") and s.name[1:].isdigit()]
    return sorted(found, key=lambda s: int(s.name[1:]))
