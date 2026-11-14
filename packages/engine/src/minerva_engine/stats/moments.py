"""Moments of a probability family, derived by integration or summation and checked by quadrature.

The derivation uses SymPy's `integrate` (continuous) and `summation` (discrete), never a stored formula.
A binomial-type size parameter (symbolic upper limit) is handled by fitting the polynomial in the size
from exact sums at small sizes, which is justified because the k-th raw moment is a polynomial of degree
k in the size, and verified at further sizes. Before anything is derived, absolute convergence of the
moment is established numerically (`Numeric.absolute_moment_exists`): a moment of the Cauchy family is
`convergence_not_established`, whatever a principal value would say.
"""

from __future__ import annotations

import time

import sympy as sp

from ..errors import LimitError
from .families import Family, x

t = sp.Symbol("t", integer=True, nonnegative=True)


class NoClosedForm(Exception):
    """SymPy returned no closed form (an unevaluated integral or sum, or a Piecewise)."""


def _clean(expr: sp.Expr) -> sp.Expr:
    result = sp.simplify(sp.gammasimp(expr))
    if result.has(sp.Integral, sp.Sum, sp.Piecewise, sp.zoo, sp.nan):
        raise NoClosedForm
    return result


def _size_symbol(upper: sp.Expr) -> sp.Symbol | None:
    for s in upper.free_symbols:
        if isinstance(s, sp.Symbol) and s.is_integer:
            return s
    return None


def raw_moment(family: Family, parametrization: str | None, k: int, deadline: float) -> sp.Expr:
    symbols = {p.name: family.symbols[p.name] for p in family.params(parametrization)}
    variable = t if family.discrete else x
    density = family.density(symbols, variable)
    lower, upper = family.support(symbols)
    if time.monotonic() > deadline:
        raise LimitError("time budget exhausted")
    if family.discrete:
        size = _size_symbol(upper)
        if size is not None:
            return _polynomial_in_size(density, variable, lower, upper, size, k)
        return _clean(sp.summation(variable**k * density, (variable, lower, upper)))
    return _clean(sp.integrate(variable**k * density, (variable, lower, upper), conds="none"))


def _polynomial_in_size(
    density: sp.Expr, variable: sp.Symbol, lower: sp.Expr, upper: sp.Expr, size: sp.Symbol, k: int
) -> sp.Expr:
    """E[X^k] for a size parameter n: a polynomial of degree k in n, fitted and then verified."""

    def exact(n: int) -> sp.Expr:
        terms = [
            sp.expand((j**k * density).subs({variable: j, size: n}))
            for j in range(int(lower), n + 1)
        ]
        return sp.expand(sp.Add(*terms))

    points = [(n, exact(n)) for n in range(1, k + 2)]
    fitted = sp.expand(sp.interpolate(points, size))
    for n in (k + 2, k + 3):
        if sp.simplify(fitted.subs(size, n) - exact(n)) != 0:
            raise NoClosedForm
    return sp.factor(fitted)
