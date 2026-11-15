"""Dimensional consistency of vector equations (units only).

The scalar checker in `dimensions.py` reads `x = y` over numbers. Here an equation may contain the
operators of vector calculus, so the dimension of each operator is part of the check: the gradient, the
divergence and the curl divide by a length, the Laplacian by a length squared, a time derivative by a time,
a dot or cross product multiplies the dimensions of its factors. Both sides of an equation must agree,
every sum must add like with like, and the arguments of transcendental functions must be dimensionless.

This catches a missing `1/c^2` or a wrong power of the permittivity. It proves nothing about the physics: a
match means the equation is dimensionally consistent, nothing more.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from ..dimensions import (
    ZERO,
    Dim,
    DimensionError,
    UnsupportedDimensionConstruct,
    dim_div,
    dim_mul,
    dim_pow,
    read_unit,
)
from ..errors import ParseError
from ..models import Limits
from .expressions import Adv, Cross, Curl, Div, Dot, Grad, Lap, Pd, Pos
from .parse import parse_vector_text

LENGTH: Dim = tuple(Fraction(1 if u == "m" else 0) for u in ("kg", "m", "s", "A", "K", "mol", "cd"))
TIME: Dim = tuple(Fraction(1 if u == "s" else 0) for u in ("kg", "m", "s", "A", "K", "mol", "cd"))


def vector_dimension(expr: sp.Expr, units: dict[str, Dim]) -> Dim:
    """The dimension of a vector expression; raises `DimensionError` or
    `UnsupportedDimensionConstruct`."""
    if isinstance(expr, sp.Symbol):
        if expr.name not in units:
            raise UnsupportedDimensionConstruct(f"no unit was given for {expr.name}")
        return units[expr.name]
    if expr.is_Number:
        return ZERO
    if isinstance(expr, Grad | Div | Curl):
        return dim_div(vector_dimension(expr.args[0], units), LENGTH)
    if isinstance(expr, Lap):
        return dim_div(dim_div(vector_dimension(expr.args[0], units), LENGTH), LENGTH)
    if isinstance(expr, Pd):
        return dim_div(vector_dimension(expr.args[0], units), TIME)
    if isinstance(expr, Dot | Cross):
        return dim_mul(vector_dimension(expr.args[0], units), vector_dimension(expr.args[1], units))
    if isinstance(expr, Adv):
        return dim_div(
            dim_mul(vector_dimension(expr.args[0], units), vector_dimension(expr.args[1], units)),
            LENGTH,
        )
    if isinstance(expr, Pos):
        return LENGTH
    if isinstance(expr, sp.Add):
        dims = [(a, vector_dimension(a, units)) for a in expr.args if a != 0]
        first = dims[0][1]
        if any(d != first for _, d in dims):
            raise DimensionError("the terms of a sum have different dimensions", "sum", dims)
        return first
    if isinstance(expr, sp.Mul):
        result = ZERO
        for factor in expr.args:
            result = dim_mul(result, vector_dimension(factor, units))
        return result
    if isinstance(expr, sp.Pow):
        exponent = expr.args[1]
        if not isinstance(exponent, sp.Rational):
            raise UnsupportedDimensionConstruct("a power needs a rational exponent")
        base = vector_dimension(expr.args[0], units)
        return dim_pow(base, Fraction(int(exponent.p), int(exponent.q)))
    # a function of dimensionless arguments: sin, exp, ...
    for arg in expr.args:
        if vector_dimension(arg, units) != ZERO:
            raise DimensionError(
                f"the argument of {expr.func.__name__} must be dimensionless", "function"
            )
    return ZERO


def check_vector_equation(
    lhs: str,
    rhs: str,
    fields: dict[str, str],
    units: dict[str, str],
    limits: Limits | None = None,
) -> Dim:
    """Read `lhs = rhs` (text syntax) and check that both sides have the same dimension, which is
    returned. Raises `DimensionError` on a mismatch and `ParseError` on input it cannot read."""
    limits = limits or Limits()
    vectors = [n for n, k in fields.items() if k == "vector"]
    names = list(units)
    known = {n: read_unit(text) for n, text in units.items()}
    sides = [
        parse_vector_text(
            text,
            vectors=vectors,
            names=names,
            directions=(),
            assumptions={"t": ["real"]},
            limits=limits,
        )
        for text in (lhs, rhs)
    ]
    for side in sides:
        stray = {s.name for s in side.free_symbols} - set(units) - {"t"}
        if stray:
            raise ParseError(f"no unit was given for {', '.join(sorted(stray))}")
    if sides[0] == 0 or sides[1] == 0:  # a zero has every dimension
        return vector_dimension(sides[1] if sides[0] == 0 else sides[0], known)
    left, right = (vector_dimension(s, known) for s in sides)
    if left != right:
        raise DimensionError("the two sides have different dimensions", "sides")
    return left
