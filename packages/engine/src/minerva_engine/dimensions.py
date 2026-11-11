"""Dimensional-consistency checking of physics equations.

Each symbol is mapped to a unit expression in SI base units (kg, m, s, A, K, mol, cd).
`check_equation_dimensions` verifies that both sides have the same dimensions, that all
terms of every sum agree, and that arguments of transcendental functions are
dimensionless. This catches dimensional mistakes; it does not prove an equation is
physically correct.
"""

from __future__ import annotations

import sympy as sp

from .errors import MinervaMathError, ParseError
from .textparse import parse_text

BASE_UNITS = ("kg", "m", "s", "A", "K", "mol", "cd")
_UNIT_SYMBOLS = {u: sp.Symbol(f"u_{u}", positive=True) for u in BASE_UNITS}


class DimensionError(MinervaMathError):
    """Dimensions are inconsistent."""


def _unit(text: str) -> sp.Expr:
    expr = parse_text(text)
    unknown = {s.name for s in expr.free_symbols} - set(BASE_UNITS)
    if unknown:
        raise ParseError(f"unknown unit symbol(s) {sorted(unknown)}")
    return expr.xreplace({sp.Symbol(u): _UNIT_SYMBOLS[u] for u in BASE_UNITS})


def _same(a: sp.Expr, b: sp.Expr) -> bool:
    return bool(sp.simplify(a / b) == 1)


def _dim(expr: sp.Expr, units: dict[sp.Symbol, sp.Expr]) -> sp.Expr:
    if expr.is_number:
        return sp.Integer(1)
    if isinstance(expr, sp.Symbol):
        if expr not in units:
            raise ParseError(f"no unit given for symbol {expr.name}")
        return units[expr]
    if isinstance(expr, sp.Add):
        dims = [_dim(a, units) for a in expr.args]
        for other in dims[1:]:
            if not _same(dims[0], other):
                raise DimensionError("terms of a sum have different dimensions")
        return dims[0]
    if isinstance(expr, sp.Mul):
        result: sp.Expr = sp.Integer(1)
        for a in expr.args:
            result = result * _dim(a, units)
        return result
    if isinstance(expr, sp.Pow):
        if not expr.exp.is_number:
            if not _same(_dim(expr.exp, units), sp.Integer(1)):
                raise DimensionError("exponent must be dimensionless")
            if not _same(_dim(expr.base, units), sp.Integer(1)):
                raise DimensionError("symbolic power of a dimensional quantity")
            return sp.Integer(1)
        return _dim(expr.base, units) ** expr.exp
    if isinstance(expr, sp.Abs):
        return _dim(expr.args[0], units)
    if isinstance(expr, sp.Function) and len(expr.args) == 1:
        if not _same(_dim(expr.args[0], units), sp.Integer(1)):
            raise DimensionError(f"argument of {expr.func.__name__} must be dimensionless")
        return sp.Integer(1)
    raise ParseError(f"unsupported construct in dimensional analysis: {type(expr).__name__}")


def check_equation_dimensions(lhs: str, rhs: str, units: dict[str, str]) -> None:
    """Raise `DimensionError` (or `ParseError`) unless `lhs` and `rhs` agree dimensionally."""
    unit_map = {sp.Symbol(name): _unit(text) for name, text in units.items()}
    left = _dim(parse_text(lhs), unit_map)
    right = _dim(parse_text(rhs), unit_map)
    if not _same(left, right):
        raise DimensionError("the two sides have different dimensions")
