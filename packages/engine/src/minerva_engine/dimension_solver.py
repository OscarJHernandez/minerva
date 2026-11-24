"""The dimension of a quantity nobody gave a unit for, found from the equation it is in.

`dimensions.dimension_of` checks an equation whose symbols all have a dimension. Here some symbols (the
`unknown` ones) have none, and the equation itself says what each must be. A dimension is a vector of
seven exponents; for an expression it is *affine* in the unknowns' vectors: a constant vector plus, for
each unknown, a rational multiple of that unknown's vector. Products add, a rational power scales. Every
sum (its terms agree), every function argument (dimensionless) and the two sides (agree) then give one
linear equation, the same for each of the seven exponents, solved exactly over the rationals.

The outcome is one of three, never a guess:

* `determined`: one solution, the dimension of each unknown;
* `inconsistent`: no assignment makes the equation consistent (the first condition that breaks it is named);
* `underdetermined`: some unknowns stay free (`a b = F`: only the product is fixed).

An unknown under a symbolic power or inside an exponent is refused (`UnsupportedDimensionConstruct`).

This module only proposes. `pi_groups.check_units` puts the dimensions it returns into the ordinary checker
(`dimension_of` on both sides), which shares no code with this one, and shows them only if that accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import sympy as sp

from .dimensions import ZERO, Dim, UnsupportedDimensionConstruct, dim_latex, dim_text
from .errors import ParseError

Outcome = Literal["determined", "inconsistent", "underdetermined"]


@dataclass(frozen=True)
class Affine:
    """`constant + sum(coefficient * dimension of unknown)`: the dimension of an expression."""

    constant: Dim = ZERO
    coefficients: tuple[tuple[str, Fraction], ...] = ()

    def __add__(self, other: Affine) -> Affine:
        merged = dict(self.coefficients)
        for name, c in other.coefficients:
            merged[name] = merged.get(name, Fraction(0)) + c
        return Affine(
            tuple(a + b for a, b in zip(self.constant, other.constant, strict=True)),
            tuple(sorted((n, c) for n, c in merged.items() if c != 0)),
        )

    def scaled(self, factor: Fraction) -> Affine:
        return Affine(
            tuple(x * factor for x in self.constant),
            tuple((n, c * factor) for n, c in self.coefficients if c * factor != 0),
        )

    def __sub__(self, other: Affine) -> Affine:
        return self + other.scaled(Fraction(-1))


def affine_text(a: Affine) -> str:
    """`kg m [k]^2`: the constant part, then the unknowns' dimensions in brackets with their powers."""
    parts = [] if a.constant == ZERO else [dim_text(a.constant)]
    for name, c in a.coefficients:
        parts.append(f"[{name}]" if c == 1 else f"[{name}]^{c}")
    return " ".join(parts) or "1"


def affine_latex(a: Affine) -> str:
    parts = [] if a.constant == ZERO else [dim_latex(a.constant)]
    for name, c in a.coefficients:
        base = rf"\left[{name}\right]"
        parts.append(base if c == 1 else f"{base}^{{{c}}}")
    return r"\,".join(parts) or "1"


@dataclass
class Condition:
    """`left = right` as dimensions (`right` is None: `left` must be dimensionless)."""

    kind: str  # sum | function | sides
    left_expr: sp.Expr
    left: Affine
    right_expr: sp.Expr | None = None
    right: Affine = field(default_factory=Affine)

    def difference(self) -> Affine:
        return self.left - self.right


@dataclass
class Solution:
    outcome: Outcome
    dimensions: dict[str, Dim] = field(default_factory=dict)  # `determined`
    conflict: Condition | None = None  # `inconsistent`: the first condition that cannot hold
    free: list[str] = field(default_factory=list)  # `underdetermined`


def _rational(value: sp.Expr) -> Fraction:
    if isinstance(value, sp.Rational):
        return Fraction(int(value.p), int(value.q))
    raise ParseError("exponents in units and dimensions must be rational numbers")


def _walk(
    expr: sp.Expr, known: dict[str, Dim], unknown: frozenset[str], out: list[Condition]
) -> Affine:
    """The affine dimension of `expr`; the conditions its sums and functions impose go into `out`."""
    if expr.is_number:
        return Affine()
    if isinstance(expr, sp.Symbol):
        if expr.name in unknown:
            return Affine(ZERO, ((expr.name, Fraction(1)),))
        if expr.name in known:
            return Affine(known[expr.name])
        raise ParseError(f"no unit given for symbol {expr.name}")
    if isinstance(expr, sp.Add):
        parts = [_walk(a, known, unknown, out) for a in expr.args]
        for arg, part in zip(expr.args[1:], parts[1:], strict=True):
            out.append(Condition("sum", expr.args[0], parts[0], arg, part))
        return parts[0]
    if isinstance(expr, sp.Mul):
        total = Affine()
        for a in expr.args:
            total = total + _walk(a, known, unknown, out)
        return total
    if isinstance(expr, sp.Pow):
        if not expr.exp.is_number:
            if {s.name for s in expr.free_symbols} & unknown:
                raise UnsupportedDimensionConstruct(
                    "an unknown cannot be found from a symbolic power or exponent"
                )
            out.append(Condition("function", expr.exp, _walk(expr.exp, known, unknown, out)))
            out.append(Condition("function", expr.base, _walk(expr.base, known, unknown, out)))
            return Affine()
        return _walk(expr.base, known, unknown, out).scaled(_rational(expr.exp))
    if isinstance(expr, sp.Abs):
        return _walk(expr.args[0], known, unknown, out)
    if isinstance(expr, sp.Function) and len(expr.args) == 1:
        out.append(Condition("function", expr.args[0], _walk(expr.args[0], known, unknown, out)))
        return Affine()
    raise UnsupportedDimensionConstruct(
        f"unsupported construct in dimensional analysis: {type(expr).__name__}"
    )


def solve_unknown_dimensions(
    lhs: sp.Expr, rhs: sp.Expr, known: dict[str, Dim], unknown: list[str]
) -> Solution:
    """Solve for the dimension of every name in `unknown` (all of them appear in `lhs = rhs`)."""
    names = list(unknown)
    conditions: list[Condition] = []
    left = _walk(lhs, known, frozenset(names), conditions)
    right = _walk(rhs, known, frozenset(names), conditions)
    conditions.append(Condition("sides", lhs, left, rhs, right))

    # Add the conditions one at a time: the first that makes the system inconsistent is the one named.
    rows: list[Affine] = []
    for condition in conditions:
        rows.append(condition.difference())
        reduced, pivots = _rref(rows, names)
        if any(p >= len(names) for p in pivots):
            return Solution("inconsistent", conflict=condition)

    n = len(names)
    free = [names[j] for j in range(n) if j not in pivots]
    for i, p in enumerate(
        pivots
    ):  # a pivot unknown that still depends on a free one is not fixed either
        if any(reduced[i, j] != 0 for j in range(n) if j not in pivots) and names[p] not in free:
            free.append(names[p])
    if free:
        return Solution("underdetermined", free=[x for x in names if x in free])
    dims: dict[str, Dim] = {}
    for i, p in enumerate(pivots):
        dims[names[p]] = tuple(
            Fraction(int(v.p), int(v.q)) for v in (reduced[i, n + k] for k in range(len(ZERO)))
        )
    return Solution("determined", dims)


def _rref(rows: list[Affine], names: list[str]) -> tuple[sp.Matrix, tuple[int, ...]]:
    def entry(r: Affine, name: str) -> sp.Rational:
        c = dict(r.coefficients).get(name, Fraction(0))
        return sp.Rational(c.numerator, c.denominator)

    matrix = sp.Matrix(
        [
            [entry(r, n) for n in names]
            + [sp.Rational(-c.numerator, c.denominator) for c in r.constant]
            for r in rows
        ]
    )
    reduced, pivots = matrix.rref()
    return reduced, tuple(int(p) for p in pivots)
