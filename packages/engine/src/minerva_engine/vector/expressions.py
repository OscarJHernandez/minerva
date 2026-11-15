"""Typed vector expressions as SymPy trees.

There is no second symbolic system: a vector expression is an ordinary SymPy tree whose vector
fields are noncommutative symbols and whose operators are inert heads (`Grad`, `Div`, `Curl`, `Lap`,
`Dot`, `Cross`, `Pd`, `Adv`, `Pos`, `Unit`). The heads never evaluate; they are only given meaning
by `lower.py` (exact, through scale factors) and `numeric.py` (an independent Cartesian reading).

`kind_of` types every node as `scalar` or `vector`. A product of two vectors, a sum of a scalar and
a vector, a gradient of a vector and the like are `VectorTypeError`s (a `ParseError`): the parsers
never guess which product was meant.
"""

from __future__ import annotations

from typing import Literal

import sympy as sp

from ..errors import ParseError

Kind = Literal["scalar", "vector"]


class VectorTypeError(ParseError):
    """A vector expression that does not type-check (the message is safe to show)."""


class _Head(sp.Function):  # type: ignore[misc]
    """Inert operator: never evaluates, so SymPy cannot rewrite or cancel it."""

    nargs: int | tuple[int, ...] = 1
    # The result is a vector (noncommutative, so products keep their place) or a scalar.
    _vector_valued: bool | None = None

    @classmethod
    def eval(cls, *args: sp.Basic) -> None:
        return None

    def _eval_is_commutative(self) -> bool | None:
        if self._vector_valued is None:  # follows the arguments
            return all(a.is_commutative for a in self.args)
        return not self._vector_valued


class Grad(_Head):
    nargs = 1
    _vector_valued = True


class Div(_Head):
    nargs = 1
    _vector_valued = False


class Curl(_Head):
    nargs = 1
    _vector_valued = True


class Lap(_Head):
    """Laplacian: of a scalar it is a scalar, of a vector the vector Laplacian."""

    nargs = 1


class Dot(_Head):
    nargs = 2
    _vector_valued = False


class Cross(_Head):
    nargs = 2
    _vector_valued = True


class Pd(_Head):
    """Partial derivative of a scalar or vector field with respect to one coordinate or `t`."""

    nargs = 2


class Adv(_Head):
    """`(A . nabla) B`: the directional derivative of a scalar or vector field along A."""

    nargs = 2

    def _eval_is_commutative(self) -> bool | None:
        return bool(self.args[1].is_commutative)


class Pos(_Head):
    """The position vector (its one argument is a placeholder, `Pos(0)`)."""

    nargs = 1
    _vector_valued = True


class Unit(_Head):
    """A unit vector of the coordinate system, by coordinate name (`x`, `rho`, `theta`)."""

    nargs = 1
    _vector_valued = True


class AdvOperator(_Head):
    """`(A . nabla)` before the field it acts on is read; never survives parsing."""

    nargs = 1
    _vector_valued = True


HEADS: tuple[type[_Head], ...] = (Grad, Div, Curl, Lap, Dot, Cross, Pd, Adv, Pos, Unit, AdvOperator)


def kind_of(expr: sp.Basic) -> Kind:
    """The kind of `expr`; raises `VectorTypeError` if it does not type-check."""
    if isinstance(expr, sp.Symbol):
        return "vector" if expr.is_commutative is False else "scalar"
    if expr.is_Number or expr.is_Atom:
        return "scalar"
    if isinstance(expr, Grad):
        _need(expr.args[0], "scalar", "the gradient is taken of a scalar field")
        return "vector"
    if isinstance(expr, Div):
        _need(expr.args[0], "vector", "the divergence is taken of a vector field")
        return "scalar"
    if isinstance(expr, Curl):
        _need(expr.args[0], "vector", "the curl is taken of a vector field")
        return "vector"
    if isinstance(expr, Lap):
        return kind_of(expr.args[0])
    if isinstance(expr, Dot):
        _need(expr.args[0], "vector", "a dot product needs two vectors")
        _need(expr.args[1], "vector", "a dot product needs two vectors")
        return "scalar"
    if isinstance(expr, Cross):
        _need(expr.args[0], "vector", "a cross product needs two vectors")
        _need(expr.args[1], "vector", "a cross product needs two vectors")
        return "vector"
    if isinstance(expr, Pd):
        return kind_of(expr.args[0])
    if isinstance(expr, Adv):
        _need(expr.args[0], "vector", "(A . nabla) needs a vector A")
        return kind_of(expr.args[1])
    if isinstance(expr, Pos | Unit):
        return "vector"
    if isinstance(expr, AdvOperator):
        raise VectorTypeError("(A . nabla) must be followed by the field it acts on")
    if isinstance(expr, sp.Add):
        kinds = {kind_of(a) for a in expr.args if a != 0}
        if len(kinds) > 1:
            raise VectorTypeError("a scalar and a vector cannot be added")
        return kinds.pop() if kinds else "scalar"
    if isinstance(expr, sp.Mul):
        factor_kinds = [kind_of(a) for a in expr.args]
        vectors = factor_kinds.count("vector")
        if vectors > 1:
            raise VectorTypeError(
                "the product of two vectors is ambiguous: write a dot product (\\cdot) "
                "or a cross product (\\times)"
            )
        return "vector" if vectors else "scalar"
    if isinstance(expr, sp.Pow):
        if kind_of(expr.args[0]) != "scalar" or kind_of(expr.args[1]) != "scalar":
            raise VectorTypeError("a power of a vector is ambiguous; write A \\cdot A for |A|^2")
        return "scalar"
    # a function of scalars: sin, exp, Abs, ...
    for arg in expr.args:
        if kind_of(arg) != "scalar":
            raise VectorTypeError(f"{expr.func.__name__} needs scalar arguments")
    return "scalar"


def _need(expr: sp.Basic, kind: Kind, message: str) -> None:
    if kind_of(expr) != kind:
        raise VectorTypeError(message)


def is_number(expr: sp.Basic) -> bool:
    """A numeral or a constant expression: `3`, `10^{3}`, `\\pi`."""
    return bool(not expr.free_symbols and expr.is_number)


def product(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """Scalar times anything. Two vectors are refused."""
    if kind_of(left) == "vector" and kind_of(right) == "vector":
        raise VectorTypeError(
            "the product of two vectors is ambiguous: write a dot product (\\cdot) "
            "or a cross product (\\times)"
        )
    return left * right


def dot(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """`left \\cdot right`: a dot product of two vectors, else a scalar product."""
    if kind_of(left) == "vector" and kind_of(right) == "vector":
        return Dot(left, right)
    return left * right


def cross(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    """`left \\times right`: the cross product of two vectors. A numeral on either side is an
    ordinary product (`2 \\times x`); anything else is refused, never read as a scalar product."""
    if kind_of(left) == "vector" and kind_of(right) == "vector":
        return Cross(left, right)
    if is_number(left) or is_number(right):
        return left * right
    raise VectorTypeError(
        "\\times is the cross product of two vectors; use \\cdot for a scalar product "
        "(or declare the vector fields)"
    )


def contains_head(expr: sp.Basic) -> bool:
    return any(isinstance(node, _Head) for node in sp.preorder_traversal(expr))


def depth_of_operators(expr: sp.Basic) -> int:
    """Largest number of nested differential operators (grad, div, curl, lap, pd, adv) in `expr`."""
    differential = (Grad, Div, Curl, Lap, Pd, Adv)
    if not expr.args:
        return 0
    inner = max((depth_of_operators(a) for a in expr.args), default=0)
    if isinstance(expr, differential):
        weight = 2 if isinstance(expr, Lap) else 1
        return inner + weight
    return inner
