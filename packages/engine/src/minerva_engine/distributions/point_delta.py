"""The three-dimensional point delta, as a typed inert head.

`PointDelta(a1, a2, a3)` stands for the distribution `delta^(3)(r - a)` on R^3 with the source at the
Cartesian point `a = (a1, a2, a3)` (`PointDelta(0, 0, 0)` is `delta^(3)(r)`). It is deliberately not a
`DiracDelta`: a one-variable delta has no source *point* in R^3, and reading `delta^(3)` as a product or a
composition of one-dimensional deltas would invent rules (`delta(x) delta(y) delta(z)` is a different, more
general object than the one supported here).

The head never evaluates: SymPy cannot simplify, differentiate, multiply or integrate it, and every
pointwise machine in Minerva (`verify_equal`, the derivative and integral checkers, the vector evaluator)
refuses it through `distributions.guard.contains_distribution`, because a delta is zero at every point a
sampler can draw. It is created only by `vector.singular_fields`, which establishes it by a distributional
argument; it is never produced by a parser. Unrestricted operations on it (products with functions,
derivatives, composition) are not supported. SymPy itself would treat the head as a constant (its arguments are
the source point, not the variable), which is why no operation is ever allowed to hand it to SymPy: the guard
(`contains_distribution`) refuses it everywhere except in the one action that builds it.
"""

from __future__ import annotations

import sympy as sp


class PointDelta(sp.Function):  # type: ignore[misc]
    nargs = 3

    @classmethod
    def eval(cls, *args: sp.Basic) -> None:
        return None

    @property
    def source(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        a, b, c = self.args
        return a, b, c

    def _eval_is_commutative(self) -> bool:
        return True
