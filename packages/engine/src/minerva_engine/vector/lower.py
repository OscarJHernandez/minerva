"""Exact lowering of a typed vector expression to components.

The result of a calculation is built here, symbolically, in the orthonormal frame of the chosen
coordinate system with the usual scale-factor formulas (gradient, divergence, curl; the Laplacian
of a scalar as div grad and the vector Laplacian as the scalar Laplacian of the Cartesian components). The independent check of
that result is `numeric.py`, which shares none of this code.

Derivatives of an abstract field are *jets*: the derivative of component `F` along the word
`("y", "x")` (first y, then x) is the plain symbol `F|y.x`. Symbols carry no symmetry, so SymPy cannot
silently identify `d/dx d/dy F` with `d/dy d/dx F`. The two are merged only for a field the user
declared `c2` (words up to length 2) or `smooth`; a result that would need a merge for another field is
`blocked` (see `Lowering.blocked`). Nothing here assumes smoothness the user did not declare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from ..errors import ParseError
from .coordinates import TIME, System
from .expressions import (
    Adv,
    Cross,
    Curl,
    Div,
    Dot,
    Grad,
    Lap,
    Pd,
    Pos,
    Unit,
    kind_of,
)
from .models import Blocked, FieldDecl

Vec = tuple[Any, Any, Any]  # sympy expressions are untyped (Any) for mypy
Lowered = Any

ZERO = sp.Integer(0)


class Undeclared(ParseError):
    """A symbol that is neither a coordinate, a declared field nor a declared constant."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(
            "undeclared symbol(s): "
            + ", ".join(names)
            + ". Declare a field (scalar or vector) or list it in `constants`; an undeclared "
            "symbol is never silently treated as a constant."
        )
        self.names = names


@dataclass
class Declared:
    decl: FieldDecl
    comps: tuple[str, ...]  # component function names (one for a scalar)
    depends: tuple[str, ...]
    explicit: tuple[sp.Expr, ...] | None = None


@dataclass
class Lowering:
    system: System
    declared: dict[str, Declared]
    constants: dict[str, sp.Symbol]
    q: tuple[sp.Symbol, sp.Symbol, sp.Symbol] = field(init=False)
    t: sp.Symbol = field(init=False)
    h: tuple[sp.Expr, sp.Expr, sp.Expr] = field(init=False)

    def __post_init__(self) -> None:
        self.q = self.system.symbols()
        self.t = sp.Symbol(TIME, real=True)
        self.h = self.system.scale_factors()
        self._order = {**{c: i for i, c in enumerate(self.system.coords)}, TIME: 3}
        self._owner: dict[str, Declared] = {}
        for d in self.declared.values():
            for comp in d.comps:
                self._owner[comp] = d
        self.jets: dict[sp.Symbol, tuple[str, tuple[str, ...]]] = {}
        self._jet_symbols: dict[tuple[str, tuple[str, ...]], sp.Symbol] = {}
        self._frame = self.system.frame()

    # -- jets ------------------------------------------------------------------------------
    def _sortable(self, comp: str, word: tuple[str, ...]) -> bool:
        smoothness = self._owner[comp].decl.smoothness
        return smoothness == "smooth" or (smoothness == "c2" and len(word) <= 2)

    def jet(self, comp: str, word: tuple[str, ...]) -> sp.Symbol:
        if self._sortable(comp, word):
            word = tuple(sorted(word, key=lambda v: self._order[v]))
        key = (comp, word)
        if key not in self._jet_symbols:
            symbol = sp.Symbol(f"{comp}|{'.'.join(word)}", real=True)
            self._jet_symbols[key] = symbol
            self.jets[symbol] = key
        return self._jet_symbols[key]

    def D(self, expr: sp.Expr, var: sp.Symbol) -> sp.Expr:
        """Total derivative with respect to a coordinate (or `t`): the explicit dependence plus the
        chain rule through the jets of the fields the expression contains."""
        out = sp.diff(expr, var)
        for s in expr.free_symbols:
            info = self.jets.get(s)
            if info is not None and var.name in self._owner[info[0]].depends:
                out += sp.diff(expr, s) * self.jet(info[0], (*info[1], var.name))
        return out

    def merge_blocks(self, results: list[sp.Expr]) -> list[Blocked]:
        """Fields whose jets appear in the result in two orders that only a symmetric mixed partial
        derivative would identify, and the smoothness that would justify it."""
        present: set[sp.Symbol] = set()
        for r in results:
            present |= {s for s in r.free_symbols if s in self.jets}
        groups: dict[tuple[str, tuple[str, ...]], set[tuple[str, ...]]] = {}
        for s in present:
            comp, word = self.jets[s]
            key = (comp, tuple(sorted(word, key=lambda v: self._order[v])))
            groups.setdefault(key, set()).add(word)
        needs: dict[str, set[str]] = {}
        for (comp, sorted_word), words in groups.items():
            if len(words) > 1:
                need = "c2" if len(sorted_word) <= 2 else "smooth"
                needs.setdefault(need, set()).add(self._owner[comp].decl.name)
        return [
            Blocked(
                needs=need,
                fields=sorted(names),
                note=(
                    "the result contains mixed partial derivatives in both orders; they are equal "
                    + (
                        "only if the field is twice continuously differentiable"
                        if need == "c2"
                        else "only if the field is differentiable often enough (smooth)"
                    )
                ),
            )
            for need, names in sorted(needs.items())
        ]

    def display(self, expr: sp.Expr) -> sp.Expr:
        """Replace jets by the derivatives of undefined functions SymPy prints."""
        mapping: dict[sp.Symbol, sp.Expr] = {}
        for symbol, (comp, word) in self.jets.items():
            if symbol not in expr.free_symbols:
                continue
            owner = self._owner[comp]
            args = [self._variable(v) for v in owner.depends]
            function = sp.Function(comp, real=True)(*args)
            mapping[symbol] = (
                sp.Derivative(function, *[self._variable(v) for v in word]) if word else function
            )
        return expr.xreplace(mapping)

    def _variable(self, name: str) -> sp.Symbol:
        if name == TIME:
            return self.t
        return self.q[self.system.coords.index(name)]

    # -- names -----------------------------------------------------------------------------
    def unknown_symbols(self, expr: sp.Basic) -> list[str]:
        known = set(self.system.coords) | {TIME} | set(self.declared) | set(self.constants)
        return sorted({s.name for s in expr.free_symbols if s.name not in known})

    def component(self, comp: str) -> sp.Expr:
        owner = self._owner[comp]
        if owner.explicit is not None:
            return owner.explicit[owner.comps.index(comp)]
        return self.jet(comp, ())

    def _symbol(self, e: sp.Symbol) -> Lowered:
        name = e.name
        if name in self.declared:
            d = self.declared[name]
            parts = tuple(self.component(c) for c in d.comps)
            if d.decl.kind == "vector":
                return parts[0], parts[1], parts[2]
            return parts[0]
        if name == TIME:
            return self.t
        if name in self.system.coords:
            return self.q[self.system.coords.index(name)]
        if name in self.constants:
            return self.constants[name]
        raise Undeclared([name])

    # -- the operators -----------------------------------------------------------------------
    def grad(self, f: sp.Expr) -> Vec:
        return (
            self.D(f, self.q[0]) / self.h[0],
            self.D(f, self.q[1]) / self.h[1],
            self.D(f, self.q[2]) / self.h[2],
        )

    def div(self, v: Vec) -> sp.Expr:
        h1, h2, h3 = self.h
        big = h1 * h2 * h3
        total = (
            self.D(big / h1 * v[0], self.q[0])
            + self.D(big / h2 * v[1], self.q[1])
            + self.D(big / h3 * v[2], self.q[2])
        )
        return total / big

    def curl(self, v: Vec) -> Vec:
        h1, h2, h3 = self.h
        q1, q2, q3 = self.q
        return (
            (self.D(h3 * v[2], q2) - self.D(h2 * v[1], q3)) / (h2 * h3),
            (self.D(h1 * v[0], q3) - self.D(h3 * v[2], q1)) / (h3 * h1),
            (self.D(h2 * v[1], q1) - self.D(h1 * v[0], q2)) / (h1 * h2),
        )

    def lap_vector(self, v: Vec) -> Vec:
        """The vector Laplacian: the scalar Laplacian of each *Cartesian* component, turned back into
        physical components. (grad div - curl curl is equal to it only for fields whose mixed partial
        derivatives commute; this form needs no such assumption.)"""
        cart = self._cartesian(v)
        laps = tuple(self.div(self.grad(c)) for c in cart)
        return self._physical(laps)

    def _cartesian(self, v: Vec) -> Vec:
        m = self._frame * sp.Matrix(v)
        return m[0], m[1], m[2]

    def _physical(self, v: Vec) -> Vec:
        m = self._frame.T * sp.Matrix(v)
        return m[0], m[1], m[2]

    def pd(self, value: Lowered, var: sp.Symbol) -> Lowered:
        if var.name != TIME and var.name not in self.system.coords:
            raise ParseError(f"cannot differentiate with respect to {var.name}: not a coordinate")
        target = self.t if var.name == TIME else self.q[self.system.coords.index(var.name)]
        if not isinstance(value, tuple):
            return self.D(value, target)
        if var.name == TIME or self.system.name == "cartesian":
            return tuple(self.D(c, target) for c in value)
        cart = self._cartesian(value)
        return self._physical(tuple(self.D(c, target) for c in cart))

    def adv(self, a: Vec, b: Lowered) -> Lowered:
        def along(f: sp.Expr) -> sp.Expr:
            return sum(
                (a[i] / self.h[i] * self.D(f, self.q[i]) for i in range(3)),
                ZERO,
            )

        if not isinstance(b, tuple):
            return along(b)
        if self.system.name == "cartesian":
            return along(b[0]), along(b[1]), along(b[2])
        cart = self._cartesian(b)
        return self._physical((along(cart[0]), along(cart[1]), along(cart[2])))

    # -- the walk ----------------------------------------------------------------------------
    def lower(self, e: sp.Basic) -> Lowered:
        if isinstance(e, sp.Symbol):
            return self._symbol(e)
        if e.is_Number or e.is_Atom:
            return e
        if isinstance(e, Grad):
            return self.grad(self._scalar(e.args[0]))
        if isinstance(e, Div):
            return self.div(self._vector(e.args[0]))
        if isinstance(e, Curl):
            return self.curl(self._vector(e.args[0]))
        if isinstance(e, Lap):
            inner = self.lower(e.args[0])
            if isinstance(inner, tuple):
                return self.lap_vector(inner)
            return self.div(self.grad(inner))
        if isinstance(e, Dot):
            a, b = self._vector(e.args[0]), self._vector(e.args[1])
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        if isinstance(e, Cross):
            a, b = self._vector(e.args[0]), self._vector(e.args[1])
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
        if isinstance(e, Pd):
            return self.pd(self.lower(e.args[0]), e.args[1])
        if isinstance(e, Adv):
            return self.adv(self._vector(e.args[0]), self.lower(e.args[1]))
        if isinstance(e, Pos):
            return self.system.position_vector()
        if isinstance(e, Unit):
            index = self.system.coords.index(e.args[0].name)
            return tuple(sp.Integer(1 if i == index else 0) for i in range(3))
        if isinstance(e, sp.Add):
            if kind_of(e) == "vector":
                parts = [self._vector(a) for a in e.args]
                return (
                    sp.Add(*[p[0] for p in parts]),
                    sp.Add(*[p[1] for p in parts]),
                    sp.Add(*[p[2] for p in parts]),
                )
            return sp.Add(*[self._scalar(a) for a in e.args])
        if isinstance(e, sp.Mul):
            scalars: list[sp.Expr] = []
            vector: Vec | None = None
            for a in e.args:
                low = self.lower(a)
                if isinstance(low, tuple):
                    vector = low
                else:
                    scalars.append(low)
            factor = sp.Mul(*scalars)
            if vector is None:
                return factor
            return factor * vector[0], factor * vector[1], factor * vector[2]
        if isinstance(e, sp.Pow):
            return self._scalar(e.args[0]) ** self._scalar(e.args[1])
        return e.func(*[self._scalar(a) for a in e.args])

    def _scalar(self, e: sp.Basic) -> sp.Expr:
        value = self.lower(e)
        if isinstance(value, tuple):
            raise ParseError("a scalar was expected here")
        return value

    def _vector(self, e: sp.Basic) -> Vec:
        value = self.lower(e)
        if not isinstance(value, tuple):
            raise ParseError("a vector was expected here")
        return value
