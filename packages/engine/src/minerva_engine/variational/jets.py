"""Jet space: the dependent functions of one independent variable and their derivatives.

A functional `S[x] = ∫ L(t, x_i, x_i', ..., x_i^(n_i)) dt` is an ordinary function of the
*jet coordinates* `u_i^(k)` (the value and the k-th derivative of the i-th dependent function at
one point) and of the independent variable. The total derivative along a path is

    D f = ∂f/∂t + Σ_{i,k} u_i^(k+1) ∂f/∂u_i^(k),

which is what distinguishes the jets from unrelated symbols: `∂L/∂x` and `D(∂L/∂ẋ)` are different
operations, and nothing here lets one be mistaken for the other.

Jets are registered by (function index, order) in this object, never recognised by a name suffix,
and `eta` holds the matching jets of the admissible perturbation. The independent variable is a
tuple of length one today; field functionals (several independent variables, jets with
multi-indices) extend `D` and the registry without changing the callers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import sympy as sp

from ..build import JetBinding, spelling
from ..errors import LimitError, ParseError

MAX_FUNCTIONS = 4
MAX_ORDER = 4  # highest derivative a user may write
MAX_DEPTH = 2 * MAX_ORDER  # highest jet represented internally (n_i + n_j)


def latex_jet_name(name: str, order: int) -> str:
    """`x`, `\\dot{x}`, `\\ddot{x}`, `x^{(3)}`; a subscripted name keeps its subscript outside."""
    base, _, sub = name.partition("_")
    head = {0: base, 1: rf"\dot{{{base}}}", 2: rf"\ddot{{{base}}}"}.get(
        order, rf"{base}^{{({order})}}"
    )
    return rf"{head}_{{{sub}}}" if sub else head


def latex_eta_name(name: str, order: int) -> str:
    base, _, sub = name.partition("_")
    root = rf"\eta_{{{name}}}" if sub else rf"\eta_{{{base}}}"
    return {0: root, 1: rf"\dot{{\eta}}_{{{name}}}", 2: rf"\ddot{{\eta}}_{{{name}}}"}.get(
        order, rf"\eta_{{{name}}}^{{({order})}}"
    )


@dataclass(frozen=True)
class JetSpace:
    variable: sp.Symbol
    names: tuple[str, ...]
    orders: tuple[int, ...]  # n_i: highest derivative of function i in the integrand
    depth: int  # highest jet order that exists
    u: tuple[tuple[sp.Symbol, ...], ...] = field(repr=False)
    eta: tuple[tuple[sp.Symbol, ...], ...] = field(repr=False)

    # -- construction --------------------------------------------------------------------
    @staticmethod
    def build(
        variable: str, names: tuple[str, ...], orders: tuple[int, ...], depth: int
    ) -> JetSpace:
        if not 1 <= len(names) <= MAX_FUNCTIONS:
            raise LimitError(f"between 1 and {MAX_FUNCTIONS} dependent functions are supported")
        if len(set(names)) != len(names):
            raise ParseError("the dependent functions must have different names")
        if len(orders) != len(names) or any(o < 0 or o > MAX_ORDER for o in orders):
            raise LimitError(f"derivatives of order above {MAX_ORDER} are not supported")
        depth = max(depth, max(orders, default=0))
        if depth > MAX_DEPTH:
            raise LimitError("the Euler-Lagrange equations would need derivatives beyond the cap")
        t = sp.Symbol(variable, real=True)
        u = tuple(
            tuple(sp.Symbol(spelling(n, k), real=True) for k in range(depth + 1)) for n in names
        )
        eta = tuple(
            tuple(
                sp.Symbol(f"eta_{n}" + ("" if k == 0 else f"_{k}"), real=True)
                for k in range(depth + 1)
            )
            for n in names
        )
        return JetSpace(t, names, orders, depth, u, eta)

    def with_orders(self, orders: tuple[int, ...]) -> JetSpace:
        depth = max(orders) + max(orders) if orders else 0
        return JetSpace.build(self.variable.name, self.names, orders, max(depth, self.depth))

    def binding(self) -> JetBinding:
        """What the parsers need: every spelling a user may write, up to the parse cap."""
        symbols: dict[str, sp.Symbol] = {self.variable.name: self.variable}
        for i, name in enumerate(self.names):
            for k in range(min(MAX_ORDER, self.depth) + 1):
                symbols[spelling(name, k)] = self.u[i][k]
        return JetBinding(
            variable=self.variable.name,
            symbols=symbols,
            dependent=frozenset(self.names),
            orders={n: MAX_ORDER for n in self.names},
        )

    # -- lookups -------------------------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise ParseError(f"{name!r} is not one of the dependent functions") from None

    @property
    def state_symbols(self) -> frozenset[sp.Symbol]:
        return frozenset(s for row in self.u for s in row)

    @property
    def perturbation_symbols(self) -> frozenset[sp.Symbol]:
        return frozenset(s for row in self.eta for s in row)

    def order_in(self, expr: sp.Expr, i: int) -> int:
        """Highest derivative of function i present in `expr` (-1 if it does not occur)."""
        found = -1
        free = expr.free_symbols
        for k, symbol in enumerate(self.u[i]):
            if symbol in free:
                found = k
        return found

    def jets_in(self, expr: sp.Expr) -> list[tuple[int, int]]:
        free = expr.free_symbols
        return [
            (i, k) for i in range(self.count) for k in range(self.depth + 1) if self.u[i][k] in free
        ]

    # -- calculus ------------------------------------------------------------------------
    def D(self, expr: sp.Expr) -> sp.Expr:
        """The total derivative along a path: the independent variable and every jet."""
        free = expr.free_symbols
        total = sp.diff(expr, self.variable) if self.variable in free else sp.Integer(0)
        for i in range(self.count):
            for k in range(self.depth + 1):
                symbol = self.u[i][k]
                if symbol not in free:
                    continue
                if k == self.depth:
                    raise LimitError("a total derivative needs a jet beyond the represented depth")
                total += sp.diff(expr, symbol) * self.u[i][k + 1]
            for k in range(self.depth + 1):
                symbol = self.eta[i][k]
                if symbol not in free:
                    continue
                if k == self.depth:
                    raise LimitError("a total derivative needs a jet beyond the represented depth")
                total += sp.diff(expr, symbol) * self.eta[i][k + 1]
        return total

    def Dn(self, expr: sp.Expr, n: int) -> sp.Expr:
        for _ in range(n):
            expr = self.D(expr)
        return expr

    def partial(self, expr: sp.Expr, i: int, k: int) -> sp.Expr:
        """∂/∂u_i^(k) with every other jet and the variable held fixed (not a total derivative)."""
        return sp.diff(expr, self.u[i][k])

    def vary(self, expr: sp.Expr) -> sp.Expr:
        """δ expr = Σ ∂expr/∂u_i^(k) η_i^(k)."""
        free = expr.free_symbols
        total = sp.Integer(0)
        for i in range(self.count):
            for k in range(self.depth + 1):
                if self.u[i][k] in free:
                    total += sp.diff(expr, self.u[i][k]) * self.eta[i][k]
        return total

    # -- display -------------------------------------------------------------------------
    def symbol_names(self) -> dict[sp.Symbol, str]:
        names: dict[sp.Symbol, str] = {}
        for i, name in enumerate(self.names):
            for k in range(self.depth + 1):
                names[self.u[i][k]] = latex_jet_name(name, k)
                names[self.eta[i][k]] = latex_eta_name(name, k)
        return names


# -- numeric stand-ins for given functions ------------------------------------------------------
def standins(
    expr: sp.Expr, rng: random.Random, depth: int
) -> dict[sp.core.function.UndefinedFunction, sp.Lambda]:
    """Random real analytic functions for every given function in `expr`.

    Each is `P(w·z) + c sin(v·z + b)` with `P` a polynomial of degree `2 depth + 2` (so that the
    derivatives that appear in an Euler-Lagrange expression do not vanish identically) and `z` the
    vector of the arguments. Numeric legs substitute these and never see the symbolic result.
    """
    replacements: dict[sp.core.function.UndefinedFunction, sp.Lambda] = {}
    for applied in sorted(expr.atoms(sp.core.function.AppliedUndef), key=str):
        function = type(applied)
        if function in replacements:
            continue
        dummies = sp.symbols(f"z0:{len(applied.args)}", real=True)
        weights = [sp.Rational(rng.randint(2, 9), 7) for _ in dummies]
        phases = [sp.Rational(rng.randint(2, 9), 5) for _ in dummies]
        w = sum((a * z for a, z in zip(weights, dummies, strict=True)), sp.Integer(0))
        v = sum((a * z for a, z in zip(phases, dummies, strict=True)), sp.Integer(0))
        degree = 2 * depth + 2
        poly = sum(
            (sp.Rational(rng.randint(-5, 5), 3 + j) * w**j / 2**j for j in range(degree + 1)),
            sp.Integer(0),
        )
        body = poly + sp.Rational(rng.randint(1, 4), 3) * sp.sin(
            v + sp.Rational(rng.randint(1, 9), 4)
        )
        replacements[function] = sp.Lambda(dummies, body)
    return replacements


def apply_standins(
    expr: sp.Expr, replacements: dict[sp.core.function.UndefinedFunction, sp.Lambda]
) -> sp.Expr:
    for function, body in replacements.items():
        expr = expr.replace(function, body)
    return sp.sympify(expr).doit()
