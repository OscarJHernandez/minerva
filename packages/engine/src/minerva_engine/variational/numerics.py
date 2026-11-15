"""Numeric evaluation of integrands along polynomial paths (shared by every numeric leg).

Expressions are compiled with `lambdify(modules="mpmath")` after the given functions are replaced
by random analytic stand-ins, so what is evaluated is the *original* integrand and not a
symbolic result. Jet values come from exact polynomial derivatives (`polypath`).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import mpmath as mp
import sympy as sp

from ..errors import LimitError, ParseError
from ..verify import _sample
from . import polypath
from .jets import JetSpace, apply_standins, standins
from .setup import Problem

DPS = 30


@dataclass
class Compiled:
    symbols: tuple[sp.Symbol, ...]
    fn: Callable[..., Any]

    def __call__(self, values: Mapping[sp.Symbol, Any]) -> Any:
        try:
            return self.fn(*[values[s] for s in self.symbols])
        except KeyError as exc:  # a symbol without a value is a programming error, not user input
            raise ParseError("an expression has a symbol without a value") from exc


@dataclass
class Poly:
    """One polynomial per function: x_i(s) = Σ_k c[i][k] s^k, with its derivative table."""

    coefficients: list[polypath.Coeffs]

    def table(self, depth: int) -> list[list[polypath.Coeffs]]:
        return [polypath.derivatives_table(c, depth) for c in self.coefficients]

    def plus(self, other: Poly, scale: Any) -> Poly:
        return Poly(
            [
                polypath.add(a, b, scale)
                for a, b in zip(self.coefficients, other.coefficients, strict=True)
            ]
        )


class Numerics:
    """A problem with sampled parameter values, stand-ins for given functions and compiled pieces."""

    def __init__(
        self,
        problem: Problem,
        seed: int,
        *,
        values: Mapping[sp.Symbol, Any] | None = None,
        definitions: Mapping[Any, sp.Lambda] | None = None,
    ):
        self.problem = problem
        self.jets: JetSpace = problem.jets
        self.rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
        self.depth = self.jets.depth
        mp.mp.dps = DPS
        expr_all = sum([problem.lagrangian, problem.lower_term, problem.upper_term], sp.Integer(0))
        self.replacements = standins(expr_all, self.rng, self.depth + 2)
        if definitions:
            self.replacements.update(definitions)
        self.params: dict[sp.Symbol, Any]
        self.lower: Any
        self.upper: Any
        self._choose_values(values)

    # -- sampling -----------------------------------------------------------------------
    def _choose_values(self, given: Mapping[sp.Symbol, Any] | None) -> None:
        """Parameter values (given, or sampled consistently with the assumptions) with lower < upper."""
        problem = self.problem
        for _ in range(1 if given is not None else 600):
            params: dict[sp.Symbol, Any] = (
                dict(given)
                if given is not None
                else {p: _to_mp(_sample(p, self.rng)) for p in problem.parameters}
            )
            lo = self._number(problem.lower, params)
            hi = self._number(problem.upper, params)
            fixed = given is not None or problem.bounds_numeric
            if lo < hi and (fixed or (hi - lo >= 0.3 and max(abs(lo), abs(hi)) <= 1.5)):
                self.params, self.lower, self.upper = params, lo, hi
                return
        raise ParseError("could not choose parameter values with lower bound < upper bound")

    @staticmethod
    def _number(expr: sp.Expr, params: Mapping[sp.Symbol, Any]) -> mp.mpf:
        if expr.is_number:
            return mp.mpf(sp.N(expr, 40))
        value = compile_expr(expr, ())({s: params[s] for s in expr.free_symbols})
        if mp.im(value) != 0:
            raise ParseError("a bound is not real for these parameter values")
        return mp.mpf(mp.re(value))

    # -- compiling ----------------------------------------------------------------------
    def compile(self, expr: sp.Expr) -> Compiled:
        expr = sp.sympify(expr)
        explicit = apply_standins(expr, self.replacements) if self.replacements else expr
        return compile_expr(explicit, ())

    # -- paths --------------------------------------------------------------------------
    def jet_values(
        self, s: Any, path: JetField, perturbation: JetField | None = None
    ) -> dict[sp.Symbol, Any]:
        values: dict[sp.Symbol, Any] = {self.jets.variable: s, **self.params}
        u = path.at(s)
        for i in range(self.jets.count):
            for k in range(min(self.depth, len(u[i]) - 1) + 1):
                values[self.jets.u[i][k]] = u[i][k]
        if perturbation is not None:
            e = perturbation.at(s)
            for i in range(self.jets.count):
                for k in range(min(self.depth, len(e[i]) - 1) + 1):
                    values[self.jets.eta[i][k]] = e[i][k]
        return values

    def random_path(self, degree: int, scale: float = 0.8) -> Poly:
        return Poly(
            [polypath.random_coefficients(self.rng, degree, scale) for _ in self.jets.names]
        )

    def field(self, poly: Poly) -> PolyField:
        return PolyField(poly, self.depth)


class JetField(Protocol):
    """A path (or a perturbation): the value and every derivative up to the jet depth at a point."""

    def at(self, s: Any) -> list[list[Any]]: ...


class PolyField:
    def __init__(self, poly: Poly, depth: int) -> None:
        self.depth = depth
        self.table = poly.table(depth)

    def at(self, s: Any) -> list[list[Any]]:
        return [
            [polypath.evaluate(self.table[i][k], s) for k in range(self.depth + 1)]
            for i in range(len(self.table))
        ]


class ShiftedField:
    """base + scale * other, derivative by derivative."""

    def __init__(self, base: JetField, other: JetField, scale: Any) -> None:
        self.base, self.other, self.scale = base, other, scale

    def at(self, s: Any) -> list[list[Any]]:
        b, o = self.base.at(s), self.other.at(s)
        return [[b[i][k] + self.scale * o[i][k] for k in range(len(b[i]))] for i in range(len(b))]


def compile_expr(expr: sp.Expr, extra: Sequence[sp.Symbol]) -> Compiled:
    symbols = tuple(sorted(expr.free_symbols | set(extra), key=lambda s: s.name))
    if expr.has(sp.Derivative, sp.Subs):
        raise LimitError("an expression still contains an unevaluated derivative")
    try:
        fn = sp.lambdify(symbols, expr, modules="mpmath")
    except (TypeError, ValueError, SyntaxError) as exc:  # pragma: no cover - lambdify failure
        raise ParseError("an expression could not be compiled for numerical evaluation") from exc
    return Compiled(symbols, fn)


def _to_mp(value: sp.Expr) -> Any:
    number = sp.N(value, 30)
    re_, im_ = number.as_real_imag()
    re_f = mp.mpf(str(re_))
    if im_ == 0:
        return re_f
    return mp.mpc(re_f, mp.mpf(str(im_)))
