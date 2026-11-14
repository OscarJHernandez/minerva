"""Independent numerical evaluation of a finite-spectrum canonical ensemble.

Nothing here uses the symbolic route that produces the displayed formulas: the energies and
degeneracies are evaluated at a sample point with mpmath, the weights are computed with log-sum-exp
(no overflow at large beta times energy), and every quantity is a direct sum over the levels. The
derivative checks differentiate these numbers, never the symbolic expression.

Numerical agreement is evidence, not proof.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any

import mpmath as mp
import sympy as sp

from ..stats.numeric import sample_symbol
from .setup import Problem

DPS = 50
POINTS = 8
MIN_POINTS = 4
REL_TOL = mp.mpf(10) ** -18
DIFF_TOL = mp.mpf(10) ** -12
SEED = "minerva-statmech-v1"

Point = dict[sp.Symbol, Any]  # symbol -> mpmath number


def sample_points(
    problem: Problem, count: int = POINTS, *, extra: Sequence[sp.Symbol] = ()
) -> list[Point]:
    """Assumption-respecting sample points: exact rationals, then mpmath numbers.

    ``beta``, ``k_B`` and ``T = 1/(k_B beta)`` are always consistent with each other.
    """
    rng = random.Random(  # noqa: S311 - deterministic sampling, not cryptographic
        f"{SEED}:{len(problem.energies)}:{len(problem.sample_symbols)}"
    )
    symbols = sorted(set(problem.sample_symbols) | set(extra), key=lambda s: s.name)
    out: list[Point] = []
    with mp.workdps(DPS):
        for _ in range(count):
            exact = {s: sample_symbol(s, rng) for s in symbols}
            point: Point = {s: _mp(v) for s, v in exact.items()}
            point[problem.T] = 1 / (point[problem.k_B] * point[problem.beta])
            out.append(point)
    return out


def _mp(value: sp.Expr) -> Any:
    return mp.mpf(value.p) / mp.mpf(value.q) if value.is_Rational else mp.mpf(str(sp.N(value, DPS)))


def compile_expr(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> Callable[..., Any]:
    return sp.lambdify(list(symbols), expr, modules="mpmath")  # type: ignore[no-any-return]


class Oracle:
    """Direct sums over the levels at one sample point."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self._symbols = sorted(
            {s for e in (*problem.energies, *problem.degeneracies) for s in e.free_symbols}
            | {s for col in problem.observables.values() for a in col for s in a.free_symbols},
            key=lambda s: s.name,
        )
        self._energy = [compile_expr(e, self._symbols) for e in problem.energies]
        self._degeneracy = [compile_expr(g, self._symbols) for g in problem.degeneracies]
        self._observable = {
            name: [compile_expr(a, self._symbols) for a in col]
            for name, col in problem.observables.items()
        }

    def _args(self, point: Point) -> list[Any]:
        return [point[s] for s in self._symbols]

    def energies(self, point: Point) -> list[Any]:
        args = self._args(point)
        return [f(*args) for f in self._energy]

    def degeneracies(self, point: Point) -> list[Any]:
        args = self._args(point)
        return [f(*args) for f in self._degeneracy]

    def observable(self, name: str, point: Point) -> list[Any]:
        args = self._args(point)
        return [f(*args) for f in self._observable[name]]

    def beta_of(self, point: Point) -> Any:
        return point[self.problem.beta]

    def log_weights(self, point: Point, beta: Any | None = None) -> tuple[list[Any], Any]:
        """Per-level log-weights ln(g e^{-beta E}) and their log-sum-exp, ln Z."""
        b = self.beta_of(point) if beta is None else beta
        terms = [
            mp.log(g) - b * e
            for e, g in zip(self.energies(point), self.degeneracies(point), strict=True)
        ]
        top = max(terms)
        lse = top + mp.log(sum(mp.exp(t - top) for t in terms))
        return terms, lse

    def probabilities(self, point: Point) -> list[Any]:
        """Probability of each *level* (all of its degenerate microstates together)."""
        terms, lse = self.log_weights(point)
        return [mp.exp(t - lse) for t in terms]

    def mean(self, values: Sequence[Any], point: Point) -> Any:
        return sum(p * v for p, v in zip(self.probabilities(point), values, strict=True))

    def central(self, values: Sequence[Any], order: int, point: Point) -> Any:
        m = self.mean(values, point)
        return sum(
            p * (v - m) ** order for p, v in zip(self.probabilities(point), values, strict=True)
        )

    def raw(self, values: Sequence[Any], order: int, point: Point) -> Any:
        return sum(p * v**order for p, v in zip(self.probabilities(point), values, strict=True))

    def quantity(self, name: str, point: Point, *, beta: Any | None = None) -> Any:
        """A base quantity by direct summation; ``beta`` overrides the point's (for differentiation)."""
        p = point if beta is None else {**point, self.problem.beta: beta}
        b = self.beta_of(p)
        _, lse = self.log_weights(p)
        energies = self.energies(p)
        match name:
            case "lnZ":
                return lse
            case "Z":
                return mp.exp(lse)
            case "F":
                return -lse / b
            case "U":
                return self.mean(energies, p)
            case "Var_E":
                return self.central(energies, 2, p)
            case "S":  # Gibbs: -k_B sum_microstates p ln p, with p = P_level / g
                weights = self.probabilities(p)
                gs = self.degeneracies(p)
                return -p[self.problem.k_B] * sum(
                    w * mp.log(w / g) for w, g in zip(weights, gs, strict=True)
                )
        raise ValueError(name)

    def eval_expr(self, expr: sp.Expr, point: Point) -> Any:
        """Evaluate a displayed formula at a sample point."""
        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        return compile_expr(expr, symbols)(*(point[s] for s in symbols))


def close(a: Any, b: Any, tol: Any = REL_TOL) -> bool:
    a, b = mp.mpmathify(a), mp.mpmathify(b)
    if not (mp.isfinite(a) and mp.isfinite(b)):
        return False
    scale = max(mp.mpf(1), abs(a), abs(b))
    return bool(abs(a - b) <= tol * scale)
