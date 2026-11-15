"""Independent numerical evaluation of a canonical ensemble.

Nothing here uses the symbolic route that produces the displayed formulas. The energies and
degeneracies (or the density of states, or the Hamiltonian) are evaluated at a sample point with
mpmath: finite and infinite sums by direct enumeration with a tail test, integrals by adaptive
quadrature with an error estimate, weights by log-sum-exp. Derivative checks differentiate these
numbers with a fourth-order central stencil, never the symbolic expression.

Every oracle gives, for ONE copy of the system, ``ln Z`` and the cumulants of the energy; copies,
free energy, entropy and the rest are built from those here. Numerical agreement is evidence, not proof.
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
SEED = "minerva-statmech-v1"

Point = dict[sp.Symbol, Any]  # symbol -> mpmath number


def sample_points(
    problem: Problem, count: int = POINTS, *, extra: Sequence[sp.Symbol] = ()
) -> list[Point]:
    """Assumption-respecting sample points: exact rationals, then mpmath numbers.

    ``beta``, ``k_B`` and ``T = 1/(k_B beta)`` are always consistent with each other.
    """
    rng = random.Random(  # noqa: S311 - deterministic sampling, not cryptographic
        f"{SEED}:{len(problem.energies)}:{len(problem.sample_symbols)}:{problem.kind}"
    )
    symbols = sorted(set(problem.sample_symbols) | set(extra), key=lambda s: s.name)
    out: list[Point] = []
    with mp.workdps(DPS):
        for _ in range(count):
            exact = {s: _generic(s, rng) for s in symbols}
            point: Point = {s: _mp(v) for s, v in exact.items()}
            point[problem.T] = 1 / (point[problem.k_B] * point[problem.beta])
            out.append(point)
    return out


def _generic(symbol: sp.Symbol, rng: random.Random) -> sp.Rational:
    """A sample that respects the assumptions and is not zero: closed forms from eigenvalues and
    derivatives are 0/0 at special points, and the parameters are generic."""
    value = sample_symbol(symbol, rng)
    for _ in range(12):
        if value != 0:
            break
        value = sample_symbol(symbol, rng)
    return value


def _mp(value: sp.Expr) -> Any:
    return mp.mpf(value.p) / mp.mpf(value.q) if value.is_Rational else mp.mpf(str(sp.N(value, DPS)))


def compile_expr(expr: sp.Expr, symbols: Sequence[sp.Symbol]) -> Callable[..., Any]:
    return sp.lambdify(list(symbols), expr, modules="mpmath")  # type: ignore[no-any-return]


def close(a: Any, b: Any, tol: Any) -> bool:
    a, b = mp.mpmathify(a), mp.mpmathify(b)
    if not (mp.isfinite(a) and mp.isfinite(b)):
        return False
    scale = max(mp.mpf(1), abs(a), abs(b))
    return bool(abs(a - b) <= tol * scale)


# Fourth-order central differences: order -> (offsets, (numerator, denominator) coefficients).
# Exact fractions, turned into numbers at the working precision when used.
_STENCILS: dict[int, tuple[tuple[int, ...], tuple[tuple[int, int], ...]]] = {
    1: ((-2, -1, 1, 2), ((1, 12), (-2, 3), (2, 3), (-1, 12))),
    2: ((-2, -1, 0, 1, 2), ((-1, 12), (4, 3), (-5, 2), (4, 3), (-1, 12))),
    3: ((-3, -2, -1, 1, 2, 3), ((1, 8), (-1, 1), (13, 8), (-13, 8), (1, 1), (-1, 8))),
    4: (
        (-3, -2, -1, 0, 1, 2, 3),
        ((-1, 6), (2, 1), (-13, 2), (28, 3), (-13, 2), (2, 1), (-1, 6)),
    ),
}


def derivative(f: Callable[[Any], Any], x: Any, order: int, h: Any) -> Any:
    """Fourth-order central difference of a numerically evaluated function (order 1 to 4)."""
    offsets, coefficients = _STENCILS[order]
    total = sum(
        (mp.mpf(num) / den) * f(x + k * h)
        for k, (num, den) in zip(offsets, coefficients, strict=True)
    )
    return total / h**order


def cumulants_from_raw(raw: Sequence[Any], upto: int) -> list[Any]:
    """Cumulants kappa_1..kappa_upto from raw moments m_0..m_upto (m_0 is the normalisation)."""
    m = [r / raw[0] for r in raw[: upto + 1]]
    out: list[Any] = []
    if upto >= 1:
        out.append(m[1])
    if upto >= 2:
        out.append(m[2] - m[1] ** 2)
    if upto >= 3:
        out.append(m[3] - 3 * m[2] * m[1] + 2 * m[1] ** 3)
    if upto >= 4:
        out.append(m[4] - 4 * m[3] * m[1] - 3 * m[2] ** 2 + 12 * m[2] * m[1] ** 2 - 6 * m[1] ** 4)
    return out


class BaseOracle:
    """Canonical-ensemble quantities from a single-copy ``ln Z`` and energy cumulants."""

    precision = DPS
    exact = True  # enumeration to working precision; numerical derivatives up to order 4
    tol = mp.mpf(10) ** -18  # agreement of a formula with the enumeration
    diff_tol = mp.mpf(10) ** -12  # agreement of a formula with a numerical derivative
    step = mp.mpf(10) ** -6

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        count = problem.copies_count
        self._count = (
            compile_expr(count, sorted(count.free_symbols, key=lambda s: s.name))
            if count is not None
            else None
        )
        self._count_symbols = (
            sorted(count.free_symbols, key=lambda s: s.name) if count is not None else []
        )

    # -- to be provided -----------------------------------------------------------------
    def lnZ1(self, point: Point, beta: Any) -> Any:
        raise NotImplementedError

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        raise NotImplementedError

    # -- derived ------------------------------------------------------------------------
    def copies(self, point: Point) -> Any:
        if self._count is None:
            return mp.mpf(1)
        return self._count(*(point[s] for s in self._count_symbols))

    def quantity(self, name: str, point: Point, *, beta: Any | None = None) -> Any:
        p = point if beta is None else {**point, self.problem.beta: beta}
        b = p[self.problem.beta]
        with mp.workdps(self.precision):
            n = self.copies(p)
            ln_z = n * self.lnZ1(p, b)
            if self.problem.identical and self.problem.copies_count is not None:
                ln_z -= mp.loggamma(n + 1)
            match name:
                case "lnZ":
                    return ln_z
                case "Z":
                    return mp.exp(ln_z)
                case "F":
                    return -ln_z / b
                case "U":
                    return n * self.cumulants1(p, b, 1)[0]
                case "Var_E":
                    return n * self.cumulants1(p, b, 2)[1]
                case "S":
                    u = n * self.cumulants1(p, b, 1)[0]
                    return p[self.problem.k_B] * (ln_z + b * u)
        raise ValueError(name)

    def energy_cumulant(self, point: Point, order: int) -> Any:
        b = point[self.problem.beta]
        with mp.workdps(self.precision):
            return self.copies(point) * self.cumulants1(point, b, order)[order - 1]

    def eval_expr(self, expr: sp.Expr, point: Point) -> Any:
        """Evaluate a displayed formula at a sample point."""
        symbols = sorted(expr.free_symbols, key=lambda s: s.name)
        with mp.workdps(self.precision):
            return compile_expr(expr, symbols)(*(point[s] for s in symbols))

    def derivative_of(self, name: str, point: Point, variable: str, order: int) -> Any:
        """Numerical derivative of an enumerated base quantity with respect to a state variable."""
        p = self.problem
        with mp.workdps(self.precision):
            if variable == p.beta.name:
                x = point[p.beta]
                return derivative(
                    lambda t: self.quantity(name, point, beta=t),
                    x,
                    order,
                    self.step * max(1, abs(x)),
                )
            if variable == p.T.name:
                x = point[p.T]
                return derivative(
                    lambda t: self.quantity(name, point, beta=1 / (point[p.k_B] * t)),
                    x,
                    order,
                    self.step * max(1, abs(x)),
                )
            symbol = next(s for s in p.params if s.name == variable)
            x = point[symbol]
            return derivative(
                lambda t: self.quantity(name, {**point, symbol: t}),
                x,
                order,
                self.step * max(1, abs(x)),
            )


class Oracle(BaseOracle):
    """Direct sums over an explicit list of levels (log-sum-exp)."""

    def __init__(self, problem: Problem) -> None:
        super().__init__(problem)
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

    # -- single copy ---------------------------------------------------------------------
    def lnZ1(self, point: Point, beta: Any) -> Any:
        return self.log_weights(point, beta)[1]

    def cumulants1(self, point: Point, beta: Any, upto: int) -> list[Any]:
        p = {**point, self.problem.beta: beta}
        e = self.energies(p)
        c2 = self.central(e, 2, p)
        out = [self.mean(e, p), c2, self.central(e, 3, p), self.central(e, 4, p) - 3 * c2**2]
        return out[:upto]

    def quantity(self, name: str, point: Point, *, beta: Any | None = None) -> Any:
        if name == "S" and self.problem.copies_count is None:
            # Gibbs entropy of one system: -k_B sum over microstates of p ln p, with p = P_level / g
            p = point if beta is None else {**point, self.problem.beta: beta}
            with mp.workdps(self.precision):
                weights = self.probabilities(p)
                gs = self.degeneracies(p)
                return -p[self.problem.k_B] * sum(
                    w * mp.log(w / g) for w, g in zip(weights, gs, strict=True)
                )
        return super().quantity(name, point, beta=beta)
