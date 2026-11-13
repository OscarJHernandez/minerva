"""Independent numerical evaluation for probability families: sampling, sums, quadrature.

These are the checks that do not use the route that produced a symbolic result: mpmath quadrature or
summation of the density at sampled parameter values. Numerical agreement is evidence, not a proof.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping

import mpmath as mp
import sympy as sp

from .domain import decide
from .families import Family, Param, x

DPS = 30
REL_TOL = mp.mpf(10) ** -8
TAIL_R1 = mp.mpf(10) ** 3
TAIL_R2 = mp.mpf(10) ** 5


def sample_symbol(symbol: sp.Symbol, rng: random.Random, unit: bool = False) -> sp.Rational:
    if unit:
        return sp.Rational(rng.randint(1, 9), 10)
    if symbol.is_integer:
        if symbol.is_positive:
            return sp.Integer(rng.randint(1, 6))
        return sp.Integer(rng.randint(0, 6) if symbol.is_nonnegative else rng.randint(-4, 4))
    if symbol.is_positive:
        return sp.Rational(rng.randint(2, 16), 4)
    if symbol.is_nonnegative:
        return sp.Rational(rng.randint(0, 16), 4)
    if symbol.is_negative:
        return -sp.Rational(rng.randint(2, 16), 4)
    return sp.Rational(rng.randint(-12, 12), 4)


def sample_parameters(
    exprs: Mapping[str, sp.Expr],
    params: tuple[Param, ...],
    rng: random.Random,
    count: int,
    attempts: int = 200,
) -> list[tuple[dict[sp.Symbol, sp.Rational], dict[str, sp.Expr]]]:
    """Assignments of the free symbols of the parameter expressions that respect every constraint."""
    unit = {
        e
        for p in params
        if p.constraint == "unit_open" and isinstance((e := exprs[p.name]), sp.Symbol)
    }
    symbols = sorted({s for e in exprs.values() for s in e.free_symbols}, key=lambda s: s.name)
    out: list[tuple[dict[sp.Symbol, sp.Rational], dict[str, sp.Expr]]] = []
    for _ in range(attempts):
        if len(out) >= count:
            break
        point = {s: sample_symbol(s, rng, unit=s in unit) for s in symbols}
        values = {name: sp.nsimplify(e.subs(point)) for name, e in exprs.items()}
        if all(decide(p.constraint, values[p.name]) is not False for p in params):
            out.append((point, values))
    return out


def _lo_hi(
    family: Family, symbols: dict[str, sp.Symbol], values: dict[str, sp.Expr]
) -> tuple[mp.mpf, mp.mpf]:
    lo, hi = family.support(symbols)
    subs = {symbols[k]: v for k, v in values.items() if k in symbols}
    return _mp(lo.subs(subs)), _mp(hi.subs(subs))


def _mp(value: sp.Expr) -> mp.mpf:
    if value == sp.oo:
        return mp.inf
    if value == -sp.oo:
        return -mp.inf
    return mp.mpf(str(sp.N(value, DPS + 5)))


class Numeric:
    """Density of a family as an mpmath function, with moments by quadrature or summation."""

    def __init__(
        self, family: Family, parametrization: str | None, values: dict[str, sp.Expr]
    ) -> None:
        self.family = family
        names = [p.name for p in family.params(parametrization)]
        self.symbols = {n: family.symbols[n] for n in names}
        density = family.density(self.symbols, x)
        # dummies: a parameter called `beta` would shadow mpmath's beta function in the generated code
        dummies = {name: sp.Dummy(f"p{k}") for k, name in enumerate(self.symbols)}
        density = density.xreplace({self.symbols[n]: d for n, d in dummies.items()})
        self._f = sp.lambdify([x, *dummies.values()], density, "mpmath")
        self.args = [_mp(values[n]) for n in names]
        self.lo, self.hi = _lo_hi(family, self.symbols, values)

    def pdf(self, t: mp.mpf) -> mp.mpf:
        return self._f(t, *self.args)

    def _breaks(self) -> list[mp.mpf]:
        lo, hi = self.lo, self.hi
        if lo == -mp.inf and hi == mp.inf:
            return [lo, mp.mpf(-10), mp.mpf(0), mp.mpf(10), hi]
        if hi == mp.inf:
            return [lo, lo + 1, lo + 10, hi]
        return [lo, hi]

    def expectation(self, g: Callable[[mp.mpf], mp.mpf]) -> tuple[mp.mpf, mp.mpf]:
        """(E g(X), error estimate) by summation or adaptive quadrature."""
        with mp.workdps(DPS):
            if self.family.discrete:
                if self.hi == mp.inf:
                    # terms decay factorially for the families here; nsum has no error estimate
                    return mp.nsum(lambda t: g(t) * self.pdf(t), [self.lo, mp.inf]), mp.mpf(0)
                total = mp.fsum(
                    g(mp.mpf(t)) * self.pdf(mp.mpf(t))
                    for t in range(int(self.lo), int(self.hi) + 1)
                )
                return total, mp.mpf(0)
            value, err = mp.quad(lambda t: g(t) * self.pdf(t), self._breaks(), error=True)
            return value, err

    def moment(self, k: int) -> tuple[mp.mpf, mp.mpf]:
        return self.expectation(lambda t: t**k)

    def absolute_moment_exists(self, k: int) -> bool:
        """Absolute convergence of E|X|^k: a finite support is enough; otherwise the tails must shrink.

        Quadrature "agrees" with a divergent integral (principal value of the Cauchy mean is 0, with an
        error estimate of 0), so convergence is established first, from the tails of |x|^k f.
        """
        if self.family.discrete and self.hi != mp.inf:
            return True
        if not self.family.discrete and self.lo != -mp.inf and self.hi != mp.inf:
            return True
        with mp.workdps(DPS):
            ends = []
            if self.hi == mp.inf:
                ends.append(1)
            if self.lo == -mp.inf:
                ends.append(-1)
            for sign in ends:
                if self.family.discrete:
                    return True  # the discrete families here have terms that decay factorially
                g = lambda t, s=sign: abs(t) ** k * self.pdf(s * t)  # noqa: E731
                t1, e1 = mp.quad(g, [TAIL_R1, mp.inf], error=True)
                t2, e2 = mp.quad(g, [TAIL_R2, mp.inf], error=True)
                if t1 <= mp.mpf(10) ** -40 and t2 <= mp.mpf(10) ** -40:
                    continue
                if not (
                    t2 <= t1 / 20 and e1 <= REL_TOL * max(1, t1) and e2 <= REL_TOL * max(1, t1)
                ):
                    return False
            return True
