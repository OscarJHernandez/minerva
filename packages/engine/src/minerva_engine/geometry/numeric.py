"""Independent numerical differential geometry (mpmath).

Nothing here uses SymPy differentiation or simplification: the metric (and any tensor field) is lambdified,
evaluated at assumption-respecting sample points, and every derivative is a central difference. The
formulas are written again in numerical form, so a result from `compute.py` is compared with a route that
shares only the definitions, not the code that produced it. Declared functions are replaced by random
polynomials (a different assignment per evaluator); the symbolic route keeps them abstract.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import mpmath as mp
import sympy as sp
from sympy.core.function import AppliedUndef

from ..verify import _sample
from .setup import Geometry
from .tensors import Tensor, apply_matrix

DPS = 50
STEP = "1e-9"
MAX_CONDITION = 1e8  # points where the metric is nearly singular are not used

Point = tuple[list[Any], list[Any]]  # (coordinate values, parameter values)


def random_polynomial(args: tuple[sp.Symbol, ...], rng: random.Random) -> sp.Lambda:
    """A random real cubic with a positive constant term, as a function of `args`."""
    poly: sp.Expr = sp.Rational(100 + rng.randint(0, 100), 100)
    for a in args:
        poly += (
            sp.Rational(rng.randint(-30, 30), 100) * a
            + sp.Rational(rng.randint(-5, 5), 100) * a**2
            + sp.Rational(rng.randint(-1, 1), 100) * a**3
        )
    return sp.Lambda(args, poly)


class Evaluator:
    """Evaluates the metric, candidate expressions and the numerical geometry at points."""

    def __init__(self, geo: Geometry, assignment: dict[str, sp.Lambda] | None) -> None:
        self.geo = geo
        self.n = geo.n
        self.assignment = assignment or {}
        self._compiled: dict[str, Callable[..., Any]] = {}
        self._riemann: dict[Any, Tensor] = {}
        self._metric = self._compile("metric", list(geo.g))

    # -- compiling -----------------------------------------------------------------------
    def substitute(self, expr: sp.Expr) -> sp.Expr:
        if not self.assignment:
            return expr
        out = expr
        for name, lam in self.assignment.items():
            out = out.replace(
                lambda e, name=name: isinstance(e, AppliedUndef) and type(e).__name__ == name,
                lambda e, lam=lam: lam(*e.args),
            )
        return out.doit()

    def _compile(
        self, key: str, exprs: list[sp.Expr], extra: list[sp.Symbol] | None = None
    ) -> Callable[..., Any]:
        if key not in self._compiled:
            subs = [self.substitute(sp.sympify(e)) for e in exprs]
            self._compiled[key] = sp.lambdify(
                [*self.geo.symbols, *(extra or [])], subs, modules="mpmath"
            )
        return self._compiled[key]

    def candidate(
        self,
        key: str,
        items: list[tuple[Any, sp.Expr]],
        point: Point,
        extra_symbols: list[sp.Symbol] | None = None,
        extra_values: list[Any] | None = None,
    ) -> dict[Any, Any]:
        """Values of `items` ((index, expression) pairs, cached under `key`) at `point`.

        `extra_symbols` are further arguments of the expressions (the velocities of a geodesic).
        """
        f = self._compile(key, [e for _, e in items], extra_symbols or [])
        x, p = point
        values = f(*x, *p, *(extra_values or []))
        return {idx: v for (idx, _), v in zip(items, values, strict=True)}

    # -- metric --------------------------------------------------------------------------
    def metric(self, x: list[Any], p: list[Any]) -> list[list[Any]]:
        values = self._metric(*x, *p)
        n = self.n
        return [[values[i * n + j] for j in range(n)] for i in range(n)]

    def inverse_metric(self, x: list[Any], p: list[Any]) -> Any:
        return mp.inverse(mp.matrix(self.metric(x, p)))

    def is_regular(self, point: Point) -> bool:
        x, p = point
        try:
            g = mp.matrix(self.metric(x, p))
            if any(not mp.isfinite(v) for v in g):
                return False
            ginv = mp.inverse(g)
            return bool(mp.mnorm(g, 1) * mp.mnorm(ginv, 1) < MAX_CONDITION)
        except (ZeroDivisionError, ValueError, ArithmeticError, TypeError):
            return False

    # -- derivatives ------------------------------------------------------------------------
    def _shift(self, x: list[Any], axis: int, h: Any) -> list[Any]:
        out = list(x)
        out[axis] = out[axis] + h
        return out

    def partial_metric(self, x: list[Any], p: list[Any], h: Any) -> list[list[list[Any]]]:
        """`dg[s][m][v] = d_s g_{mv}` by central differences."""
        out = []
        for s in range(self.n):
            gp = self.metric(self._shift(x, s, h), p)
            gm = self.metric(self._shift(x, s, -h), p)
            out.append(
                [[(gp[m][v] - gm[m][v]) / (2 * h) for v in range(self.n)] for m in range(self.n)]
            )
        return out

    def christoffel(self, x: list[Any], p: list[Any], h: Any) -> Tensor:
        n = self.n
        ginv = self.inverse_metric(x, p)
        dg = self.partial_metric(x, p, h)
        out = Tensor(n, ("up", "down", "down"))
        for rho in range(n):
            for mu in range(n):
                for nu in range(n):
                    total = mp.mpf(0)
                    for s in range(n):
                        total += ginv[rho, s] * (dg[mu][s][nu] + dg[nu][s][mu] - dg[s][mu][nu])
                    out.data[(rho, mu, nu)] = total / 2
        return out

    def riemann(self, x: list[Any], p: list[Any], h: Any) -> Tensor:
        """`R^r_{smn}` from central differences of the numerical Christoffel symbols (cached per point)."""
        key = (tuple(x), tuple(p), h)
        if key not in self._riemann:
            self._riemann[key] = self._riemann_at(x, p, h)
        return self._riemann[key]

    def _riemann_at(self, x: list[Any], p: list[Any], h: Any) -> Tensor:
        n = self.n
        gam = self.christoffel(x, p, h)
        dgam = []
        for m in range(n):
            plus = self.christoffel(self._shift(x, m, h), p, h)
            minus = self.christoffel(self._shift(x, m, -h), p, h)
            dgam.append({k: (plus.data[k] - minus.data[k]) / (2 * h) for k in plus.data})
        out = Tensor(n, ("up", "down", "down", "down"))
        for rho in range(n):
            for sigma in range(n):
                for mu in range(n):
                    for nu in range(n):
                        total = dgam[mu][(rho, nu, sigma)] - dgam[nu][(rho, mu, sigma)]
                        for lam in range(n):
                            total += gam.data[(rho, mu, lam)] * gam.data[(lam, nu, sigma)]
                            total -= gam.data[(rho, nu, lam)] * gam.data[(lam, mu, sigma)]
                        out.data[(rho, sigma, mu, nu)] = total
        return out

    # -- curvature contractions (from the numerical Riemann tensor) ---------------------------
    def ricci(self, riem: Tensor) -> Tensor:
        n = self.n
        out = Tensor(n, ("down", "down"))
        for s in range(n):
            for v in range(n):
                out.data[(s, v)] = sum((riem.data[(r, s, r, v)] for r in range(n)), mp.mpf(0))
        return out

    def scalar(self, ric: Tensor, ginv: Any) -> Any:
        n = self.n
        return sum((ginv[m, v] * ric.data[(m, v)] for m in range(n) for v in range(n)), mp.mpf(0))

    def einstein(self, ric: Tensor, g: list[list[Any]], scalar: Any) -> Tensor:
        n = self.n
        out = Tensor(n, ("down", "down"))
        for m in range(n):
            for v in range(n):
                out.data[(m, v)] = ric.data[(m, v)] - g[m][v] * scalar / 2
        return out

    def kretschmann(self, riem: Tensor, g: list[list[Any]], ginv: Any) -> Any:
        lowered = apply_matrix(riem, 0, lambda a, b: g[a][b], "down")
        raised = lowered
        for axis in range(4):
            raised = apply_matrix(raised, axis, lambda a, b: ginv[a, b], "up")
        return sum((v * raised.get(i) for i, v in lowered.data.items()), mp.mpf(0))

    def ricci_squared(self, ric: Tensor, ginv: Any) -> Any:
        up = apply_matrix(ric, 0, lambda a, b: ginv[a, b], "up")
        up = apply_matrix(up, 1, lambda a, b: ginv[a, b], "up")
        return sum((v * up.get(i) for i, v in ric.data.items()), mp.mpf(0))


def step() -> Any:
    return mp.mpf(STEP)


def sample_point(geo: Geometry, rng: random.Random) -> Point:
    """A random point consistent with the declared assumptions (real coordinates, parameters as declared)."""
    x = [mp.mpf(float(sp.re(_sample(c, rng)))) for c in geo.coords]
    p: list[Any] = []
    for s in geo.params:
        value = complex(_sample(s, rng))
        p.append(mp.mpf(value.real) if s.is_real else mp.mpc(value))
    return x, p


def make_evaluators(geo: Geometry, rng: random.Random, count: int) -> list[Evaluator]:
    """One evaluator without declared functions; `count` with different random polynomials otherwise."""
    if not geo.function_classes:
        return [Evaluator(geo, None)]
    by_name = {c.name: c for c in geo.coords}
    evaluators = []
    for _ in range(count):
        assignment = {
            name: random_polynomial(tuple(by_name[a] for a in geo.functions[name]), rng)
            for name in geo.function_classes
        }
        evaluators.append(Evaluator(geo, assignment))
    return evaluators


def close(a: Any, b: Any, tol: float, scale: Any = 0) -> bool:
    """`|a - b| <= tol * max(1, |a|, |b|, scale)`.

    `scale` is the size of the terms that cancel in an identity: a residual of 1e-9 is noise when it
    is the difference of terms of size 1e3, and is only measured against 1 when nothing else is known.
    """
    try:
        size = max(1.0, float(abs(a)), float(abs(b)), float(abs(scale)))
        return bool(abs(a - b) <= tol * size)
    except (TypeError, ValueError, OverflowError):
        return False
