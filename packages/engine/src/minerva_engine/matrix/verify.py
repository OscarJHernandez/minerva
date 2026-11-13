"""Independent checks for matrix results.

Two legs, reported separately:

* **numeric** (decides): random matrices of the declared sizes (declared commuting matrices are
  sampled as polynomials of one random matrix, explicit matrices at sampled parameter values);
  exponentials by `mpmath.expm`, integrals by `mpmath.quad`, derivatives by a central difference.
  None of this uses the code path that produced the candidate.
* **symbolic** (exact, but only through a finite order): both sides are expanded as formal power
  series in the matrices, in the free algebra modulo the declared commutations, and compared
  coefficient by coefficient up to total degree `SERIES_ORDER`. An agreement is evidence, not a
  proof; a disagreement is a refutation.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

from ..errors import LimitError
from ..verify import MIN_VALID_POINTS, REL_TOL, SAMPLE_ATTEMPTS, _sample
from .expressions import Commutation, Exp, Sym, Terms

DIGITS = 25
QUAD_DIGITS = 16  # quadrature needs no more than the 1e-9 comparison; this keeps it fast
STEP = mp.mpf("1e-12")
SERIES_ORDER = 4
QUAD_ERROR = mp.mpf("1e-12")


@dataclass
class Setup:
    """What the checks need to know about the declared matrices."""

    shapes: dict[str, tuple[int, int]]
    explicit: dict[str, sp.Matrix] = field(default_factory=dict)
    comm: Commutation = field(default_factory=Commutation)
    cliques: list[list[str]] = field(default_factory=list)  # mutually commuting groups (>= 2)


@dataclass
class Env:
    scalars: dict[sp.Symbol, mp.mpc]
    mats: dict[str, mp.matrix]


class Evaluator:
    """Evaluates scalar SymPy expressions at mpmath values (compiled once with lambdify)."""

    def __init__(self, symbols: list[sp.Symbol]) -> None:
        self.symbols = sorted(set(symbols), key=lambda s: s.name)
        self._cache: dict[sp.Basic, Callable[..., mp.mpc]] = {}

    def scalar(self, expr: sp.Expr, values: dict[sp.Symbol, mp.mpc]) -> mp.mpc:
        fn = self._cache.get(expr)
        if fn is None:
            fn = sp.lambdify(self.symbols, expr, modules="mpmath")
            self._cache[expr] = fn
        out = fn(*[values[s] for s in self.symbols])
        if not mp.isfinite(out):
            raise ArithmeticError("not finite")
        return mp.mpmathify(out)

    # -- matrices ------------------------------------------------------------------------
    def factor(self, f: Sym | Exp, env: Env) -> mp.matrix:
        if isinstance(f, Sym):
            m = env.mats[f.name]
            return m if f.power == 1 else m**f.power
        total: mp.matrix | None = None
        for name, c in f.arg:
            piece = self.scalar(c, env.scalars) * env.mats[name]
            total = piece if total is None else total + piece
        assert total is not None  # noqa: S101 - an Exp always has at least one matrix
        return mp.expm(total)

    def terms(self, terms: Terms, env: Env, size: int | None = None) -> mp.matrix:
        """Value of the terms; a term without matrix factors is `size` times the identity."""
        total: mp.matrix | None = None
        for t in terms:
            product: mp.matrix | None = None
            for f in t.factors:
                value = self.factor(f, env)
                product = value if product is None else product * value
            if product is None:
                if size is None:
                    raise ArithmeticError("scalar term")
                product = mp.eye(size)
            piece = self.scalar(t.coeff, env.scalars) * product
            total = piece if total is None else total + piece
        if total is None:
            raise ArithmeticError("empty expression")
        return total


# -- sampling ---------------------------------------------------------------------------------
def _to_mp(v: sp.Expr) -> mp.mpc:
    return mp.mpc(complex(v))


def _random_matrix(rows: int, cols: int, rng: random.Random) -> mp.matrix:
    m = mp.matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m[i, j] = mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1))
    return m


def make_env(setup: Setup, ev: Evaluator, rng: random.Random) -> Env:
    scalars = {s: _to_mp(_sample(s, rng)) for s in ev.symbols}
    mats: dict[str, mp.matrix] = {}
    for name, m in setup.explicit.items():
        shape = m.shape
        out = mp.matrix(*shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[i, j] = ev.scalar(sp.sympify(m[i, j]), scalars)
        mats[name] = out
    for group in setup.cliques:
        fixed = [n for n in group if n in mats]
        n = setup.shapes[group[0]][0]
        base = mats[fixed[0]] if fixed else _random_matrix(n, n, rng)
        for name in group:
            if name in mats:
                continue
            a0, a1, a2 = (mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3))
            mats[name] = a0 * mp.eye(n) + a1 * base + a2 * base * base
    for name, (r, c) in setup.shapes.items():
        if name not in mats:
            mats[name] = _random_matrix(r, c, rng)
    return Env(scalars, mats)


def max_abs(m: mp.matrix) -> mp.mpf:
    return max((abs(m[i, j]) for i in range(m.rows) for j in range(m.cols)), default=mp.mpf(0))


def close(a: mp.matrix, b: mp.matrix, tol: float = REL_TOL) -> bool:
    if a.rows != b.rows or a.cols != b.cols:
        return False
    return bool(max_abs(a - b) <= tol * max(1, max_abs(a), max_abs(b)))


@dataclass
class Count:
    checked: int = 0
    agreed: int = 0

    @property
    def passed(self) -> bool:
        return self.checked >= MIN_VALID_POINTS and self.agreed == self.checked


def numeric_points(
    setup: Setup,
    ev: Evaluator,
    check: Callable[[Env], bool | None],
    *,
    deadline: float,
    max_points: int = 8,
    seed: int = 20260101,
    digits: int = DIGITS,
) -> Count:
    """Run `check` at random environments. It returns None for an unusable point."""
    rng = random.Random(seed)  # noqa: S311 - test points, not security
    count = Count()
    with mp.workdps(digits):
        for _ in range(SAMPLE_ATTEMPTS):
            if count.checked >= max_points:
                break
            if time.monotonic() > deadline:
                raise LimitError("The time budget was used up before the check finished.")
            env = make_env(setup, ev, rng)
            try:
                verdict = check(env)
            except (ArithmeticError, ValueError, ZeroDivisionError, mp.libmp.NoConvergence):
                continue
            if verdict is None:
                continue
            count.checked += 1
            count.agreed += bool(verdict)
    return count


# -- specific checks -------------------------------------------------------------------------
def check_equal(
    setup: Setup,
    ev: Evaluator,
    left: Callable[[Env], mp.matrix],
    right: Callable[[Env], mp.matrix],
    *,
    deadline: float,
) -> Count:
    return numeric_points(setup, ev, lambda env: close(left(env), right(env)), deadline=deadline)


def check_derivative(
    setup: Setup,
    ev: Evaluator,
    var: sp.Symbol,
    f: Callable[[Env], mp.matrix],
    candidate: Callable[[Env], mp.matrix],
    *,
    deadline: float,
) -> tuple[Count, list[str]]:
    """Central difference of `f` against `candidate`, along the imaginary direction too unless real."""
    directions = ["real"] + ([] if var.is_real is True else ["imaginary"])

    def check(env: Env) -> bool | None:
        x0 = env.scalars[var]
        want = candidate(env)
        for direction in directions:
            unit = mp.mpf(1) if direction == "real" else mp.mpc(0, 1)
            plus = f(Env({**env.scalars, var: x0 + unit * STEP}, env.mats))
            minus = f(Env({**env.scalars, var: x0 - unit * STEP}, env.mats))
            if not close((plus - minus) / (2 * unit * STEP), want):
                return False
        return True

    return numeric_points(setup, ev, check, deadline=deadline), directions


def quad_matrix(
    f: Callable[[mp.mpc], mp.matrix], a: mp.mpc, b: mp.mpc, shape: tuple[int, int]
) -> tuple[mp.matrix, mp.mpf]:
    """Entrywise Gauss-Legendre quadrature of a matrix function; nodes are shared by entries."""
    cache: dict[mp.mpc, mp.matrix] = {}

    def node(t: mp.mpc) -> mp.matrix:
        if t not in cache:
            cache[t] = f(t)
        return cache[t]

    out = mp.zeros(*shape)
    worst = mp.mpf(0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            value, err = mp.quad(
                lambda t, i=i, j=j: node(t)[i, j],
                [a, b],
                error=True,
                method="gauss-legendre",
                maxdegree=6,
            )
            out[i, j] = value
            worst = max(worst, err / max(1, abs(value)))
    return out, worst


def check_integral(
    setup: Setup,
    ev: Evaluator,
    var: sp.Symbol,
    integrand: Callable[[Env], mp.matrix],
    lower: sp.Expr,
    upper: sp.Expr,
    shape: tuple[int, int],
    judge: Callable[[Env, mp.matrix], bool],
    *,
    deadline: float,
) -> Count:
    """Quadrature of the integrand at sampled parameters; `judge(env, X)` decides agreement."""

    def check(env: Env) -> bool | None:
        a = ev.scalar(lower, env.scalars)
        b = ev.scalar(upper, env.scalars)

        def f(t: mp.mpc) -> mp.matrix:
            return integrand(Env({**env.scalars, var: t}, env.mats))

        value, err = quad_matrix(f, a, b, shape)
        if err > QUAD_ERROR:
            return None  # the quadrature itself did not converge: not a usable point
        return judge(env, value)

    return numeric_points(
        setup, ev, check, deadline=deadline, max_points=MIN_VALID_POINTS, digits=QUAD_DIGITS
    )


# -- exact leg: truncated formal series -----------------------------------------------------
Poly = dict[tuple[str, ...], sp.Expr]


def _trace_key(word: tuple[str, ...], comm: Commutation) -> tuple[tuple[int, str], ...]:
    """Canonical form of a word modulo the declared commutations (Foata normal form)."""
    levels: list[int] = []
    for k, x in enumerate(word):
        level = 1
        for j in range(k):
            if not comm.commutes(word[j], x):
                level = max(level, levels[j] + 1)
        levels.append(level)
    return tuple(sorted(zip(levels, word, strict=True)))


def poly_mul(p: Poly, q: Poly, order: int = SERIES_ORDER) -> Poly:
    out: Poly = {}
    for w1, c1 in p.items():
        for w2, c2 in q.items():
            if len(w1) + len(w2) <= order:
                w = w1 + w2
                out[w] = out.get(w, sp.Integer(0)) + c1 * c2
    return out


def poly_add(p: Poly, q: Poly, sign: int = 1) -> Poly:
    out = dict(p)
    for w, c in q.items():
        out[w] = out.get(w, sp.Integer(0)) + sign * c
    return out


def poly_scale(p: Poly, c: sp.Expr) -> Poly:
    return {w: c * v for w, v in p.items()}


def letter(name: str, coeff: sp.Expr | None = None) -> Poly:
    return {(name,): sp.Integer(1) if coeff is None else coeff}


def series_of_terms(terms: Terms, order: int = SERIES_ORDER) -> Poly:
    total: Poly = {}
    for t in terms:
        product: Poly = {(): t.coeff}
        for f in t.factors:
            if isinstance(f, Sym):
                if f.power < 1:
                    raise ValueError("no series for an inverse")
                product = poly_mul(product, {(f.name,) * f.power: sp.Integer(1)}, order)
            else:
                x: Poly = {}
                for name, c in f.arg:
                    x = poly_add(x, letter(name, c))
                series: Poly = {(): sp.Integer(1)}
                power: Poly = {(): sp.Integer(1)}
                for k in range(1, order + 1):
                    power = poly_scale(poly_mul(power, x, order), sp.Rational(1, k))
                    series = poly_add(series, power)
                product = poly_mul(product, series, order)
        total = poly_add(total, product)
    return total


def poly_integrate(p: Poly, var: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> Poly:
    return {w: sp.integrate(sp.expand(c), (var, lower, upper)) for w, c in p.items()}


def poly_diff(p: Poly, var: sp.Symbol) -> Poly:
    return {w: sp.diff(c, var) for w, c in p.items()}


def poly_subs(p: Poly, var: sp.Symbol, value: sp.Expr) -> Poly:
    return {w: c.subs(var, value) for w, c in p.items()}


def poly_equal(p: Poly, q: Poly, comm: Commutation, order: int = SERIES_ORDER) -> bool:
    """Equality of truncated series modulo the declared commutations."""

    def normal(poly: Poly) -> dict[tuple[tuple[int, str], ...], sp.Expr]:
        out: dict[tuple[tuple[int, str], ...], sp.Expr] = {}
        for w, c in poly.items():
            if len(w) <= order:
                key = _trace_key(w, comm)
                out[key] = out.get(key, sp.Integer(0)) + c
        return out

    a, b = normal(p), normal(q)
    for key in a.keys() | b.keys():
        diff = sp.expand(a.get(key, sp.Integer(0)) - b.get(key, sp.Integer(0)))
        if diff != 0 and sp.simplify(diff) != 0:
            return False
    return True


def series_terms_equal(a: Terms, b: Terms, comm: Commutation) -> bool | None:
    """Exact comparison through SERIES_ORDER; None when a series cannot be formed (inverses)."""
    try:
        return poly_equal(series_of_terms(a), series_of_terms(b), comm)
    except ValueError:
        return None
