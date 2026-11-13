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
from .expressions import Commutation, Exp, Frechet, Sym, Term, Terms

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
    var: sp.Symbol | None = None  # the variable the time-dependent matrices depend on
    dependent: frozenset[str] = frozenset()  # matrices that are functions of `var`
    _derivatives: dict[tuple[str, int], sp.Matrix] = field(default_factory=dict)

    def derivative_entries(self, name: str, order: int) -> sp.Matrix:
        """Entries of an explicit time-dependent matrix differentiated `order` times."""
        key = (name, order)
        if key not in self._derivatives:
            assert self.var is not None  # noqa: S101 - set whenever a matrix is time dependent
            self._derivatives[key] = self.explicit[name].diff(self.var, order)
        return self._derivatives[key]


Rebuild = Callable[[dict[sp.Symbol, mp.mpc]], dict[str, mp.matrix]]


@dataclass
class Env:
    scalars: dict[sp.Symbol, mp.mpc]
    mats: dict[str, mp.matrix]
    rebuild: Rebuild | None = None  # recomputes the time-dependent matrices for new scalars

    def at(self, var: sp.Symbol, value: mp.mpc) -> Env:
        """The same sample with the variable moved: time-dependent matrices follow it."""
        scalars = {**self.scalars, var: value}
        mats = dict(self.mats)
        if self.rebuild is not None:
            mats.update(self.rebuild(scalars))
        return Env(scalars, mats, self.rebuild)


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
    def factor(self, f: Sym | Exp | Frechet, env: Env) -> mp.matrix:
        if isinstance(f, Sym):
            m = env.mats[f.name + "'" * f.deriv]
            return m if f.power == 1 else m**f.power
        if isinstance(f, Frechet):
            return self.frechet_block(f, env)
        total: mp.matrix | None = None
        for name, c in f.arg:
            piece = self.scalar(c, env.scalars) * env.mats[name]
            total = piece if total is None else total + piece
        assert total is not None  # noqa: S101 - an Exp always has at least one matrix
        return mp.expm(total)

    def exponent(self, f: Exp, env: Env) -> mp.matrix:
        total: mp.matrix | None = None
        for name, c in f.arg:
            piece = self.scalar(c, env.scalars) * env.mats[name]
            total = piece if total is None else total + piece
        assert total is not None  # noqa: S101 - an Exp always has at least one matrix
        return total

    def direction(self, f: Frechet, env: Env) -> mp.matrix:
        total: mp.matrix | None = None
        for name, d, c in f.direction:
            piece = self.scalar(c, env.scalars) * env.mats[name + "'" * d]
            total = piece if total is None else total + piece
        if total is None:
            n = env.mats[f.base.names[0]].rows
            return mp.zeros(n, n)
        return total

    def frechet_block(self, f: Frechet, env: Env) -> mp.matrix:
        """The Frechet derivative through the exponential of [[X, D], [0, X]] (upper right block)."""
        x, d = self.exponent(f.base, env), self.direction(f, env)
        n = x.rows
        block = mp.zeros(2 * n, 2 * n)
        for i in range(n):
            for j in range(n):
                block[i, j] = x[i, j]
                block[n + i, n + j] = x[i, j]
                block[i, n + j] = d[i, j]
        e = mp.expm(block)
        out = mp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                out[i, j] = e[i, n + j]
        return out

    def frechet_quadrature(self, f: Frechet, env: Env) -> tuple[mp.matrix, mp.mpf]:
        """The same quantity from its definition: the integral of exp((1-s)X) D exp(sX) ds."""
        x, d = self.exponent(f.base, env), self.direction(f, env)
        return quad_matrix(
            lambda s: mp.expm((1 - s) * x) * d * mp.expm(s * x),
            mp.mpf(0),
            mp.mpf(1),
            (x.rows, x.cols),
        )

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


def _poly_of(base: mp.matrix, rng: random.Random) -> mp.matrix:
    n = base.rows
    a0, a1, a2 = (mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3))
    return a0 * mp.eye(n) + a1 * base + a2 * base * base


def _explicit_matrix(m: sp.Matrix, ev: Evaluator, scalars: dict[sp.Symbol, mp.mpc]) -> mp.matrix:
    out = mp.matrix(*m.shape)
    for i in range(m.rows):
        for j in range(m.cols):
            out[i, j] = ev.scalar(sp.sympify(m[i, j]), scalars)
    return out


def make_env(setup: Setup, ev: Evaluator, rng: random.Random) -> Env:
    scalars = {s: _to_mp(_sample(s, rng)) for s in ev.symbols}
    mats: dict[str, mp.matrix] = {}
    for name, m in setup.explicit.items():
        mats[name] = _explicit_matrix(m, ev, scalars)
    bases: dict[str, mp.matrix] = {}
    for group in setup.cliques:
        fixed = [n for n in group if n in mats]
        n = setup.shapes[group[0]][0]
        base = mats[fixed[0]] if fixed else _random_matrix(n, n, rng)
        for name in group:
            bases[name] = base
            if name in mats or name in setup.dependent:
                continue
            mats[name] = _poly_of(base, rng)
    # abstract time-dependent matrices: A(t) = C0 + t C1 + t^2 C2, with every C_k in the
    # commutant the declarations describe (a polynomial of the clique's base matrix, or of a
    # private base matrix when the matrix is declared to commute with itself at all times)
    coeffs: dict[str, list[mp.matrix]] = {}
    for name in sorted(setup.dependent):
        if name in setup.explicit:
            continue
        rows, cols = setup.shapes[name]
        if name in bases:
            coeffs[name] = [_poly_of(bases[name], rng) for _ in range(3)]
        elif name in setup.comm.self_commuting:
            base = _random_matrix(rows, rows, rng)
            coeffs[name] = [_poly_of(base, rng) for _ in range(3)]
        else:
            coeffs[name] = [_random_matrix(rows, cols, rng) for _ in range(3)]
    for name, (r, c) in setup.shapes.items():
        if name not in mats and name not in coeffs:
            mats[name] = _random_matrix(r, c, rng)

    var = setup.var
    moving = [n for n in setup.dependent if n in setup.explicit]

    def rebuild(values: dict[sp.Symbol, mp.mpc]) -> dict[str, mp.matrix]:
        assert var is not None  # noqa: S101 - rebuild exists only with a variable
        t = values[var]
        out: dict[str, mp.matrix] = {}
        for name, (c0, c1, c2) in coeffs.items():
            out[name] = c0 + t * c1 + t * t * c2
            out[name + "'"] = c1 + 2 * t * c2
            out[name + "''"] = 2 * c2
        for name in moving:
            out[name] = _explicit_matrix(setup.explicit[name], ev, values)
            out[name + "'"] = _explicit_matrix(setup.derivative_entries(name, 1), ev, values)
            out[name + "''"] = _explicit_matrix(setup.derivative_entries(name, 2), ev, values)
        return out

    env = Env(scalars, mats, rebuild if (coeffs or moving) else None)
    if env.rebuild is not None and var is not None:
        env.mats.update(rebuild(scalars))
    return env


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
            plus = f(env.at(var, x0 + unit * STEP))
            minus = f(env.at(var, x0 - unit * STEP))
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
            return integrand(env.at(var, t))

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


def _exp_series(x: Poly, order: int) -> Poly:
    series: Poly = {(): sp.Integer(1)}
    power: Poly = {(): sp.Integer(1)}
    for k in range(1, order + 1):
        power = poly_scale(poly_mul(power, x, order), sp.Rational(1, k))
        series = poly_add(series, power)
    return series


def _frechet_series(f: Frechet, order: int) -> Poly:
    """Series of the integral of exp((1-s)X) D exp(sX): the Beta integral of (1-s)^i s^j gives
    the coefficient 1/(i+j+1)! on X^i D X^j (derived here, not by differentiating a series)."""
    x: Poly = {}
    for name, c in f.base.arg:
        x = poly_add(x, letter(name, c))
    d: Poly = {}
    for name, deriv, c in f.direction:
        if deriv:
            raise ValueError("expand time-dependent matrices before taking a series")
        d = poly_add(d, letter(name, c))
    powers: list[Poly] = [{(): sp.Integer(1)}]
    for _ in range(order):
        powers.append(poly_mul(powers[-1], x, order))
    out: Poly = {}
    for i in range(order):
        for j in range(order - i):
            piece = poly_mul(poly_mul(powers[i], d, order), powers[j], order)
            out = poly_add(out, poly_scale(piece, sp.Rational(1, sp.factorial(i + j + 1))))
    return out


def series_of_terms(terms: Terms, order: int = SERIES_ORDER) -> Poly:
    total: Poly = {}
    for t in terms:
        product: Poly = {(): t.coeff}
        for f in t.factors:
            if isinstance(f, Sym):
                if f.power < 1:
                    raise ValueError("no series for an inverse")
                if f.deriv:
                    raise ValueError("expand time-dependent matrices before taking a series")
                product = poly_mul(product, {(f.name,) * f.power: sp.Integer(1)}, order)
            elif isinstance(f, Frechet):
                product = poly_mul(product, _frechet_series(f, order), order)
            else:
                x: Poly = {}
                for name, c in f.arg:
                    x = poly_add(x, letter(name, c))
                product = poly_mul(product, _exp_series(x, order), order)
        total = poly_add(total, product)
    return total


# -- time-dependent matrices for the exact leg -------------------------------------------------
DEGREE = 2  # A(t) is expanded as A@0 + t A@1 + t^2 A@2: the same family the numeric leg samples


def _pieces(name: str, dependent: frozenset[str]) -> list[str]:
    return [f"{name}@{k}" for k in range(DEGREE + 1)] if name in dependent else [name]


def expand_dependent(
    terms: Terms, var: sp.Symbol, dependent: frozenset[str], comm: Commutation
) -> tuple[Terms, Commutation]:
    """Replace each time-dependent matrix by a polynomial in `var` with constant matrix
    coefficients, so that a formal series (letters are constant) can differentiate it. The identities
    checked this way hold for every such polynomial, hence for the entries given."""

    def derived(name: str, d: int) -> list[tuple[str, sp.Expr]]:
        out = []
        for k in range(d, DEGREE + 1):
            out.append((f"{name}@{k}", sp.ff(k, d) * var ** (k - d)))
        return out

    def exp_items(items: tuple[tuple[str, sp.Expr], ...]) -> tuple[tuple[str, sp.Expr], ...]:
        merged: dict[str, sp.Expr] = {}
        for n, c in items:
            if n in dependent:
                for piece, factor in derived(n, 0):
                    merged[piece] = sp.expand(merged.get(piece, sp.Integer(0)) + c * factor)
            else:
                merged[n] = sp.expand(merged.get(n, sp.Integer(0)) + c)
        return tuple(sorted((n, c) for n, c in merged.items() if c != 0))

    out_terms: list[Term] = []
    for term in terms:
        partial: list[tuple[sp.Expr, tuple[Sym | Exp | Frechet, ...]]] = [(term.coeff, ())]
        for f in term.factors:
            options: list[tuple[sp.Expr, tuple[Sym | Exp | Frechet, ...]]]
            if isinstance(f, Sym) and f.name in dependent:
                if f.power > 3:
                    raise ValueError("power of a time-dependent matrix too large to expand")
                seqs: list[tuple[sp.Expr, tuple[Sym | Exp | Frechet, ...]]] = [(sp.Integer(1), ())]
                for index in range(f.power):
                    d = f.deriv if index == 0 else 0
                    seqs = [
                        (c * coef, (*syms, Sym(piece)))
                        for c, syms in seqs
                        for piece, coef in derived(f.name, d)
                    ]
                options = seqs
            elif isinstance(f, Exp):
                options = [(sp.Integer(1), (Exp(exp_items(f.arg)),))]
            elif isinstance(f, Frechet):
                direction: list[tuple[str, int, sp.Expr]] = []
                for n, d, c in f.direction:
                    if n in dependent:
                        direction += [(p, 0, c * coef) for p, coef in derived(n, d)]
                    else:
                        direction.append((n, 0, c))
                frechet = Frechet(Exp(exp_items(f.base.arg)), tuple(direction))
                options = [(sp.Integer(1), (frechet,))]
            else:
                options = [(sp.Integer(1), (f,))]
            partial = [(c0 * c1, (*factors, *g)) for c0, factors in partial for c1, g in options]
        out_terms += [Term(sp.expand(c), fs) for c, fs in partial]
    pairs: set[frozenset[str]] = set()
    for pair in comm.pairs:
        a, b = sorted(pair)
        for x in _pieces(a, dependent):
            for y in _pieces(b, dependent):
                pairs.add(frozenset((x, y)))
    for name in comm.self_commuting:
        letters = _pieces(name, dependent)
        for x in letters:
            for y in letters:
                if x != y:
                    pairs.add(frozenset((x, y)))
    return tuple(out_terms), Commutation(frozenset(pairs))


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
