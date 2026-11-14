"""Independent numerical reading of a vector expression (mpmath, Cartesian finite differences).

This module never sees the symbolic result's route. The *input* expression is evaluated directly at
Cartesian points: a field value at X is read in the coordinates of the system (the analytic inverse map
`q(X)`), its components are rotated into Cartesian components with a frame found by differencing the map
`x(q)`, and every operator is a central difference in Cartesian directions, nested for nested
operators. No scale factor, no analytic frame and no SymPy derivative of the expression is involved, so a
dropped `1/sin(theta)` or a componentwise vector Laplacian in curvilinear coordinates cannot agree with it.

Abstract fields are replaced by random smooth functions (a polynomial plus sine, cosine and exponential
terms); the symbolic result is evaluated on the same functions. Precision grows with the nesting depth.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mpmath as mp
import sympy as sp

from ..geometry.numeric import close
from .coordinates import TIME, System
from .expressions import Adv, Cross, Curl, Div, Dot, Grad, Lap, Pd, Pos, Unit
from .lower import Declared, Lowering
from .models import VectorCheck, VectorVerification

BASE_DPS = 60
STEP = "1e-10"
TOL = 1e-8
MAX_ATTEMPTS = 30
WANT_POINTS = 6
ASSIGNMENTS = 2

Value = Any  # mp.mpf or a 3x1 mp.matrix


class Skip(Exception):
    """A point where the expression cannot be evaluated (pole, branch cut, non-real value)."""


def _vec(a: Any, b: Any, c: Any) -> Any:
    return mp.matrix([a, b, c])


def _is_vec(v: Value) -> bool:
    return isinstance(v, mp.matrix)


def _dotv(a: Any, b: Any) -> Any:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _real(v: Value) -> Value:
    """The real value, or Skip if a number has a non-negligible imaginary part or is not finite."""
    if _is_vec(v):
        return _vec(*[_real(v[i]) for i in range(3)])
    if isinstance(v, mp.mpc):
        if abs(v.imag) > mp.mpf("1e-30") * max(1, abs(v.real)):
            raise Skip
        v = v.real
    if not mp.isfinite(v):
        raise Skip
    return v


@lru_cache(maxsize=64)
def _function_impl(cls: Any, nargs: int) -> Callable[..., Any]:
    dummies = [sp.Dummy() for _ in range(nargs)]
    return sp.lambdify(dummies, cls(*dummies), modules="mpmath")  # type: ignore[no-any-return]


def random_field(args: list[sp.Symbol], rng: random.Random) -> sp.Expr:
    """A random smooth real function of `args`: entire, with no vanishing high derivatives."""

    def r(lo: int, hi: int, den: int) -> sp.Rational:
        return sp.Rational(rng.randint(lo, hi), den)

    expr: sp.Expr = r(-8, 8, 8)
    for a in args:
        expr += r(-8, 8, 8) * a + r(-3, 3, 10) * a**2 + r(-2, 2, 20) * a**3
    phase1 = sum((r(-6, 6, 4) * a for a in args), sp.Integer(0)) + r(0, 7, 4)
    phase2 = sum((r(-6, 6, 4) * a for a in args), sp.Integer(0)) + r(0, 7, 4)
    expr += r(2, 8, 8) * sp.sin(phase1) + r(1, 6, 8) * sp.cos(phase2)
    if args:
        expr += r(1, 4, 8) * sp.exp(r(-3, 3, 8) * args[0])
    return expr


@dataclass
class Setup:
    """One random assignment of the abstract fields and of the constants."""

    lowering: Lowering
    fields: dict[str, sp.Expr]  # component name -> concrete expression
    params: dict[sp.Symbol, sp.Expr]


def _param_value(symbol: sp.Symbol, rng: random.Random) -> sp.Rational:
    magnitude = sp.Rational(rng.randint(5, 20), 8)
    if symbol.is_negative or symbol.is_nonpositive:
        return -magnitude
    if symbol.is_positive or symbol.is_nonnegative:
        return magnitude
    return magnitude if rng.random() < 0.5 else -magnitude


def make_setup(lowering: Lowering, rng: random.Random) -> Setup:
    params = {s: _param_value(s, rng) for s in lowering.constants.values()}
    fields: dict[str, sp.Expr] = {}
    for d in lowering.declared.values():
        for i, comp in enumerate(d.comps):
            if d.explicit is not None:
                fields[comp] = d.explicit[i].subs(params)
            else:
                args = [lowering._variable(v) for v in d.depends]
                fields[comp] = random_field(args, rng)
    return Setup(lowering, fields, params)


def fd_frame(x_of_q: Callable[..., Any], q: tuple[Any, Any, Any], h: Any) -> list[Any]:
    """Unit vectors e_j at q: normalised central differences of the map x(q)."""
    out = []
    for j in range(3):
        up, down = list(q), list(q)
        up[j] += h
        down[j] -= h
        a, b = x_of_q(*up), x_of_q(*down)
        d = _vec(*[(a[i] - b[i]) / (2 * h) for i in range(3)])
        out.append(d / mp.norm(d))
    return out


class Evaluator:
    """Evaluates an input expression at Cartesian points by finite differences."""

    def __init__(self, setup: Setup, system: System, t: Any) -> None:
        self.setup = setup
        self.system = system
        self.t = t
        self.h = mp.mpf(STEP)
        self._x = system.x_of_q()
        self._q = system.q_of_x()
        self._compiled: dict[str, Callable[..., Any]] = {}
        symbols = [*setup.lowering.q, setup.lowering.t]
        for comp, expr in setup.fields.items():
            self._compiled[comp] = sp.lambdify(symbols, expr, modules="mpmath")

    # -- geometry (map and frame by differencing) -------------------------------------------
    def frame(self, q: tuple[Any, Any, Any]) -> list[Any]:
        return fd_frame(self._x, q, self.h)

    def physical(self, cart: Value, q: tuple[Any, Any, Any]) -> list[Any]:
        return [_dotv(e, cart) for e in self.frame(q)]

    def _field_at(self, comp: str, q: tuple[Any, Any, Any]) -> Any:
        return self._compiled[comp](*q, self.t)

    # -- evaluation -------------------------------------------------------------------------
    def value(self, e: sp.Basic, X: Value) -> Value:
        return _real(self._value(e, X))

    def _scalar_field(self, e: sp.Symbol, X: Value) -> Any:
        q = self._q(X[0], X[1], X[2])
        d = self.setup.lowering.declared[e.name]
        return self._field_at(d.comps[0], q)

    def _value(self, e: sp.Basic, X: Value) -> Value:
        low = self.setup.lowering
        if isinstance(e, sp.Symbol):
            name = e.name
            if e.is_commutative is False:
                q = self._q(X[0], X[1], X[2])
                d = low.declared[name]
                comps = [self._field_at(c, q) for c in d.comps]
                frame = self.frame(q)
                out = _vec(0, 0, 0)
                for j in range(3):
                    out += frame[j] * comps[j]
                return out
            if name == TIME:
                return self.t
            if name in self.system.coords:
                return self._q(X[0], X[1], X[2])[self.system.coords.index(name)]
            if name in low.declared:
                return self._scalar_field(e, X)
            raise Skip
        if isinstance(e, sp.Rational):
            return mp.mpf(e.p) / mp.mpf(e.q)
        if e.is_Number or e.is_Atom:
            return mp.mpmathify(sp.N(e, BASE_DPS))
        if isinstance(e, Grad):
            return self._grad(e.args[0], X)
        if isinstance(e, Div):
            return self._div(e.args[0], X)
        if isinstance(e, Curl):
            return self._curl(e.args[0], X)
        if isinstance(e, Lap):
            return self._lap(e.args[0], X)
        if isinstance(e, Dot):
            return _dotv(self.value(e.args[0], X), self.value(e.args[1], X))
        if isinstance(e, Cross):
            a, b = self.value(e.args[0], X), self.value(e.args[1], X)
            return _vec(
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
        if isinstance(e, Pd):
            return self._pd(e.args[0], e.args[1], X)
        if isinstance(e, Adv):
            return self._adv(e.args[0], e.args[1], X)
        if isinstance(e, Pos):
            return X
        if isinstance(e, Unit):
            q = self._q(X[0], X[1], X[2])
            return self.frame(q)[self.system.coords.index(e.args[0].name)]
        if isinstance(e, sp.Add):
            total: Value = 0
            for a in e.args:
                total = total + self.value(a, X)
            return total
        if isinstance(e, sp.Mul):
            scalar: Any = mp.mpf(1)
            vector: Value | None = None
            for a in e.args:
                v = self.value(a, X)
                if _is_vec(v):
                    vector = v
                else:
                    scalar = scalar * v
            return scalar if vector is None else vector * scalar
        if isinstance(e, sp.Pow):
            base, exponent = self.value(e.args[0], X), self.value(e.args[1], X)
            if base == 0 and exponent < 0:
                raise Skip
            return mp.power(base, exponent) if base >= 0 else mp.mpmathify(base**exponent)
        args = [self.value(a, X) for a in e.args]
        try:
            return _function_impl(type(e), len(args))(*args)  # type: ignore[arg-type]
        except (ValueError, ZeroDivisionError, TypeError, OverflowError) as exc:
            raise Skip from exc

    # -- Cartesian finite differences -------------------------------------------------------
    def _partial(self, e: sp.Basic, X: Value, k: int) -> Value:
        up, down = X.copy(), X.copy()
        up[k] += self.h
        down[k] -= self.h
        return (self.value(e, up) - self.value(e, down)) / (2 * self.h)

    def _grad(self, e: sp.Basic, X: Value) -> Value:
        return _vec(*[self._partial(e, X, k) for k in range(3)])

    def _jacobian_rows(self, e: sp.Basic, X: Value) -> list[Value]:
        return [self._partial(e, X, k) for k in range(3)]  # d_k of the vector

    def _div(self, e: sp.Basic, X: Value) -> Value:
        return sum((self._partial(e, X, k)[k] for k in range(3)), mp.mpf(0))

    def _curl(self, e: sp.Basic, X: Value) -> Value:
        d = self._jacobian_rows(e, X)  # d[k][i] = d_k e_i
        return _vec(d[1][2] - d[2][1], d[2][0] - d[0][2], d[0][1] - d[1][0])

    def _lap(self, e: sp.Basic, X: Value) -> Value:
        """Componentwise Cartesian Laplacian: the sum of second differences, nested."""
        total: Value = 0
        for k in range(3):
            up, down = X.copy(), X.copy()
            up[k] += self.h
            down[k] -= self.h
            total = total + (self._partial_at(e, up, k) - self._partial_at(e, down, k)) / (
                2 * self.h
            )
        return total

    def _partial_at(self, e: sp.Basic, X: Value, k: int) -> Value:
        return self._partial(e, X, k)

    def _pd(self, e: sp.Basic, var: sp.Symbol, X: Value) -> Value:
        if var.name == TIME:
            saved = self.t
            try:
                self.t = saved + self.h
                up = self.value(e, X)
                self.t = saved - self.h
                down = self.value(e, X)
            finally:
                self.t = saved
            return (up - down) / (2 * self.h)
        j = self.system.coords.index(var.name)
        q = list(self._q(X[0], X[1], X[2]))
        qp, qm = list(q), list(q)
        qp[j] += self.h
        qm[j] -= self.h
        xp, xm = self._x(*qp), self._x(*qm)
        return (self.value(e, _vec(*xp)) - self.value(e, _vec(*xm))) / (2 * self.h)

    def _adv(self, a: sp.Basic, b: sp.Basic, X: Value) -> Value:
        direction = self.value(a, X)
        total: Value = 0
        for k in range(3):
            total = total + self._partial(b, X, k) * direction[k]
        return total


# -- symbolic result on the same assignment ---------------------------------------------------
def result_function(
    setup: Setup, results: list[sp.Expr]
) -> Callable[[tuple[Any, Any, Any], Any], list[Any]]:
    low = setup.lowering
    jets: dict[sp.Symbol, sp.Expr] = {}
    for symbol, (comp, word) in low.jets.items():
        expr = setup.fields[comp]
        for v in word:
            expr = sp.diff(expr, low._variable(v))
        jets[symbol] = expr
    concrete = [r.xreplace(jets).subs(setup.params) for r in results]
    fn = sp.lambdify([*low.q, low.t], concrete, modules="mpmath")

    def call(q: tuple[Any, Any, Any], t: Any) -> list[Any]:
        return list(fn(*q, t))

    return call


def verify_result(
    lowering: Lowering,
    expression: sp.Basic,
    results: list[sp.Expr],
    is_vector: bool,
    *,
    seed: int,
    deadline: float,
    depth: int,
) -> VectorVerification:
    """Compare the symbolic components `results` with the independent evaluation of `expression`."""
    system = lowering.system
    rng = random.Random(seed)  # noqa: S311  (sampling points, not security)
    checked = agreed = skipped = 0
    worst = 0.0
    failed_note = ""
    with mp.workdps(BASE_DPS + 10 * max(0, depth - 2)):
        for _ in range(ASSIGNMENTS):
            setup = make_setup(lowering, rng)
            expr = expression.subs(setup.params)
            fn = result_function(setup, results)
            attempts = 0
            target = WANT_POINTS // ASSIGNMENTS
            got = 0
            while got < target and attempts < MAX_ATTEMPTS:
                if time.monotonic() > deadline:
                    break
                attempts += 1
                q = system.sample(rng)
                t = mp.mpf(rng.uniform(0.1, 1.0))
                ev = Evaluator(setup, system, t)
                try:
                    X = _vec(*ev._x(*q))
                    numeric = ev.value(expr, X)
                    symbolic = [_real(c) for c in fn(q, t)]
                    reference = ev.physical(numeric, q) if is_vector else [numeric]
                except (Skip, ValueError, ZeroDivisionError, TypeError, OverflowError):
                    skipped += 1
                    continue
                got += 1
                checked += 1
                scale = max([1, *[abs(c) for c in reference], *[abs(c) for c in symbolic]])
                residual = max(abs(a - b) for a, b in zip(reference, symbolic, strict=True))
                worst = max(worst, float(residual / scale))
                if all(close(a, b, TOL, scale) for a, b in zip(reference, symbolic, strict=True)):
                    agreed += 1
                elif not failed_note:
                    failed_note = "the symbolic result disagrees with the independent evaluation"
    passed = checked >= WANT_POINTS // 2 and agreed == checked
    note = failed_note or (
        "" if passed else f"only {checked} usable sample point(s) (needed {WANT_POINTS // 2})"
    )
    return VectorVerification(
        numeric_passed=passed,
        points_checked=checked,
        points_agreed=agreed,
        points_skipped=skipped,
        max_residual=worst,
        checks=[
            VectorCheck(
                name="cartesian_finite_difference",
                passed=passed,
                note=note or f"agreed at {agreed} points on {ASSIGNMENTS} random field assignments",
            )
        ],
        note=note,
    )


def declared_fields(lowering: Lowering) -> dict[str, Declared]:
    return lowering.declared


def _compare(reference: list[Any], symbolic: list[Any]) -> tuple[bool, float]:
    scale = max([1, *[abs(c) for c in reference], *[abs(c) for c in symbolic]])
    residual = max(abs(a - b) for a, b in zip(reference, symbolic, strict=True))
    ok = all(close(a, b, TOL, scale) for a, b in zip(reference, symbolic, strict=True))
    return ok, float(residual / scale)


def _verdict(
    name: str, checked: int, agreed: int, skipped: int, worst: float, wanted: int, detail: str
) -> VectorVerification:
    passed = checked >= wanted and agreed == checked
    note = (
        ""
        if passed
        else (
            "the symbolic result disagrees with the independent evaluation"
            if agreed != checked
            else f"only {checked} usable sample point(s) (needed {wanted})"
        )
    )
    return VectorVerification(
        method=name,
        numeric_passed=passed,
        points_checked=checked,
        points_agreed=agreed,
        points_skipped=skipped,
        max_residual=worst,
        checks=[VectorCheck(name=name, passed=passed, note=note or detail)],
        note=note,
    )


def verify_jacobian(
    system: System,
    matrix: sp.Matrix,
    determinant: sp.Expr,
    *,
    seed: int,
    deadline: float,
) -> VectorVerification:
    """The symbolic Jacobian dx/dq and its determinant against central differences of x(q)."""
    rng = random.Random(seed)  # noqa: S311
    fn = sp.lambdify(list(system.symbols()), [matrix, determinant], modules="mpmath")
    x_of_q = system.x_of_q()
    checked = agreed = skipped = 0
    worst = 0.0
    with mp.workdps(BASE_DPS):
        h = mp.mpf(STEP)
        for _ in range(WANT_POINTS * 2):
            if checked >= WANT_POINTS or time.monotonic() > deadline:
                break
            q = system.sample(rng)
            try:
                sym_matrix, sym_det = fn(*q)
                rows = [[mp.mpf(0)] * 3 for _ in range(3)]
                for j in range(3):
                    up, down = list(q), list(q)
                    up[j] += h
                    down[j] -= h
                    a, b = x_of_q(*up), x_of_q(*down)
                    for i in range(3):
                        rows[i][j] = (a[i] - b[i]) / (2 * h)
                reference = [rows[i][j] for i in range(3) for j in range(3)]
                reference.append(mp.det(mp.matrix(rows)))
                symbolic = [_real(sym_matrix[i, j]) for i in range(3) for j in range(3)]
                symbolic.append(_real(sym_det))
            except (Skip, ValueError, ZeroDivisionError, TypeError, OverflowError):
                skipped += 1
                continue
            ok, residual = _compare(reference, symbolic)
            checked += 1
            agreed += int(ok)
            worst = max(worst, residual)
    return _verdict(
        "central_difference_jacobian",
        checked,
        agreed,
        skipped,
        worst,
        WANT_POINTS // 2,
        f"agreed at {agreed} points",
    )


def verify_conversion(
    source: System,
    target: System,
    source_values: list[sp.Expr],
    target_values: list[sp.Expr],
    is_vector: bool,
    symbols: tuple[sp.Symbol, sp.Symbol],
    constants: list[sp.Symbol],
    *,
    seed: int,
    deadline: float,
) -> VectorVerification:
    """Components in the target system against the same field read in the source system.

    The route is independent of the symbolic conversion: a target point is mapped to Cartesian
    coordinates and back into the source coordinates by the analytic inverse map, the source
    components are evaluated there and turned into Cartesian components with a frame found by
    differencing the source map, and those are projected on a frame found by differencing the target map.
    """
    rng = random.Random(seed)  # noqa: S311
    params = {c: _param_value(c, rng) for c in constants}
    t_symbol = symbols[1]
    src_fn = sp.lambdify(
        [*source.symbols(), t_symbol], [v.subs(params) for v in source_values], modules="mpmath"
    )
    dst_fn = sp.lambdify(
        [*target.symbols(), t_symbol], [v.subs(params) for v in target_values], modules="mpmath"
    )
    x_target, x_source, q_source = target.x_of_q(), source.x_of_q(), source.q_of_x()
    checked = agreed = skipped = 0
    worst = 0.0
    with mp.workdps(BASE_DPS):
        h = mp.mpf(STEP)
        for _ in range(WANT_POINTS * 4):
            if checked >= WANT_POINTS or time.monotonic() > deadline:
                break
            qt = target.sample(rng)
            t = mp.mpf(rng.uniform(0.1, 1.0))
            try:
                X = x_target(*qt)
                qs = q_source(*X)
                values = [_real(v) for v in src_fn(*qs, t)]
                if is_vector:
                    frame_source = fd_frame(x_source, qs, h)
                    cart = _vec(0, 0, 0)
                    for j in range(3):
                        cart += frame_source[j] * values[j]
                    reference = [_dotv(e, cart) for e in fd_frame(x_target, qt, h)]
                else:
                    reference = values
                symbolic = [_real(v) for v in dst_fn(*qt, t)]
            except (Skip, ValueError, ZeroDivisionError, TypeError, OverflowError):
                skipped += 1
                continue
            ok, residual = _compare(reference, symbolic)
            checked += 1
            agreed += int(ok)
            worst = max(worst, residual)
    return _verdict(
        "cartesian_roundtrip",
        checked,
        agreed,
        skipped,
        worst,
        WANT_POINTS // 2,
        f"agreed at {agreed} points",
    )
