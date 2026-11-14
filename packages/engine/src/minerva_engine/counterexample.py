"""Search for a counterexample to a proposed identity `a = b`.

A counterexample *disproves* a universal identity. Failing to find one proves nothing: the result says which
probe families were tried and how many points were evaluated, never "the identity holds".

Points are exact rationals (complex rationals for symbols without assumptions), so a counterexample can be
reproduced and re-evaluated at any precision. A disagreement found at 30 digits only counts if it survives
60 and 120 digits with a stable residual well above rounding noise; one that vanishes with precision is
cancellation, counted but never reported as a counterexample.

Probe families (every point respects the declared assumptions of its symbols):
  random    ordinary and wide regions, like `verify` but at exact rational points.
  boundary  where an argument of Abs, Max/Min, Piecewise, floor/ceiling, sign, Heaviside, KroneckerDelta, log
            or a fractional power changes sign or touches zero, probed on both sides of that point.
  branch    large imaginary parts (multiples of pi) for symbols without assumptions: log(exp(z)) and the
            inverse trigonometric functions only fail beyond the principal strip.
  special   integers outside the small pool `verify` draws from; for other symbols simple values (0, +-1, 1/2...).
  scale     very large and very small magnitudes.

A point where exactly one side is undefined is *not* a counterexample here (it can be a removable
singularity); it is counted in `domain_differences`. Noncommuting symbols and distributions are `unsupported`:
use `ncverify` and the distribution operation. The module is deterministic for a given seed and point budget;
the wall-clock limit only truncates, and `exhausted` says so. A `Counterexample` holds expressions: it is returned
to the caller and never logged (a test checks that no field name is in a log allow-list).
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import mpmath
import sympy as sp

from .distributions.guard import contains_distribution
from .ncverify import noncommuting_symbols

ProbeKind = Literal["random", "boundary", "branch", "special", "scale"]
KINDS: tuple[ProbeKind, ...] = ("random", "boundary", "branch", "special", "scale")
# What `verify_equal` runs after its own random points agreed: `random` would repeat them.
TARGETED: tuple[ProbeKind, ...] = ("boundary", "scale", "branch", "special")

SCREEN_DPS = 30
CONFIRM_DPS = (60, 120)
NOISE_FLOOR = (
    1e-20  # relative to max(1, |a|, |b|); far above 30-digit rounding, far below any real error
)
STABILITY = 1e-3  # the residual at 60 and 120 digits must agree to this relative amount
GRID_INTEGERS = 60
BOUNDARY_ROOTS_PER_ARGUMENT = 6


@dataclass(frozen=True)
class Budget:
    """`points` caps evaluated probe points (deterministic); `seconds` is a safety stop only."""

    points: int = 96
    seconds: float = 2.0


@dataclass(frozen=True)
class Counterexample:
    """A point where the two sides differ, with what is needed to reproduce it. Returned, never logged."""

    expression_a: str
    expression_b: str
    assumptions: dict[str, list[str]]
    substitution: dict[str, str]
    value_a: str
    value_b: str
    residual: str
    probe: ProbeKind
    dps: int
    seed: int
    minimised: bool
    values: dict[sp.Symbol, sp.Expr] = field(repr=False, compare=False)


@dataclass(frozen=True)
class SearchResult:
    counterexample: Counterexample | None
    probes_tried: int = 0
    kinds_tried: tuple[ProbeKind, ...] = ()
    cancellation_suspected: int = 0  # screened as different at 30 digits, equal at higher precision
    domain_differences: int = 0  # defined on one side only: reported, not a counterexample
    exhausted: bool = False  # the budget ended the search before every probe was tried
    unsupported: str | None = None  # "distribution" or "noncommutative": nothing was searched


# ---------------------------------------------------------------------------------------------------------
# Exact points that respect assumptions


_FLAGS = (
    "integer",
    "rational",
    "real",
    "positive",
    "negative",
    "nonnegative",
    "nonpositive",
    "nonzero",
)


def _assumption_names(symbol: sp.Symbol) -> list[str]:
    return [flag for flag in _FLAGS if getattr(symbol, f"is_{flag}") is True]


def _admits(symbol: sp.Symbol, value: sp.Expr) -> bool:
    """Whether `value` is an allowed value for `symbol` under its declared assumptions."""
    return all(getattr(value, f"is_{flag}") is True for flag in _assumption_names(symbol))


def _snap(x: float, digits: int = 6) -> sp.Rational:
    """A few significant digits, as an exact rational: reproducible and short to print."""
    return sp.Rational(f"{x:.{digits}g}")


def _draw(symbol: sp.Symbol, rng: random.Random, magnitude: float) -> sp.Expr:
    if symbol.is_integer:
        top = max(1, int(magnitude))
        pool = [n for n in range(-top, top + 1) if _admits(symbol, sp.Integer(n))]
        return sp.Integer(rng.choice(pool)) if pool else sp.Integer(1)
    real = rng.uniform(-magnitude, magnitude)
    if symbol.is_positive:
        real = abs(real) + magnitude / 1000
    elif symbol.is_negative:
        real = -abs(real) - magnitude / 1000
    elif symbol.is_nonnegative:
        real = abs(real)
    elif symbol.is_nonpositive:
        real = -abs(real)
    value: sp.Expr = _snap(real)
    if symbol.is_real is not True:
        value = value + sp.I * _snap(rng.uniform(-min(magnitude, 10.0), min(magnitude, 10.0)))
    if value == 0 and symbol.is_nonzero:
        value = sp.Integer(1)
    return value if _admits(symbol, value) else sp.Integer(1)


def _draw_all(
    symbols: list[sp.Symbol], rng: random.Random, magnitude: float
) -> dict[sp.Symbol, sp.Expr]:
    return {s: _draw(s, rng, magnitude) for s in symbols}


# ---------------------------------------------------------------------------------------------------------
# Evaluation at a given precision

_POLE = object()  # genuinely undefined here (zoo, oo, nan, division by zero)


def _has_symbolic_power(expr: sp.Expr) -> bool:
    """A power whose exponent contains a symbol (a**b): after substitution it is an arbitrary rational."""
    return any(
        isinstance(node, sp.Pow) and not node.exp.is_number for node in sp.preorder_traversal(expr)
    )


GUARD_DIGITS = 80


def _inputs(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr], dps: int) -> dict[sp.Symbol, sp.Expr]:
    """Exact points keep `evalf` adaptive: it adds precision against cancellation (sin^2 + cos^2 far from
    the real axis loses hundreds of digits). A power with a symbolic exponent (885**(30864/25)) is the exception: with exact
    inputs SymPy extracts roots symbolically, which can take minutes, so those go in as floats with guard digits.
    Fixed exponents such as sqrt stay exact: float noise on a branch cut (an argument that is exactly negative
    real) would flip the branch and reject sqrt(x^2 + 2x + 1) = sqrt((x + 1)^2).
    """
    if not _has_symbolic_power(expr):
        return values
    return {s: v if v.is_Integer else v.evalf(dps + GUARD_DIGITS) for s, v in values.items()}


def _value(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr], dps: int) -> mpmath.mpc | object | None:
    """The value at `dps` digits, `_POLE` when undefined, `None` when it could not be evaluated."""
    try:
        number = expr.xreplace(_inputs(expr, values, dps)).evalf(dps)
        if number is sp.nan or number.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return _POLE
        if not number.is_number:
            return None
        re_, im_ = number.as_real_imag()
        with mpmath.workdps(dps):
            return mpmath.mpc(
                mpmath.mpf(sp.Float(re_, dps)._mpf_), mpmath.mpf(sp.Float(im_, dps)._mpf_)
            )
    except ZeroDivisionError:
        return _POLE
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None


@dataclass(frozen=True)
class _Gap:
    residual: mpmath.mpf
    scale: mpmath.mpf
    value_a: mpmath.mpc
    value_b: mpmath.mpc


def _gap(a: sp.Expr, b: sp.Expr, values: dict[sp.Symbol, sp.Expr], dps: int) -> _Gap | str | None:
    """The difference at `dps` digits; "pole" if exactly one side is undefined; None if unusable."""
    va, vb = _value(a, values, dps), _value(b, values, dps)
    if (va is _POLE) != (vb is _POLE):
        return "pole"
    if not isinstance(va, mpmath.mpc) or not isinstance(vb, mpmath.mpc):
        return None
    with mpmath.workdps(dps):
        scale = max(mpmath.mpf(1), abs(va), abs(vb))
        return _Gap(abs(va - vb), scale, va, vb)


def _confirmed(a: sp.Expr, b: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> _Gap | None:
    """A gap that persists at 60 and 120 digits with a stable residual; None for cancellation noise."""
    gaps: list[_Gap] = []
    for dps in CONFIRM_DPS:
        g = _gap(a, b, values, dps)
        if not isinstance(g, _Gap):
            return None
        gaps.append(g)
    low, high = gaps
    with mpmath.workdps(CONFIRM_DPS[-1]):
        if high.residual <= NOISE_FLOOR * high.scale:
            return None
        if abs(low.residual - high.residual) > STABILITY * high.residual:
            return None
    return high


# ---------------------------------------------------------------------------------------------------------
# Probe families


def _critical_arguments(expr: sp.Expr) -> list[sp.Expr]:
    """Expressions whose sign change or zero is where the formula changes branch or form."""
    found: list[sp.Expr] = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Abs | sp.sign | sp.Heaviside | sp.log):
            found.append(node.args[0])
        elif isinstance(node, sp.floor | sp.ceiling):
            g = node.args[0]
            found.extend([g, g - 1, g + 1])
        elif isinstance(node, sp.Max | sp.Min):
            found.extend(x - y for x, y in itertools.combinations(node.args, 2))
        elif isinstance(node, sp.KroneckerDelta):
            found.append(node.args[0] - node.args[1])
        elif isinstance(node, sp.Pow) and node.exp.is_integer is not True:
            found.append(node.base)
        elif isinstance(node, sp.Piecewise):
            for _, cond in node.args:
                found.extend(rel.lhs - rel.rhs for rel in cond.atoms(sp.core.relational.Relational))
    unique: list[sp.Expr] = []
    for g in found:
        g = sp.sympify(g)
        if g.free_symbols and g not in unique:
            unique.append(g)
    return unique


def _scan_grid(symbol: sp.Symbol) -> list[float]:
    if symbol.is_integer:
        return [float(n) for n in range(-GRID_INTEGERS, GRID_INTEGERS + 1)]
    magnitudes = [10 ** (k / 4) for k in range(-12, 25)]
    return sorted({*magnitudes, *(-m for m in magnitudes), 0.0})


def _real_at(f: Callable[[mpmath.mpf], object], x: float) -> float | None:
    """`f(x)` as a finite real float, or None where it is undefined, complex or overflows."""
    try:
        y = f(mpmath.mpf(x))
        if abs(mpmath.im(y)) > 1e-12 * (1 + abs(mpmath.re(y))):
            return None
        r = float(mpmath.re(y))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, NameError, AttributeError):
        return None
    return r if math.isfinite(r) else None


def _bisect(f: Callable[[mpmath.mpf], object], lo: float, hi: float, flo: float) -> float:
    """A point between `lo` and `hi` where `f` changes sign (`flo` is f(lo))."""
    for _ in range(50):
        mid = (lo + hi) / 2
        fm = _real_at(f, mid)
        if fm is None:
            break
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi) / 2


def _refine_extremum(
    f: Callable[[mpmath.mpf], object], lo: float, hi: float, minimise: bool
) -> tuple[float, float] | None:
    """Ternary search for the extremum of `f` on [lo, hi]: a region too narrow for the scan grid."""
    sign = 1.0 if minimise else -1.0
    for _ in range(60):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        fa, fb = _real_at(f, a), _real_at(f, b)
        if fa is None or fb is None:
            return None
        if sign * fa < sign * fb:
            hi = b
        else:
            lo = a
    x = (lo + hi) / 2
    y = _real_at(f, x)
    return None if y is None else (x, y)


def _roots_in(g: sp.Expr, symbol: sp.Symbol) -> list[float]:
    """Real points where `g` (a function of `symbol` alone) changes sign or touches zero."""
    roots: list[float] = []
    try:
        poly = sp.Poly(g, symbol)
        if 1 <= poly.degree() <= 8:
            roots.extend(
                float(sp.re(r))
                for r in poly.nroots(n=15, maxsteps=100)
                if abs(float(sp.im(r))) < 1e-9
            )
    except (sp.PolynomialError, sp.polys.polyerrors.BasePolynomialError, ValueError, TypeError):
        pass
    try:
        f = sp.lambdify(symbol, g, "mpmath")
    except (TypeError, ValueError, NameError, AttributeError):
        f = None
    if f is not None:
        with mpmath.workdps(15):
            grid = [
                x
                for x in _scan_grid(symbol)
                if not ((symbol.is_positive and x <= 0) or (symbol.is_negative and x >= 0))
            ]
            samples = [(x, _real_at(f, x)) for x in grid]
            for (x0, y0), (x1, y1) in itertools.pairwise(samples):
                if y0 is None or y1 is None:
                    continue  # undefined at one end: a pole or a gap, not a sign change we can bracket
                if y1 == 0.0:
                    roots.append(x1)
                elif y0 != 0.0 and (y0 < 0) != (y1 < 0):
                    if symbol.is_integer:
                        roots.extend([x0, x1])
                    else:
                        roots.append(_bisect(f, x0, x1, y0))
            if not symbol.is_integer:
                roots.extend(_narrow_roots(f, samples))
    unique = sorted({round(r, 9) for r in roots if math.isfinite(r)})
    return unique[:BOUNDARY_ROOTS_PER_ARGUMENT]


def _narrow_roots(
    f: Callable[[mpmath.mpf], object], samples: list[tuple[float, float | None]]
) -> list[float]:
    """Roots of a region narrower than the grid spacing: look near the grid points closest to zero that
    are local extrema (|x - 1/3| - 1e-6 is positive on every grid point and negative between two)."""
    extrema: list[tuple[float, int]] = []
    for i in range(1, len(samples) - 1):
        (_, lo), (_, mid), (_, hi) = samples[i - 1], samples[i], samples[i + 1]
        if lo is None or mid is None or hi is None or mid == 0.0:
            continue
        if (mid <= lo and mid <= hi) or (mid >= lo and mid >= hi):
            extrema.append((abs(mid), i))
    found: list[float] = []
    for _, i in sorted(extrema)[:3]:
        (xl, yl), (xm, ym), (xr, yr) = samples[i - 1], samples[i], samples[i + 1]
        assert yl is not None and ym is not None and yr is not None  # noqa: S101 - filtered above
        peak = _refine_extremum(f, xl, xr, minimise=ym > 0)
        if peak is None or (peak[1] < 0) == (ym < 0):
            continue  # the extremum does not cross zero
        xe, ye = peak
        found.append(_bisect(f, xl, xe, yl) if (yl < 0) != (ye < 0) else _bisect(f, xm, xe, ym))
        found.append(_bisect(f, xe, xr, ye) if (ye < 0) != (yr < 0) else _bisect(f, xe, xm, ye))
    return found


def _around(symbol: sp.Symbol, root: float) -> list[sp.Expr]:
    """Admissible values on both sides of `root`, near and far."""
    if symbol.is_integer:
        base = math.floor(root)
        candidates: list[sp.Expr] = [sp.Integer(base + k) for k in (-1, 0, 1, 2)]
    else:
        spread = max(1.0, abs(root))
        candidates = [
            _snap(root + e * spread) for e in (0.001, -0.001, 0.01, -0.01, 0.5, -0.5, 1, -1)
        ]
        # A tiny root (a threshold at 1/1000 for a positive symbol) needs offsets relative to itself.
        candidates += [_snap(root * (1 + e)) for e in (0.01, -0.01, 0.5, -0.5)]
        if symbol.is_real is not True:  # off the real axis on both sides too
            candidates += [_snap(root) + sp.I * _snap(e) for e in (-0.01, 0.01)]
    return [c for c in candidates if _admits(symbol, c)]


def _boundary_points(
    a: sp.Expr, b: sp.Expr, symbols: list[sp.Symbol], rng: random.Random
) -> Iterator[dict[sp.Symbol, sp.Expr]]:
    for g in _critical_arguments(a) + _critical_arguments(b):
        for symbol in sorted(g.free_symbols & set(symbols), key=lambda s: s.name):
            for _ in range(2):  # two draws for the other symbols
                base = _draw_all(symbols, rng, 3.5)
                reduced = g.xreplace({s: v for s, v in base.items() if s != symbol})
                if reduced.free_symbols - {symbol}:
                    continue
                roots = _roots_in(reduced, symbol)
                # Between two adjacent roots lies a region that may be too narrow for a fixed offset.
                for low, high in itertools.pairwise(roots):
                    middle = _snap((low + high) / 2, digits=12)
                    if _admits(symbol, middle) or symbol.is_integer:
                        mid = middle if not symbol.is_integer else sp.Integer(math.floor(middle))
                        if _admits(symbol, mid):
                            yield {**base, symbol: mid}
                for root in roots:
                    for candidate in _around(symbol, root):
                        yield {**base, symbol: candidate}


def _random_points(
    symbols: list[sp.Symbol], rng: random.Random
) -> Iterator[dict[sp.Symbol, sp.Expr]]:
    for magnitude in (3.5,) * 8 + (50.0,) * 8:
        yield _draw_all(symbols, rng, magnitude)


def _branch_points(
    symbols: list[sp.Symbol], rng: random.Random
) -> Iterator[dict[sp.Symbol, sp.Expr]]:
    for symbol in symbols:
        if symbol.is_real is True:
            continue
        for multiple in (2, 8, 41):
            base = _draw_all(symbols, rng, 3.5)
            imaginary = _snap(math.pi * multiple * 1.5)
            for sign in (1, -1):
                yield {**base, symbol: _snap(rng.uniform(-3.5, 3.5)) + sign * sp.I * imaginary}


SPECIAL_INTEGERS = (7, 8, 10, 15, 31, 100, -7, -8, -40)
SPECIAL_VALUES = (
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    sp.Rational(1, 2),
    10,
    100,
)  # where a narrow bump is likely centred


def _special_points(
    symbols: list[sp.Symbol], rng: random.Random
) -> Iterator[dict[sp.Symbol, sp.Expr]]:
    for symbol in symbols:
        values = SPECIAL_INTEGERS if symbol.is_integer is True else SPECIAL_VALUES
        for raw in values:
            value = sp.sympify(raw)
            if _admits(symbol, value):
                yield {**_draw_all(symbols, rng, 3.5), symbol: value}


def _scale_points(
    symbols: list[sp.Symbol], rng: random.Random
) -> Iterator[dict[sp.Symbol, sp.Expr]]:
    for magnitude in (1e2, 1e3, 1e-3, 1e-6):
        yield _draw_all(symbols, rng, magnitude)


def _streams(
    a: sp.Expr, b: sp.Expr, symbols: list[sp.Symbol], rng: random.Random
) -> dict[ProbeKind, Iterator[dict[sp.Symbol, sp.Expr]]]:
    return {
        "random": _random_points(symbols, rng),
        "boundary": _boundary_points(a, b, symbols, rng),
        "branch": _branch_points(symbols, rng),
        "special": _special_points(symbols, rng),
        "scale": _scale_points(symbols, rng),
    }


# ---------------------------------------------------------------------------------------------------------
# Minimisation


def _simpler(value: sp.Expr) -> list[sp.Expr]:
    re_, im_ = value.as_real_imag()
    out: list[sp.Expr] = [sp.Integer(n) for n in (0, 1, -1, 2, -2)]
    out.extend(sp.Rational(round(float(re_) * step), step) for step in (1, 2, 10))
    if im_ != 0:  # a complex point: drop the imaginary part, or round it
        out.append(re_)
        out.append(re_ + sp.I * round(float(im_)))
    return [c for c in dict.fromkeys(out) if c != value]


def _minimise(
    a: sp.Expr, b: sp.Expr, symbols: list[sp.Symbol], values: dict[sp.Symbol, sp.Expr]
) -> tuple[dict[sp.Symbol, sp.Expr], bool]:
    """Greedily replace each value by a simpler one while the point stays a confirmed counterexample."""
    current = dict(values)
    changed = False
    for symbol in symbols:
        for candidate in _simpler(current[symbol]):
            if not _admits(symbol, candidate):
                continue
            trial = {**current, symbol: candidate}
            if _confirmed(a, b, trial) is not None:
                current, changed = trial, True
                break
    return current, changed


# ---------------------------------------------------------------------------------------------------------
# Entry point


def _format(value: mpmath.mpc) -> str:
    return str(mpmath.nstr(value, 20))


def check_counterexample(a: sp.Expr, b: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> bool:
    """Re-check a reported point independently of the search: confirmed at 60 and 120 digits."""
    return _confirmed(a, b, values) is not None


def find_counterexample(
    a: sp.Expr,
    b: sp.Expr,
    *,
    seed: int = 20260101,
    budget: Budget | None = None,
    kinds: tuple[ProbeKind, ...] = KINDS,
) -> SearchResult:
    """Look for a point where `a` and `b` differ. `counterexample=None` means none was found, not that none exists.

    `kinds` restricts the probe families (tests use it to check each family on its own target).
    """
    budget = budget or Budget()
    if contains_distribution(a) or contains_distribution(b):
        return SearchResult(None, unsupported="distribution")
    if noncommuting_symbols(a, b):
        return SearchResult(None, unsupported="noncommutative")
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic probing, not cryptographic
    deadline = time.monotonic() + budget.seconds
    streams = {k: it for k, it in _streams(a, b, symbols, rng).items() if k in kinds}
    if not symbols:
        streams = {"random": iter([{}])}  # a constant: one point
    tried = 0
    cancelled = 0
    domain = 0
    seen: set[tuple[str, ...]] = set()
    kinds_tried: list[ProbeKind] = []
    active = list(streams)
    while active:
        for kind in list(active):
            point = next(streams[kind], None)
            if point is None:
                active.remove(kind)
                continue
            if kind not in kinds_tried:
                kinds_tried.append(kind)
            key = tuple(f"{s.name}={point[s]}" for s in symbols)
            if key in seen:
                continue
            seen.add(key)
            if tried >= budget.points or time.monotonic() > deadline:
                return SearchResult(
                    None, tried, tuple(kinds_tried), cancelled, domain, exhausted=True
                )
            tried += 1
            screen = _gap(a, b, point, SCREEN_DPS)
            if screen == "pole":
                domain += 1
                continue
            if not isinstance(screen, _Gap) or screen.residual <= NOISE_FLOOR * screen.scale:
                continue
            gap = _confirmed(a, b, point)
            if gap is None:
                cancelled += 1
                continue
            final, minimised = _minimise(a, b, symbols, point)
            gap = _confirmed(a, b, final) or gap
            return SearchResult(
                Counterexample(
                    expression_a=str(a),
                    expression_b=str(b),
                    assumptions={s.name: _assumption_names(s) for s in symbols},
                    substitution={s.name: str(final[s]) for s in symbols},
                    value_a=_format(gap.value_a),
                    value_b=_format(gap.value_b),
                    residual=mpmath.nstr(gap.residual, 8),
                    probe=kind,
                    dps=CONFIRM_DPS[-1],
                    seed=seed,
                    minimised=minimised,
                    values=final,
                ),
                tried,
                tuple(kinds_tried),
                cancelled,
                domain,
            )
    return SearchResult(None, tried, tuple(kinds_tried), cancelled, domain)
