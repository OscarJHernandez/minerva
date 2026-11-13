"""Distributional verification: pair both sides with test functions. Never pointwise sampling.

The input is *regularised* (``DiracDelta(u)`` becomes a Gaussian of width ``eps`` in ``u``,
``Heaviside(u)`` becomes ``(1 + erf(u/eps))/2``) and integrated against test functions with a fixed
quadrature, for three halving widths. All errors of such a mollification are even powers of
``eps``, so two rounds of Richardson extrapolation give the limit to about 1e-8. That limit is
compared with the action of the *result* on the same test function (its deltas evaluated, its
ordinary part integrated). Nothing here reuses the route that produced the result, and the
location of the deltas comes from a numerical polynomial root finder, not from the exact roots.

This is supporting evidence, not a proof: the exact conditions (`normal_form.py`) are the other,
separately reported leg. Test functions are Gaussian-windowed so every integral is over a finite
range.
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import mpmath
import sympy as sp
from mpmath.calculus.quadrature import GaussLegendre

from ..verify import _sample
from .models import Action, DistributionVerification, Endpoint
from .normal_form import Refusal, split_terms

MP_DPS = 20
GL_DEGREE = 4  # 24 Gauss-Legendre nodes per sub-interval
EPS_LEVELS = 3
TOLERANCE = 1e-7  # relative to max(1, |value|, |regularised values|)
MAX_SAMPLES = 3
SAMPLE_ATTEMPTS = 24
WINDOW = 10  # a regularised delta is integrated to this many widths either side

METHOD = (
    "pairing with Gaussian-windowed test functions; Gaussian/erf regularisation at three "
    "halving widths with Richardson extrapolation (supporting evidence, not a proof)"
)
TEST_FUNCTIONS = [
    "Gaussian window",
    "Gaussian window times x",
    "Gaussian window times (x^2 - 1)",
    "Gaussian window times exp(0.3 x)",
    "Gaussian window times cos(1.3 x)",
    "Gaussian window times sin(0.7 x)",
]

_Fn = Callable[..., Any]


def _nodes() -> list[tuple[Any, Any]]:
    return [(t, w) for t, w in GaussLegendre(mpmath.mp).calc_nodes(GL_DEGREE, mpmath.mp.prec)]


def _quad(f: _Fn, points: list[Any], nodes: list[tuple[Any, Any]]) -> Any:
    total = mpmath.mpf(0)
    for a, b in itertools.pairwise(points):
        if b <= a:
            continue
        half = (b - a) / 2
        mid = (a + b) / 2
        total += half * mpmath.fsum(w * f(mid + half * t) for t, w in nodes)
    return total


def _regularise(expr: sp.Expr, eps: sp.Symbol) -> sp.Expr:
    return expr.replace(
        lambda e: isinstance(e, sp.DiracDelta),
        lambda e: sp.exp(-(e.args[0] ** 2) / eps**2) / (eps * sp.sqrt(sp.pi)),
    ).replace(
        lambda e: isinstance(e, sp.Heaviside),
        lambda e: (1 + sp.erf(e.args[0] / eps)) / 2,
    )


def _as_step(expr: sp.Expr) -> sp.Expr:
    """Heaviside as an exact step (its value on the jump never matters to a distribution)."""
    return expr.replace(
        lambda e: isinstance(e, sp.Heaviside),
        lambda e: sp.Piecewise((1, e.args[0] > 0), (0, True)),
    )


def _number(value: Any) -> sp.Expr:
    if isinstance(value, sp.Basic):
        return value
    if isinstance(value, mpmath.mpc):
        return sp.Float(value.real, MP_DPS) + sp.I * sp.Float(value.imag, MP_DPS)
    return sp.Float(value, MP_DPS)


def _numeric_roots(g: sp.Expr, x: sp.Symbol, values: dict[sp.Symbol, sp.Expr]) -> list[Any]:
    """Real roots of the polynomial ``g`` at the sampled parameters, by a numerical root finder."""
    poly = sp.Poly(g.xreplace(values), x)
    coefficients = [mpmath.mpmathify(complex(sp.N(c, MP_DPS))) for c in poly.all_coeffs()]
    if len(coefficients) < 2:
        return []
    roots = mpmath.polyroots(coefficients, maxsteps=400, extraprec=120)
    return [
        mpmath.re(r)
        for r in (roots if isinstance(roots, list) else [roots])
        if abs(mpmath.im(r)) < mpmath.mpf(10) ** -8 * (1 + abs(r))
    ]


def _plan(
    original: sp.Expr, x: sp.Symbol, values: dict[sp.Symbol, sp.Expr]
) -> list[tuple[Any, Any]]:
    """(position, |slope|) of every delta root and step jump of the input, from numerical roots."""
    points: list[tuple[Any, Any]] = []
    for delta in original.atoms(sp.DiracDelta):
        g = delta.args[0]
        slope_expr = sp.diff(g, x).xreplace(values)
        for root in _numeric_roots(g, x, values):
            slope = abs(mpmath.mpmathify(complex(sp.N(slope_expr.subs(x, _number(root)), MP_DPS))))
            points.append((root, slope if slope > 0 else mpmath.mpf(1)))
    for step in original.atoms(sp.Heaviside):
        g = step.args[0]
        slope = abs(mpmath.mpmathify(complex(sp.N(sp.diff(g, x).xreplace(values), MP_DPS))))
        points.extend(
            (root, slope if slope > 0 else mpmath.mpf(1)) for root in _numeric_roots(g, x, values)
        )
    return points


def _result_positions(result: sp.Expr, x: sp.Symbol, values: dict[sp.Symbol, sp.Expr]) -> list[Any]:
    """Where the result's deltas and step jumps are (they need breakpoints, not widths)."""
    found: list[Any] = []
    for atom in result.atoms(sp.DiracDelta, sp.Heaviside):
        found.extend(_numeric_roots(atom.args[0], x, values))
    return found


def _test_functions(centre: float, scale: float, x: sp.Symbol) -> list[sp.Expr]:
    window = sp.exp(-((x - sp.Float(centre)) ** 2) / (2 * sp.Float(scale) ** 2))
    return [
        window,
        window * x,
        window * (x**2 - 1),
        window * sp.exp(sp.Float("0.3") * x),
        window * sp.cos(sp.Float("1.3") * x),
        window * sp.sin(sp.Float("0.7") * x),
    ]


def _extrapolate(values: list[Any]) -> Any:
    first = (4 * values[1] - values[0]) / 3
    second = (4 * values[2] - values[1]) / 3
    return (16 * second - first) / 15


def _breakpoints(
    points: list[tuple[Any, Any]],
    eps: Any,
    lo: Any,
    hi: Any,
    *,
    coarse: int,
    extra: list[Any] = (),  # type: ignore[assignment]
) -> list[Any]:
    marks = [lo + (hi - lo) * i / coarse for i in range(coarse + 1)] + list(extra)
    for position, slope in points:
        width = eps / slope
        for m in (-WINDOW, -5, -2, -1, 0, 1, 2, 5, WINDOW):
            marks.append(position + m * width)
    cleaned = sorted({mpmath.nstr(m, MP_DPS): m for m in marks if lo <= m <= hi}.values())
    return list(cleaned)


def _result_parts(result: sp.Expr, x: sp.Symbol) -> tuple[list[tuple[sp.Expr, sp.Expr]], sp.Expr]:
    """(coefficient, position) of each delta in the result, and its ordinary part."""
    deltas: list[tuple[sp.Expr, sp.Expr]] = []
    regular: list[sp.Expr] = []
    for term in split_terms(result):
        if term.arg is None:
            regular.append(term.coeff)
            continue
        (position,) = sp.solve(term.arg, x)
        deltas.append((term.coeff, position))
    return deltas, sp.Add(*regular)


@dataclasses.dataclass
class _Tally:
    checked: int = 0
    agreed: int = 0
    convergent: bool = True

    def add(self, limit: Any, expected: Any, levels: list[Any]) -> None:
        scale = max(1, abs(expected), *(abs(v) for v in levels))
        self.checked += 1
        self.agreed += int(abs(limit - expected) <= TOLERANCE * scale)
        self.convergent &= abs(levels[-1] - expected) <= abs(levels[0] - expected) + 1e-12 * scale


def _out_of_time(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise Refusal("resource_limit", "The verification ran out of time.")


def _pair_with_test_functions(
    *,
    action: Action,
    regular_input: sp.Expr,
    result: sp.Expr,
    x: sp.Symbol,
    eps_symbol: sp.Symbol,
    values: dict[sp.Symbol, sp.Expr],
    points: list[tuple[Any, Any]],
    extra: list[Any],
    eps_levels: list[Any],
    window: tuple[float, float],
    nodes: list[tuple[Any, Any]],
    deadline: float,
    tally: _Tally,
) -> None:
    """simplify and differentiate: the regularised input against the action of the result."""
    centre, scale = window
    lo, hi = centre - 14 * scale, centre + 14 * scale
    deltas, regular = _result_parts(result, x)
    input_fn = sp.lambdify((x, eps_symbol), regular_input.xreplace(values), "mpmath")
    regular_fn = sp.lambdify(x, _as_step(regular).xreplace(values), "mpmath")
    exact_marks = _breakpoints(points, eps_levels[-1], lo, hi, coarse=28, extra=extra)
    for phi in _test_functions(centre, scale, x):
        phi_fn = sp.lambdify(x, phi, "mpmath")
        weight_fn = sp.lambdify(x, sp.diff(phi, x) if action == "differentiate" else phi, "mpmath")
        sign = -1 if action == "differentiate" else 1
        levels = [
            sign
            * _quad(
                partial(_product, input_fn, weight_fn, e),
                _breakpoints(points, e, lo, hi, coarse=28, extra=extra),
                nodes,
            )
            for e in eps_levels
        ]
        expected = _quad(partial(_product2, regular_fn, phi_fn), exact_marks, nodes)
        for coefficient, position in deltas:
            expected += mpmath.mpmathify(
                complex(sp.N(coefficient.xreplace(values), MP_DPS))
            ) * phi_fn(mpmath.mpmathify(float(sp.N(position.xreplace(values), MP_DPS))))
        tally.add(_extrapolate(levels), expected, levels)
        _out_of_time(deadline)


def _product(f: _Fn, g: _Fn, eps: Any, t: Any) -> Any:
    return f(t, eps) * g(t)


def _product2(f: _Fn, g: _Fn, t: Any) -> Any:
    return f(t) * g(t)


def _pair_with_interval(
    *,
    regular_input: sp.Expr,
    result: sp.Expr,
    x: sp.Symbol,
    eps_symbol: sp.Symbol,
    values: dict[sp.Symbol, sp.Expr],
    points: list[tuple[Any, Any]],
    eps_levels: list[Any],
    bounds: tuple[sp.Expr, sp.Expr],
    endpoint: Endpoint | None,
    nodes: list[tuple[Any, Any]],
    tally: _Tally,
) -> None:
    """integrate: the regularised integral over the interval against the value found."""
    lower = _limit(bounds[0])
    upper = _limit(bounds[1])
    expected = mpmath.mpmathify(complex(sp.N(result.xreplace(values), MP_DPS)))
    levels = [
        _integral_over_windows(
            regular_input, x, eps_symbol, e, values, points, lower, upper, endpoint, nodes
        )
        for e in eps_levels
    ]
    tally.add(_extrapolate(levels), expected, levels)


def _limit(bound: sp.Expr) -> Any:
    if bound == sp.oo:
        return mpmath.inf
    if bound == -sp.oo:
        return -mpmath.inf
    return mpmath.mpmathify(float(sp.N(bound, MP_DPS)))


def verify_pairing(
    original: sp.Expr,
    result: sp.Expr,
    x: sp.Symbol,
    *,
    action: Action,
    bounds: tuple[sp.Expr, sp.Expr] | None = None,
    endpoint: Endpoint | None = None,
    deadline: float,
) -> DistributionVerification:
    """Check ``result`` against ``original`` by test-function pairing (see the module docstring)."""
    mpmath.mp.dps = MP_DPS
    nodes = _nodes()
    eps_symbol = sp.Symbol("eps", positive=True)
    parameters = sorted((original.free_symbols | result.free_symbols) - {x}, key=lambda s: s.name)
    rng = random.Random(20240613)  # noqa: S311 - deterministic sampling, not cryptographic
    wanted = MAX_SAMPLES if parameters else 1
    regular_input = _regularise(original, eps_symbol)
    tally = _Tally()
    samples = 0
    widths: list[float] = []
    note = ""
    for _ in range(SAMPLE_ATTEMPTS if parameters else 1):
        if samples >= wanted:
            break
        _out_of_time(deadline)
        raw = {p: _sample(p, rng) for p in parameters}
        values = {p: _number(v) for p, v in raw.items()}
        try:
            points = _plan(original, x, values)
            extra = _result_positions(result, x, values)
        except (ValueError, TypeError, ZeroDivisionError, mpmath.libmp.NoConvergence):
            continue
        if not points:
            note = "no singular point was found numerically"
            continue
        input_positions = sorted(mpmath.mpf(p) for p, _ in points)
        positions = sorted(input_positions + [mpmath.mpf(p) for p in extra])
        separation = min(
            (b - a for a, b in itertools.pairwise(input_positions) if b - a > 1e-9),
            default=mpmath.mpf(4),
        )
        if separation < 1e-2:  # nearly coinciding features leave no room for the regularisation
            continue
        scale = max(1.0, float((positions[-1] - positions[0]) * 0.75) + 0.5)
        slope_min = min(s for _, s in points)
        eps0 = min(
            mpmath.mpf("0.02"), mpmath.mpf("0.02") * min(separation, mpmath.mpf(scale)) * slope_min
        )
        if eps0 < mpmath.mpf(10) ** -5:
            continue
        eps_levels = [eps0 / 2**k for k in range(EPS_LEVELS)]
        widths = [float(e) for e in eps_levels]
        samples += 1
        if action == "integrate":
            if bounds is None:
                raise Refusal("verification_failed", "The integration limits are missing.")
            _pair_with_interval(
                regular_input=regular_input,
                result=result,
                x=x,
                eps_symbol=eps_symbol,
                values=values,
                points=points,
                eps_levels=eps_levels,
                bounds=bounds,
                endpoint=endpoint,
                nodes=nodes,
                tally=tally,
            )
        else:
            _pair_with_test_functions(
                action=action,
                regular_input=regular_input,
                result=result,
                x=x,
                eps_symbol=eps_symbol,
                values=values,
                points=points,
                extra=extra,
                eps_levels=eps_levels,
                window=(float((positions[0] + positions[-1]) / 2), scale),
                nodes=nodes,
                deadline=deadline,
                tally=tally,
            )
    passed = tally.checked > 0 and tally.agreed == tally.checked
    if tally.checked == 0:
        note = note or "no parameter sample allowed a regularised check (features too close)"
    elif passed:
        note = (
            "equal as distributions on every test function tried; the regularised input "
            "converges to the result"
        )
    else:
        note = "the regularised input does not converge to the result on some test function"
    return DistributionVerification(
        method=METHOD,
        test_functions=["indicator of the integration interval"]
        if action == "integrate"
        else TEST_FUNCTIONS,
        pairings_checked=tally.checked,
        pairings_agreed=tally.agreed,
        parameter_samples=samples,
        widths=widths,
        convergence_observed=bool(tally.checked and tally.convergent),
        passed=passed,
        note=note,
    )


def _integral_over_windows(
    regular_input: sp.Expr,
    x: sp.Symbol,
    eps_symbol: sp.Symbol,
    eps: Any,
    values: dict[sp.Symbol, sp.Expr],
    points: list[tuple[Any, Any]],
    lower: Any,
    upper: Any,
    endpoint: Endpoint | None,
    nodes: list[tuple[Any, Any]],
) -> Any:
    """The regularised integral over ``[lower, upper]``, one window around each delta."""
    fn = sp.lambdify(
        x, regular_input.xreplace(values).xreplace({eps_symbol: _number(eps)}), "mpmath"
    )
    total = mpmath.mpf(0)
    tolerance = mpmath.mpf(10) ** -9
    for position, slope in points:
        width = eps / slope
        start, end = position - WINDOW * width, position + WINDOW * width
        at_lower = lower != -mpmath.inf and abs(position - lower) < tolerance * (1 + abs(lower))
        at_upper = upper != mpmath.inf and abs(position - upper) < tolerance * (1 + abs(upper))
        if (at_lower or at_upper) and endpoint == "include":
            pass  # the whole window counts
        elif (at_lower or at_upper) and endpoint == "exclude":
            continue
        else:  # inside, outside, or on a limit with "half" (the symmetric regularisation gives 1/2)
            start, end = max(start, lower), min(end, upper)
        if end <= start:
            continue
        marks = [start, end] + [
            position + k * width
            for k in (-WINDOW, -5, -2, -1, 0, 1, 2, 5, WINDOW)
            if start < position + k * width < end
        ]
        unique = sorted({mpmath.nstr(m, MP_DPS): m for m in marks}.values())
        total += _quad(fn, unique, nodes)
    return total
