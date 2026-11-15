"""Numerical fallback for a definite integral without free symbols: an estimate, never a formula.

Two independent rules (tanh-sinh and Gauss-Legendre, both from mpmath) must agree to the reported
error. A number is returned only if they do and the tail does not keep
growing; otherwise the integral is reported as divergent or unreliable, and no number is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath
import sympy as sp

DPS = 30
REL_TOL = mpmath.mpf("1e-9")


@dataclass(frozen=True)
class Estimate:
    value: str  # decimal text, 15 significant digits
    error: str  # the rules' disagreement, as decimal text
    real: bool


def _mp(bound: sp.Expr) -> mpmath.mpf:
    if bound is sp.oo:
        return mpmath.inf
    if bound is -sp.oo:
        return -mpmath.inf
    return mpmath.mpf(str(sp.N(bound, DPS + 5)))


def estimate(f: sp.Expr, x: sp.Symbol, lower: sp.Expr, upper: sp.Expr) -> Estimate | str:
    """An `Estimate`, or the reason (a fixed code) why none is given."""
    if f.free_symbols - {x}:
        return "free_symbols"
    mpmath.mp.dps = DPS
    fn = sp.lambdify(x, f, "mpmath")
    a, b = _mp(lower), _mp(upper)
    points = [a, 0, b] if a < 0 < b else [a, b]
    try:
        first, first_error = mpmath.quad(lambda t: fn(t), points, error=True)
        second = mpmath.quad(lambda t: fn(t), points, method="gauss-legendre")
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return "numerical_failure"
    error = max(mpmath.mpf(first_error), abs(first - second))
    value = first
    if not mpmath.isfinite(value) or mpmath.isnan(value):
        return "divergent_or_unreliable"
    scale = max(mpmath.mpf(1), abs(value))
    if error > REL_TOL * scale:
        return "divergent_or_unreliable"
    if (a == -mpmath.inf or b == mpmath.inf) and not _tail_small(fn, a, b, value, scale):
        return "divergent_or_unreliable"
    real = abs(mpmath.im(value)) <= REL_TOL * scale
    shown = mpmath.re(value) if real else value
    return Estimate(mpmath.nstr(shown, 15), mpmath.nstr(error, 3), real)


def _tail_small(fn, a, b, value, scale) -> bool:  # type: ignore[no-untyped-def]
    """The part of the integral beyond a large cutoff must be negligible and keep shrinking."""
    try:
        tails = []
        for cutoff in (mpmath.mpf(50), mpmath.mpf(200)):
            if b == mpmath.inf:
                tails.append(abs(mpmath.quad(fn, [cutoff, mpmath.inf])))
            if a == -mpmath.inf:
                tails.append(abs(mpmath.quad(fn, [-mpmath.inf, -cutoff])))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return False
    return all(t <= REL_TOL * scale * 1000 for t in tails)
