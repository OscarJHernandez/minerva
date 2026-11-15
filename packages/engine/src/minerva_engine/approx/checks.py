"""Numerical checks used by several approximation methods.

All of them evaluate the ORIGINAL expression with mpmath at high precision and compare it with the
candidate; none uses the route that built the candidate. Results are measurements.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from typing import Any

import mpmath
import sympy as sp

from ..series import _eval
from ..verify import _sample

DPS = 60
RATIO_SLACK = 1.4  # a residual that falls at the claimed rate may be this much above 2^-(N+1)
EXACT_BELOW = mpmath.mpf(10) ** -(DPS - 15)


def compile_mp(expr: sp.Expr, var: sp.Symbol) -> Callable[[Any], Any]:
    """`expr` as a function of `var` for mpmath numbers (other symbols must already be numbers)."""
    fn = sp.lambdify(var, expr, modules="mpmath")

    def call(x: Any) -> Any:
        return _eval(fn, x)

    return call


def parameter_sets(
    expr: sp.Expr, var: sp.Symbol, count: int, seed: int
) -> list[dict[sp.Symbol, sp.Expr]]:
    """Assumption-respecting values for the symbols other than `var` (one empty set if there are none)."""
    others = sorted(expr.free_symbols - {var}, key=lambda s: s.name)
    if not others:
        return [{}]
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    sets: list[dict[sp.Symbol, sp.Expr]] = []
    for _ in range(40 * count):
        sets.append({s: _sample(s, rng) for s in others})
        if len(sets) >= count:
            break
    return sets


def falls_at_rate(
    residuals: list[Any],
    claimed: int,
    *,
    slack: float = RATIO_SLACK,
    exact_below: Any = EXACT_BELOW,
) -> bool | None:
    """Do successive residuals (the step halved or the argument doubled each time) fall at 2^-claimed?

    None: too few usable values to decide. A residual that is zero to working precision is exact.
    """
    usable = [r for r in residuals if r is not None]
    if len(usable) < 3:
        return None
    mags = [abs(r) for r in usable]
    if all(m < exact_below for m in mags):
        return True
    target = mpmath.mpf(2) ** -claimed * slack
    for big, small in itertools.pairwise(mags):
        if big < exact_below:
            continue
        if small / big > target:
            return False
    return True


def with_precision(fn: Callable[[], Any]) -> Any:
    with mpmath.workdps(DPS):
        return fn()


def local_residual_check(
    expr: sp.Expr,
    approx: sp.Expr,
    var: sp.Symbol,
    x0: sp.Expr,
    claimed: int,
    *,
    seed: int = 20260906,
) -> tuple[bool | None, int, str]:
    """Does f - approx fall like h^claimed near x0, on both sides, at every parameter set?

    h = 1e-12 * 2^-j. Evaluated at a precision that resolves a residual of size h^claimed, so a
    coefficient wrong by one part in 1e9 is seen. Uses f and the approximant directly, not the
    series they were built from.
    """
    digits = 12 * claimed + 60
    exact_below = mpmath.mpf(10) ** -(12 * claimed + 40)
    decided = 0
    sets = parameter_sets(expr, var, 4, seed)
    with mpmath.workdps(digits):
        centre = mpmath.mpmathify(sp.N(x0, digits))
        for values in sets:
            f = compile_mp(expr.subs(values), var)
            a = compile_mp(approx.subs(values), var)
            for sign in (1, -1):
                residuals = []
                for j in range(5):
                    x = centre + sign * mpmath.mpf(10) ** -12 / 2**j
                    fv, av = f(x), a(x)
                    residuals.append(None if fv is None or av is None else fv - av)
                # the step is halved each time, so the residual must fall by 2^claimed
                verdict = falls_at_rate(residuals, claimed, exact_below=exact_below)
                if verdict is False:
                    return False, len(sets), "the residual does not fall at the claimed rate"
                if verdict:
                    decided += 1
    if decided == 0:
        return None, len(sets), "no point could be evaluated"
    return True, decided, f"the residual falls like h^{claimed} as the step is halved from 1e-12"
