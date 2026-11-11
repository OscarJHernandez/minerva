"""Independent checks that two expressions are equal.

`simplify(a - b) == 0` is reported separately from numeric sampling because both can be
fooled in different ways: the symbolic check shares SymPy's simplifier with the operation
being verified, and numeric sampling can miss isolated differences. A verification only
`passed` when sampling found enough valid points and *all* of them agreed.
"""

from __future__ import annotations

import random

import sympy as sp

from .models import Verification

MIN_VALID_POINTS = 4
SAMPLE_ATTEMPTS = 40
REL_TOL = 1e-9


def _sample(symbol: sp.Symbol, rng: random.Random) -> sp.Expr:
    """A random value consistent with the symbol's assumptions."""
    if symbol.is_integer:
        pool = [
            n
            for n in range(-6, 7)
            if not (symbol.is_positive and n <= 0)
            and not (symbol.is_negative and n >= 0)
            and not (symbol.is_nonnegative and n < 0)
            and not (symbol.is_nonpositive and n > 0)
            and not (symbol.is_nonzero and n == 0)
        ]
        return sp.Integer(rng.choice(pool))
    if symbol.is_positive:
        return sp.Float(rng.uniform(0.3, 3.5))
    if symbol.is_negative:
        return sp.Float(-rng.uniform(0.3, 3.5))
    if symbol.is_nonnegative:
        return sp.Float(rng.uniform(0.0, 3.5))
    if symbol.is_nonpositive:
        return sp.Float(-rng.uniform(0.0, 3.5))
    if symbol.is_real:
        return sp.Float(rng.uniform(-3.5, 3.5))
    # No assumptions: sympy symbols are complex by default, so sample complex numbers.
    return sp.Float(rng.uniform(-3.0, 3.0)) + sp.I * sp.Float(rng.uniform(-3.0, 3.0))


def _evaluate(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> complex | None:
    try:
        number = expr.xreplace(values).evalf(30)
        if not number.is_number or number.has(sp.zoo, sp.oo, sp.nan):
            return None
        value = complex(number)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
        return None
    if value != value or abs(value) == float("inf"):
        return None
    return value


def numeric_check(a: sp.Expr, b: sp.Expr, *, seed: int = 20260101) -> tuple[int, int]:
    """Return (points_checked, points_agreed) over random assumption-respecting points."""
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    for _ in range(SAMPLE_ATTEMPTS):
        values = {s: _sample(s, rng) for s in symbols}
        va = _evaluate(a, values)
        vb = _evaluate(b, values)
        if va is None or vb is None:
            continue
        checked += 1
        scale = max(1.0, abs(va), abs(vb))
        if abs(va - vb) <= REL_TOL * scale:
            agreed += 1
        if checked >= 12:
            break
    return checked, agreed


def verify_equal(original: sp.Expr, candidate: sp.Expr) -> Verification:
    checked, agreed = numeric_check(original, candidate)
    try:
        symbolic_zero = sp.simplify(original - candidate) == 0
    except Exception:
        symbolic_zero = False
    passed = checked >= MIN_VALID_POINTS and agreed == checked
    if checked < MIN_VALID_POINTS:
        note = "too few valid sample points to verify numerically"
    elif agreed != checked:
        note = "numeric check found a disagreement"
    else:
        note = "numeric check agreed at all sampled points"
    return Verification(
        symbolic_zero=symbolic_zero,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=passed,
        note=note,
    )
