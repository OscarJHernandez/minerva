"""Independent checks that two expressions are equal.

`simplify(a - b) == 0` is reported separately from numeric sampling because both can be
fooled in different ways: the symbolic check shares SymPy's simplifier with the operation
being verified, and numeric sampling can miss isolated differences. A verification only
`passed` when sampling found enough valid points and *all* of them agreed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import mpmath
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


def verify_equal(original: sp.Expr, candidate: sp.Expr, *, symbolic: bool = True) -> Verification:
    """Numeric agreement is what decides `passed`; the symbolic zero test is extra evidence.

    Pass `symbolic=False` where many verifications share one time budget (rule application):
    `sp.simplify(a - b)` can be slower than everything else together.
    """
    checked, agreed = numeric_check(original, candidate)
    symbolic_zero = False
    if symbolic:
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


EL_POINTS = 6  # test paths per coordinate
EL_REL_TOL = 1e-7  # nested numerical differentiation is less exact than plain evaluation
EL_DPS = 40


def _real_sample(symbol: sp.Symbol, rng: random.Random) -> float:
    """A real value consistent with the symbol's assumptions (a point on a test path)."""
    return float(sp.re(_sample(symbol, rng)))


Coeffs = list[float]


def _el_reference(
    f: Callable[..., mpmath.mpf],
    i: int,
    a: Coeffs,
    b: Coeffs,
    c: Coeffs,
    d: Coeffs,
    t0: float,
    pmp: list[mpmath.mpf],
) -> complex | None:
    """`d/ds (dL/dqdot_i) - dL/dq_i` on the cubic path `q(s) = a + b s + c s^2 + d s^3`, at s = 0."""
    n = len(a)

    def momentum(s: mpmath.mpf) -> mpmath.mpf:
        qs = [a[j] + b[j] * s + c[j] * s**2 + d[j] * s**3 for j in range(n)]
        qds = [b[j] + 2 * c[j] * s + 3 * d[j] * s**2 for j in range(n)]

        def along(x: mpmath.mpf) -> mpmath.mpf:
            v = list(qds)
            v[i] = x
            return f(*qs, *v, t0 + s, *pmp)

        return mpmath.diff(along, qds[i])

    def along_q(x: mpmath.mpf) -> mpmath.mpf:
        v = list(a)
        v[i] = x
        return f(*v, *b, t0, *pmp)

    try:
        value = complex(mpmath.diff(momentum, 0) - mpmath.diff(along_q, a[i]))
    except (ArithmeticError, ValueError, TypeError):
        return None
    return value if value == value else None


def euler_lagrange_check(
    lagrangian: sp.Expr,
    q: Sequence[sp.Symbol],
    qd: Sequence[sp.Symbol],
    qdd: Sequence[sp.Symbol],
    t: sp.Symbol,
    derived: Sequence[sp.Expr],
    *,
    seed: int = 20260102,
) -> list[Verification]:
    """Check each derived Euler-Lagrange expression without using SymPy's differentiation.

    Along a random cubic path `q_i(s)` (with `t = t0 + s`) the value `d/ds (dL/dqdot_i) - dL/dq_i` is
    computed by nested numerical differentiation of the lambdified Lagrangian (mpmath, 40 digits) and
    compared with the derived expression evaluated at `q, qdot, qddot` of that path. One
    `Verification` per coordinate; `passed` needs enough valid points and all of them agreeing.
    """
    n = len(q)
    symbols_in = lagrangian.free_symbols.union(*(e.free_symbols for e in derived))
    fixed = {*q, *qd, *qdd, t}
    params = sorted(symbols_in - fixed, key=lambda s: s.name)
    note = "numerical differentiation of L along random paths agrees with the derived expression"
    try:
        f = sp.lambdify([*q, *qd, t, *params], lagrangian, "mpmath")
    except Exception:
        return [
            Verification(note="the Lagrangian could not be evaluated numerically") for _ in range(n)
        ]
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = [0] * n
    agreed = [0] * n
    with mpmath.workdps(EL_DPS):
        for _ in range(SAMPLE_ATTEMPTS):
            if min(checked) >= EL_POINTS:
                break
            a = [_real_sample(s, rng) for s in q]
            b = [rng.uniform(-2.0, 2.0) for _ in q]
            c = [rng.uniform(-2.0, 2.0) for _ in q]
            d = [rng.uniform(-2.0, 2.0) for _ in q]
            t0 = _real_sample(t, rng)
            pvals = [_sample(s, rng) for s in params]
            pmp = [mpmath.mpmathify(complex(v)) for v in pvals]

            values: dict[sp.Symbol, sp.Expr] = {}
            for i in range(n):
                values[q[i]] = sp.Float(a[i])
                values[qd[i]] = sp.Float(b[i])
                values[qdd[i]] = sp.Float(2 * c[i])
            values[t] = sp.Float(t0)
            values.update(zip(params, pvals, strict=True))

            for i in range(n):
                reference = _el_reference(f, i, a, b, c, d, t0, pmp)
                if reference is None:
                    continue
                got = _evaluate(derived[i], values)
                if got is None:
                    continue
                checked[i] += 1
                scale = max(1.0, abs(got), abs(reference))
                if abs(got - reference) <= EL_REL_TOL * scale:
                    agreed[i] += 1
    return [
        Verification(
            symbolic_zero=False,
            numeric_points_checked=checked[i],
            numeric_points_agreed=agreed[i],
            passed=checked[i] >= MIN_VALID_POINTS and agreed[i] == checked[i],
            note=note
            if checked[i] >= MIN_VALID_POINTS and agreed[i] == checked[i]
            else "too few valid sample points or a disagreement was found",
        )
        for i in range(n)
    ]
