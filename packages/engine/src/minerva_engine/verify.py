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

from .distributions.guard import contains_distribution
from .models import Verification

MIN_VALID_POINTS = 4
SAMPLE_ATTEMPTS = 40
REL_TOL = 1e-9


def _flt(value: float, wide: bool) -> sp.Float:
    """Wide points carry extra digits: sin(x)**2 + cos(x)**2 at Im x = 10 cancels ~1e9 against 1."""
    return sp.Float(value, 80) if wide else sp.Float(value)


def _sample(
    symbol: sp.Symbol, rng: random.Random, *, wide: bool = False, spread: int = 1
) -> sp.Expr:
    """A random value consistent with the symbol's assumptions.

    `wide=True` draws from a much larger region (|x| up to 50, |Im| up to 10, values just off
    integers). The ordinary region stays inside the principal branches of log, asin and atan, so
    identities that only hold there (`log(exp(z)) = z`, `|x-10| = 10-x`) would pass it.
    """
    # With k symbols the products of values grow like top**k, and sin(x)**2 + cos(x)**2 at a
    # large imaginary part cancels to nothing: keep the spread of a point bounded.
    top = max(4.0, 50.0 ** (1.0 / spread)) if wide else 3.5
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
        return _flt(rng.uniform(0.01 if wide else 0.3, top), wide)
    if symbol.is_negative:
        return _flt(-rng.uniform(0.01 if wide else 0.3, top), wide)
    if symbol.is_nonnegative:
        return _flt(rng.uniform(0.0, top), wide)
    if symbol.is_nonpositive:
        return _flt(-rng.uniform(0.0, top), wide)
    if symbol.is_real:
        if wide and rng.random() < 0.3:  # just off an integer: where cancelled singularities hide
            return _flt(rng.randint(-12, 12) + rng.choice([-1, 1]) * 1e-3, wide)
        return _flt(rng.uniform(-top, top), wide)
    # No assumptions: sympy symbols are complex by default, so sample complex numbers.
    span = max(4.0, 10.0 ** (1.0 / spread)) if wide else 3.0
    return _flt(rng.uniform(-top, top), wide) + sp.I * _flt(rng.uniform(-span, span), wide)


_POLE = object()  # sentinel: the expression is genuinely undefined here (zoo, oo, nan)


def _evaluate_ex(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> complex | object | None:
    """A complex value, `_POLE` for a genuine pole, or None when evaluation failed or overflowed."""
    try:
        number = expr.xreplace(values).evalf(30)
        if number.has(sp.zoo, sp.oo, sp.nan, -sp.oo):
            return _POLE
        if not number.is_number:
            return None
        value = complex(number)
    except ZeroDivisionError:
        return _POLE
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    if value != value or abs(value) == float("inf"):
        return None
    return value


def _evaluate(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> complex | None:
    value = _evaluate_ex(expr, values)
    return None if value is _POLE else value  # type: ignore[return-value]


NARROW_POINTS = 12
WIDE_POINTS = 8


def numeric_detail(
    a: sp.Expr, b: sp.Expr, *, seed: int = 20260101, max_points: int = 12
) -> tuple[int, int, int]:
    """Return (points_checked, points_agreed, points_skipped).

    Two tranches: the ordinary region and a much wider one. A point where exactly one side is a
    genuine pole counts as a disagreement (the two expressions have different domains); a point
    where both are undefined, or evaluation failed, is skipped and counted.
    """
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = skipped = 0
    if max_points >= NARROW_POINTS:
        tranches = ((False, NARROW_POINTS), (True, WIDE_POINTS))
    else:  # a screening check: split the smaller budget between both regions
        tranches = ((False, (max_points + 1) // 2), (True, max_points // 2))
    for wide, limit in tranches:
        got = 0
        for _ in range(SAMPLE_ATTEMPTS):
            values = {s: _sample(s, rng, wide=wide, spread=max(1, len(symbols))) for s in symbols}
            va = _evaluate_ex(a, values)
            vb = _evaluate_ex(b, values)
            if (va is _POLE) != (vb is _POLE):
                checked += 1  # defined on one side only: different domains
                got += 1
            elif not isinstance(va, complex) or not isinstance(vb, complex):
                skipped += 1
                continue
            else:
                checked += 1
                got += 1
                scale = max(1.0, abs(va), abs(vb))
                if abs(va - vb) <= REL_TOL * scale:
                    agreed += 1
            if got >= limit:
                break
    return checked, agreed, skipped


def numeric_check(
    a: sp.Expr, b: sp.Expr, *, seed: int = 20260101, max_points: int = 12
) -> tuple[int, int]:
    """Return (points_checked, points_agreed) over random assumption-respecting points."""
    checked, agreed, _ = numeric_detail(a, b, seed=seed, max_points=max_points)
    return checked, agreed


def verify_equal(
    original: sp.Expr, candidate: sp.Expr, *, symbolic: bool = True, max_points: int = 12
) -> Verification:
    """Numeric agreement is what decides `passed`; the symbolic zero test is extra evidence.

    Pass `symbolic=False` where many verifications share one time budget (rule application):
    `sp.simplify(a - b)` can be slower than everything else together. `max_points` (at least
    `MIN_VALID_POINTS`) lowers the sample size for the many screening checks of a search; the
    reported chain is re-verified with the default.
    """
    if contains_distribution(original) or contains_distribution(candidate):
        # Sampling cannot see a delta (it is 0 at every sample), so it proves nothing here.
        return Verification(
            symbolic_zero=False,
            numeric_points_checked=0,
            numeric_points_agreed=0,
            numeric_points_skipped=0,
            passed=False,
            note="distributions cannot be verified by sampling; use the distribution operation",
        )
    checked, agreed, skipped = numeric_detail(
        original, candidate, max_points=max(max_points, MIN_VALID_POINTS)
    )
    symbolic_zero = False
    if symbolic:
        try:
            symbolic_zero = sp.simplify(original - candidate) == 0
        except Exception:
            symbolic_zero = False
    passed = checked >= MIN_VALID_POINTS and agreed == checked and skipped <= checked
    if checked < MIN_VALID_POINTS:
        note = "too few valid sample points to verify numerically"
    elif agreed != checked:
        note = "numeric check found a disagreement"
    elif skipped > checked:
        note = "too many sample points were undefined to verify numerically"
    else:
        note = "numeric check agreed at all sampled points"
    return Verification(
        symbolic_zero=symbolic_zero,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        numeric_points_skipped=skipped,
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
