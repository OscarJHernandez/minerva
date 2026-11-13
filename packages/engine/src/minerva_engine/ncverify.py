"""Checking identities between polynomials in noncommuting symbols.

Sampling numbers cannot check `AB + BA = 2AB`: numbers commute. Noncommuting symbols are replaced by
random complex n x n matrices and both sides are evaluated with matrix products in the original order.
Matrices of size n satisfy no polynomial identity of degree below 2n (Amitsur-Levitzki), so a
difference of degree d cannot vanish on random n x n matrices by accident once n >= d/2 + 1; with
smaller matrices an identity like Hall's would pass. The size follows the degree for that reason.

Admissible expressions are sums, products and nonnegative integer powers of the noncommuting symbols,
with arbitrary commuting coefficients (numbers, other symbols, functions of them). Anything else
(a function of a noncommuting symbol, an inverse, a symbolic exponent) is outside the supported subset:
`nc_degree` raises `UnsupportedInput` and no result is produced. Agreement at all sampled points is
numerical evidence, as everywhere else in Minerva; the symbolic test (`expand(a - b) == 0`) is
reported separately.
"""

from __future__ import annotations

import random
from typing import Any

import mpmath
import sympy as sp

from .errors import LimitError, UnsupportedInput
from .models import Verification
from .verify import _POLE, MIN_VALID_POINTS, _evaluate_ex, _sample

MAX_NC_DEGREE = 10  # matrices up to 6 x 6
POINTS = 6
ATTEMPTS = 24
DPS = 30
REL_TOL = 1e-9

Value = Any  # an mpmath scalar or matrix (mpmath ships no type information)


def noncommuting_symbols(*exprs: sp.Expr) -> set[sp.Symbol]:
    return {s for e in exprs for s in e.free_symbols if s.is_commutative is False}


def nc_degree(expr: sp.Expr, nc: set[sp.Symbol] | None = None) -> int:
    """Total degree in the noncommuting symbols; `UnsupportedInput` outside the polynomial subset."""
    nc = noncommuting_symbols(expr) if nc is None else nc

    def deg(node: sp.Basic) -> int:
        if not (node.free_symbols & nc):
            return 0
        if node.is_Symbol:
            return 1
        if node.is_Add:
            return max(deg(a) for a in node.args)
        if node.is_Mul:
            return sum(deg(a) for a in node.args)
        if node.is_Pow and node.exp.is_Integer and node.exp >= 0:
            return int(node.exp) * deg(node.base)
        raise UnsupportedInput(
            "noncommuting symbols are supported only in sums, products and nonnegative integer "
            "powers; functions of them, inverses and symbolic exponents are not"
        )

    return deg(expr)


def matrix_size(degree: int) -> int:
    if degree > MAX_NC_DEGREE:
        raise LimitError(
            f"degree {degree} in the noncommuting symbols is above the limit of {MAX_NC_DEGREE}"
        )
    return max(3, degree // 2 + 1)


class _Skip(Exception):
    """A sample point where a commuting coefficient is undefined."""


def _promote(value: Value, n: int) -> mpmath.matrix:
    return value if isinstance(value, mpmath.matrix) else value * mpmath.eye(n)


def _add(a: Value, b: Value, n: int) -> Value:
    if isinstance(a, mpmath.matrix) or isinstance(b, mpmath.matrix):
        return _promote(a, n) + _promote(b, n)
    return a + b


def _evaluate(
    node: sp.Basic,
    matrices: dict[sp.Symbol, mpmath.matrix],
    scalars: dict[sp.Symbol, sp.Expr],
    n: int,
) -> Value:
    if not (node.free_symbols & set(matrices)):
        value = _evaluate_ex(node, scalars)
        if value is _POLE or not isinstance(value, complex):
            raise _Skip
        return mpmath.mpmathify(value)
    if node.is_Symbol:
        return matrices[node]
    if node.is_Add:
        total = _evaluate(node.args[0], matrices, scalars, n)
        for term in node.args[1:]:
            total = _add(total, _evaluate(term, matrices, scalars, n), n)
        return total
    if node.is_Mul:  # SymPy keeps the noncommuting factors of a product in their original order
        product = _evaluate(node.args[0], matrices, scalars, n)
        for factor in node.args[1:]:
            product = product * _evaluate(factor, matrices, scalars, n)
        return product
    if node.is_Pow:
        base = _evaluate(node.base, matrices, scalars, n)
        return base ** int(node.exp)
    raise UnsupportedInput("outside the supported noncommutative polynomial subset")


def _random_matrix(n: int, rng: random.Random) -> mpmath.matrix:
    return mpmath.matrix(
        [
            [mpmath.mpc(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)) for _ in range(n)]
            for _ in range(n)
        ]
    )


def _close(a: Value, b: Value, n: int) -> bool:
    x, y = _promote(a, n), _promote(b, n)
    scale = max(1.0, float(mpmath.mnorm(x, 1)), float(mpmath.mnorm(y, 1)))
    return bool(float(mpmath.mnorm(x - y, 1)) <= REL_TOL * scale)


def verify_equal_nc(a: sp.Expr, b: sp.Expr, *, seed: int = 20260103) -> Verification:
    """Do `a` and `b` agree on random matrices substituted for the noncommuting symbols?"""
    nc = noncommuting_symbols(a, b)
    n = matrix_size(max(nc_degree(a, nc), nc_degree(b, nc)))
    commuting = sorted((a.free_symbols | b.free_symbols) - nc, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = skipped = 0
    with mpmath.workdps(DPS):
        for _ in range(ATTEMPTS):
            if checked >= POINTS:
                break
            matrices = {s: _random_matrix(n, rng) for s in sorted(nc, key=lambda s: s.name)}
            scalars = {s: _sample(s, rng, spread=max(1, len(commuting))) for s in commuting}
            try:
                va = _evaluate(a, matrices, scalars, n)
                vb = _evaluate(b, matrices, scalars, n)
            except _Skip:
                skipped += 1
                continue
            checked += 1
            agreed += int(_close(va, vb, n))
    try:
        symbolic_zero = bool(sp.expand(a - b) == 0)
    except Exception:
        symbolic_zero = False
    passed = checked >= MIN_VALID_POINTS and agreed == checked
    if checked < MIN_VALID_POINTS:
        note = "too few valid sample points to verify with matrices"
    elif agreed != checked:
        note = f"matrix sampling ({n}x{n}) found a disagreement"
    else:
        note = f"matrix sampling ({n}x{n}) agreed at all sampled points"
    return Verification(
        symbolic_zero=symbolic_zero,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        numeric_points_skipped=skipped,
        passed=passed,
        note=note,
    )
