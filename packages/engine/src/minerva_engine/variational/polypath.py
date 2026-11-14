"""Polynomial paths and perturbations in exact-derivative form.

Numeric legs never differentiate a path numerically: a path is a polynomial, its derivatives are
polynomials, and an admissible perturbation is a polynomial whose coefficients solve the linear
conditions `η^(j)(end) = 0` exactly (to working precision). Nothing here uses SymPy's `diff`, so it
shares no code with the symbolic variation it checks.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import mpmath as mp

Coeffs = list[mp.mpf]  # c[k] multiplies s^k


def derivative(c: Sequence[mp.mpf], order: int = 1) -> Coeffs:
    out = list(c)
    for _ in range(order):
        out = [k * out[k] for k in range(1, len(out))] or [mp.mpf(0)]
    return out


def evaluate(c: Sequence[mp.mpf], s: mp.mpf) -> mp.mpf:
    total = mp.mpf(0)
    for coefficient in reversed(c):
        total = total * s + coefficient
    return total


def add(a: Sequence[mp.mpf], b: Sequence[mp.mpf], scale: mp.mpf | int = 1) -> Coeffs:
    n = max(len(a), len(b))
    return [
        (a[k] if k < len(a) else mp.mpf(0)) + scale * (b[k] if k < len(b) else mp.mpf(0))
        for k in range(n)
    ]


def derivatives_table(c: Sequence[mp.mpf], depth: int) -> list[Coeffs]:
    table = [list(c)]
    for _ in range(depth):
        table.append(derivative(table[-1]))
    return table


def random_coefficients(rng: random.Random, degree: int, scale: float = 1.0) -> Coeffs:
    return [mp.mpf(rng.uniform(-1, 1)) * scale / (1 + 0.5 * k) for k in range(degree + 1)]


def admissible(
    rng: random.Random,
    degree: int,
    a: mp.mpf,
    b: mp.mpf,
    vanishing: Iterable[tuple[str, int]],
    scale: float = 1.0,
) -> Coeffs:
    """A random polynomial of the given degree with `p^(j)(end) = 0` for every (end, j) listed.

    The random polynomial is projected onto the null space of the conditions with an exact solve of
    the (small) normal equations in the working precision. Raises ValueError if the conditions
    cannot be met at this degree.
    """
    rows = sorted(set(vanishing))
    p0 = random_coefficients(rng, degree, scale)
    if not rows:
        return p0
    if len(rows) > degree:
        raise ValueError("the degree is too small for the conditions")
    # constraint row for (end, j): value of the j-th derivative of s^k at the end
    constraint = mp.matrix(len(rows), degree + 1)
    for r, (end, j) in enumerate(rows):
        point = a if end == "lower" else b
        for k in range(degree + 1):
            if k >= j:
                constraint[r, k] = mp.factorial(k) / mp.factorial(k - j) * point ** (k - j)
    vector = mp.matrix(p0)
    gram = constraint * constraint.T
    try:
        multiplier = mp.lu_solve(gram, constraint * vector)
    except ZeroDivisionError as exc:
        raise ValueError("the conditions are not independent at this degree") from exc
    projected = vector - constraint.T * multiplier
    return [projected[k] for k in range(degree + 1)]


def hermite_cardinal(
    degree: int,
    a: mp.mpf,
    b: mp.mpf,
    data: dict[tuple[str, int], mp.mpf],
    bound: int,
) -> Coeffs:
    """The polynomial of least degree-`degree` solution of `p^(j)(end) = data` for all j < bound.

    Entries of `data` not listed are zero. The unique Hermite interpolant needs `degree + 1 = 2 bound`.
    """
    unknowns = degree + 1
    rows = [(end, j) for end in ("lower", "upper") for j in range(bound)]
    if len(rows) != unknowns:
        raise ValueError("a Hermite cardinal needs degree = 2*bound - 1")
    matrix = mp.matrix(len(rows), unknowns)
    rhs = mp.matrix(len(rows), 1)
    for r, (end, j) in enumerate(rows):
        point = a if end == "lower" else b
        for k in range(j, unknowns):
            matrix[r, k] = mp.factorial(k) / mp.factorial(k - j) * point ** (k - j)
        rhs[r] = data.get((end, j), mp.mpf(0))
    solution = mp.lu_solve(matrix, rhs)
    return [solution[k] for k in range(unknowns)]
