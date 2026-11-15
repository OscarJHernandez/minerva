"""Noncommutative polynomials and small explicit matrices with controlled algebraic properties."""

from __future__ import annotations

import sympy as sp
from hypothesis import strategies as st

A = sp.Symbol("A", commutative=False)
B = sp.Symbol("B", commutative=False)


@st.composite
def nc_polynomials(draw: st.DrawFn, *, max_terms: int = 3, max_degree: int = 3) -> sp.Expr:
    """An integer combination of words in the noncommuting symbols A and B (degree at most `max_degree`)."""
    terms = []
    for _ in range(draw(st.integers(1, max_terms))):
        word = draw(st.lists(st.sampled_from([A, B]), min_size=1, max_size=max_degree))
        terms.append(draw(st.integers(-3, 3)) * sp.Mul(*word))
    return sp.Add(*terms)


def _matrix(rows: list[list[int]]) -> sp.Matrix:
    return sp.Matrix(rows)


@st.composite
def explicit_matrices(draw: st.DrawFn, *, kind: str = "any", n: int = 2) -> sp.Matrix:
    """Small integer matrices with a stated property: any, symmetric, singular, nilpotent, defective, orthogonal-like.

    `defective` is a Jordan block with a repeated eigenvalue; `nilpotent` is strictly upper triangular.
    """
    entry = st.integers(-3, 3)
    if kind == "symmetric":
        a, b, c = draw(entry), draw(entry), draw(entry)
        return _matrix([[a, b], [b, c]])
    if kind == "singular":
        row = [draw(entry), draw(entry)]
        k = draw(st.integers(-2, 2))
        return _matrix([row, [k * v for v in row]])
    if kind == "nilpotent":
        return _matrix([[0, draw(entry)], [0, 0]])
    if kind == "defective":
        lam = draw(entry)
        return _matrix([[lam, 1], [0, lam]])
    if kind == "orthogonal":  # a rotation by a Pythagorean angle
        c, s = draw(st.sampled_from([(3, 4), (5, 12), (8, 15), (0, 1)]))
        h = sp.sqrt(c**2 + s**2)
        return sp.Matrix([[c / h, -s / h], [s / h, c / h]])
    return _matrix([[draw(entry) for _ in range(n)] for _ in range(n)])
