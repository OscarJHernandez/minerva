"""Polynomials and rational expressions with known structure (roots, cancelled factors)."""

from __future__ import annotations

import sympy as sp
from hypothesis import strategies as st

X = sp.Symbol("x")
Y = sp.Symbol("y")


@st.composite
def polynomials(
    draw: st.DrawFn,
    symbol: sp.Symbol = X,
    *,
    max_degree: int = 4,
    coeff: int = 5,
    min_degree: int = 0,
) -> sp.Expr:
    """An integer-coefficient polynomial whose degree is exactly `degree` (leading coefficient nonzero)."""
    degree = draw(st.integers(min_degree, max_degree))
    lower = [draw(st.integers(-coeff, coeff)) for _ in range(degree)]
    leading = draw(st.integers(1, coeff) | st.integers(-coeff, -1))
    return (
        sp.Add(*(c * symbol**i for i, c in enumerate([*lower, leading])), evaluate=True)
        if degree or leading
        else sp.Integer(0)
    )


@st.composite
def nonconstant_polynomials(
    draw: st.DrawFn, symbol: sp.Symbol = X, *, max_degree: int = 4
) -> sp.Expr:
    return draw(polynomials(symbol, max_degree=max_degree, min_degree=1))


@st.composite
def factored_polynomials(draw: st.DrawFn) -> tuple[sp.Expr, list[int]]:
    """c * prod (x - r_i)^m_i with distinct integer roots, expanded; returns (polynomial, distinct roots)."""
    roots = draw(st.lists(st.integers(-4, 4), min_size=1, max_size=3, unique=True))
    multiplicities = [draw(st.integers(1, 2)) for _ in roots]
    c = draw(st.integers(1, 3) | st.integers(-3, -1))
    expr = c * sp.Mul(*((X - r) ** m for r, m in zip(roots, multiplicities, strict=True)))
    return sp.expand(expr), roots


@st.composite
def cancelled_rationals(draw: st.DrawFn) -> tuple[str, int]:
    """((x - r)(x - s)) / ((x - r)(x - t)) as text, with r, s, t distinct; returns (text, r).

    Simplifying cancels (x - r), so the result must report x != r.
    """
    r, s, t = draw(st.lists(st.integers(-4, 4), min_size=3, max_size=3, unique=True))
    return f"(x-({r}))*(x-({s}))/((x-({r}))*(x-({t})))", r
