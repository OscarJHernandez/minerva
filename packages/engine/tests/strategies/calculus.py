"""Integrands built by differentiating a known function, so an antiderivative is known to exist."""

from __future__ import annotations

import sympy as sp
from hypothesis import strategies as st

from .algebra import X


@st.composite
def antiderivable(draw: st.DrawFn) -> tuple[sp.Expr, sp.Expr]:
    """(integrand, F) with integrand = dF/dx and F a small sum of polynomial, exponential, sine/cosine terms."""
    a = draw(st.integers(1, 3))
    candidates = [
        lambda k: X**k,
        lambda k: sp.exp(k * X),
        lambda k: sp.sin(k * X),
        lambda k: sp.cos(k * X),
        lambda k: X * sp.exp(k * X),
    ]
    terms = draw(
        st.lists(
            st.tuples(
                st.integers(0, len(candidates) - 1), st.integers(1, a + 1), st.integers(-3, 3)
            ),
            min_size=1,
            max_size=3,
        )
    )
    F = sp.Add(*(c * candidates[i](k) for i, k, c in terms))
    return sp.diff(F, X), F
