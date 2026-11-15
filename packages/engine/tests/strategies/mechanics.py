"""Lagrangians with known equations of motion and conserved quantities."""

from __future__ import annotations

import sympy as sp
from hypothesis import strategies as st


@st.composite
def anharmonic_lagrangians(draw: st.DrawFn) -> tuple[str, sp.Expr, sp.Expr]:
    """L = m/2 xdot^2 - (a/2) x^2 - (b/4) x^4 with integer m, a, b; returns (latex, m, V(x)).

    The acceleration is -V'(x)/m, and H = p^2/(2m) + V(x) by the Legendre transformation: both known here
    without the engine.
    """
    m = draw(st.integers(1, 4))
    a = draw(st.integers(0, 4))
    b = draw(st.integers(0, 3))
    x = sp.Symbol("x")
    potential = sp.Rational(a, 2) * x**2 + sp.Rational(b, 4) * x**4
    latex = rf"\frac{{{m}}}{{2}} \dot{{x}}^2 - \frac{{{a}}}{{2}} x^2 - \frac{{{b}}}{{4}} x^4"
    return latex, sp.Integer(m), potential
