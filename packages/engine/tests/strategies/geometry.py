"""Metrics and charts: diagonal metrics with polynomial entries, and two charts of the same sphere."""

from __future__ import annotations

from hypothesis import strategies as st

COORDS = ["x", "y", "z"]
coefficient = st.integers(min_value=-2, max_value=2)


@st.composite
def positive_entry(draw: st.DrawFn) -> str:
    """A quadratic polynomial entry in x, y, z that is positive near the origin."""
    a, b, c = draw(coefficient), draw(coefficient), draw(coefficient)
    return f"(3 + {a}*x + {b}*y^2/2 + {c}*z)"


@st.composite
def diagonal_metric(draw: st.DrawFn) -> tuple[list[str], list[list[str]]]:
    n = draw(st.integers(min_value=2, max_value=3))
    entries = [draw(positive_entry()) for _ in range(n)]
    grid = [[entries[i] if i == j else "0" for j in range(n)] for i in range(n)]
    return COORDS[:n], grid


def sphere_charts(
    radius: str,
) -> tuple[tuple[list[str], list[list[str]]], tuple[list[str], list[list[str]]]]:
    """The round sphere of radius `radius` (LaTeX) in angles and in stereographic coordinates.

    Both have scalar curvature 2 / radius^2 (a textbook value, not computed by Minerva).
    """
    angles = (["\\theta", "\\phi"], [[f"({radius})^2", "0"], ["0", f"({radius})^2 \\sin^2\\theta"]])
    conformal = f"\\frac{{4 ({radius})^2}}{{(1+x^2+y^2)^2}}"
    return angles, (["x", "y"], [[conformal, "0"], ["0", conformal]])
