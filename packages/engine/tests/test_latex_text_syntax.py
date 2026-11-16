"""Plain-text syntax written where LaTeX is expected is refused, never read as a product of letters.

Regression: `theta_dot^2` parsed as `t*h*e*t*a_d*o*t**3` and `sin(theta)` as `s*i*n*theta`; the analysis then
failed (or worse, answered) about another expression. Each refused form has an accepted twin.
"""

import pytest
import sympy as sp
from minerva_engine.errors import ParseError
from minerva_engine.latexparse import parse_latex
from minerva_engine.textparse import parse_text


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (r"theta_dot^2", "_dot"),
        (r"x_ddot + 1", "_ddot"),
        (r"a_bc", "_bc"),
        (r"sin(x)", r"\sin("),
        (r"\frac12 m R^2 (\dot{\theta}^2 + sin(\theta)^2)", r"\sin("),
        (r"exp(x) + 1", r"\exp("),
        (r"sqrt(x)", r"\sqrt("),
        (r"theta + 1", r"\theta"),
        (r"m omega^2", r"\omega"),
    ],
)
def test_text_syntax_in_latex_is_refused_with_the_fix(source: str, fragment: str) -> None:
    with pytest.raises(ParseError) as info:
        parse_latex(source)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "source",
    [
        r"\dot{\theta}^2 + \sin(\theta)^2",
        r"\sin(x) + \exp(x)",
        r"x_{dot} + 1",
        r"x_i^2 + x_{ab}",
        r"\theta + \omega^2",
        r"\mathrm{e}^{x}",
        r"a\,b",
    ],
)
def test_the_latex_twins_are_accepted(source: str) -> None:
    parse_latex(source)


def test_text_syntax_still_reads_the_same_names() -> None:
    expr = parse_text("theta_dot^2 + sin(theta)^2*phi_dot^2")
    assert {s.name for s in expr.free_symbols} == {"theta_dot", "theta", "phi_dot"}
    assert expr.has(sp.sin)
