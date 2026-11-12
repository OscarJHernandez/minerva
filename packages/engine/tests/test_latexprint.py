r"""Printed LaTeX parses back to the same expression: constants stay constants, symbols stay symbols."""

import pytest
import sympy as sp
from minerva_engine import ParseError, parse_latex
from minerva_engine.latexprint import to_latex

x, a, e, i, alpha, w0 = sp.symbols("x a e i alpha omega_0")

ROUND_TRIP = [
    sp.exp(sp.I * x),
    sp.E,
    sp.I,
    2 * sp.I * sp.pi,
    sp.exp(-a * x**2),
    sp.exp(x),
    alpha * x + w0,
    sp.Rational(3, 7) * x**2,
    sp.sqrt(x + 1),
    sp.asin(x),
    sp.acos(x) + sp.atan(x),
    sp.log(x),
    sp.Abs(x),
    x ** sp.Rational(1, 3),
    sp.E * x,
    e * sp.E,  # symbol e and constant E stay distinct
    i * sp.I,  # symbol i and constant I stay distinct
    e**x * sp.exp(x),
]


@pytest.mark.parametrize("expr", ROUND_TRIP, ids=lambda s: sp.srepr(s)[:40])
def test_printed_latex_parses_back_to_the_same_expression(expr: sp.Expr) -> None:
    back = parse_latex(to_latex(expr))
    assert sp.simplify(back - expr) == 0, to_latex(expr)
    assert back.free_symbols == expr.free_symbols


def test_constants_are_upright_and_symbols_are_not() -> None:
    assert to_latex(sp.exp(sp.I * x)) == r"\mathrm{e}^{\mathrm{i} x}"
    assert to_latex(e**x) == "e^{x}"
    assert to_latex(i * x) == "i x"
    assert to_latex(sp.E) == r"\mathrm{e}"


def test_inverse_trig_uses_the_commands_the_parser_accepts() -> None:
    assert to_latex(sp.asin(x)) == r"\arcsin{\left(x \right)}"


def test_settings_pass_through() -> None:
    assert to_latex(alpha * x, symbol_names={alpha: r"\beta"}) == r"\beta x"


@pytest.mark.parametrize("source", [r"\int x\,dx", r"\infty", r"\sum_{n} x^n"])
def test_unsupported_commands_are_rejected_not_reinterpreted(source: str) -> None:
    with pytest.raises(ParseError):
        parse_latex(source)
