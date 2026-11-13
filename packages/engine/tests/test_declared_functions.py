r"""Declared functions of named symbols (differential geometry): `f(r)` is a function only if declared."""

import pytest
import sympy as sp
from minerva_engine import ParseError, parse_latex, parse_text
from minerva_engine.latexprint import to_latex

r, t = sp.symbols("r t")
F = {"f": ("r",)}
f = sp.Function("f", real=True)


def test_a_declared_function_is_read_as_a_function_not_a_product() -> None:
    expr = parse_latex("f(r)^2 + 1", functions=F)
    assert expr == f(r) ** 2 + 1
    assert expr.has(sp.Function("f", real=True))
    assert parse_text("f(r)**2 + 1", functions=F) == expr


def test_without_a_declaration_latex_keeps_the_implicit_product() -> None:
    # Existing behaviour of every other operation is unchanged.
    assert parse_latex("f(r)") == sp.Symbol("f") * r


def test_strict_calls_refuse_an_undeclared_call_instead_of_reading_a_product() -> None:
    with pytest.raises(ParseError, match="ambiguous"):
        parse_latex("f(r)", strict_calls=True)
    # An explicit product is still fine, and so is a parenthesis after an operator or a number.
    assert parse_latex(r"f\cdot(r)", strict_calls=True) == sp.Symbol("f") * r
    assert parse_latex("2(r+1)", strict_calls=True) == 2 * r + 2


def test_text_parser_still_rejects_an_undeclared_function() -> None:
    with pytest.raises(ParseError, match="unsupported function"):
        parse_text("f(r)")


def test_a_declared_function_cannot_be_used_as_a_bare_symbol() -> None:
    with pytest.raises(ParseError, match="declared function"):
        parse_latex("f + 1", functions=F)
    with pytest.raises(ParseError, match="declared function"):
        parse_text("f + 1", functions=F)


def test_arguments_must_be_exactly_the_declared_ones() -> None:
    for source in ("f(t)", "f(2r)", "f(r, t)", "f(r+1)"):
        with pytest.raises(ParseError):
            parse_latex(source, functions=F)


def test_greek_and_subscripted_function_names() -> None:
    expr = parse_latex(r"e^{2\Phi(r)} + a_1(t)", functions={"Phi": ("r",), "a_1": ("t",)})
    assert expr.has(sp.Function("Phi", real=True)(r))
    assert expr.has(sp.Function("a_1", real=True)(t))


def test_two_argument_functions() -> None:
    expr = parse_latex("h(r, t)", functions={"h": ("r", "t")})
    assert expr == sp.Function("h", real=True)(r, t)
    with pytest.raises(ParseError):
        parse_latex("h(t, r)", functions={"h": ("r", "t")})


@pytest.mark.parametrize(
    "declaration",
    [
        {"sin": ("r",)},  # reserved name
        {"f": ()},  # no argument
        {"f": ("r", "r")},  # repeated argument
        {"f": ("f",)},  # a function of itself
        {"f": ("pi",)},  # reserved argument
        {"f g": ("r",)},  # not a name
        {"f": ("a", "b", "c", "d", "e")},  # too many arguments
    ],
)
def test_bad_declarations_are_rejected(declaration: dict[str, tuple[str, ...]]) -> None:
    with pytest.raises(ParseError):
        parse_latex("1", functions=declaration)


def test_assumptions_carry_over_to_the_arguments() -> None:
    expr = parse_latex("f(r)", assumptions={"r": ["positive"]}, functions=F)
    (arg,) = expr.args
    assert arg.is_positive is True


def test_printed_functions_parse_back() -> None:
    expr = f(r) ** 2 + sp.exp(2 * f(r)) / r
    printed = to_latex(expr)
    assert r"f\left(r\right)" in printed
    back = parse_latex(printed, functions=F)
    assert sp.simplify(back - expr) == 0


def test_printed_derivatives_use_primes() -> None:
    assert to_latex(sp.Derivative(f(r), r)) == r"f'\left(r\right)"
    assert to_latex(sp.Derivative(f(r), (r, 2))) == r"f''\left(r\right)"
    assert to_latex(sp.Derivative(f(r), (r, 4))) == r"f^{(4)}\left(r\right)"


def test_ordinary_functions_print_as_before() -> None:
    assert to_latex(sp.sin(r)) == r"\sin{\left(r \right)}"
    assert to_latex(sp.Derivative(sp.sin(r), r)) == r"\frac{d}{d r} \sin{\left(r \right)}"
