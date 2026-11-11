import pytest
import sympy as sp

from minerva_engine import LimitError, MinervaMathError, ParseError, parse_latex, parse_text

x, y = sp.symbols("x y")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\sin^2 x + \cos^2 x", sp.sin(x) ** 2 + sp.cos(x) ** 2),
        (r"\frac{x^2-1}{x-1}", (x**2 - 1) / (x - 1)),
        (r"\frac{1}{2}x", x / 2),
        (r"2x^2", 2 * x**2),
        (r"x^23", 3 * x**2),  # TeX: an unbraced exponent is one digit
        (r"x^{y^2}", x ** (y**2)),
        (r"-x^2", -(x**2)),
        (r"\sqrt[3]{x}", x ** sp.Rational(1, 3)),
        (r"|x|", sp.Abs(x)),
        (r"\left(x+1\right)^2", (x + 1) ** 2),
        (r"\Gamma(x+1)", sp.gamma(x + 1)),
        (r"\ln(x)", sp.log(x)),
        (r"\sinh^2 x - \cosh^2 x", sp.sinh(x) ** 2 - sp.cosh(x) ** 2),
        (r"\exponentialE^{\imaginaryI\pi}", sp.Integer(-1)),
        (r"x \cdot y", x * y),
        (r"\sin x \cos x", sp.sin(x) * sp.cos(x)),
    ],
)
def test_latex_parses(source: str, expected: sp.Expr) -> None:
    assert sp.simplify(parse_latex(source) - expected) == 0


def test_latex_e_and_i_are_plain_symbols() -> None:
    expr = parse_latex(r"e^{i}")
    assert {s.name for s in expr.free_symbols} == {"e", "i"}


def test_latex_subscripts_and_greek() -> None:
    expr = parse_latex(r"\alpha_1 + \beta_{ij}")
    assert {s.name for s in expr.free_symbols} == {"alpha_1", "beta_ij"}


@pytest.mark.parametrize(
    "source",
    [
        r"\sin 2x",  # ambiguous bare argument
        r"\sin^{-1} x",  # ambiguous inverse
        r"\int x\,dx",
        r"\unknowncommand",
        r"x^",
        r"\frac{1}{",
        r"((x)",
        r"x = 1",
        r"\sin",
        "x # y",
        r"\frac{1}{0}",
    ],
)
def test_latex_rejects(source: str) -> None:
    with pytest.raises(MinervaMathError):
        parse_latex(source)


def test_latex_assumptions_reach_symbols() -> None:
    expr = parse_latex(r"\sqrt{x^2}", {"x": ["positive"]})
    assert expr == sp.Symbol("x", positive=True)


def test_unknown_assumption_rejected() -> None:
    with pytest.raises(ParseError):
        parse_latex("x", {"x": ["hackable"]})


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "x.real",
        "(lambda: 1)()",
        "[x for x in range(3)]",
        "a[0]",
        "open('/etc/passwd')",
        "x if y else 1",
        "f'{x}'",
        "'a' + 'b'",
        "x; y",
        "import os",
        "sin(x, y)",
        "sin(x=1)",
        "x__class__",
    ],
)
def test_text_parser_rejects_code(source: str) -> None:
    with pytest.raises(MinervaMathError):
        parse_text(source)


def test_text_parser_accepts_math() -> None:
    assert parse_text("sin(x)**2 + cos(x)^2") == sp.sin(x) ** 2 + sp.cos(x) ** 2
    assert parse_text("2**0.5") == sp.sqrt(2)
    assert parse_text("log(8, 2)") == 3


@pytest.mark.parametrize("source", ["9**9**9", "10**100000", "2**(2**(2**5))", "9" * 200])
def test_numeric_explosions_rejected_before_evaluation(source: str) -> None:
    with pytest.raises(LimitError):
        parse_text(source)


def test_latex_numeric_explosion_rejected() -> None:
    with pytest.raises(LimitError):
        parse_latex(r"9^{9^{9}}")


def test_input_length_and_depth_limits() -> None:
    with pytest.raises(LimitError):
        parse_text("x+" * 5000 + "x")
    with pytest.raises(MinervaMathError):
        parse_text("(" * 500 + "x" + ")" * 500)
    with pytest.raises(MinervaMathError):
        parse_latex("{" * 500 + "x" + "}" * 500)


def test_reserved_names_cannot_be_symbols() -> None:
    with pytest.raises(ParseError):
        parse_text("sin + 1")
