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


@pytest.mark.parametrize(
    "source",
    [
        "9**9**9",
        "10**100000",
        "2**(2**(2**5))",
        "9" * 200,
        "(x+1)**9999999",
        "x**100000",
        "factorial(10**6)",
        "gamma(500000)",
    ],
)
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


# -- Leibniz derivative notation is never read as a fraction of letters ---------------------------
@pytest.mark.parametrize(
    "source",
    [
        r"\frac{dx}{dy}",
        r"\frac{dy}{dx} + y",
        r"\frac{d^2x}{dt^2}",
        r"\frac{d}{dx} x^2",
        r"\dfrac{dx}{dt}",
        r"\tfrac{d^{2}y}{dx^{2}}",
        r"\sqrt{1+\left(\frac{dx}{dy}\right)^2}",
    ],
)
def test_leibniz_notation_is_refused_not_cancelled(source: str) -> None:
    # `\frac{dx}{dy}` used to be read as x/y: the d's cancelled and a derivative became a ratio.
    with pytest.raises(ParseError, match="derivative notation"):
        parse_latex(source)


def test_leibniz_shape_with_a_declared_d_is_an_ordinary_fraction() -> None:
    d, a, b = sp.symbols("d a b")
    assumed = {"d": ["positive"]}
    assert parse_latex(r"\frac{da}{db}", assumptions=assumed) == d * a / (d * b)
    assert parse_latex(r"\frac{d}{d+1}") == sp.Symbol("d") / (sp.Symbol("d") + 1)
    assert parse_latex(r"\frac{a}{b}") == a / b
    assert parse_latex(r"\frac{x_d}{d}") == sp.Symbol("x_d") / sp.Symbol("d")


# -- square and curly brackets group like parentheses ---------------------------------------------
_PHI, _TH = sp.symbols("phi theta")


@pytest.mark.parametrize(
    ("bracketed", "plain"),
    [
        (r"\frac{1}{2} \left[ a + b \right]^2", r"\frac{1}{2} \left( a + b \right)^2"),
        (r"\left[ a + b \right] c", r"\left( a + b \right) c"),
        (r"[a+b]", r"(a+b)"),
        (r"\sin\left[ a + b \right]", r"\sin\left( a + b \right)"),
        (r"\frac{1}{2} \left\{ a + b \right\}", r"\frac{1}{2} \left( a + b \right)"),
        (r"\left\lbrace a + b \right\rbrace", r"(a+b)"),
        (r"2 - [a - [b + c]]", r"2 - (a - (b + c))"),
    ],
)
def test_square_and_curly_brackets_group_like_parentheses(bracketed: str, plain: str) -> None:
    assert parse_latex(bracketed) == parse_latex(plain)


@pytest.mark.parametrize(
    "source",
    [
        r"[a, b]",  # a commutator or an interval: never guessed
        r"\left[ a + b \right)",  # mismatched pair
        r"\left( a + b \right]",
        r"\{ a + b \right)",
        r"x[2]",  # would be x*2: indexed data is written x_2
        r"2[a+b]",
        r"[a + b",
    ],
)
def test_ambiguous_or_mismatched_brackets_are_refused(source: str) -> None:
    with pytest.raises(ParseError):
        parse_latex(source)


# --- unreadable notation is named, not just refused -------------------------------------------------
def test_a_prime_is_said_to_be_derivative_notation_read_elsewhere() -> None:
    with pytest.raises(ParseError, match=r"derivative notation.*variation.*differential equations"):
        parse_latex(r"f'(r) + 1")


def test_an_evaluation_bar_and_an_integral_are_named() -> None:
    with pytest.raises(ParseError, match="evaluation bar"):
        parse_latex(r"\left. x^2 \right|_{x=1}")
    with pytest.raises(ParseError, match=r"integral \(\\int\) is read only by"):
        parse_latex(r"\int_0^1 x \, dx")


def test_the_derivative_message_does_not_send_the_user_back_to_the_failing_tool_alone() -> None:
    with pytest.raises(ParseError) as caught:
        parse_latex(r"\frac{dx}{dy} + 1")
    text = str(caught.value)
    assert (
        "`variation`" in text
        and "differential equation tools" in text
        and "time_derivative" in text
    )


def test_ell_is_an_ordinary_letter() -> None:
    ell = sp.Symbol("ell")
    assert parse_latex(r"\ell^2 + \ell") == ell**2 + ell
