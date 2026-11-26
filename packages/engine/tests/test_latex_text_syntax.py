"""Plain-text syntax written where LaTeX is expected is refused, never read as a product of letters.

Regression: `theta_dot^2` parsed as `t*h*e*t*a_d*o*t**3` and `sin(theta)` as `s*i*n*theta`; the analysis then
failed (or worse, answered) about another expression. Each refused form has an accepted twin.
"""

import pytest
import sympy as sp
from minerva_engine.errors import ParseError
from minerva_engine.latexparse import parse_latex
from minerva_engine.latexprint import to_latex
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


# -- braced exponents in text syntax (`x^{-1}` is `x^(-1)`; no set literal is ever accepted) ------------------

x, y, n, a, b, M, r = sp.symbols("x y n a b M r")


@pytest.mark.parametrize(
    ("braced", "parenthesised"),
    [
        ("x^{-1}", "x^(-1)"),
        ("(x+y)^{n+1}", "(x+y)^(n+1)"),
        ("(1-2*M/r)^{-1}", "1/(1-2*M/r)"),
        ("x^{a^{b}+1}", "x^(a^(b)+1)"),  # nested: the inner group is read the same way
        ("x^{ -2 }*y", "x^(-2)*y"),
        ("2^{x^{2}}", "2^(x^2)"),
    ],
)
def test_a_braced_exponent_is_the_parenthesised_one(braced: str, parenthesised: str) -> None:
    assert parse_text(braced) == parse_text(parenthesised)


def test_a_braced_exponent_is_a_power_not_a_product_or_a_set() -> None:
    assert parse_text("x^{-1}") == 1 / x
    assert parse_text("x^{-1}") != -x and parse_text("x^{-1}") != x * -1
    assert parse_text("(x+y)^{n+1}") == (x + y) ** (n + 1)


@pytest.mark.parametrize(
    "source",
    [
        "x^{a,b}",  # a top-level comma: a set or tuple, never an exponent
        "x^{a",  # unbalanced
        "x^a}",
        "x^{}",  # empty
        "{1,2}",  # a set literal elsewhere
        "x*{2}",  # a brace group not directly after `^`
        "x^{2}*{3}",
        "x {^2}",
        "x^{(a,b)}",  # a tuple inside
    ],
)
def test_braces_that_are_not_a_plain_exponent_stay_refused(source: str) -> None:
    with pytest.raises(ParseError):
        parse_text(source)


def test_the_rule_never_touches_a_text_without_a_braced_exponent() -> None:
    from minerva_engine.notation.transport import read_braced_exponents

    for source in ["x^2", "x^(-1)", "sin(x)^2 + 1/(1-2*M/r)", "x**2"]:
        assert read_braced_exponents(source) == (source, ())
    text, notes = read_braced_exponents("(1-2*M/r)^{-1}")
    assert text == "(1-2*M/r)^(-1)" and len(notes) == 1
    assert notes[0].kind.value == "transport"


# -- velocities print with the dot they were written with (debug report: a unit card showed theta_{dot}) ----


@pytest.mark.parametrize(
    ("name", "printed"),
    [
        ("r_dot", r"\dot{r}"),
        ("theta_dot", r"\dot{\theta}"),
        ("x_ddot", r"\ddot{x}"),  # SymPy's default printed x_{\dot{d}}
        ("theta_1_dot", r"\dot{\theta}_{1}"),
        ("q_12_dot", r"\dot{q}_{12}"),
        ("x_alpha_dot", r"\dot{x}_{\alpha}"),
        ("Phi_dot", r"\dot{\Phi}"),
    ],
)
def test_a_velocity_prints_with_its_dot_and_reads_back(name: str, printed: str) -> None:
    symbol = sp.Symbol(name)
    assert to_latex(symbol) == printed
    assert parse_latex(printed) == symbol


def test_the_written_velocities_survive_a_round_trip_in_an_expression() -> None:
    expr = parse_latex(r"m r^2 \dot{r} \dot{\theta} - \ddot{x}_{1}")
    text = to_latex(expr)
    assert "dot}" not in text and r"\dot{\theta}" in text and r"\ddot{x}_{1}" in text
    assert parse_latex(text) == expr


@pytest.mark.parametrize("name", ["x_d", "dot", "x_dotty", "x_1"])
def test_names_that_are_not_velocities_print_as_before(name: str) -> None:
    assert r"\dot" not in to_latex(sp.Symbol(name))


def test_a_caller_name_map_still_wins() -> None:
    v = sp.Symbol("x_dot")
    assert to_latex(v, symbol_names={v: "v"}) == "v"
