"""Every problem in an input is named in one error, with the form that would be read.

Regression for two chats (2026-11-26): a Schwarzschild metric written as `-(1-2M/r)` next to
`r^2*sin(theta)^2` took eight geometry calls, because LaTeX errors named one token at a time (`sin(`, then
`theta`) and the text parser said only "could not parse the expression" for `2M`; and a Lagrangian written as
`1/2 m1 (x1_dot^2 + ...) - k/2 * [ ... ]` got the same bare message. Nothing here accepts more input: each
refused form is still refused, and each accepted twin still reads the same.
"""

import pytest
import sympy as sp
from minerva_engine import Limits, Status
from minerva_engine.errors import ParseError
from minerva_engine.geometry import GeometryInput, geometry
from minerva_engine.lagrangian import DerivationError, derive
from minerva_engine.latexparse import parse_latex, reject_text_syntax, reject_text_syntax_in
from minerva_engine.notation.models import InputCode
from minerva_engine.notation.names import near_misses, spelling_key
from minerva_engine.textparse import parse_text

# --- text syntax: products without `*` and square brackets --------------------------------------------------


@pytest.mark.parametrize(
    ("source", "symbols", "suggestions"),
    [
        ("-(1-2M/r)", ["2M"], ["2*M"]),
        ("(1-2M/r)^{-1}", ["2M"], ["2*M"]),
        ("2 M", ["2 M"], ["2*M"]),
        ("2(x+1)", ["2("], ["2*("]),
        ("m1 (x+1)", ["m1 ("], ["m1*("]),
        ("x y", ["x y"], ["x*y"]),
        ("(a)(b)", [")("], [")*("]),
        ("x^2y", ["2y"], ["2*y"]),
        ("k/2 * [x - a]", ["["], ["("]),
        (
            "1/2 m1 (x1_dot^2 + y1_dot^2) - k/2 * [ (sqrt((x2-x1)^2) - a)^2 ]",
            ["2 m1", "m1 (", "["],
            ["2*m1", "m1*(", "("],
        ),
    ],
)
def test_a_product_without_a_star_is_refused_naming_every_spot(
    source: str, symbols: list[str], suggestions: list[str]
) -> None:
    with pytest.raises(ParseError) as info:
        parse_text(source)
    assert info.value.code is InputCode.IMPLICIT_MULTIPLICATION
    assert list(info.value.symbols) == symbols
    assert list(info.value.suggestions) == suggestions
    assert info.value.recoverable


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2e3*x", "2000*x"),
        ("2*M", "2*M"),
        ("sin (x)", "sin(x)"),
        ("(1-2*M/r)^{-1}", "1/(1 - 2*M/r)"),
        ("m1*(x+1)", "m1*(x + 1)"),
    ],
)
def test_the_explicit_twins_still_read_the_same(source: str, expected: str) -> None:
    assert parse_text(source) == sp.sympify(expected)  # test-only parsing of trusted literals


def test_a_call_written_without_a_space_keeps_its_own_error() -> None:
    with pytest.raises(ParseError) as info:
        parse_text("f(x)")
    assert info.value.code is None and "f" in str(info.value)


def test_any_other_syntax_error_still_carries_a_code() -> None:
    with pytest.raises(ParseError) as info:
        parse_text("x +* y")
    assert info.value.code is InputCode.SYNTAX
    assert str(info.value).startswith("could not parse the expression")


# --- LaTeX: every piece of text syntax at once --------------------------------------------------------------


def test_every_text_syntax_problem_in_a_latex_input_is_named() -> None:
    with pytest.raises(ParseError) as info:
        parse_latex(r"r^2*sin(theta)^2 + omega_dot")
    assert info.value.code is InputCode.TEXT_SYNTAX_IN_LATEX
    assert list(info.value.symbols) == ["_dot", "sin(", "theta", "omega"]
    assert list(info.value.suggestions)[-1] == "format text"


def test_one_problem_keeps_the_single_message() -> None:
    with pytest.raises(ParseError) as one:
        reject_text_syntax("sin(x)")
    with pytest.raises(ParseError) as many:
        reject_text_syntax_in(["sin(x)", "1"])
    assert str(one.value) == str(many.value)
    assert one.value.symbols == many.value.symbols == ("sin(",)


def test_a_repeated_problem_is_named_once() -> None:
    with pytest.raises(ParseError) as info:
        reject_text_syntax_in(["sin(x)", "sin(y)", "theta"])
    assert list(info.value.symbols) == ["sin(", "theta"]


# --- geometry: the metric of the chat -----------------------------------------------------------------------

MIXED = [
    ["-(1-2M/r)", "0", "0", "0"],
    ["0", "(1-2M/r)^{-1}", "0", "0"],
    ["0", "0", "r^2", "0"],
    ["0", "0", "0", "r^2*sin(theta)^2"],
]


def run(metric: list[list[str]], fmt: str):  # type: ignore[no-untyped-def]
    data = GeometryInput(
        coordinates=["t", "r", "theta", "phi"],
        metric=metric,
        format=fmt,
        quantities=["christoffel"],
    )
    return geometry(data, Limits())


def test_a_latex_metric_names_every_text_syntax_problem_in_every_cell() -> None:
    result = run(MIXED, "latex")
    assert result.status is Status.PARSE_ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code is InputCode.TEXT_SYNTAX_IN_LATEX
    assert result.diagnostic.symbols == ["sin(", "theta"]
    assert result.diagnostic.field == "metric"


def test_a_text_metric_names_every_product_without_a_star() -> None:
    metric = [row[:] for row in MIXED]
    metric[2][2] = "2r^2"
    result = run(metric, "text")
    assert result.status is Status.PARSE_ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code is InputCode.IMPLICIT_MULTIPLICATION
    assert result.diagnostic.symbols == ["2M", "2r"]
    assert result.diagnostic.suggestions == ["2*M", "2*r"]


def test_the_corrected_text_metric_is_computed() -> None:
    metric = [[c.replace("2M", "2*M").replace("{-1}", "(-1)") for c in row] for row in MIXED]
    assert run(metric, "text").status is Status.OK


# --- declared coordinates: suggest the name the Lagrangian uses ----------------------------------------------


# In LaTeX a declared `x1` or `\\theta_1` is now read as the `x_1` / `theta_1` the input writes
# (`test_mixed_syntax_acceptance.py`); in text syntax `x1` and `x_1` are two names, so it is still refused.
@pytest.mark.parametrize(
    ("declared", "lagrangian", "fmt", "suggestions"),
    [
        (
            ["x1", "y1"],
            "1/2*m*(x_1_dot^2 + y_1_dot^2) - k*x_1^2 - k*y_1^2",
            "text",
            ["x_1", "y_1"],
        ),
        (["theta1"], "1/2*m*theta_1_dot^2 - k*theta_1^2", "text", ["theta_1"]),
        (["q"], r"\frac12 m v^2 - k x^2", "latex", ["q_dot"]),
        (["x", "m1"], r"\frac12 m_1 \dot{x}^2 - k x^2", "latex", ["m_1"]),
    ],
)
def test_a_misspelt_coordinate_is_refused_with_the_name_read(
    declared: list[str], lagrangian: str, fmt: str, suggestions: list[str]
) -> None:
    with pytest.raises(DerivationError) as info:
        derive(lagrangian, declared, "t", {}, Limits(), fmt)  # type: ignore[arg-type]
    diagnostic = info.value.diagnostic
    assert diagnostic is not None and diagnostic.code is InputCode.DECLARED_NAME_ABSENT
    assert diagnostic.symbols == [n for n in declared if n not in ("x",)]
    assert diagnostic.suggestions == suggestions


def test_two_names_spelt_alike_are_both_suggested_and_none_is_chosen() -> None:
    assert near_misses("x1", ["x_1", "x1_dot", "k"]) == ["x_1"]
    assert near_misses("ab", ["a_b", "ab_dot"]) == ["a_b"]
    assert near_misses("ab", ["a_b", "a_b_dot", "q"]) == ["a_b"]
    assert near_misses("xy1", ["x_y1", "xy_1"]) == ["x_y1", "xy_1"]
    assert spelling_key(r"\theta_{1}") == spelling_key("theta1") == "theta1"
