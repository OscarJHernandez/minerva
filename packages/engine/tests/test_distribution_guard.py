"""Distributions are readable, print back exactly, and are refused by every pointwise operation."""

import pytest
import sympy as sp
from minerva_engine import (
    DiffVariable,
    Engine,
    ExpressionInput,
    ParseError,
    Status,
    parse_latex,
    parse_text,
)
from minerva_engine.latexprint import to_latex
from minerva_engine.verify import verify_equal

x = sp.Symbol("x", real=True)
a = sp.Symbol("a", real=True)


def test_explicit_names_parse_in_both_syntaxes() -> None:
    assert parse_text("DiracDelta(x - 1)") == sp.DiracDelta(sp.Symbol("x") - 1)
    assert parse_text("Heaviside(x)") == sp.Heaviside(sp.Symbol("x"))
    assert parse_latex(r"\operatorname{DiracDelta}(x-1)") == sp.DiracDelta(sp.Symbol("x") - 1)
    assert parse_latex(r"\operatorname{Heaviside}(x)") == sp.Heaviside(sp.Symbol("x"))


def test_greek_letters_stay_symbols() -> None:
    # \delta(x) must keep meaning "delta times x"; distributions need the explicit names.
    d, t, y = sp.symbols("delta theta x")
    assert parse_latex(r"\delta(x)") == d * y
    assert parse_latex(r"\theta(x)") == t * y


def test_a_non_real_argument_is_a_parse_error() -> None:
    with pytest.raises(ParseError):
        parse_text("DiracDelta(I)")


@pytest.mark.parametrize(
    "expr",
    [sp.DiracDelta(x - a), sp.Heaviside(x), 3 * sp.DiracDelta(2 * x + 1) * sp.exp(x)],
    ids=["delta", "heaviside", "product"],
)
def test_printed_distributions_read_back(expr: sp.Expr) -> None:
    printed = to_latex(expr)
    assert r"\operatorname" in printed
    back = parse_latex(printed, {s.name: ["real"] for s in expr.free_symbols})
    assert back == expr


def test_pointwise_verification_never_passes_a_distribution() -> None:
    # Sampling sees 0 = 5*delta(x) at every point: it must refuse instead of agreeing.
    v = verify_equal(sp.Integer(0), 5 * sp.DiracDelta(x))
    assert v.passed is False
    assert "distribution" in v.note
    assert verify_equal(sp.Heaviside(x), (sp.sign(x) + 1) / 2).passed is False


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine()


def test_other_operations_refuse_distributions(engine: Engine) -> None:
    data = ExpressionInput(input=r"\operatorname{DiracDelta}(x)", assumptions={"x": ["real"]})
    simplified = engine.simplify(data)
    assert simplified.status is Status.UNSUPPORTED
    assert simplified.result_latex is None
    assert "distribution" in simplified.message
    assert engine.integrate(data, "x").status is Status.UNSUPPORTED
    assert (
        engine.differentiate(data, [DiffVariable(name="x", order=1)]).status is Status.UNSUPPORTED
    )


def test_a_derivative_that_produces_a_delta_is_not_verified_pointwise(engine: Engine) -> None:
    data = ExpressionInput(input="Heaviside(x)", format="text", assumptions={"x": ["real"]})
    assert (
        engine.differentiate(data, [DiffVariable(name="x", order=1)]).status is Status.UNSUPPORTED
    )


def test_features_of_a_distribution_expression_are_still_read(engine: Engine) -> None:
    # Retrieval needs `f:diracdelta`; reading the structure is not evaluating the expression.
    status, _, features, _ = engine.features(
        ExpressionInput(input=r"\operatorname{DiracDelta}(x-1)", assumptions={"x": ["real"]})
    )
    assert status is Status.OK
    assert "f:diracdelta" in features
