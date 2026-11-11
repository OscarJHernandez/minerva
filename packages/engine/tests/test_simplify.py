"""Regression tests for ordinary symbolic simplification through the sandboxed engine."""

import pytest
from minerva_engine import Engine, ExpressionInput, Status


def run(engine: Engine, source: str, fmt: str = "latex", **assumptions: list[str]):
    return engine.simplify(ExpressionInput(input=source, format=fmt, assumptions=assumptions))


@pytest.mark.parametrize(
    ("source", "expected_text"),
    [
        (r"\sin^2 x + \cos^2 x", "1"),
        (r"\frac{x^2-1}{x-1}", "x + 1"),
        (r"\cosh^2 x - \sinh^2 x", "1"),
        (r"\frac{\sin x}{\cos x} - \tan x", "0"),
        (r"(x+1)^2 - x^2 - 2x - 1", "0"),
        (r"\ln(\exponentialE^{x})", None),  # x complex: must NOT become x
    ],
)
def test_known_simplifications(engine: Engine, source: str, expected_text: str | None) -> None:
    result = run(engine, source)
    if expected_text is None:
        assert result.status in (Status.UNCHANGED, Status.OK)
        assert result.result_text != "x"
        return
    assert result.status is Status.OK
    assert result.result_text == expected_text
    assert result.verification is not None
    assert result.verification.passed
    assert result.verification.numeric_points_agreed >= 4


def test_unchanged_is_reported_explicitly(engine: Engine) -> None:
    result = run(engine, "x + y", "text")
    assert result.status is Status.UNCHANGED
    assert "unchanged" in result.message.lower()


def test_sqrt_of_square_depends_on_assumptions(engine: Engine) -> None:
    plain = run(engine, r"\sqrt{x^2}")
    assert plain.status is Status.UNCHANGED  # not x: that is false for negative/complex x
    assert plain.result_text == "sqrt(x**2)"
    positive = run(engine, r"\sqrt{x^2}", x=["positive"])
    assert positive.result_text == "x"
    real = run(engine, r"\sqrt{x^2}", x=["real"])
    assert real.result_text == "Abs(x)"


def test_log_of_product_not_split_without_assumptions(engine: Engine) -> None:
    result = run(engine, "log(x*y)", "text")
    assert result.status is Status.UNCHANGED


def test_parse_errors_are_explicit(engine: Engine) -> None:
    result = run(engine, r"\foo{x}")
    assert result.status is Status.PARSE_ERROR
    assert result.result_latex is None


def test_resource_limit_is_reported_not_guessed(engine: Engine) -> None:
    from minerva_engine import Limits

    quick = Engine(Limits(timeout_seconds=2))
    result = quick.simplify(ExpressionInput(input="(x+1)**9000*(y+1)**9000 - 1", format="text"))
    assert result.status is Status.RESOURCE_LIMIT
    assert result.result_latex is None


def test_contradictory_assumptions_are_a_clear_error(engine: Engine) -> None:
    result = run(engine, "x + 1", "text", x=["positive", "negative"])
    assert result.status is Status.PARSE_ERROR
    assert "contradict" in result.message


def test_expansion_bomb_is_rejected_before_computation(engine: Engine) -> None:
    result = run(engine, "(x+1)**9999999", "text")
    assert result.status is Status.RESOURCE_LIMIT
    assert result.result_latex is None
