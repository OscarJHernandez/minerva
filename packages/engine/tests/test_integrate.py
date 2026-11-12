"""Regression tests for verified antiderivatives, run through the sandboxed engine."""

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.integrate import check_antiderivative


def i(engine: Engine, source: str, var: str = "x", fmt: str = "text", **assumptions: list[str]):
    return engine.integrate(ExpressionInput(input=source, format=fmt, assumptions=assumptions), var)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x**2", "x**3/3"),
        ("cos(x)", "sin(x)"),
        ("exp(2*x)", "exp(2*x)/2"),
        ("x*cos(x)", "x*sin(x) + cos(x)"),
        ("1/(x**2 + 1)", "atan(x)"),
        ("exp(-x**2)", "sqrt(pi)*erf(x)/2"),
    ],
)
def test_known_antiderivatives(engine: Engine, source: str, expected: str) -> None:
    result = i(engine, source)
    assert result.status is Status.OK
    assert result.verification is not None
    assert result.verification.passed
    assert sp.simplify(sp.sympify(result.result_text) - sp.sympify(expected)) == 0
    assert any("constant" in n for n in result.notes)


def test_special_function_is_noted(engine: Engine) -> None:
    result = i(engine, "exp(-x**2)")
    assert any("special functions" in n for n in result.notes)


def test_one_over_x_with_positive_and_complex_variable(engine: Engine) -> None:
    assert i(engine, "1/x", x=["positive"]).status is Status.OK
    assert i(engine, "1/x").status is Status.OK


def test_latex_input_and_notation(engine: Engine) -> None:
    result = i(engine, r"x \cdot \exponentialE^{x}", fmt="latex")
    assert result.status is Status.OK
    assert result.notation_latex is not None
    assert "int" in result.notation_latex


def test_power_with_symbolic_exponent_checks_every_case(engine: Engine) -> None:
    result = i(engine, "x**n")
    assert result.status is Status.OK
    assert result.verification is not None
    assert result.verification.branches_checked == 2
    assert "log" in (result.result_text or "")


def test_no_closed_form_is_unsupported_not_a_guess(engine: Engine) -> None:
    result = i(engine, "exp(x**x)")
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None
    assert "does not prove" in result.message


def test_integer_variable_is_unsupported(engine: Engine) -> None:
    assert i(engine, "x**2", x=["integer"]).status is Status.UNSUPPORTED


def test_parse_errors_are_explicit(engine: Engine) -> None:
    assert i(engine, "x +* 2").status is Status.PARSE_ERROR


def test_constant_integrand(engine: Engine) -> None:
    result = i(engine, "y", "x")
    assert result.status is Status.OK
    assert result.result_text == "x*y"
    assert any("does not depend" in n for n in result.notes)


def test_timeout_ends_in_a_status_not_a_crash() -> None:
    small = Engine(Limits(timeout_seconds=3, max_concurrency=1))
    result = i(small, "sin(x)**7*exp(x)*tan(x)/(x**3 + 2*x + 1)")
    assert result.status in (Status.OK, Status.UNSUPPORTED, Status.RESOURCE_LIMIT)
    if result.status is not Status.OK:
        assert result.result_latex is None


# --- the verifier itself ---------------------------------------------------------------------
x = sp.Symbol("x")


def test_verifier_accepts_the_right_antiderivative() -> None:
    assert check_antiderivative(sp.cos(x), x, sp.sin(x)).passed


def test_verifier_rejects_a_wrong_antiderivative() -> None:
    assert not check_antiderivative(sp.cos(x), x, -sp.sin(x)).passed


def test_verifier_rejects_a_near_miss() -> None:
    assert not check_antiderivative(sp.cos(x), x, sp.sin(x) + sp.Rational(1, 10**6) * x).passed


def test_verifier_refuses_a_case_it_cannot_sample() -> None:
    n = sp.Symbol("n")
    bad = sp.Piecewise((x**2, n > 0), (x, True))
    assert not check_antiderivative(2 * x, x, bad).passed
