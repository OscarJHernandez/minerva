"""Regression tests for verified differentiation, run through the sandboxed engine."""

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Limits, Status
from minerva_engine.differentiate import DiffVariable, check_derivative
from pydantic import ValidationError


def d(
    engine: Engine,
    source: str,
    *variables: tuple[str, int],
    fmt: str = "text",
    **assumptions: list[str],
):
    return engine.differentiate(
        ExpressionInput(input=source, format=fmt, assumptions=assumptions),
        [DiffVariable(name=n, order=o) for n, o in variables],
    )


@pytest.mark.parametrize(
    ("source", "variables", "expected"),
    [
        ("x**3", (("x", 1),), "3*x**2"),
        ("x**3", (("x", 2),), "6*x"),
        ("x**3", (("x", 4),), "0"),
        ("sin(x)", (("x", 4),), "sin(x)"),
        ("x**x", (("x", 1),), "x**x*(log(x) + 1)"),
        ("sin(x)*exp(x)", (("x", 10),), "32*exp(x)*cos(x)"),
        ("1/x", (("x", 5),), "-120/x**6"),
        ("gamma(x)", (("x", 1),), "gamma(x)*polygamma(0, x)"),
        ("x**2*y**3", (("x", 1), ("y", 1)), "6*x*y**2"),
        ("x**2*y**3", (("y", 1), ("x", 1)), "6*x*y**2"),  # order of differentiation is free
        ("y", (("x", 1),), "0"),
        ("x**2", (("x", 1), ("y", 1)), "0"),
    ],
)
def test_known_derivatives(engine: Engine, source, variables, expected) -> None:
    result = d(engine, source, *variables)
    assert result.status is Status.OK
    assert result.result_text == expected
    assert result.verification is not None
    assert result.verification.passed


def test_latex_input_and_notation(engine: Engine) -> None:
    result = d(engine, r"\frac{x^2-1}{x-1}", ("x", 1), fmt="latex")
    assert result.status is Status.OK
    assert result.result_text == "1"
    assert result.notation_latex is not None
    assert "partial" in result.notation_latex or "frac" in result.notation_latex


def test_every_step_is_verified_and_reported(engine: Engine) -> None:
    result = d(engine, "x**5", ("x", 3))
    assert result.status is Status.OK
    assert [s.latex for s in result.steps] == ["5 x^{4}", "20 x^{3}", "60 x^{2}"]
    v = result.verification
    assert v is not None
    assert v.steps_checked == 3
    assert v.points_agreed == v.points_checked >= 3 * 4


def test_variable_that_does_not_occur_is_exactly_zero(engine: Engine) -> None:
    result = d(engine, "y", ("x", 1))
    assert result.status is Status.OK
    assert result.result_text == "0"
    assert result.verification is not None
    assert result.verification.steps_checked == 0


# --- assumptions ----------------------------------------------------------------------------
def test_absolute_value_needs_a_real_variable(engine: Engine) -> None:
    plain = d(engine, "abs(x)", ("x", 1))
    assert plain.status is Status.UNSUPPORTED
    assert plain.result_text is None
    assert "real" in plain.message
    real = d(engine, "abs(x)", ("x", 1), x=["real"])
    assert real.status is Status.OK
    assert real.result_text == "sign(x)"


def test_second_derivative_of_abs_is_unsupported_here(engine: Engine) -> None:
    # The pointwise derivative check compares 0 with 0 away from the kink: it proves nothing
    # about the delta, so the distribution operation owns this derivative.
    result = d(engine, "abs(x)", ("x", 2), x=["real"])
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None
    assert "distribution" in (result.message or "")


def test_assumption_ladder_for_sqrt_of_square(engine: Engine) -> None:
    assert d(engine, "sqrt(x**2)", ("x", 1), x=["positive"]).result_text == "1"
    assert d(engine, "sqrt(x**2)", ("x", 1), x=["real"]).result_text == "sign(x)"
    # Complex x: sqrt(x**2) is locally holomorphic (+-x), so sqrt(x**2)/x is a verified derivative
    # that is NOT 1 or sign(x); both directions of the complex check agree on it.
    plain = d(engine, "sqrt(x**2)", ("x", 1))
    assert plain.status is Status.OK
    assert plain.result_text == "sqrt(x**2)/x"
    assert plain.verification is not None
    assert plain.verification.directions == ["real", "imaginary"]


def test_nonnegative_variable_samples_stay_in_the_domain(engine: Engine) -> None:
    result = d(engine, "sqrt(x)", ("x", 3), x=["nonnegative"])
    assert result.status is Status.OK
    assert result.result_text == "3/(8*x**(5/2))"


# --- honesty --------------------------------------------------------------------------------
@pytest.mark.parametrize("assumption", ["integer", "rational"])
def test_discrete_variable_has_no_derivative(engine: Engine, assumption: str) -> None:
    result = d(engine, "x**2", ("x", 1), x=[assumption])
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None


def test_total_order_is_capped(engine: Engine) -> None:
    result = d(engine, "x*y", ("x", 6), ("y", 5))
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None
    with pytest.raises(ValidationError):
        DiffVariable(name="x", order=11)
    with pytest.raises(ValidationError):
        DiffVariable(name="x", order=0)


def test_parse_errors_are_explicit(engine: Engine) -> None:
    assert d(engine, r"\foo{x}", ("x", 1), fmt="latex").status is Status.PARSE_ERROR
    assert d(engine, "x", ("pi", 1)).status is Status.PARSE_ERROR
    assert d(engine, "x + ", ("x", 1)).status is Status.PARSE_ERROR


def test_no_or_too_many_variables_are_unsupported(engine: Engine) -> None:
    assert d(engine, "x").status is Status.UNSUPPORTED
    five = tuple((n, 1) for n in "abcde")
    assert d(engine, "a*b*c*d*e", *five).status is Status.UNSUPPORTED


def test_derivative_tower_ends_in_a_status_not_a_crash() -> None:
    small = Engine(Limits(timeout_seconds=4, max_concurrency=1))
    result = d(small, "x**(x**(x**x))", ("x", 10))
    assert result.status in (Status.OK, Status.RESOURCE_LIMIT)
    if result.status is not Status.OK:
        assert result.result_latex is None


# --- the verifier itself ---------------------------------------------------------------------
x = sp.Symbol("x")
z = sp.Symbol("z")
xr = sp.Symbol("xr", real=True)


def test_verifier_accepts_the_right_derivative() -> None:
    assert check_derivative(sp.sin(x) * x, x, sp.cos(x) * x + sp.sin(x)).passed


def test_verifier_rejects_a_wrong_derivative() -> None:
    wrong = check_derivative(sp.cos(x), x, sp.cos(x))  # the derivative of cos is -sin
    assert not wrong.passed
    assert wrong.points_agreed < wrong.points_checked


def test_verifier_rejects_a_near_miss() -> None:
    assert not check_derivative(x**3, x, 3 * x**2 + sp.Rational(1, 10**6)).passed


def test_verifier_detects_non_holomorphic_functions() -> None:
    # d|z|/dz "=" conj(z)/(2|z|) holds along the real axis but not the imaginary one.
    claimed = sp.conjugate(z) / (2 * sp.Abs(z))
    result = check_derivative(sp.Abs(z), z, claimed)
    assert result.directions == ["real", "imaginary"]
    assert not result.passed
    # With a real variable only the real direction applies, and the formula is right.
    assert check_derivative(sp.Abs(xr), xr, sp.sign(xr)).passed


def test_verifier_needs_enough_valid_points() -> None:
    result = check_derivative(sp.log(x - x), x, sp.Integer(0))  # zoo everywhere
    assert not result.passed
    assert result.points_checked < 4
