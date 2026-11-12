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


# --- definite integrals ----------------------------------------------------------------------
def definite(
    engine: Engine, source: str, lower: str, upper: str, fmt: str = "text", **assumptions: list[str]
):
    return engine.integrate(
        ExpressionInput(input=source, format=fmt, assumptions=assumptions),
        "x",
        lower=lower,
        upper=upper,
    )


@pytest.mark.parametrize(
    ("source", "lower", "upper", "expected"),
    [
        ("x**2", "0", "3", "9"),
        ("sin(x)", "0", "pi", "2"),
        ("exp(-x)", "0", "oo", "1"),
        ("1/(1 + x**2)", "-oo", "oo", "pi"),
        ("1/sqrt(x)", "0", "1", "2"),  # integrable singularity at the end point
        ("log(x)", "0", "1", "-1"),
    ],
)
def test_known_definite_integrals(engine: Engine, source, lower, upper, expected) -> None:
    result = definite(engine, source, lower, upper)
    assert result.status is Status.OK, result.message
    assert result.definite
    assert result.quadrature is not None and result.quadrature.passed
    assert sp.simplify(sp.sympify(result.result_text) - sp.sympify(expected)) == 0
    assert result.numeric is not None


def test_parameters_are_checked_at_sampled_values(engine: Engine) -> None:
    result = definite(engine, "exp(-a*x**2)", "-oo", "oo", a=["positive"])
    assert result.status is Status.OK
    assert result.quadrature is not None and result.quadrature.points_checked >= 4
    assert result.numeric is None  # has a free parameter


def test_convergence_conditions_are_never_assumed(engine: Engine) -> None:
    result = definite(engine, "exp(-a*x)", "0", "oo")  # a is complex: needs Re(a) > 0
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.result_latex is None
    assert result.conditions_latex
    with_assumption = definite(engine, "exp(-a*x)", "0", "oo", a=["positive"])
    assert with_assumption.status is Status.OK


def test_pole_inside_the_interval_is_never_a_number(engine: Engine) -> None:
    # The fundamental theorem would give -2 here; the integral diverges.
    result = definite(engine, "1/x**2", "-1", "1")
    assert result.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)
    assert result.result_latex is None
    assert result.numeric is None


def test_divergent_integrals_show_no_value(engine: Engine) -> None:
    for source, lower, upper in [("1/x", "0", "1"), ("x", "0", "oo")]:
        result = definite(engine, source, lower, upper)
        assert result.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)
        assert result.result_latex is None


def test_latex_bounds_and_infinity(engine: Engine) -> None:
    result = definite(engine, r"\exponentialE^{-x}", "0", r"\infty", fmt="latex")
    assert result.status is Status.OK
    assert result.lower_latex == "0"
    result = definite(engine, r"\sin x", "0", r"\pi", fmt="latex")
    assert result.status is Status.OK
    assert result.result_text == "2"


def test_reversed_bounds_flip_the_sign(engine: Engine) -> None:
    assert definite(engine, "x", "2", "0").result_text == "-2"


def test_bad_bounds_are_explicit(engine: Engine) -> None:
    assert definite(engine, "x", "a", "1").status is Status.PARSE_ERROR  # symbol, not a number
    assert definite(engine, "x", "I", "1").status is Status.PARSE_ERROR  # not real
    assert definite(engine, "x", "0", "1 +* 2").status is Status.PARSE_ERROR
    half = engine.integrate(ExpressionInput(input="x", format="text"), "x", lower="0")
    assert half.status is Status.PARSE_ERROR


def test_no_closed_form_definite_is_unsupported(engine: Engine) -> None:
    result = definite(engine, "exp(x**x)", "0", "1")
    assert result.status in (Status.UNSUPPORTED, Status.NOT_VERIFIED)
    assert result.result_latex is None


def test_quadrature_verifier_rejects_a_wrong_value() -> None:
    import time

    from minerva_engine.integrate import check_definite

    deadline = time.monotonic() + 20
    f = sp.sin(x)
    good = check_definite(f, x, sp.Integer(0), sp.pi, sp.Integer(2), deadline=deadline)
    assert good.passed
    bad = check_definite(f, x, sp.Integer(0), sp.pi, sp.Integer(3), deadline=deadline)
    assert not bad.passed
    near = check_definite(f, x, sp.Integer(0), sp.pi, sp.Float("2.000001"), deadline=deadline)
    assert not near.passed
    pole = check_definite(
        1 / x**2, x, sp.Integer(-1), sp.Integer(1), sp.Integer(-2), deadline=deadline
    )
    assert not pole.passed
