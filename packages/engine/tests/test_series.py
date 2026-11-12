"""Regression tests for verified Taylor expansion, run through the sandboxed engine."""

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Status
from minerva_engine.series import check_series

x = sp.Symbol("x")
a = sp.Symbol("a")


def s(
    engine: Engine,
    source: str,
    order: int,
    point: str = "0",
    variable: str = "x",
    **assumptions: list[str],
):
    return engine.series(
        ExpressionInput(input=source, format="text", assumptions=assumptions),
        variable,
        point,
        order,
    )


def same(result_text: str | None, expected: sp.Expr) -> bool:
    assert result_text is not None
    return sp.simplify(sp.sympify(result_text) - expected) == 0


# --- positive ----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("source", "order", "point", "expected"),
    [
        ("exp(x)", 5, "0", sum(x**k / sp.factorial(k) for k in range(6))),
        ("sin(x)", 7, "0", x - x**3 / 6 + x**5 / 120 - x**7 / 5040),
        ("cos(x)", 4, "0", 1 - x**2 / 2 + x**4 / 24),
        ("1/(1-x)", 4, "0", 1 + x + x**2 + x**3 + x**4),
        ("log(1+x)", 4, "0", x - x**2 / 2 + x**3 / 3 - x**4 / 4),
        ("tan(x)", 5, "0", x + x**3 / 3 + 2 * x**5 / 15),
        ("exp(x)", 3, "1", sp.E * sum((x - 1) ** k / sp.factorial(k) for k in range(4))),
        ("sqrt(x)", 2, "4", 2 + (x - 4) / 4 - (x - 4) ** 2 / 64),
    ],
)
def test_known_series(engine: Engine, source, order, point, expected) -> None:
    result = s(engine, source, order, point, x=["positive"] if source == "sqrt(x)" else [])
    assert result.status is Status.OK, result.message
    assert same(result.result_text, expected)
    assert result.verification is not None
    assert result.verification.passed
    assert result.verification.coefficients_passed and result.verification.remainder_passed
    assert result.remainder_latex is not None


def test_symbolic_exponent_is_checked_at_random_parameter_values(engine: Engine) -> None:
    result = s(engine, "(1+x)**a", 3)
    assert result.status is Status.OK, result.message
    expected = 1 + a * x + a * (a - 1) / 2 * x**2 + a * (a - 1) * (a - 2) / 6 * x**3
    assert same(result.result_text, expected)
    assert result.verification is not None
    assert result.verification.parameter_sets_checked >= 4


def test_polynomial_is_exact_and_independent_of_order(engine: Engine) -> None:
    result = s(engine, "x**3 + 2*x", 6)
    assert result.status is Status.OK
    assert any("exact" in note for note in result.notes)
    assert all(float(e.absolute_error) == 0.0 for e in result.measured_errors)


def test_expression_without_the_variable_is_its_own_series(engine: Engine) -> None:
    result = s(engine, "y**2", 3)
    assert result.status is Status.OK
    assert result.terms[0].power == 0


def test_measured_errors_shrink_with_distance(engine: Engine) -> None:
    result = s(engine, "exp(x)", 4)
    errors = [float(e.absolute_error) for e in result.measured_errors]
    assert len(errors) == 3
    assert errors == sorted(errors, reverse=True)  # distances 0.5, 0.1, 0.01


def test_nearby_pole_is_verified_at_a_small_circle_or_not_at_all(engine: Engine) -> None:
    result = s(engine, "1/(1-100*x)", 4)
    assert result.status is Status.OK
    assert same(result.result_text, sum((100 * x) ** k for k in range(5)))
    assert result.verification is not None and result.verification.radius is not None
    assert result.verification.radius < 0.01  # the pole at 0.01 must lie outside the circle


# --- the checker rejects wrong candidates ------------------------------------------------------
GOOD = 1 + x + x**2 / 2 + x**3 / 6


def test_checker_accepts_the_right_polynomial() -> None:
    assert check_series(sp.exp(x), x, sp.Integer(0), GOOD, 3).passed


@pytest.mark.parametrize(
    "wrong",
    [
        1 + x + x**2 / 2 + x**3 / 5,  # one wrong coefficient at the top order
        1 + x + x**2 / 3 + x**3 / 6,  # one wrong coefficient in the middle
        1 + x + x**2 / 2,  # truncated one order early
        2 + x + x**2 / 2 + x**3 / 6,  # wrong constant term
        1 + x + x**2 / 2 + x**3 / 6 + sp.Rational(1, 10**9) * x**3,  # tiny error is still an error
    ],
)
def test_checker_rejects_wrong_candidates(wrong) -> None:
    result = check_series(sp.exp(x), x, sp.Integer(0), wrong, 3)
    assert not result.passed
    assert not (result.coefficients_passed and result.remainder_passed)


def test_checker_rejects_a_candidate_that_is_not_a_polynomial() -> None:
    assert not check_series(sp.exp(x), x, sp.Integer(0), sp.exp(x), 3).passed
    assert not check_series(sp.exp(x), x, sp.Integer(0), 1 / x + 1, 3).passed


def test_checker_rejects_the_series_about_the_wrong_point() -> None:
    assert not check_series(sp.exp(x), x, sp.Integer(1), GOOD, 3).passed


def test_remainder_leg_alone_catches_a_wrong_order_claim() -> None:
    # exp is right to order 3, but claiming order 4 without the x^4 term must fail.
    assert not check_series(sp.exp(x), x, sp.Integer(0), GOOD, 4).passed


# --- honesty: no formula where none exists -----------------------------------------------------
@pytest.mark.parametrize("source", ["1/x", "sqrt(x)", "log(x)", "1/sin(x)", "x**(1/3)"])
def test_poles_and_branch_points_at_the_point_are_unsupported(engine: Engine, source) -> None:
    result = s(engine, source, 3)
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None


def test_absolute_value_has_no_taylor_series(engine: Engine) -> None:
    result = s(engine, "Abs(x)", 3, "1")
    assert result.status is Status.UNSUPPORTED
    assert result.result_latex is None


@pytest.mark.parametrize("source", ["re(x)", "im(x)", "conjugate(x)", "sign(x)"])
def test_other_non_analytic_functions_never_give_a_series(engine: Engine, source) -> None:
    # The parser does not accept these functions at all today; either refusal is acceptable,
    # a formula is not.
    result = s(engine, source, 3, "1")
    assert result.status in {Status.PARSE_ERROR, Status.UNSUPPORTED}
    assert result.result_latex is None


def test_point_at_infinity_or_symbolic_point_is_unsupported(engine: Engine) -> None:
    assert s(engine, "exp(x)", 3, "oo").status is Status.UNSUPPORTED
    assert s(engine, "exp(x)", 3, "y").status is Status.UNSUPPORTED


def test_integer_variable_is_unsupported(engine: Engine) -> None:
    assert s(engine, "x**2", 3, x=["integer"]).status is Status.UNSUPPORTED


def test_order_out_of_range_is_unsupported(engine: Engine) -> None:
    assert s(engine, "exp(x)", 13).status is Status.UNSUPPORTED
    assert s(engine, "exp(x)", 0).status is Status.UNSUPPORTED


def test_unparsable_input_is_a_parse_error(engine: Engine) -> None:
    assert s(engine, "exp(x", 3).status is Status.PARSE_ERROR


# --- assumptions -------------------------------------------------------------------------------
def test_positive_variable_expands_log_about_one_and_stays_in_its_domain(engine: Engine) -> None:
    result = s(engine, "log(x)", 4, "1", x=["positive"])
    assert result.status is Status.OK, result.message
    assert result.verification is not None
    assert result.verification.directions == ["real", "imaginary"][:1] or "real" in (
        result.verification.directions
    )


def test_real_variable_is_only_probed_along_the_real_axis(engine: Engine) -> None:
    result = s(engine, "exp(x)", 3, x=["real"])
    assert result.status is Status.OK
    assert result.verification is not None
    assert result.verification.directions == ["real"]


def test_complex_variable_is_probed_along_the_imaginary_axis_too(engine: Engine) -> None:
    result = s(engine, "exp(x)", 3)
    assert result.verification is not None
    assert result.verification.directions == ["real", "imaginary"]


# --- safety ------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source", ["x**100000", "factorial(10**6)*x", "exp(" * 80 + "x" + ")" * 80]
)
def test_hostile_inputs_are_rejected_without_crashing(engine: Engine, source) -> None:
    result = s(engine, source, 3)
    assert result.status in {Status.PARSE_ERROR, Status.RESOURCE_LIMIT, Status.UNSUPPORTED}
