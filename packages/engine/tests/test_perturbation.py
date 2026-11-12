"""Regression tests for the Poincare-Lindstedt perturbation solution (sandboxed engine)."""

import copy

import pytest
import sympy as sp
from minerva_engine import CompareRequest, Engine, Status
from minerva_engine.perturbation import _check_residual, _solve_orders

DUFFING = r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4"
PENDULUM = r"\frac12 m l^2 \dot{\theta}^2 + m g l \cos\theta"
POSITIVE = {name: ["positive"] for name in ("m", "g", "l", "k", "b")}


def solve(
    engine: Engine,
    latex,
    coordinates,
    small=None,
    order=2,
    equilibrium="0",
    assumptions=None,
    compare=None,
):
    return engine.perturbation_solve(
        latex,
        coordinates,
        "t",
        POSITIVE if assumptions is None else assumptions,
        small,
        equilibrium,
        order,
        compare,
    )


# --- positive: known results ---------------------------------------------------------------------
def test_duffing_frequency_to_second_order(engine: Engine) -> None:
    report = solve(engine, DUFFING, ["x"], small="b", order=2)
    assert report.status is Status.OK, report.message
    # omega = 1 + 3 b A^2 / 8 - 21 b^2 A^4 / 256
    assert report.frequency_latex == (
        r"\omega = 1 + \frac{3 b A^{2}}{8} - \frac{21 A^{4} b^{2}}{256}"
    )
    assert report.period_ratio_latex is not None
    assert r"1 - \frac{3 A^{2} b}{8} + \frac{57 A^{4} b^{2}}{256}" in report.period_ratio_latex
    assert report.mode == "parameter" and report.amplitude_latex == "A"
    verification = report.verification
    assert verification is not None and verification.passed
    assert verification.residual_passed and verification.initial_conditions_passed
    assert verification.bounded and verification.parameter_sets_checked >= 4


def test_duffing_first_order_shape_has_the_third_harmonic(engine: Engine) -> None:
    report = solve(engine, DUFFING, ["x"], small="b", order=1)
    assert report.status is Status.OK
    assert report.orders[1].shape_latex == (
        r"w_{1}(\tau) = - \frac{A^{3} \cos{\left(\tau \right)}}{32} "
        r"+ \frac{A^{3} \cos{\left(3 \tau \right)}}{32}"
    )
    assert report.orders[1].frequency_latex == r"\omega_{1} = \frac{3 A^{2}}{8}"


def test_pendulum_period_series_about_the_bottom(engine: Engine) -> None:
    report = solve(engine, PENDULUM, ["theta"], order=4)
    assert report.status is Status.OK, report.message
    assert report.mode == "amplitude"
    assert report.period_ratio_latex == (
        r"\frac{T}{T_0} = 1 + \frac{\epsilon^{2}}{16} + \frac{11 \epsilon^{4}}{3072} "
        r"+ O\left(\epsilon^{5}\right)"
    )
    assert report.linear_frequency_latex == r"\omega_0 = \frac{\sqrt{g}}{\sqrt{l}}"


def test_pendulum_sixth_order_matches_the_classical_series(engine: Engine) -> None:
    report = solve(engine, PENDULUM, ["theta"], order=6)
    assert report.status is Status.OK, report.message
    assert report.period_ratio_latex is not None
    assert r"\frac{173 \epsilon^{6}}{737280}" in report.period_ratio_latex


def test_harmonic_oscillator_has_no_corrections(engine: Engine) -> None:
    report = solve(engine, r"\frac12 m \dot{x}^2 - \frac12 k x^2", ["x"], order=3)
    assert report.status is Status.OK
    assert report.frequency_latex == r"\omega = \frac{\sqrt{k}}{\sqrt{m}}"
    assert [o.shape_latex for o in report.orders[1:]] == [
        rf"w_{{{k}}}(\tau) = 0" for k in (1, 2, 3)
    ]


def test_an_equilibrium_that_the_perturbation_moves_is_refused(engine: Engine) -> None:
    latex = r"\frac12 m \dot{x}^2 - \frac12 k x^2 + m g x - \frac14 b x^4"
    # x0 = m g / k is the equilibrium of the unperturbed oscillator only; the quartic term moves it.
    report = solve(engine, latex, ["x"], small="b", equilibrium="m*g/k")
    assert report.status is Status.UNSUPPORTED  # not an equilibrium for b != 0


# --- the numerical comparison ------------------------------------------------------------------
def test_duffing_error_against_the_numerical_solution_falls_at_the_predicted_order(
    engine: Engine,
) -> None:
    compare = CompareRequest(parameters={}, epsilon=0.1, amplitude=1.0, t_end=30.0)
    report = solve(engine, DUFFING, ["x"], small="b", order=2, compare=compare)
    assert report.status is Status.OK, report.message
    c = report.comparison
    assert c is not None and c.status is Status.OK
    assert len(c.t) == len(c.numerical) == len(c.perturbative) > 20
    assert c.max_error_half is not None and c.max_error is not None
    assert c.max_error_half < c.max_error
    assert c.observed_order is not None and 2.4 < c.observed_order < 3.6  # predicted 3
    assert any("not a claim for all times" in n for n in c.notes)


def test_pendulum_error_against_the_numerical_solution(engine: Engine) -> None:
    compare = CompareRequest(parameters={"m": 1.0, "g": 9.81, "l": 1.0}, epsilon=0.5, t_end=20.0)
    report = solve(engine, PENDULUM, ["theta"], order=4, compare=compare)
    c = report.comparison
    assert c is not None and c.status is Status.OK
    assert c.observed_order is not None and c.observed_order > 4.5  # predicted 5
    assert c.max_error is not None and c.max_error < 1e-3
    assert c.scaling.endswith("divided by eps")


def test_comparison_without_numbers_it_needs_shows_nothing(engine: Engine) -> None:
    no_amplitude = solve(
        engine,
        DUFFING,
        ["x"],
        small="b",
        compare=CompareRequest(parameters={}, epsilon=0.1, t_end=5.0),
    )
    assert no_amplitude.status is Status.OK  # the solution itself does not depend on it
    assert no_amplitude.comparison is not None
    assert no_amplitude.comparison.status is Status.PARSE_ERROR
    assert no_amplitude.comparison.numerical == []

    missing = solve(
        engine,
        PENDULUM,
        ["theta"],
        compare=CompareRequest(parameters={"m": 1.0, "g": 9.81}, epsilon=0.3, t_end=5.0),
    )
    assert missing.comparison is not None and missing.comparison.status is not Status.OK
    assert "l" in missing.comparison.message and missing.comparison.numerical == []


def test_a_numerical_run_that_fails_its_own_checks_gives_no_comparison(engine: Engine) -> None:
    compare = CompareRequest(parameters={"m": 1.0, "g": 9.81, "l": 1.0}, epsilon=0.3, t_end=1.0e7)
    report = solve(engine, PENDULUM, ["theta"], order=2, compare=compare)
    assert report.status is Status.OK  # the solution stands on its own checks
    c = report.comparison
    assert c is not None and c.status is not Status.OK
    assert c.numerical == [] and c.perturbative == [] and c.max_error is None


# --- the checks reject wrong solutions ------------------------------------------------------------
def _duffing_solution():
    w, a_amp = sp.Symbol("w"), sp.Symbol("A", positive=True)
    b = sp.Symbol("b", positive=True)
    solved = _solve_orders([-w, -(w**3), sp.Integer(0)], w, sp.Integer(1), sp.Integer(1), a_amp, b)
    q = sp.Symbol("q")
    acceleration = -q - b * q**3
    return solved, acceleration, q, b


def test_the_residual_check_accepts_the_right_solution() -> None:
    solved, acceleration, q, b = _duffing_solution()
    passed, sets = _check_residual(acceleration, q, sp.Integer(0), solved, b, "parameter", 2)
    assert passed and sets >= 4


def test_a_wrong_frequency_correction_is_rejected() -> None:
    solved, acceleration, q, b = _duffing_solution()
    wrong = copy.copy(solved)
    wrong.omegas = [solved.omegas[0], solved.omegas[1] * sp.Rational(101, 100), solved.omegas[2]]
    assert not _check_residual(acceleration, q, sp.Integer(0), wrong, b, "parameter", 2)[0]


def test_a_wrong_harmonic_coefficient_is_rejected() -> None:
    solved, acceleration, q, b = _duffing_solution()
    wrong = copy.copy(solved)
    wrong.w = [dict(h) for h in solved.w]
    wrong.w[1][3] = wrong.w[1][3] * sp.Rational(11, 10)
    assert not _check_residual(acceleration, q, sp.Integer(0), wrong, b, "parameter", 2)[0]


def test_the_unstrained_expansion_with_its_secular_term_is_rejected() -> None:
    # Keeping omega = omega0 (no strained time) cannot cancel the cos(tau) resonance.
    solved, acceleration, q, b = _duffing_solution()
    wrong = copy.copy(solved)
    wrong.omegas = [solved.omegas[0], sp.Integer(0), sp.Integer(0)]
    assert not _check_residual(acceleration, q, sp.Integer(0), wrong, b, "parameter", 2)[0]


def test_a_solution_of_the_wrong_equation_is_rejected() -> None:
    solved, _, q, b = _duffing_solution()
    other = -q - 2 * b * q**3  # the checker uses the acceleration it is given, not the solver's
    assert not _check_residual(other, q, sp.Integer(0), solved, b, "parameter", 2)[0]


# --- honesty -----------------------------------------------------------------------------------
def test_two_coordinates_are_unsupported(engine: Engine) -> None:
    latex = r"\frac12 (\dot{x}^2+\dot{y}^2) - \frac12 (x^2+y^2)"
    report = solve(engine, latex, ["x", "y"])
    assert report.status is Status.UNSUPPORTED and "one degree of freedom" in report.message
    assert report.frequency_latex is None


def test_velocity_dependent_acceleration_is_unsupported(engine: Engine) -> None:
    report = solve(engine, r"\frac12 (1+x^2)\dot{x}^2 - \frac12 x^2", ["x"], assumptions={})
    assert report.status is Status.UNSUPPORTED and "velocity" in report.message


def test_non_polynomial_perturbation_in_parameter_mode_is_unsupported(engine: Engine) -> None:
    latex = r"\frac12 \dot{x}^2 - \frac12 x^2 - s(1-\cos x)"
    report = solve(engine, latex, ["x"], small="s", assumptions={"s": ["positive"]})
    assert report.status is Status.UNSUPPORTED and "polynomial" in report.message
    assert report.solution_latex is None


def test_amplitude_mode_handles_the_same_non_polynomial_potential(engine: Engine) -> None:
    latex = r"\frac12 \dot{x}^2 - (1-\cos x)"
    report = solve(engine, latex, ["x"], order=2, assumptions={})
    assert report.status is Status.OK, report.message
    assert (
        report.period_ratio_latex is not None
        and r"\frac{\epsilon^{2}}{16}" in report.period_ratio_latex
    )


def test_unperturbed_system_must_be_a_linear_oscillator(engine: Engine) -> None:
    latex = r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac13 a x^3 - \frac14 b x^4"
    report = solve(
        engine, latex, ["x"], small="b", assumptions={"a": ["positive"], "b": ["positive"]}
    )
    assert report.status is Status.UNSUPPORTED and "linear oscillator" in report.message


def test_the_top_of_a_potential_is_not_an_oscillation(engine: Engine) -> None:
    report = solve(engine, PENDULUM, ["theta"], equilibrium="pi")
    assert report.status is Status.UNSUPPORTED and "positive" in report.message


def test_the_sign_of_the_frequency_must_follow_from_the_assumptions(engine: Engine) -> None:
    report = solve(engine, PENDULUM, ["theta"], assumptions={})
    assert report.status is Status.UNSUPPORTED and "declared assumptions" in report.message


def test_a_point_that_is_not_an_equilibrium_is_refused(engine: Engine) -> None:
    report = solve(engine, PENDULUM, ["theta"], equilibrium="1")
    assert report.status is Status.UNSUPPORTED and "not an equilibrium" in report.message


def test_the_small_parameter_must_be_a_parameter_of_the_lagrangian(engine: Engine) -> None:
    report = solve(engine, DUFFING, ["x"], small="zeta")
    assert report.status is Status.PARSE_ERROR and "zeta" in report.message


def test_time_dependent_lagrangian_is_unsupported(engine: Engine) -> None:
    report = solve(engine, r"\frac12 \dot{x}^2 - \frac12 t x^2", ["x"], assumptions={})
    assert report.status is Status.UNSUPPORTED


@pytest.mark.parametrize("order", [0, 7, -1])
def test_order_out_of_range_is_unsupported(engine: Engine, order: int) -> None:
    assert solve(engine, DUFFING, ["x"], small="b", order=order).status is Status.UNSUPPORTED


def test_hostile_input_is_rejected_without_crashing(engine: Engine) -> None:
    for latex in (r"\nonsense", r"\frac12 \dot{x}^2 + x^{100000}"):
        report = solve(engine, latex, ["x"], assumptions={})
        assert report.status in {Status.PARSE_ERROR, Status.RESOURCE_LIMIT, Status.UNSUPPORTED}
