"""Total time derivative along a motion: the chain rule over coordinates, velocities and time, checked
against a central difference of the input along random polynomial motions, and (with a Lagrangian)
evaluated on the verified equations of motion."""

from __future__ import annotations

import pytest
import sympy as sp
from minerva_engine import Engine, ExpressionInput, Status
from minerva_engine.models import Limits
from minerva_engine.time_derivative import time_derivative

LIMITS = Limits(timeout_seconds=60)
KEPLER = r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}"
ASSUME = {"mu": ["positive"], "k": ["positive"], "r": ["positive"], "theta": ["real"]}


def run(source: str, coords: list[str], *, lagrangian: str | None = None, **kw: object):  # type: ignore[no-untyped-def]
    data = ExpressionInput(input=source, assumptions=kw.pop("assumptions", ASSUME))  # type: ignore[arg-type]
    return time_derivative(data, coords, "t", lagrangian, LIMITS)


def parsed(text: str | None) -> sp.Expr:
    assert text is not None
    return sp.sympify(text)


def test_angular_momentum_total_derivative_is_the_product_rule() -> None:
    r = run(r"\mu r^2 \dot{\theta}", ["r", "theta"])
    assert r.status == Status.OK
    mu, rr, thd, rd, thdd = sp.symbols("mu r theta_dot r_dot theta_ddot")
    expected = mu * (2 * rr * rd * thd + rr**2 * thdd)  # hand-written: d/dt(mu r^2 thdot)
    assert sp.simplify(parsed(r.result_text) - expected) == 0
    assert r.verification is not None and r.verification.passed
    assert r.verification.points_agreed == r.verification.points_checked >= 6


def test_explicit_time_dependence_is_included() -> None:
    r = run(r"t x + \dot{x}^2", ["x"], assumptions={})
    assert r.status == Status.OK
    x, xd, xdd, t = sp.symbols("x x_dot x_ddot t")
    assert sp.simplify(parsed(r.result_text) - (x + t * xd + 2 * xd * xdd)) == 0


def test_kepler_constants_are_zero_on_the_motion_and_a_non_constant_is_not() -> None:
    angular = run(r"\mu r^2 \dot{\theta}", ["r", "theta"], lagrangian=KEPLER)
    assert angular.status == Status.OK and angular.zero_on_motion is True
    assert angular.on_motion_latex == "0"
    energy = run(
        r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) - \frac{k}{r}",
        ["r", "theta"],
        lagrangian=KEPLER,
    )
    assert energy.zero_on_motion is True
    radial_momentum_like = run(r"r \dot{r}", ["r", "theta"], lagrangian=KEPLER)
    assert radial_momentum_like.status == Status.OK
    assert radial_momentum_like.zero_on_motion is False
    assert radial_momentum_like.on_motion_latex not in (None, "0")


def test_a_wrong_candidate_cannot_pass_the_check() -> None:
    from minerva_engine.time_derivative import check_total_derivative

    x, xd, xdd = sp.symbols("x x_dot x_ddot")
    f = x * xd
    exact = xd**2 + x * xdd
    assert check_total_derivative(f, exact, [x], [xd], [xdd], sp.Symbol("t")).passed
    for wrong in (exact * (1 + sp.Rational(1, 10**6)), exact + sp.Rational(1, 10**9), xd**2):
        assert not check_total_derivative(f, wrong, [x], [xd], [xdd], sp.Symbol("t")).passed


@pytest.mark.parametrize(
    ("source", "coords", "needle"),
    [
        (r"\ddot{x} + x", ["x"], "acceleration"),  # needs the third derivative
        (r"\dot{y} x", ["x"], "y"),  # a velocity of something that is not a coordinate
        (r"m g", ["x"], "not a function of"),  # nothing that changes along a motion
    ],
)
def test_unsupported_inputs_are_refused_with_a_reason(
    source: str, coords: list[str], needle: str
) -> None:
    r = run(source, coords, assumptions={})
    assert r.status == Status.UNSUPPORTED
    assert needle in r.message


def test_lagrangian_with_an_equation_is_refused_not_misread() -> None:
    r = run(r"r \dot{r}", ["r", "theta"], lagrangian=r"L = " + KEPLER)
    assert r.status == Status.PARSE_ERROR and "after the last" in r.message


def test_engine_wrapper_runs_it_in_the_sandbox(engine: Engine) -> None:
    r = engine.time_derivative(
        ExpressionInput(input=r"\mu r^2 \dot{\theta}", assumptions=ASSUME), ["r", "theta"]
    )
    assert r.status == Status.OK and r.verification is not None and r.verification.passed


# -- constants of motion named in the expression (issue #132) ---------------------------------------------
# What `find_conserved_quantities` printed for the polar Kepler Lagrangian (names C_1..C_4) and the relation
# it listed among them. The relation is an identity in the coordinates and velocities, so its derivative is 0.
POLAR = r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"
CONSTANTS = {
    "C_1": r"\mu r^{2} \dot{\theta}",
    "C_2": r"- \frac{G M \mu}{r} + \frac{\mu r^{2} \dot{\theta}^{2}}{2} + \frac{\mu \dot{r}^{2}}{2}",
    "C_3": r"- G M \cos{\theta} + r^{3} \dot{\theta}^{2} \cos{\theta} + r^{2} \dot{r} \dot{\theta} \sin{\theta}",
    "C_4": r"G M \sin{\theta} - r^{3} \dot{\theta}^{2} \sin{\theta} + r^{2} \dot{r} \dot{\theta} \cos{\theta}",
}
RELATION = r"2 C_{1}^{2} C_{2} - C_{3}^{2} \mu^{3} - C_{4}^{2} \mu^{3} + G^{2} M^{2} \mu^{3}"


def with_constants(source: str, defs: dict[str, str], *, lagrangian: str | None = POLAR):  # type: ignore[no-untyped-def]
    return time_derivative(
        ExpressionInput(input=source), ["r", "theta"], "t", lagrangian, LIMITS, defs
    )


def test_a_relation_among_verified_constants_is_zero_on_the_motion() -> None:
    r = with_constants(RELATION, CONSTANTS)
    assert r.status == Status.OK
    assert r.zero_on_motion is True and r.symbolic_zero_on_motion is True
    assert r.verification is not None and r.verification.passed
    assert len(r.definitions_used) == 4 and r.definitions_used[0].startswith("C_{1} = ")
    # the input is shown as written, the check was of the expression with the definitions inserted
    assert "C_{1}" in (r.input_latex or "") and any("Inserted, as given" in n for n in r.notes)


def test_a_quantity_with_one_constant_and_one_variable_is_zero_only_for_the_constant_part() -> None:
    assert with_constants(r"C_1^2 + C_2", CONSTANTS).zero_on_motion is True
    moving = with_constants(r"C_1 \dot{r}", CONSTANTS)
    assert moving.status == Status.OK and moving.zero_on_motion is False


def test_a_definition_wrong_by_1e_6_is_not_zero_on_the_motion() -> None:
    wrong = {**CONSTANTS, "C_2": CONSTANTS["C_2"] + r" + \frac{r}{1000000}"}
    r = with_constants(RELATION, wrong)
    assert r.status == Status.OK and r.zero_on_motion is False


def test_an_expression_that_becomes_a_constant_is_zero_and_says_so() -> None:
    r = with_constants(r"C_1 - \mu r^2 \dot{\theta}", CONSTANTS, lagrangian=None)
    assert r.status == Status.OK and r.result_latex == "0"
    assert any("constant along every motion" in n for n in r.notes)


def test_bare_names_without_a_definition_are_refused_with_a_typed_code() -> None:
    r = with_constants(RELATION, {})
    assert r.status == Status.UNSUPPORTED
    assert r.diagnostic is not None and r.diagnostic.code.value == "no_dynamic_dependence"
    assert {"C_1", "C_2", "C_3", "C_4"} <= set(r.diagnostic.symbols)


@pytest.mark.parametrize(
    "bad",
    [
        r"\ddot{r} r",  # an acceleration
        r"\dot{q} r",  # the velocity of something that is not a coordinate
        r"C_2 r",  # another defined name
    ],
)
def test_a_definition_that_hides_a_second_derivative_or_a_loop_is_refused(bad: str) -> None:
    r = with_constants(r"C_1 + C_2", {"C_1": bad, "C_2": r"r"})
    assert r.status == Status.UNSUPPORTED
    assert r.diagnostic is not None and r.diagnostic.code.value == "unsupported_definition"


def test_the_zero_test_is_not_fooled_or_loosened_by_the_extra_precision() -> None:
    from minerva_engine.lagrangian import verify_zero_derivative

    x = sp.Symbol("x")
    ok, _ = verify_zero_derivative(x**2 - x**2 + sp.Rational(1, 10**6) * x, digits=80)
    assert not ok  # a 1e-6 term is not zero at any precision
    ok, ver = verify_zero_derivative(
        sp.expand((x + 1) ** 12) - sum(sp.binomial(12, k) * x**k for k in range(13)), digits=80
    )
    assert ok and ver.numeric_points_checked >= 4
