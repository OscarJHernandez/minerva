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
        (r"m g", ["x"], "does not depend"),  # nothing that changes along a motion
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
