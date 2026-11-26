"""The Lagrangian analysis returns its verified equations of motion even when the search for constants runs out
of time, and the numerical screen that speeds that search can only rule a candidate out, never in.

Regression for the chat of 2026-11-26: three particles on a ring (three angles) timed out at the deployed 10 s and
returned nothing, not even the equations of motion. Measured on a developer machine the analysis took 3.8 s, of
which 2.75 s were the symmetry search (one `simplify` per pair of coordinates); the deployed pod is about three
times slower. With the screen the ring takes about 1.7 s, the six-coordinate triangle about 3.5 s (was 13.4 s).
"""

import time

import pytest
import sympy as sp
from minerva_engine import Engine, Limits, Status
from minerva_engine.clock import TimeUp, stop_at
from minerva_engine.lagrangian import analyze_lagrangian
from minerva_engine.verify import clearly_nonzero

RING = (
    r"\frac{1}{2} R^2 (m_1 \dot{\theta}_1^2 + m_2 \dot{\theta}_2^2 + m_3 \dot{\theta}_3^2) - \frac{k}{2} \left[ "
    r"\left(2R\sin\frac{\theta_2-\theta_1}{2} - a\right)^2 + \left(2R\sin\frac{\theta_3-\theta_2}{2} - a\right)^2 "
    r"+ \left(2R\sin\frac{\theta_1-\theta_3}{2} - a\right)^2 \right]"
)
RING_COORDINATES = ["theta_1", "theta_2", "theta_3"]
TRIANGLE = (
    r"\frac{1}{2} m_1 (\dot{x}_1^2 + \dot{y}_1^2) + \frac{1}{2} m_2 (\dot{x}_2^2 + \dot{y}_2^2) + "
    r"\frac{1}{2} m_3 (\dot{x}_3^2 + \dot{y}_3^2) - \frac{k}{2} \left[ "
    r"\left(\sqrt{(x_2-x_1)^2 + (y_2-y_1)^2} - a\right)^2 + \left(\sqrt{(x_3-x_2)^2 + (y_3-y_2)^2} - a\right)^2 "
    r"+ \left(\sqrt{(x_1-x_3)^2 + (y_1-y_3)^2} - a\right)^2 \right]"
)
TRIANGLE_COORDINATES = ["x_1", "y_1", "x_2", "y_2", "x_3", "y_3"]


def kinds(report) -> list[str]:  # type: ignore[no-untyped-def]
    return sorted(c.kind for c in report.conserved if c.verified)


# --- the screen only rules out ------------------------------------------------------------------------------

x, y, z = sp.symbols("x y z")


@pytest.mark.parametrize(
    "zero",
    [
        sp.Integer(0),
        10**30 * sp.sin(x) ** 2 + 10**30 * sp.cos(x) ** 2 - 10**30,  # huge terms cancelling exactly
        sp.Rational(1, 10**12) * (x * y - y * x),
        sp.Rational(1, 10**12) * (sp.sin(2 * x) - 2 * sp.sin(x) * sp.cos(x)),
        sp.sqrt(x**2 + y**2) ** 2 - x**2 - y**2,
        sp.exp(x + y) - sp.exp(x) * sp.exp(y),
    ],
)
def test_an_expression_that_is_zero_is_never_called_clearly_nonzero(zero: sp.Expr) -> None:
    assert not clearly_nonzero(zero)


@pytest.mark.parametrize(
    "nonzero",
    [
        x,
        sp.sin(x) ** 2
        + sp.cos(x) ** 2
        - 1
        + sp.Rational(1, 10**6) * x,  # broken at 1e-6 relative size
    ],
)
def test_a_clearly_nonzero_expression_is_ruled_out(nonzero: sp.Expr) -> None:
    assert clearly_nonzero(nonzero)


@pytest.mark.parametrize(
    "small",
    [
        sp.sin(x) ** 2 + sp.cos(x) ** 2 - 1 + sp.Rational(1, 10**12) * x,
        10**30 * sp.sin(x) ** 2 + 10**30 * sp.cos(x) ** 2 - 10**30 + x,
    ],
)
def test_a_difference_far_below_the_terms_is_left_to_the_symbolic_test(small: sp.Expr) -> None:
    # Nonzero, but far below the size of the terms: not "clearly" nonzero, so the caller runs simplify as before.
    assert not clearly_nonzero(small)


def test_a_rotation_symmetry_with_a_tiny_coefficient_is_still_found() -> None:
    lag = r"\frac12 m(\dot{x}^2 + \dot{y}^2) - 10^{-12} (x^2+y^2)^3 - \frac{k}{\sqrt{x^2+y^2}}"
    report = analyze_lagrangian(lag, ["x", "y"], "t", {}, Limits(timeout_seconds=60))
    assert kinds(report) == ["energy", "rotation"]


def test_a_rotation_broken_at_one_part_in_a_million_is_not_reported() -> None:
    lag = r"\frac12 m(\dot{x}^2 + \dot{y}^2) - k (x^2+y^2) - 10^{-6} k x^4"
    report = analyze_lagrangian(lag, ["x", "y"], "t", {}, Limits(timeout_seconds=60))
    assert kinds(report) == ["energy"]
    assert any("rotations of each pair" in t for t in report.tested)


# --- partial results ----------------------------------------------------------------------------------------


def test_the_ring_gives_equations_and_energy_and_rules_out_rotations() -> None:
    report = analyze_lagrangian(RING, RING_COORDINATES, "t", {}, Limits(timeout_seconds=60))
    assert report.status is Status.OK
    assert len(report.equations) == 3 and all(e.verification for e in report.equations)
    assert kinds(report) == ["energy"]
    assert report.not_searched == []
    assert any("rotations of each pair" in t for t in report.tested)


def test_a_tight_limit_keeps_the_equations_and_names_what_was_not_searched() -> None:
    report = analyze_lagrangian(TRIANGLE, TRIANGLE_COORDINATES, "t", {}, Limits(timeout_seconds=1))
    assert report.status is Status.OK
    assert len(report.equations) == 6 and len(report.accelerations) == 6
    assert report.not_searched, report.message
    assert "not searched" in report.message
    assert not any("none leaves L unchanged" in t for t in report.tested if "first" not in t)
    every_pair = "rotations of each pair of coordinates (as Cartesian): none leaves L unchanged"
    assert every_pair not in report.tested


def test_stop_at_interrupts_the_block_and_a_past_deadline_never_enters_it() -> None:
    started = time.monotonic()
    with pytest.raises(TimeUp), stop_at(time.monotonic() + 0.05):
        while True:
            sp.factor(sp.expand((x + y + z) ** 12))
    assert time.monotonic() - started < 5
    entered = False
    with pytest.raises(TimeUp), stop_at(time.monotonic() - 1):
        entered = True
    assert not entered


@pytest.mark.extended
@pytest.mark.parametrize(
    ("lagrangian", "coordinates"),
    [(RING, RING_COORDINATES), (TRIANGLE, TRIANGLE_COORDINATES)],
    ids=["ring", "triangle"],
)
def test_three_particles_fit_the_production_limit(lagrangian: str, coordinates: list[str]) -> None:
    """Wall-clock dependent, so not part of the pull-request tier. On the deployed pods (about three times
    slower) the triangle is expected to return its equations with part of the search listed as not searched."""
    live = Engine(Limits(timeout_seconds=10, max_concurrency=1))
    report = live.analyze_lagrangian(lagrangian, coordinates, "t", {})
    assert report.status is Status.OK, report.message
    assert report.not_searched == [], report.message
