"""Numerical solution of Lagrangian systems: closed forms, invariants, and honest failures."""

import math

import pytest
from minerva_engine import Engine, Limits, MotionReport, MotionRequest, Status, Sweep
from minerva_engine.motion import integrate_motion, parse_request

OSC = r"\frac12 m\dot{x}^2 - \frac12 k x^2"
KEPLER = r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}"
PENDULUM = (
    r"\frac12 (m_1+m_2) l_1^2\dot{\theta}_1^2 + \frac12 m_2 l_2^2\dot{\theta}_2^2 "
    r"+ m_2 l_1 l_2 \dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2) "
    r"+ (m_1+m_2) g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2"
)
LIMITS = Limits(timeout_seconds=20)


def solve(**kw) -> MotionReport:  # type: ignore[no-untyped-def]
    return integrate_motion(MotionRequest(**kw), LIMITS)


def oscillator(**kw) -> MotionReport:  # type: ignore[no-untyped-def]
    base = {
        "lagrangian": OSC,
        "coordinates": ["x"],
        "parameters": {"m": 1.0, "k": 4.0},
        "initial_positions": [1.0],
        "initial_velocities": [0.0],
        "t_end": 10.0,
    }
    return solve(**{**base, **kw})


def test_harmonic_oscillator_matches_the_closed_form() -> None:
    r = oscillator()
    assert r.status is Status.OK and len(r.runs) == 1
    run = r.runs[0]
    assert run.status is Status.OK and run.verification.passed
    error = max(abs(x - math.cos(2 * t)) for x, t in zip(run.q[0], r.t, strict=True))
    assert error < 1e-7
    velocity = max(abs(v + 2 * math.sin(2 * t)) for v, t in zip(run.v[0], r.t, strict=True))
    assert velocity < 1e-7
    assert r.t[0] == 0.0 and r.t[-1] == 10.0 and len(r.t) == len(run.q[0])


def test_energy_is_followed_and_barely_drifts() -> None:
    run = oscillator().runs[0]
    assert [c.kind for c in run.conserved] == ["energy"]
    assert run.verification.drift_checked == 1
    assert run.verification.max_drift is not None and run.verification.max_drift < 1e-7
    assert run.verification.method_difference is not None
    assert run.verification.method_difference < 1e-6
    assert run.verification.reversal_error is not None and run.verification.reversal_error < 1e-6


def test_free_particle_is_a_straight_line() -> None:
    r = solve(
        lagrangian=r"\frac12 m\dot{x}^2",
        coordinates=["x"],
        parameters={"m": 2.0},
        initial_positions=[1.0],
        initial_velocities=[3.0],
        t_end=5.0,
    )
    run = r.runs[0]
    assert run.status is Status.OK
    assert max(abs(x - (1 + 3 * t)) for x, t in zip(run.q[0], r.t, strict=True)) < 1e-9
    assert {c.kind for c in run.conserved} == {"cyclic", "energy"}


def test_circular_kepler_orbit_keeps_its_radius_and_closes_after_one_period() -> None:
    # r = 1, k = m = 1: v_theta = sqrt(k/(m r)) = 1, period 2 pi
    r = solve(
        lagrangian=KEPLER,
        coordinates=["r", "theta"],
        assumptions={"r": ["positive"], "m": ["positive"], "k": ["positive"]},
        parameters={"m": 1.0, "k": 1.0},
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 1.0],
        t_end=2 * math.pi,
    )
    run = r.runs[0]
    assert run.status is Status.OK
    assert max(abs(x - 1.0) for x in run.q[0]) < 1e-8
    assert abs(run.q[1][-1] - 2 * math.pi) < 1e-8
    assert {c.kind for c in run.conserved} == {"cyclic", "energy"}


def test_eccentric_orbit_conserves_angular_momentum_and_energy() -> None:
    r = solve(
        lagrangian=KEPLER,
        coordinates=["r", "theta"],
        parameters={"m": 1.0, "k": 1.0},
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 0.8],
        t_end=6.0,
    )
    run = r.runs[0]
    assert run.status is Status.OK
    assert run.verification.max_drift is not None and run.verification.max_drift < 1e-6
    assert min(run.q[0]) < 1.0 and run.verification.drift_checked == 2


def test_eccentric_orbit_sweep_is_verified_not_rejected_for_loose_tolerance() -> None:
    # Regression: at relative tolerance 1e-9 the primary run was off by ~8e-6 here and the cross-check
    # rightly refused it. The tolerance is tight enough now that these orbits verify.
    r = solve(
        lagrangian=KEPLER,
        coordinates=["r", "theta"],
        parameters={"m": 1.0},
        sweep=Sweep(name="k", values=[0.5, 1.0, 1.5, 2.0]),
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 0.8],
        t_end=12.0,
    )
    assert [x.status for x in r.runs] == [Status.OK] * 4
    for run in r.runs:
        assert run.verification.method_difference is not None
        assert run.verification.method_difference < 1e-7


def test_double_pendulum_over_a_short_span_is_verified() -> None:
    r = solve(
        lagrangian=PENDULUM,
        coordinates=["theta_1", "theta_2"],
        parameters={"m_1": 1.0, "m_2": 1.0, "l_1": 1.0, "l_2": 1.0, "g": 9.81},
        initial_positions=[1.0, 0.5],
        initial_velocities=[0.0, 0.0],
        t_end=5.0,
    )
    assert r.status is Status.OK and r.runs[0].verification.passed


def test_driven_oscillator_with_explicit_time_dependence() -> None:
    r = solve(
        lagrangian=r"\frac12 m\dot{x}^2 - \frac12 k x^2 + F x\cos(w t)",
        coordinates=["x"],
        parameters={"m": 1.0, "k": 4.0, "F": 1.0, "w": 1.0},
        initial_positions=[0.0],
        initial_velocities=[0.0],
        t_end=8.0,
    )
    run = r.runs[0]
    assert run.status is Status.OK and run.conserved == []  # energy is not conserved here
    assert any("no verified constant" in n for n in run.verification.notes)
    exact = [(math.cos(t) - math.cos(2 * t)) / 3 for t in r.t]
    assert max(abs(x - e) for x, e in zip(run.q[0], exact, strict=True)) < 1e-7


def test_sweep_gives_one_verified_run_per_value() -> None:
    r = oscillator(parameters={"m": 1.0}, sweep=Sweep(name="k", values=[1.0, 4.0, 9.0]))
    assert r.status is Status.OK and r.sweep_name == "k"
    assert [x.parameter_value for x in r.runs] == [1.0, 4.0, 9.0]
    assert all(x.status is Status.OK for x in r.runs)
    for run, k in zip(r.runs, [1.0, 4.0, 9.0], strict=True):
        error = max(abs(x - math.cos(math.sqrt(k) * t)) for x, t in zip(run.q[0], r.t, strict=True))
        assert error < 1e-6


def test_a_sweep_keeps_good_runs_and_drops_failed_ones() -> None:
    # Radial fall from rest hits r = 0; a nonzero angular velocity avoids the collision.
    r = solve(
        lagrangian=KEPLER,
        coordinates=["r", "theta"],
        parameters={"m": 1.0, "k": 1.0},
        sweep=Sweep(name="m", values=[1.0, 2.0]),
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 0.8],
        t_end=3.0,
    )
    assert r.status is Status.OK
    assert all(x.status is Status.OK for x in r.runs)


def test_sample_count_is_reduced_to_keep_the_result_small() -> None:
    r = solve(
        lagrangian=OSC,
        coordinates=["x"],
        parameters={"m": 1.0},
        sweep=Sweep(name="k", values=[1, 2, 3, 4, 5, 6]),
        initial_positions=[1.0],
        initial_velocities=[0.0],
        t_end=3.0,
        samples=400,
    )
    total = sum(len(s.values) for run in r.runs for s in run.conserved) + sum(
        len(row) for run in r.runs for row in (*run.q, *run.v)
    )
    assert r.status is Status.OK and total <= 20_000


# --- honesty -------------------------------------------------------------------------------------
def test_missing_parameter_is_named() -> None:
    r = oscillator(parameters={"m": 1.0})
    assert r.status is Status.PARSE_ERROR and "k" in r.message and r.runs == []


def test_unknown_parameter_and_unknown_sweep_name() -> None:
    assert oscillator(parameters={"m": 1.0, "k": 1.0, "zz": 3.0}).status is Status.PARSE_ERROR
    r = oscillator(sweep=Sweep(name="zz", values=[1.0, 2.0]))
    assert r.status is Status.PARSE_ERROR and "zz" in r.message


def test_values_must_respect_declared_assumptions() -> None:
    r = oscillator(assumptions={"m": ["positive"]}, parameters={"m": -1.0, "k": 4.0})
    assert r.status is Status.PARSE_ERROR and "positive" in r.message
    r = oscillator(
        assumptions={"k": ["positive"]},
        parameters={"m": 1.0},
        sweep=Sweep(name="k", values=[1.0, -2.0]),
    )
    assert r.status is Status.PARSE_ERROR
    r = oscillator(assumptions={"x": ["positive"]}, initial_positions=[-1.0])
    assert r.status is Status.PARSE_ERROR and "initial" in r.message


def test_time_span_must_be_positive() -> None:
    assert oscillator(t_end=0.0).status is Status.PARSE_ERROR
    assert oscillator(t_start=5.0, t_end=1.0).status is Status.PARSE_ERROR


def test_singular_mass_matrix_is_unsupported() -> None:
    r = solve(
        lagrangian=r"\frac12(\dot{x}+\dot{y})^2 - x^2",
        coordinates=["x", "y"],
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 0.0],
        t_end=1.0,
    )
    assert r.status is Status.UNSUPPORTED and r.runs == []


def test_chaotic_double_pendulum_over_a_long_span_is_not_shown() -> None:
    r = solve(
        lagrangian=PENDULUM,
        coordinates=["theta_1", "theta_2"],
        parameters={"m_1": 1.0, "m_2": 1.0, "l_1": 1.0, "l_2": 1.0, "g": 9.81},
        initial_positions=[2.5, 2.0],
        initial_velocities=[0.0, 0.0],
        t_end=60.0,
    )
    assert r.status is Status.NOT_VERIFIED
    run = r.runs[0]
    assert run.status is Status.NOT_VERIFIED and run.q == [] and run.v == [] and run.conserved == []
    assert not run.verification.passed and "shorter" in run.message


def test_collision_with_the_centre_is_stopped_without_data() -> None:
    r = solve(
        lagrangian=KEPLER,
        coordinates=["r", "theta"],
        parameters={"m": 1.0, "k": 1.0},
        initial_positions=[1.0, 0.0],
        initial_velocities=[0.0, 0.0],
        t_end=3.0,
    )
    assert r.status is Status.NOT_VERIFIED and r.runs[0].q == []


def test_extremely_fast_motion_hits_the_resource_limit() -> None:
    r = oscillator(parameters={"m": 1.0, "k": 1e14}, t_end=100.0)
    assert r.status is Status.RESOURCE_LIMIT and r.runs[0].q == []


def test_bad_shapes_become_parse_errors_not_exceptions() -> None:
    bad = parse_request(
        {
            "lagrangian": OSC,
            "coordinates": ["x"],
            "initial_positions": [1.0, 2.0],
            "initial_velocities": [0.0],
            "t_end": 1.0,
        }
    )
    assert isinstance(bad, MotionReport) and bad.status is Status.PARSE_ERROR
    nan = parse_request(
        {
            "lagrangian": OSC,
            "coordinates": ["x"],
            "initial_positions": [float("nan")],
            "initial_velocities": [0.0],
            "t_end": 1.0,
        }
    )
    assert isinstance(nan, MotionReport) and nan.status is Status.PARSE_ERROR


def test_unparsable_lagrangian_is_a_parse_error() -> None:
    r = oscillator(lagrangian=r"\foo")
    assert r.status is Status.PARSE_ERROR


def test_engine_runs_the_integration_in_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=60))
    r = engine.integrate_motion(
        MotionRequest(
            lagrangian=OSC,
            coordinates=["x"],
            parameters={"m": 1.0, "k": 4.0},
            initial_positions=[1.0],
            initial_velocities=[0.0],
            t_end=3.0,
        )
    )
    assert r.status is Status.OK and r.duration_ms >= 0 and r.runs[0].verification.passed


@pytest.mark.parametrize("bad", [{"lagrangian": ""}, {"samples": 5}])
def test_request_model_bounds(bad: dict[str, object]) -> None:
    base: dict[str, object] = {
        "lagrangian": OSC,
        "coordinates": ["x"],
        "initial_positions": [1.0],
        "initial_velocities": [0.0],
        "t_end": 1.0,
    }
    assert isinstance(parse_request({**base, **bad}), MotionReport)
