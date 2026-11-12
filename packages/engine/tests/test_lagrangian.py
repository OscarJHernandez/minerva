"""Lagrangian analysis: equations of motion and verified constants of motion on classic systems."""

import pytest
from minerva_engine import Engine, Limits, Status
from minerva_engine.lagrangian import analyze_lagrangian

L = Limits()


def analyze(lag: str, coords: list[str]):  # type: ignore[no-untyped-def]
    return analyze_lagrangian(lag, coords, "t", {}, L)


def kinds(report) -> list[str]:  # type: ignore[no-untyped-def]
    return [c.kind for c in report.conserved if c.verified]


def test_free_particle_momentum_and_energy() -> None:
    r = analyze(r"\frac12 m\dot{x}^2", ["x"])
    assert r.equations[0].latex == r"m \ddot{x} = 0"
    assert r.accelerations[0].latex == r"\ddot{x} = 0"
    assert kinds(r) == ["cyclic", "energy"]
    assert "lagrangian:cyclic" in r.features and "lagrangian:time-independent" in r.features


def test_harmonic_oscillator_has_energy_only() -> None:
    r = analyze(r"\frac12 m\dot{x}^2 - \frac12 k x^2", ["x"])
    assert r.accelerations[0].latex == r"\ddot{x} = - \frac{k x}{m}"
    assert kinds(r) == ["energy"]
    assert any("cyclic" in t for t in r.tested)
    assert r.conserved[0].verification and r.conserved[0].verification.passed


def test_kepler_in_polar_coordinates_finds_angular_momentum_and_energy() -> None:
    r = analyze(r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}", ["r", "theta"])
    assert kinds(r) == ["cyclic", "energy"]
    cyclic = r.conserved[0]
    assert cyclic.expression_latex == r"p_{\theta} = m r^{2} \dot{\theta}"
    assert r.accelerations[1].latex == r"\ddot{\theta} = - \frac{2 \dot{r} \dot{\theta}}{r}"
    assert any("rotations" in t for t in r.tested)  # polar coordinates are not Cartesian


def test_kepler_in_cartesian_coordinates_finds_angular_momentum_by_rotation() -> None:
    r = analyze(r"\frac12 m(\dot{x}^2 + \dot{y}^2) + \frac{k}{\sqrt{x^2+y^2}}", ["x", "y"])
    assert sorted(kinds(r)) == [
        "energy",
        "quadratic",
        "quadratic",
        "rotation",
    ]  # the Runge-Lenz vector
    rotation = next(c for c in r.conserved if c.kind == "rotation")
    assert rotation.expression_latex == r"Q = m \left(x \dot{y} - \dot{x} y\right)"
    assert rotation.verification and rotation.verification.symbolic_zero


def test_explicit_time_dependence_removes_energy() -> None:
    r = analyze(r"\frac12 m\dot{x}^2 - \frac12 k x^2 t", ["x"])
    assert r.time_dependent and kinds(r) == []
    assert any("energy function: not conserved" in t for t in r.tested)
    assert "lagrangian:no-symmetry-found" in r.features


def test_symmetry_breaking_potential_does_not_report_rotation() -> None:
    r = analyze(r"\frac12 m(\dot{x}^2 + \dot{y}^2) + x", ["x", "y"])
    assert "rotation" not in kinds(r)
    assert "cyclic" in kinds(r)  # y is cyclic
    assert "energy" in kinds(r)


def test_double_pendulum_has_energy_and_dotted_subscripts_parse() -> None:
    lag = (
        r"\frac12 (m_1+m_2) l_1^2\dot{\theta}_1^2 + \frac12 m_2 l_2^2\dot{\theta}_2^2 "
        r"+ m_2 l_1 l_2 \dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2) "
        r"+ (m_1+m_2) g l_1\cos\theta_1 + m_2 g l_2\cos\theta_2"
    )
    r = analyze(lag, ["theta_1", "theta_2"])
    assert r.status is Status.OK and len(r.equations) == 2 and len(r.accelerations) == 2
    assert kinds(r) == ["energy"]


def test_singular_mass_matrix_is_reported_not_guessed() -> None:
    r = analyze(r"\frac12(\dot{x}+\dot{y})^2 - x^2", ["x", "y"])
    assert r.status is Status.OK and r.accelerations == []
    assert any("singular" in n for n in r.notes)
    assert all(not c.verified for c in r.conserved)  # cannot verify without the accelerations


@pytest.mark.parametrize(
    ("lag", "coords", "status"),
    [
        (r"x y", ["x"], Status.UNSUPPORTED),  # no velocities
        (r"\ddot{x}^2", ["x"], Status.UNSUPPORTED),  # higher derivatives
        (r"\foo", ["x"], Status.PARSE_ERROR),
        (r"\frac12\dot{x}^2", [], Status.UNSUPPORTED),
        (r"\frac12\dot{x}^2", ["x", "x"], Status.PARSE_ERROR),
    ],
)
def test_unsupported_inputs_are_explicit(lag: str, coords: list[str], status: Status) -> None:
    assert analyze(lag, coords).status is status


def test_engine_runs_the_analysis_in_the_sandbox() -> None:
    engine = Engine(Limits(timeout_seconds=60))
    r = engine.analyze_lagrangian(r"\frac12 m\dot{x}^2 - \frac12 k x^2", ["x"])
    assert r.status is Status.OK and r.duration_ms >= 0 and kinds(r) == ["energy"]


def test_every_equation_is_confirmed_by_numerical_differentiation() -> None:
    r = analyze(r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}", ["r", "theta"])
    assert r.status is Status.OK and len(r.equations) == 2
    for e in r.equations:
        assert e.verification and e.verification.passed
        assert e.verification.numeric_points_checked >= 4
        assert e.verification.numeric_points_agreed == e.verification.numeric_points_checked


def test_a_wrong_equation_is_rejected_by_the_independent_check() -> None:
    from minerva_engine.lagrangian import check_equations, derive
    from minerva_engine.verify import euler_lagrange_check

    d = derive(r"\frac12 m\dot{x}^2 - \frac12 k x^2", ["x"], "t", {}, L)
    assert all(v.passed for v in check_equations(d))
    wrong = [d.euler[0] + d.q[0]]  # m x'' + k x + x = 0
    assert not euler_lagrange_check(d.lagrangian, d.q, d.qd, d.qdd, d.t, wrong)[0].passed
    sign = [d.euler[0].subs(d.qdd[0], -d.qdd[0])]
    assert not euler_lagrange_check(d.lagrangian, d.q, d.qd, d.qdd, d.t, sign)[0].passed


def test_time_dependent_lagrangian_check_covers_explicit_time() -> None:
    from minerva_engine.lagrangian import check_equations, derive

    d = derive(r"\frac12 m\dot{x}^2 - \frac12 k x^2 t", ["x"], "t", {}, L)
    assert all(v.passed for v in check_equations(d))


def test_unconfirmed_equations_are_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    from minerva_engine import lagrangian
    from minerva_engine.models import Verification

    monkeypatch.setattr(
        lagrangian, "check_equations", lambda d: [Verification(passed=False) for _ in d.q]
    )
    r = analyze(r"\frac12 m\dot{x}^2 - \frac12 k x^2", ["x"])
    assert r.status is Status.NOT_VERIFIED
    assert r.equations == [] and r.accelerations == [] and r.conserved == []
