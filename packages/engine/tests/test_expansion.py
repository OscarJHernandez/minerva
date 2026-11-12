"""Regression tests for expanding equations of motion about an equilibrium (sandboxed engine)."""

import pytest
import sympy as sp
from minerva_engine import Engine, Status
from minerva_engine.expansion import _zero_check
from minerva_engine.series import check_series

POSITIVE = {name: ["positive"] for name in ("m", "g", "l", "k", "c")}
PENDULUM = r"\frac12 m l^2 \dot{\theta}^2 + m g l \cos\theta"
COUPLED = r"\frac12 m(\dot{x}^2+\dot{y}^2) - \frac12 k x^2 - \frac12 k y^2 - \frac12 c (x-y)^2"


def expand(engine: Engine, latex, coordinates, order=3, assumptions=None, **equilibrium):
    return engine.expand_motion(
        latex,
        coordinates,
        "t",
        assumptions if assumptions is not None else POSITIVE,
        equilibrium or None,
        order,
    )


# --- positive ----------------------------------------------------------------------------------
def test_pendulum_about_the_bottom_to_third_order(engine: Engine) -> None:
    report = expand(engine, PENDULUM, ["theta"], order=3)
    assert report.status is Status.OK, report.message
    (equation,) = report.equations
    degrees = {t.degree: t.latex for t in equation.terms}
    assert set(degrees) == {1, 3}  # sin is odd: no quadratic term
    assert degrees[1] == r"- \frac{\delta \theta g}{l}"
    assert degrees[3] == r"\frac{\delta \theta^{3} g}{6 l}"
    assert report.equilibrium[0].name_latex == r"\theta"
    # Terms are written in ascending degree.
    assert equation.full_latex.index("delta \\theta g") < equation.full_latex.index(
        "delta \\theta^{3}"
    )
    assert equation.verification is not None and equation.verification.passed
    assert report.equilibrium_verification is not None and report.equilibrium_verification.passed
    (mode,) = report.modes
    assert mode.sign == "positive" and mode.verified
    assert mode.omega_latex == r"\omega = \frac{\sqrt{g}}{\sqrt{l}}"


def test_harmonic_oscillator_has_no_higher_terms(engine: Engine) -> None:
    report = expand(engine, r"\frac12 m \dot{x}^2 - \frac12 k x^2", ["x"], order=5)
    assert report.status is Status.OK
    assert [t.degree for t in report.equations[0].terms] == [1]


def test_equilibrium_that_depends_on_the_parameters(engine: Engine) -> None:
    report = expand(
        engine, r"\frac12 m \dot{x}^2 - \frac12 k x^2 + m g x", ["x"], order=2, x="m*g/k"
    )
    assert report.status is Status.OK, report.message
    assert [t.degree for t in report.equations[0].terms] == [1]  # the shift removes the force
    assert report.modes[0].omega_squared_latex == r"\omega^2 = \frac{k}{m}"


def test_duffing_has_an_amplitude_term_at_third_order(engine: Engine) -> None:
    report = expand(
        engine, r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac14 b x^4", ["x"], assumptions={}
    )
    assert report.status is Status.OK
    degrees = {t.degree: t.latex for t in report.equations[0].terms}
    assert degrees == {1: r"- \delta x", 3: r"- b \delta x^{3}"}


def test_coupled_oscillators_give_both_modes_with_eigenvectors(engine: Engine) -> None:
    report = expand(engine, COUPLED, ["x", "y"], order=2)
    assert report.status is Status.OK, report.message
    squares = sorted(m.omega_squared_latex for m in report.modes)
    assert squares == sorted([r"\omega^2 = \frac{k}{m}", r"\omega^2 = \frac{2 c + k}{m}"])
    assert all(m.verified and m.sign == "positive" for m in report.modes)
    vectors = {m.eigenvector_latex for m in report.modes}
    assert vectors == {
        r"\left[\begin{matrix}1\\1\end{matrix}\right]",
        r"\left[\begin{matrix}-1\\1\end{matrix}\right]",
    }


def test_double_pendulum_modes_match_the_known_values(engine: Engine) -> None:
    latex = (
        r"\frac12 (m_1+m_2) l_1^2\dot{a}^2 + \frac12 m_2 l_2^2 \dot{b}^2 "
        r"+ m_2 l_1 l_2 \dot{a}\dot{b}\cos(a-b) + (m_1+m_2) g l_1 \cos a + m_2 g l_2 \cos b"
    )
    assumptions = {n: ["positive"] for n in ("m_1", "m_2", "g", "l_1", "l_2")}
    report = expand(engine, latex, ["a", "b"], order=1, assumptions=assumptions)
    assert report.status is Status.OK, report.message
    assert len(report.modes) == 2 and all(m.verified for m in report.modes)
    # With m1 = m2 = l1 = l2 = g = 1 the two squared frequencies are 2 -/+ sqrt(2).
    from minerva_engine import Limits, parse_latex

    values = sorted(
        float(
            parse_latex(m.omega_squared_latex.split("=", 1)[1], assumptions, Limits()).subs(
                {sp.Symbol(n, positive=True): 1 for n in ("m_1", "m_2", "g", "l_1", "l_2")}
            )
        )
        for m in report.modes
    )
    assert values == pytest.approx([2 - 2**0.5, 2 + 2**0.5])


# --- honesty ------------------------------------------------------------------------------------
def test_a_point_that_is_not_an_equilibrium_is_refused(engine: Engine) -> None:
    report = expand(engine, PENDULUM, ["theta"], theta="1")
    assert report.status is Status.UNSUPPORTED
    assert "not an equilibrium" in report.message
    assert report.equations == [] and report.equilibrium_verification is not None
    assert not report.equilibrium_verification.passed


def test_the_inverted_pendulum_has_negative_omega_squared_and_no_frequency(engine: Engine) -> None:
    report = expand(engine, PENDULUM, ["theta"], theta="pi")
    assert report.status is Status.OK, report.message
    (mode,) = report.modes
    assert mode.sign == "negative" and mode.omega_latex is None
    assert "stable" not in report.message.lower()
    assert [t.degree for t in report.equations[0].terms] == [1, 3]


def test_the_sign_is_not_claimed_without_assumptions(engine: Engine) -> None:
    report = expand(engine, PENDULUM, ["theta"], assumptions={})
    assert report.status is Status.OK
    assert report.modes[0].sign == "unknown" and report.modes[0].omega_latex is None


def test_velocity_terms_give_no_normal_modes(engine: Engine) -> None:
    latex = r"\frac12 (\dot{x}^2+\dot{y}^2) + \Omega (x\dot{y}-y\dot{x}) - \frac12 k(x^2+y^2)"
    report = expand(
        engine, latex, ["x", "y"], order=2, assumptions={"k": ["positive"], "Omega": ["real"]}
    )
    assert report.status is Status.OK, report.message
    assert report.modes == []
    assert report.velocity_latex is not None and report.characteristic_latex is not None
    assert any("gyroscopic" in n for n in report.notes)


def test_more_than_two_coordinates_get_the_characteristic_polynomial_only(engine: Engine) -> None:
    latex = r"\frac12 (\dot{x}^2+\dot{y}^2+\dot{z}^2) - \frac12 (x^2+2y^2+3z^2)"
    report = expand(engine, latex, ["x", "y", "z"], order=1, assumptions={})
    assert report.status is Status.OK
    assert report.modes == [] and report.characteristic_latex is not None
    assert any("up to 2 coordinates" in n for n in report.notes)


def test_explicit_time_dependence_is_unsupported(engine: Engine) -> None:
    report = expand(engine, r"\frac12 \dot{x}^2 - \frac12 t x^2", ["x"], assumptions={})
    assert report.status is Status.UNSUPPORTED and "time" in report.message


def test_singular_mass_matrix_is_unsupported(engine: Engine) -> None:
    report = expand(
        engine, r"\frac12 \dot{x}^2 - \frac12 x^2 - \frac12 y^2 + x y", ["x", "y"], assumptions={}
    )
    assert report.status is Status.UNSUPPORTED and "singular" in report.message


def test_singular_acceleration_at_the_point_is_unsupported(engine: Engine) -> None:
    report = expand(
        engine, r"\frac12 \dot{r}^2 + \frac{k}{r}", ["r"], assumptions={"k": ["positive"]}
    )
    assert report.status is Status.UNSUPPORTED  # r = 0 is a singularity, and not an equilibrium


@pytest.mark.parametrize(
    ("equilibrium", "status"),
    [({"z": "0"}, Status.PARSE_ERROR), ({"theta": "w"}, Status.UNSUPPORTED)],
)
def test_bad_equilibrium_input(engine: Engine, equilibrium, status) -> None:
    report = expand(engine, PENDULUM, ["theta"], **equilibrium)
    assert report.status is status
    assert report.equations == []


def test_order_and_input_limits(engine: Engine) -> None:
    assert expand(engine, PENDULUM, ["theta"], order=0).status is Status.UNSUPPORTED
    assert expand(engine, PENDULUM, ["theta"], order=13).status is Status.UNSUPPORTED
    assert expand(engine, r"\nonsense", ["x"]).status in {Status.PARSE_ERROR, Status.UNSUPPORTED}
    assert expand(engine, r"\frac12 \dot{x}^2 + x^{100000}", ["x"]).status in {
        Status.PARSE_ERROR,
        Status.RESOURCE_LIMIT,
        Status.UNSUPPORTED,
    }


# --- the checks reject wrong results --------------------------------------------------------------
def test_wrong_expansion_coefficient_is_rejected_by_the_series_check() -> None:
    eps, u = sp.symbols("eps u")
    g, ell = sp.symbols("g l", positive=True)
    acceleration = -g / ell * sp.sin(eps * u)
    right = -g / ell * (eps * u - (eps * u) ** 3 / 6)
    wrong = -g / ell * (eps * u - (eps * u) ** 3 / 5)
    assert check_series(acceleration, eps, sp.Integer(0), right, 3).passed
    assert not check_series(acceleration, eps, sp.Integer(0), wrong, 3).passed


def test_wrong_normal_mode_is_rejected_by_substitution() -> None:
    w = sp.Symbol("w")
    g, ell = sp.symbols("g l", positive=True)
    poly = sp.expand(g / ell + w)  # det(A + w I) for the pendulum about the top
    assert _zero_check(poly.subs(w, -g / ell), "").passed
    assert not _zero_check(poly.subs(w, g / ell), "").passed
    assert not _zero_check(poly.subs(w, -g / (2 * ell)), "").passed
