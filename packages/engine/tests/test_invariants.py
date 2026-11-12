"""The search for constants of motion beyond Noether's easy cases (`invariants.py`)."""

import sympy as sp
from minerva_engine.invariants import InvariantReport, search_invariants
from minerva_engine.latexparse import parse_latex
from minerva_engine.models import Limits, Status

LIMITS = Limits()
KEPLER = r"\frac{1}{2}m(\dot{r}^2+r^2\dot{\theta}^2)+\frac{k}{r}"
KEPLER_CARTESIAN = r"\frac{1}{2}m(\dot{x}^2+\dot{y}^2)+\frac{k}{\sqrt{x^2+y^2}}"
OSCILLATOR = r"\frac{1}{2}m(\dot{x}^2+\dot{y}^2)-\frac{1}{2}k(x^2+y^2)"
HENON_HEILES = r"\frac{1}{2}(\dot{x}^2+\dot{y}^2)-\frac{1}{2}(x^2+y^2)-x^2 y+\frac{1}{3}y^3"


def run(
    latex: str,
    coordinates: list[str],
    angles: list[str] | None = None,
    degree: int = 2,
    power_bound: int = 3,
    harmonics: int = 1,
) -> InvariantReport:
    return search_invariants(
        latex, coordinates, angles or [], degree, power_bound, harmonics, "t", {}, LIMITS
    )


def parsed(report: InvariantReport) -> list[sp.Expr]:
    out = []
    for item in report.found:
        rhs = item.expression_latex.split("=", 1)[1]
        out.append(parse_latex(rhs, {}, LIMITS))
    return out


def symbols(expressions: list[sp.Expr]) -> dict[str, sp.Symbol]:
    return {s.name: s for e in expressions for s in e.free_symbols}


def same_up_to_sign(a: sp.Expr, b: sp.Expr) -> bool:
    return sp.simplify(a - b) == 0 or sp.simplify(a + b) == 0


def test_kepler_polar_finds_the_runge_lenz_vector() -> None:
    report = run(KEPLER, ["r", "theta"], ["theta"])
    assert report.status is Status.OK
    assert report.known == ["Momentum conjugate to theta", "Energy function"]
    assert len(report.found) == 2
    assert all(f.verified and f.verification and f.verification.passed for f in report.found)
    found = parsed(report)
    s = symbols(found)
    r, th, rd, thd, m, k = (s[n] for n in ("r", "theta", "r_dot", "theta_dot", "m", "k"))
    # A = p x L - m k r-hat, divided by m: both Cartesian components of the Laplace-Runge-Lenz vector.
    ax = m * r**3 * thd**2 * sp.cos(th) + m * r**2 * rd * thd * sp.sin(th) - k * sp.cos(th)
    ay = m * r**3 * thd**2 * sp.sin(th) - m * r**2 * rd * thd * sp.cos(th) - k * sp.sin(th)
    assert sum(same_up_to_sign(f, ax) for f in found) == 1
    assert sum(same_up_to_sign(f, ay) for f in found) == 1
    assert "proves nothing" in " ".join(report.notes)
    assert "cos and sin of theta" in report.ansatz


def test_the_runge_lenz_vector_is_not_a_noether_constant_of_the_default_search() -> None:
    from minerva_engine.lagrangian import analyze_lagrangian

    report = analyze_lagrangian(KEPLER, ["r", "theta"], "t", {}, LIMITS)
    assert [c.kind for c in report.conserved] == ["cyclic", "energy"]


def test_kepler_needs_velocity_degree_two() -> None:
    report = run(KEPLER, ["r", "theta"], ["theta"], degree=1)
    assert report.status is Status.OK
    assert report.found == []


def test_isotropic_oscillator_finds_the_fradkin_tensor_beyond_l_and_h() -> None:
    report = run(OSCILLATOR, ["x", "y"])
    assert report.status is Status.OK
    assert report.known == ["Energy function", "Rotation symmetry in the (x, y) plane"]
    found = parsed(report)
    s = symbols(found)
    x, y, xd, yd, m, k = (s[n] for n in ("x", "y", "x_dot", "y_dot", "m", "k"))
    cross = k * x * y + m * xd * yd
    xx = k * x**2 + m * xd**2
    assert sum(same_up_to_sign(f, cross) for f in found) == 1
    assert sum(same_up_to_sign(f, xx) for f in found) == 1
    assert len(found) == 2


def test_henon_heiles_has_none_in_this_family() -> None:
    report = run(HENON_HEILES, ["x", "y"])
    assert report.status is Status.OK
    assert report.found == []
    assert report.known == ["Energy function"]
    assert report.features[-1] == "lagrangian:no-hidden-constant"


def test_known_constants_and_their_products_are_not_reported_again() -> None:
    # p_theta, h and their products lie in the ansatz and must be factored out of the count.
    report = run(KEPLER, ["r", "theta"], ["theta"], power_bound=4)
    assert len(report.found) == 2


def test_every_reported_constant_has_a_symbolic_zero_derivative() -> None:
    for latex, coordinates, angles in (
        (KEPLER, ["r", "theta"], ["theta"]),
        (OSCILLATOR, ["x", "y"], []),
    ):
        report = run(latex, coordinates, angles)
        for item in report.found:
            assert item.verification is not None
            assert item.verification.symbolic_zero and item.verification.passed


def test_time_dependent_lagrangian_is_unsupported() -> None:
    report = run(r"\frac{1}{2}m\dot{x}^2-\frac{1}{2}k t x^2", ["x"])
    assert report.status is Status.UNSUPPORTED
    assert "time" in report.message


def test_square_roots_are_unsupported_not_guessed() -> None:
    report = run(KEPLER_CARTESIAN, ["x", "y"])
    assert report.status is Status.UNSUPPORTED
    assert report.found == []


def test_an_angle_that_is_not_declared_is_not_searched_as_one() -> None:
    # theta appears through cos only: without declaring it an angle, cos(theta) is not rational.
    report = run(r"\frac{1}{2}m l^2\dot{\theta}^2+m g l\cos(\theta)", ["theta"])
    assert report.status is Status.UNSUPPORTED


def test_oversized_ansatz_is_a_resource_limit() -> None:
    report = run(
        r"\frac{1}{2}(\dot{x}^2+\dot{y}^2+\dot{z}^2)-\frac{1}{2}(x^2+y^2+z^2)",
        ["x", "y", "z"],
        degree=2,
        power_bound=4,
    )
    assert report.status is Status.RESOURCE_LIMIT
    assert report.unknowns > 500


def test_bad_arguments_are_refused() -> None:
    assert run(KEPLER, ["r", "theta"], ["phi"]).status is Status.PARSE_ERROR
    assert run(KEPLER, ["r", "theta"], ["theta"], degree=3).status is Status.UNSUPPORTED
    assert run(KEPLER, ["a", "b", "c", "d"]).status is Status.UNSUPPORTED
    assert run(r"\sin 2x", ["x"]).status is Status.PARSE_ERROR


def test_a_wrong_candidate_fails_the_verification_used_by_the_search() -> None:
    from minerva_engine.lagrangian import derive, verify_constant

    d = derive(KEPLER, ["r", "theta"], "t", {}, LIMITS)
    th, r, rd, thd = d.q[1], d.q[0], d.qd[0], d.qd[1]
    wrong = d.params[1] * r**3 * thd**2 * sp.cos(th) + r**2 * rd * thd * sp.sin(th)
    ok, _ = verify_constant(d, wrong)
    assert not ok
