"""Regression tests for the search for integrals quadratic in the velocities (Runge-Lenz and relatives)."""

import time

import pytest
import sympy as sp
from minerva_engine import Engine, Limits, Status
from minerva_engine.lagrangian import derive, find_conserved, verify_quantity
from minerva_engine.quadratic_integrals import find_quadratic

KEPLER_2D = r"\frac12 m(\dot{x}^2 + \dot{y}^2) + \frac{k}{\sqrt{x^2+y^2}}"
KEPLER_3D = r"\frac12 m(\dot{x}^2 + \dot{y}^2 + \dot{z}^2) + \frac{k}{\sqrt{x^2+y^2+z^2}}"
LIMITS = Limits(timeout_seconds=60)


def search(latex: str, coordinates: list[str], assumptions=None):
    d = derive(latex, coordinates, "t", assumptions or {}, LIMITS)
    return d, find_conserved(d, quadratic_deadline=time.monotonic() + 40)


def quadratic(found):
    return [(c, e) for c, e in found.items if c.kind == "quadratic"]


def symbols(d, *names):
    by_name = {s.name: s for s in d.params}
    return [by_name[n] for n in names]


def in_span(d, known: sp.Expr, items: list[sp.Expr], points: list[tuple[int, ...]]) -> bool:
    """Is `known` a constant-coefficient combination of `items` (plus a constant)? Checked at exact points."""
    a = sp.symbols(f"a0:{len(items)}")
    c = sp.Symbol("c")
    equations = []
    for point in points:
        n = d.n
        values = {
            **dict(zip(d.q, point[:n], strict=True)),
            **dict(zip(d.qd, point[n:], strict=True)),
        }
        difference = known - sum((ai * it for ai, it in zip(a, items, strict=True)), c)
        equations.append(sp.simplify(difference.subs(values)))
    solution = sp.solve([e for e in equations], [*a, c], dict=True)
    return bool(solution)


# --- the Runge-Lenz vector -------------------------------------------------------------------------
def test_kepler_2d_finds_both_runge_lenz_components() -> None:
    d, found = search(KEPLER_2D, ["x", "y"])
    items = quadratic(found)
    assert len(items) == 2  # the energy, L and L^2 are not new
    assert all(c.verified and c.verification and c.verification.symbolic_zero for c, _ in items)
    x, y = d.q
    xd, yd = d.qd
    (m, k) = symbols(d, "m", "k")
    r = sp.sqrt(x**2 + y**2)
    a_x = m * yd * (x * yd - y * xd) - k * x / r
    a_y = m * xd * (y * xd - x * yd) - k * y / r
    pythagorean = [(3, 4, 1, 2), (5, 12, -2, 3), (8, 15, 3, -1), (7, 24, 2, 5)]
    expressions = [e for _, e in items]
    assert in_span(d, a_x, expressions, pythagorean)
    assert in_span(d, a_y, expressions, pythagorean)


def test_kepler_3d_finds_the_three_components() -> None:
    d, found = search(KEPLER_3D, ["x", "y", "z"])
    items = quadratic(found)
    assert len(items) == 3  # the three angular-momentum components are the linear integrals
    assert all(c.verified for c, _ in items)
    x, y, z = d.q
    xd, yd, zd = d.qd
    (m, k) = symbols(d, "m", "k")
    r = sp.sqrt(x**2 + y**2 + z**2)
    a_z = m * z * (xd**2 + yd**2) - m * zd * (x * xd + y * yd) - k * z / r
    expressions = [e for _, e in items]
    points = [(2, 3, 6, 1, 2, 3), (1, 2, 2, -1, 0, 2), (6, 2, 3, 2, -1, 1), (3, 6, 2, 0, 1, -2)]
    points += [(2, 6, 3, 1, 1, 1), (4, 4, 7, -2, 2, 1), (1, 4, 8, 3, 1, 0)]
    assert in_span(d, a_z, expressions, points)


def test_the_quantities_are_conserved_on_the_equations_of_motion() -> None:
    d, found = search(KEPLER_2D, ["x", "y"])
    for _, expression in quadratic(found):
        ok, verification = verify_quantity(d, expression)
        assert ok and verification is not None and verification.symbolic_zero


def test_a_wrong_runge_lenz_vector_is_rejected() -> None:
    d, _ = search(KEPLER_2D, ["x", "y"])
    x, y = d.q
    xd, yd = d.qd
    (m, k) = symbols(d, "m", "k")
    r = sp.sqrt(x**2 + y**2)
    right = m * yd * (x * yd - y * xd) - k * x / r
    assert verify_quantity(d, right)[0]
    for wrong in (
        m * yd * (x * yd - y * xd) - 2 * k * x / r,  # wrong strength of the potential term
        m * yd * (x * yd - y * xd) + k * x / r,  # wrong sign
        m * xd * (x * yd - y * xd) - k * x / r,  # wrong velocity factor
        right + m * xd,  # a constant of motion plus something that is not
    ):
        ok, verification = verify_quantity(d, wrong)
        assert not ok and verification is not None and not verification.symbolic_zero


# --- other known cases ---------------------------------------------------------------------------
def test_isotropic_oscillator_gives_the_fradkin_tensor() -> None:
    latex = r"\frac12 m(\dot{x}^2 + \dot{y}^2) - \frac12 k (x^2+y^2)"
    d, found = search(latex, ["x", "y"])
    items = quadratic(found)
    assert len(items) == 2  # three independent components, minus the energy
    x, y = d.q
    xd, yd = d.qd
    (m, k) = symbols(d, "m", "k")
    energy = [e for c, e in found.items if c.kind == "energy"]
    expressions = [e for _, e in items] + energy  # E_x - E_y = I_1 - E
    points = [(1, 2, 3, 4), (2, -1, 1, 5), (3, 1, -2, 2), (0, 2, 1, 1)]
    assert in_span(d, m * xd * yd + k * x * y, expressions, points)
    assert in_span(d, m * xd**2 + k * x**2 - m * yd**2 - k * y**2, expressions, points)
    assert all(c.verified and c.verification and c.verification.symbolic_zero for c, _ in items)


def test_isotropic_oscillator_in_three_dimensions_has_five_new_integrals() -> None:
    latex = r"\frac12 m(\dot{x}^2 + \dot{y}^2 + \dot{z}^2) - \frac12 k (x^2+y^2+z^2)"
    _, found = search(latex, ["x", "y", "z"])
    assert len(quadratic(found)) == 5  # six components of the Fradkin tensor, minus the energy


def test_anisotropic_oscillator_has_only_the_separated_energies() -> None:
    latex = r"\frac12 m(\dot{x}^2 + \dot{y}^2) - \frac12 a x^2 - \frac12 b y^2"
    d, found = search(latex, ["x", "y"])
    items = quadratic(found)
    assert len(items) == 1  # E_x (E_y follows from it and the energy); no mixed integral
    x, _y = d.q
    xd, _yd = d.qd
    (a, m) = symbols(d, "a", "m")
    assert in_span(
        d, m * xd**2 + a * x**2, [e for _, e in items], [(1, 2, 3, 4), (2, 1, 0, 1), (3, 3, 1, 2)]
    )


def test_free_particle_has_nothing_beyond_products_of_linear_integrals() -> None:
    _, found = search(r"\frac12 m(\dot{x}^2 + \dot{y}^2)", ["x", "y"])
    assert quadratic(found) == []
    assert any("none beyond" in t for t in found.tested)


def test_a_generic_potential_has_none_and_the_report_says_what_was_searched() -> None:
    _, found = search(r"\frac12 m(\dot{x}^2 + \dot{y}^2) - x^4 - x y^2", ["x", "y"])
    assert quadratic(found) == []
    assert any("K of degree <= 2" in t and "none beyond" in t for t in found.tested)


def test_the_inverse_square_potential_has_no_extra_quadratic_integral() -> None:
    _, found = search(r"\frac12 m(\dot{x}^2 + \dot{y}^2) - \frac{k}{x^2+y^2}", ["x", "y"])
    assert quadratic(found) == []


def test_products_of_known_integrals_are_not_listed() -> None:
    _, found = search(KEPLER_2D, ["x", "y"])
    kinds = sorted(c.kind for c, _ in found.items)
    assert kinds == ["energy", "quadratic", "quadratic", "rotation"]  # not L^2, not E L


# --- honesty ---------------------------------------------------------------------------------------
def test_polar_coordinates_are_not_searched_and_the_report_says_so() -> None:
    engine = Engine(Limits(timeout_seconds=30, max_concurrency=2))
    r = engine.analyze_lagrangian(
        r"\frac12 m(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{k}{r}", ["r", "theta"]
    )
    assert r.status is Status.OK
    assert [c.kind for c in r.conserved] == ["cyclic", "energy"]
    assert any("Cartesian" in n and "polar" in n for n in r.notes)
    assert any("None found" in n and "not a proof" in n for n in r.notes)


def test_the_analyser_reports_the_runge_lenz_vector_in_cartesian_coordinates() -> None:
    engine = Engine(Limits(timeout_seconds=30, max_concurrency=2))
    r = engine.analyze_lagrangian(KEPLER_2D, ["x", "y"])
    assert "lagrangian:quadratic-integral" in r.features
    assert sum(1 for c in r.conserved if c.kind == "quadratic") == 2
    assert any("2 quadratic integrals found" in n for n in r.notes)
    assert all(c.verified for c in r.conserved)


def test_one_coordinate_is_not_searched() -> None:
    d = derive(r"\frac12 m\dot{x}^2 - \frac12 k x^2", ["x"], "t", {}, LIMITS)
    found = find_quadratic(d, time.monotonic() + 30)
    assert found.items == [] and any("at least two coordinates" in n for n in found.notes)


def test_too_many_coordinates_are_not_searched() -> None:
    latex = r"\frac12 (\dot{a}^2+\dot{b}^2+\dot{c}^2+\dot{d}^2) - \frac12 (a^2+b^2+c^2+d^2)"
    d = derive(latex, ["a", "b", "c", "d"], "t", {}, LIMITS)
    found = find_quadratic(d, time.monotonic() + 30)
    assert found.items == [] and any("more than 3" in n for n in found.notes)


def test_potentials_with_other_functions_are_not_searched() -> None:
    d = derive(r"\frac12 m(\dot{x}^2 + \dot{y}^2) - \sin(x) y", ["x", "y"], "t", {}, LIMITS)
    found = find_quadratic(d, time.monotonic() + 30)
    assert found.items == [] and any("powers and roots" in n for n in found.notes)


def test_a_non_unit_metric_is_not_searched() -> None:
    d = derive(r"\frac12 (1+y^2)\dot{x}^2 + \frac12 \dot{y}^2 - x^2", ["x", "y"], "t", {}, LIMITS)
    found = find_quadratic(d, time.monotonic() + 30)
    assert found.items == [] and any("Cartesian" in n for n in found.notes)


def test_a_potential_that_is_not_homogeneous_never_gives_an_unverified_result() -> None:
    d = derive(
        r"\frac12 m(\dot{x}^2 + \dot{y}^2) - \sqrt{1+x^2} - y^2", ["x", "y"], "t", {}, LIMITS
    )
    found = find_quadratic(d, time.monotonic() + 30)
    assert all(c.verified for c, _ in found.items)


def test_an_exhausted_time_budget_stops_the_search_and_says_so() -> None:
    d = derive(KEPLER_2D, ["x", "y"], "t", {}, LIMITS)
    found = find_quadratic(d, time.monotonic() - 1)
    assert found.items == [] and any("time budget" in n for n in found.notes)


def test_explicit_time_dependence_is_not_searched() -> None:
    d = derive(r"\frac12 m(\dot{x}^2 + \dot{y}^2) - t x^2", ["x", "y"], "t", {}, LIMITS)
    assert find_quadratic(d, time.monotonic() + 30).items == []


def test_motion_integration_does_not_pay_for_the_quadratic_search() -> None:
    d = derive(KEPLER_2D, ["x", "y"], "t", {}, LIMITS)
    plain = find_conserved(d)  # as integrate_motion calls it
    assert sorted(c.kind for c, _ in plain.items) == ["energy", "rotation"]


@pytest.mark.parametrize("assumptions", [{"k": ["positive"], "m": ["positive"]}])
def test_assumptions_do_not_change_the_result(assumptions) -> None:
    _, found = search(KEPLER_2D, ["x", "y"], assumptions)
    assert len(quadratic(found)) == 2
