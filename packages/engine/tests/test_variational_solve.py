"""Stationary paths by two methods, against closed forms and with the honest failures."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import pytest
from minerva_engine.models import Limits, Status
from minerva_engine.variational import VariationInput, variation
from minerva_engine.variational import hermite as H
from minerva_engine.variational.numeric_setup import prepare
from minerva_engine.variational.setup import read_problem


def fixed(end: str, value: str, order: int = 0, function: str = "x") -> dict[str, Any]:
    return {"function": function, "order": order, "end": end, "kind": "fixed", "value": value}


def free(end: str, order: int = 0, function: str = "x") -> dict[str, Any]:
    return {"function": function, "order": order, "end": end, "kind": "free"}


def solve(integrand: str, **kw: Any) -> Any:
    kw.setdefault("functions", ["x"])
    kw.setdefault("lower", "0")
    kw.setdefault("upper", "1")
    data = VariationInput(action="solve", integrand=integrand, **kw)
    return variation(data, Limits(timeout_seconds=120))


def action(report: Any, index: int = 0) -> mp.mpf:
    return mp.mpf(report.solution.paths[index].action)


def value_at(report: Any, t: float, name: str = "x", index: int = 0) -> float:
    s = report.solution.paths[index].samples
    k = min(range(len(s.t)), key=lambda n: abs(s.t[n] - t))
    return s.values[name][k]


def test_harmonic_oscillator_matches_the_closed_form_and_both_methods_agree() -> None:
    r = solve(
        r"\frac12\dot{x}^2-\frac12 x^2", conditions=[fixed("lower", "0"), fixed("upper", "1")]
    )
    assert r.status is Status.OK
    sol = r.solution
    assert sol.confirmed_by_both and sol.agreement is not None and sol.agreement < 1e-9
    assert abs(action(r) - mp.cot(1) / 2) < 1e-9
    assert abs(value_at(r, 0.5) - float(mp.sin(0.5) / mp.sin(1))) < 1e-9
    assert all(m.converged for m in sol.methods)
    assert "not claimed to be unique" in sol.statement and "saddle" in sol.statement


def test_cantilever_with_free_end_and_its_natural_conditions() -> None:
    r = solve(
        r"\frac12\ddot{x}^2-x",
        conditions=[fixed("lower", "0"), fixed("lower", "0", 1), free("upper"), free("upper", 1)],
    )
    assert r.status is Status.OK
    assert abs(action(r) + mp.mpf(1) / 40) < 1e-10
    t = 0.7
    assert abs(value_at(r, t) - (t**4 / 24 - t**3 / 6 + t**2 / 4)) < 1e-9


def test_simply_supported_beam_leaves_the_slopes_free() -> None:
    r = solve(
        r"\frac12\ddot{x}^2-x",
        conditions=[fixed("lower", "0"), fixed("upper", "0"), free("lower", 1), free("upper", 1)],
    )
    assert r.status is Status.OK
    assert abs(action(r) + mp.mpf(1) / 240) < 1e-10


def test_free_end_natural_condition_is_found_not_assumed() -> None:
    r = solve(r"\frac12\dot{x}^2-x", conditions=[fixed("lower", "0"), free("upper")])
    assert abs(action(r) + mp.mpf(1) / 6) < 1e-10
    slope = r.solution.paths[0].samples.slopes["x"][-1]
    assert abs(slope) < 1e-8  # x'(1) = 0 came out of the problem


def test_endpoint_term_changes_the_natural_condition() -> None:
    r = solve(
        r"\frac12\dot{x}^2",
        upper_term=r"\frac{k}{2}x^2-c x",
        values={"k": "2", "c": "3"},
        conditions=[fixed("lower", "0"), free("upper")],
    )
    assert r.status is Status.OK
    assert abs(action(r) + mp.mpf(3) / 2) < 1e-10  # -c^2 / (2 (1 + k))
    assert abs(value_at(r, 1.0) - 1.0) < 1e-9  # c / (1 + k)


def test_higher_derivative_functional_with_a_nonlinear_potential() -> None:
    r = solve(
        r"\frac12\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2+V(x)",
        given_functions=[{"name": "V", "args": ["x"]}],
        definitions={"V": "x^4/4"},
        values={"alpha": "0.01"},
        conditions=[
            fixed("lower", "0"),
            fixed("upper", "1"),
            free("lower", 1),
            free("upper", 1),
        ],
    )
    assert r.status is Status.OK and r.solution.confirmed_by_both


def test_stiff_higher_derivative_problem_is_handled_by_collocation() -> None:
    r = solve(
        r"\frac12\dot{x}^2+\frac{\alpha}{2}\ddot{x}^2",
        values={"alpha": "0.0001"},
        conditions=[fixed("lower", "0"), fixed("upper", "1"), free("lower", 1), free("upper", 1)],
    )
    assert r.status is Status.OK
    assert abs(action(r) - mp.mpf(1) / 2) < 1e-9 and abs(value_at(r, 0.3) - 0.3) < 1e-9


def test_a_stationary_path_that_is_not_a_minimum_is_reported_as_such() -> None:
    r = solve(
        r"\frac12\dot{x}^2-\frac12 x^2",
        upper="4",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
        with_second=True,
    )
    assert r.status is Status.OK
    assert "not_a_weak_minimum" in r.solution.second.facts


def test_resonant_problem_has_no_isolated_solution_and_says_so() -> None:
    r = solve(
        r"\frac12\dot{x}^2-\frac12 x^2",
        upper=r"\pi",
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    assert r.status is Status.NOT_VERIFIED
    assert r.solution.paths == []
    assert any(
        "singular" in m.note or "converge" in m.note or "settle" in m.note
        for m in r.solution.methods
    )


def test_two_stationary_paths_of_the_catenary_from_two_starting_guesses() -> None:
    def c_of(guess_c: float) -> mp.mpf:
        return mp.findroot(lambda c: c * mp.cosh(1 / c) - 2, guess_c)

    results = []
    for c0 in (1.7, 0.47):
        c = c_of(c0)
        r = solve(
            r"x\sqrt{1+\dot{x}^2}",
            lower="-1",
            upper="1",
            conditions=[fixed("lower", "2"), fixed("upper", "2")],
            paths=[{"analytic": {"x": f"{float(c)}*\\cosh(t/{float(c)})*(1+0.05*t)"}}],
        )
        assert r.status is Status.OK, (c0, r.message)
        want = c + c**2 / 2 * mp.sinh(2 / c)
        assert abs(action(r) - want) < 1e-7
        results.append(float(action(r)))
    assert abs(results[0] - results[1]) > 0.1  # genuinely different stationary paths


# -- refusals -------------------------------------------------------------------------------------------
def test_undeclared_endpoint_conditions_are_asked_for() -> None:
    r = solve(r"\frac12\dot{x}^2", conditions=[fixed("lower", "0")])
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and "upper" in r.message


def test_a_fixed_derivative_beyond_the_order_would_over_determine_the_problem() -> None:
    r = solve(
        r"\frac12\dot{x}^2",
        conditions=[fixed("lower", "0"), fixed("upper", "1"), fixed("lower", "0", 1)],
    )
    assert r.status is Status.CONDITIONS_NOT_SATISFIED and "over-determined" in r.message


def test_algebraic_unknowns_and_complex_values_are_refused() -> None:
    r = solve(
        r"p\dot{x}-\frac12(p^2+x^2)",
        functions=["x", "p"],
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    assert r.status is Status.UNSUPPORTED
    c = solve(
        "m/2*x_dot**2",
        format="text",
        values={"m": "1+I"},
        conditions=[fixed("lower", "0"), fixed("upper", "1")],
    )
    assert c.status is Status.CONDITIONS_NOT_SATISFIED


# -- the discretisation itself ---------------------------------------------------------------------------
def build_mesh(elements: int = 6) -> tuple[H.Mesh, list[float]]:
    data = VariationInput(
        action="solve", integrand=r"\frac12\ddot{x}^2+\frac12\dot{x}^2-\sin(x)", functions=["x"],
        lower="0", upper="1",
    )  # fmt: skip
    problem = read_problem(data, Limits())
    num = prepare(problem, data, 1)
    mesh = H.Mesh(H.discretise(num), elements)
    z = [0.1 * ((i * 7) % 5) - 0.2 for i in range(mesh.size)]
    return mesh, z


def test_assembled_gradient_and_hessian_match_finite_differences_of_the_action() -> None:
    mesh, z = build_mesh()
    assert H.check_discretisation(mesh, z)


def test_a_wrong_stencil_is_caught() -> None:
    mesh, z = build_mesh()

    def wrong_scale(m: H.Mesh, v: list[float]) -> Any:
        grad, rows = H.assemble(m, v)
        grad[3] *= 1.001  # one entry off by a tenth of a percent
        return grad, rows

    def dropped_term(m: H.Mesh, v: list[float]) -> Any:
        grad, rows = H.assemble(m, v)
        row = rows[4]
        row[max(row)] = row[max(row)] * 0.5  # a coupling halved
        return grad, rows

    assert not H.check_discretisation(mesh, z, assembler=wrong_scale)
    assert not H.check_discretisation(mesh, z, assembler=dropped_term)


@pytest.mark.parametrize("elements", [4, 8])
def test_hermite_dofs_are_scaled_derivatives(elements: int) -> None:
    mesh, _ = build_mesh(elements)
    assert mesh.discrete.layout.nd == 3 and mesh.size == 3 * (elements + 1)
