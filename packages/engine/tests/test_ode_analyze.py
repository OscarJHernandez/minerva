"""Structure, classification, problem kind and method suggestions for differential equations."""

from __future__ import annotations

import json
from typing import Any

import pytest
from minerva_engine import Engine, OdeAnalysis, OdeInput, Status
from minerva_engine.ode import analyze
from minerva_engine.testing import budget_limits

L = budget_limits()


def run(*equations: str, fmt: str = "latex", **kw: Any) -> OdeAnalysis:
    return analyze(OdeInput(equations=list(equations), format=fmt, **kw), L)


def methods(result: OdeAnalysis) -> dict[str, Any]:
    return {m.id: m for m in result.methods}


def ranked(result: OdeAnalysis) -> list[str]:
    return [m.id for m in sorted((m for m in result.methods if m.rank), key=lambda m: m.rank or 0)]


# --- structure -------------------------------------------------------------------------------


def test_the_nonlinear_damped_oscillator_of_the_brief() -> None:
    r = run(r"y'' + 0.2 y' + y + y^3 = 0", independent="t")
    assert r.status is Status.OK
    s = r.structure
    assert s is not None
    assert (s.kind, s.order, s.state_dimension) == ("ode", 2, 2)
    assert s.linear is False and s.autonomous is True and s.solved_for_highest is True
    assert s.position_only is False  # the acceleration depends on the velocity
    assert r.classes == []
    m = methods(r)
    assert m["numerical_ivp"].applicable and m["numerical_ivp"].rank == 1
    assert m["equilibrium_analysis"].applicable and m["phase_plane"].applicable
    assert m["power_series"].applicable
    assert not m["symbolic_closed_form"].applicable
    assert not m["poincare_lindstedt"].applicable
    assert "velocity" in m["poincare_lindstedt"].reason
    assert r.solved_latex == ["y'' = - y^{3} - y - \\frac{y'}{5}"]


@pytest.mark.parametrize(
    ("source", "linear", "homogeneous", "constant", "autonomous"),
    [
        (r"y'' + 3y' + 2y = 0", True, True, True, True),
        (r"y'' + 3y' + 2y = \sin t", True, False, True, False),
        (r"y' + t y = 0", True, True, False, False),
        (r"y' = y^2", False, None, None, True),
        (r"y'' + \sin y = 0", False, None, None, True),
        (r"y' = a y", True, True, True, True),
    ],
)
def test_exact_structure_flags(
    source: str, linear: bool, homogeneous: bool | None, constant: bool | None, autonomous: bool
) -> None:
    s = run(source, independent="t").structure
    assert s is not None
    assert (s.linear, s.homogeneous, s.constant_coefficients, s.autonomous) == (
        linear,
        homogeneous,
        constant,
        autonomous,
    )


def test_first_order_classes() -> None:
    assert "separable" in run(r"y' = x y", independent="x").classes
    assert (
        "linear_first_order"
        in run(r"y' + 2 y = e^{x}".replace("e^", r"\exp"), independent="x").classes
    )
    assert "bernoulli" in run(r"y' + y = x y^3", independent="x").classes
    assert "exact" in run(r"2 x y + (x^2 + 1) y' = 0", independent="x").classes
    assert "homogeneous_degree_zero" in run(r"y' = \frac{x + y}{x}", independent="x").classes


def test_higher_order_classes() -> None:
    r = run(r"x^2 y'' + x y' - y = 0", independent="x")
    assert "euler_cauchy" in r.classes and "second_order_linear" in r.classes
    r = run(r"y'' + 4 y' + 4 y = 0", independent="x")
    assert "linear_constant_coefficients" in r.classes


def test_a_system_is_solved_for_its_highest_derivatives() -> None:
    r = run(r"\dot{x} = -y", r"\dot{y} = x")
    assert r.structure is not None
    assert r.structure.kind == "system" and r.structure.state_dimension == 2
    assert r.solved_latex == ["x' = - y", "y' = x"]
    assert "linear_constant_coefficients" in r.classes


def test_implicit_systems_are_solved_by_inverting_the_coefficient_matrix() -> None:
    # x' + y' = 1, x' - y' = 3  ->  x' = 2, y' = -1
    r = run(r"x' + y' = 1", r"x' - y' = 3", independent="t")
    assert r.solved_latex == ["x' = 2", "y' = -1"]
    assert r.restrictions_latex == []


def test_a_variable_coefficient_is_reported_as_a_restriction() -> None:
    r = run(r"x y' = 1", independent="x")
    assert r.structure is not None and r.structure.solved_for_highest
    assert r.restrictions_latex == ["x \\neq 0"]


@pytest.mark.parametrize(
    ("equations", "fragment"),
    [
        ([r"(y')^2 = y"], "not linear in its highest derivative"),
        ([r"\sin(y') = y"], "not linear in its highest derivative"),
        ([r"x' + y' = 1", r"2x' + 2y' = 2"], "singular"),
        ([r"x' = y", r"y = x"], "algebraic"),
    ],
)
def test_equations_that_cannot_be_solved_for_the_highest_derivative(
    equations: list[str], fragment: str
) -> None:
    r = run(*equations, independent="t", unknowns=["x", "y"] if len(equations) > 1 else [])
    assert r.status is Status.OK
    assert r.structure is not None
    assert r.structure.solved_for_highest is False
    assert fragment in " ".join(r.notes)
    m = methods(r)
    assert not m["numerical_ivp"].applicable and fragment in m["numerical_ivp"].reason
    assert not m["power_series"].applicable
    assert not m["equilibrium_analysis"].applicable


def test_more_unknowns_than_equations_is_underdetermined() -> None:
    r = run(r"x' = y", independent="t", unknowns=["x", "y"])
    assert r.structure is not None and r.structure.solved_for_highest is False
    assert "number of equations" in " ".join(r.notes)


# --- the kind of problem ---------------------------------------------------------------------


def test_problem_kinds() -> None:
    assert run(r"y'' + y = 0", independent="t").problem_kind == "general"
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"])
    assert r.problem_kind == "initial_value" and r.conditions_needed == []
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 1"])
    assert r.problem_kind == "incomplete" and r.conditions_needed == ["y'(t_0)"]
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0", "y''(0) = 2"])
    assert r.problem_kind == "overdetermined"
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 0", "y(1) = 1"])
    assert r.problem_kind == "boundary_value"
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 1", "y(0) = 2"])
    assert r.problem_kind == "overdetermined"  # the same condition twice cannot be satisfied twice
    r = run(r"y'' + y = 0", independent="t", conditions=["y(0) = 1", "y(1) = 1", "y(2) = 1"])
    assert r.problem_kind == "overdetermined"


def test_two_names_for_the_same_point_are_one_point() -> None:
    r = run(r"y'' + y = 0", independent="t", conditions=["y(1/2) = 1", "y'(0.5) = 0"])
    assert r.problem_kind == "initial_value"


def test_a_system_needs_every_unknown_conditioned() -> None:
    r = run(r"\dot{x} = -y", r"\dot{y} = x", conditions=["x(0) = 1"])
    assert r.problem_kind == "incomplete" and r.conditions_needed == ["y(t_0)"]


# --- the suggestions -------------------------------------------------------------------------


def test_a_recognised_class_ranks_the_closed_form_first() -> None:
    r = run(r"y' = x y", independent="x", conditions=["y(0) = 1"])
    assert ranked(r)[:3] == ["symbolic_closed_form", "numerical_ivp", "power_series"]
    assert methods(r)["symbolic_closed_form"].tool == "ode_solve_symbolic"


def test_boundary_problems_rank_the_bvp_method_first_and_refuse_the_ivp_one() -> None:
    r = run(r"y'' + y = 0", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert ranked(r)[0] == "numerical_bvp"
    m = methods(r)
    assert not m["numerical_ivp"].applicable and not m["power_series"].applicable


def test_perturbation_methods_need_a_symbolic_parameter_and_say_so() -> None:
    r = run(r"y'' + y + \epsilon y^3 = 0", independent="t")
    m = methods(r)
    assert m["poincare_lindstedt"].applicable and m["regular_perturbation"].applicable
    assert "named by you" in m["regular_perturbation"].requires[0]
    r = run(r"y'' + y + 0.1 y^3 = 0", independent="t")
    assert not methods(r)["regular_perturbation"].applicable


def test_poincare_lindstedt_is_refused_for_forcing_and_damping() -> None:
    forced = run(r"y'' + y + \epsilon y^3 = \cos t", independent="t")
    assert "forcing" in methods(forced)["poincare_lindstedt"].reason
    damped = run(r"y'' + \epsilon y' + y = 0", independent="t")
    assert "velocity" in methods(damped)["poincare_lindstedt"].reason
    system = run(r"\dot{x} = y", r"\dot{y} = -x")
    assert not methods(system)["poincare_lindstedt"].applicable


def test_qualitative_methods_need_an_autonomous_small_system() -> None:
    driven = run(r"y'' + y = \sin t", independent="t")
    assert not methods(driven)["equilibrium_analysis"].applicable
    assert "depends explicitly" in methods(driven)["equilibrium_analysis"].reason
    big = run(r"y^{(4)} + y = 0", independent="t")
    assert not methods(big)["equilibrium_analysis"].applicable
    third = run(r"y''' + y = 0", independent="t")
    assert methods(third)["equilibrium_analysis"].applicable
    assert not methods(third)["phase_plane"].applicable


def test_unimplemented_methods_are_listed_as_unavailable() -> None:
    m = methods(run(r"y'' + y = 0", independent="t"))["multiple_scales"]
    assert m.available is False and m.applicable is False and m.rank is None


def test_suggestions_never_promise_success() -> None:
    r = run(r"y' = x y", independent="x")
    for m in r.methods:
        text = " ".join([m.reason, *m.limitations]).lower()
        assert "will succeed" not in text and "guarantee" not in text
    numeric = methods(r)["numerical_ivp"]
    assert any("stiffness" in x for x in numeric.limitations)
    assert any("approximation" in x for x in numeric.limitations)


def test_ranks_are_consecutive_and_only_applicable_methods_have_one() -> None:
    for source in (r"y'' + y = 0", r"y' = x y", r"(y')^2 = y"):
        r = run(source, independent="x")
        ranks = sorted(m.rank for m in r.methods if m.rank is not None)
        assert ranks == list(range(1, len(ranks) + 1))
        assert all((m.rank is not None) == m.applicable for m in r.methods)


# --- metamorphic: the same dynamics in another form ------------------------------------------


def test_a_higher_order_equation_and_its_first_order_system_have_the_same_structure() -> None:
    higher = run(r"y'' + 3 y' + 2 y = 0", independent="t")
    system = run(r"y' = v", r"v' = -3 v - 2 y", independent="t")
    a, b = higher.structure, system.structure
    assert a is not None and b is not None
    assert (a.state_dimension, a.linear, a.homogeneous, a.constant_coefficients, a.autonomous) == (
        b.state_dimension,
        b.linear,
        b.homogeneous,
        b.constant_coefficients,
        b.autonomous,
    )
    assert "linear_constant_coefficients" in higher.classes
    assert "linear_constant_coefficients" in system.classes


def test_equivalent_notations_give_the_same_analysis() -> None:
    forms = [
        (r"y'' + \omega^2 y = 0", "latex"),
        (r"\frac{d^2 y}{dt^2} + \omega^2 y = 0", "latex"),
        ("y'' + omega^2*y = 0", "text"),
    ]
    results = [run(s, fmt=f, independent="t") for s, f in forms]
    dumps = []
    for r in results:
        d = r.model_dump(mode="json")
        for key in ("duration_ms", "diagnostics", "interpretation"):
            d.pop(key)
        dumps.append(json.dumps(d, sort_keys=True))
    assert len(set(dumps)) == 1


# --- failure behaviour -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"equations": ["y' = y"]}, Status.PARSE_ERROR),  # independent variable not named
        ({"equations": [r"y'(t) = y(t-1)"], "independent": "t"}, Status.UNSUPPORTED),
        ({"equations": [r"\partial_t u = u"]}, Status.UNSUPPORTED),
        ({"equations": ["y' + y"], "independent": "t"}, Status.PARSE_ERROR),
        ({"equations": ["y^{(12)} = y"], "independent": "t"}, Status.UNSUPPORTED),
    ],
)
def test_unreadable_input_has_an_explicit_status_and_no_analysis(
    kwargs: dict[str, Any], status: Status
) -> None:
    r = analyze(OdeInput(**kwargs), L)
    assert r.status is status
    assert r.message
    assert r.structure is None and r.methods == []
    assert r.diagnostics.failure_reason == status.value


def test_a_comparison_of_e_warns_that_e_is_a_symbol() -> None:
    r = run(r"y' = e^{-t}", independent="t")
    assert r.interpretation is not None
    assert any("Euler" in n for n in r.interpretation.notes)


# --- the sandboxed entry point and what the diagnostics may contain --------------------------


def test_the_engine_runs_the_analysis_in_the_sandbox(engine: Engine) -> None:
    r = engine.ode_analyze(OdeInput(equations=[r"y'' + y = 0"], independent="t"))
    assert r.status is Status.OK and r.duration_ms >= 0
    assert r.structure is not None and r.structure.order == 2
    assert [s.name for s in r.diagnostics.stages][-1] == "total"


def test_the_engine_reports_a_refusal_with_a_status(engine: Engine) -> None:
    r = engine.ode_analyze(OdeInput(equations=["y' = y"]))
    assert r.status is Status.PARSE_ERROR


def test_diagnostics_hold_no_equation_name_or_value() -> None:
    secret = "Zqmark7919"
    r = run(
        rf"{secret}' = {secret}^2 + kappa_{{4242}}",
        independent="s",
        conditions=[f"{secret}(31337) = 2"],
    )
    dump = r.diagnostics.model_dump_json()
    for fragment in (secret, "4242", "31337", "kappa"):
        assert fragment not in dump
    assert set(r.diagnostics.model_dump()) == {
        "trace_id",
        "stages",
        "n_equations",
        "order",
        "state_dimension",
        "classes",
        "method",
        "methods_tried",
        "steps",
        "evaluations",
        "candidates",
        "verified",
        "dropped",
        "solution_kind",
        "max_residual",
        "failure_reason",
    }  # counters, durations and fixed words only: never an equation, a name or a value
