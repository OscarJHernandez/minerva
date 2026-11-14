"""Numerical initial-value problems: agreement with references that are not SciPy, convergence under
tolerance refinement, conservation, and every way a trajectory must NOT be shown."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mpmath as mp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from minerva_engine import Engine, Limits, OdeInput, Status
from minerva_engine.ode import numerical
from minerva_engine.ode.analyze import solve_highest
from minerva_engine.ode.models import OdeNumericInput, OdeNumericReport
from minerva_engine.ode.numerical import build_system, solve_numerical
from minerva_engine.ode.reading import read_ode

L = Limits(timeout_seconds=60)


def run(*equations: str, t_end: float, fmt: str = "latex", **kw: Any) -> OdeNumericReport:
    request = OdeNumericInput(equations=list(equations), format=fmt, t_end=t_end, **kw)  # type: ignore[arg-type]
    return solve_numerical(request, L)


def column(report: OdeNumericReport, label: str) -> np.ndarray:
    return np.array(next(s.values for s in report.series if s.label_latex == label))


def checks(report: OdeNumericReport) -> dict[str, str]:
    assert report.verification is not None
    return {c.name: c.status for c in report.verification.checks}


# --- agreement with exact and independently computed solutions -------------------------------


def test_the_harmonic_oscillator_matches_the_exact_solution() -> None:
    r = run(
        r"y'' + \omega^2 y = 0",
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
        parameters={"omega": 2.0},
        t_end=10,
    )
    assert r.status is Status.OK, r.message
    t = np.array(r.t)
    assert np.max(np.abs(column(r, "y") - np.cos(2 * t))) < 1e-6
    assert np.max(np.abs(column(r, "y'") + 2 * np.sin(2 * t))) < 1e-6
    assert r.t[0] == 0 and r.t[-1] == 10
    assert r.verification is not None and r.verification.passed
    assert checks(r) == {
        "initial_conditions": "passed",
        "independent_method": "passed",
        "tolerance_refinement": "passed",
        "defect": "passed",
        "first_integral": "passed",  # y'' = -w^2 y has the verified energy
    }


def test_the_logistic_equation_matches_its_closed_form() -> None:
    r = run(r"y' = y \cdot (1 - y)", independent="t", conditions=["y(0) = 1/10"], t_end=12)
    assert r.status is Status.OK, r.message
    t = np.array(r.t)
    exact = 1 / (1 + 9 * np.exp(-t))
    assert np.max(np.abs(column(r, "y") - exact)) < 1e-6


def test_time_can_run_backwards() -> None:
    r = run("y' = y", independent="t", conditions=["y(0) = 1"], t_end=-2)
    assert r.status is Status.OK, r.message
    assert r.t[0] == 0 and r.t[-1] == -2
    assert abs(column(r, "y")[-1] - np.exp(-2)) < 1e-7


def test_the_starting_point_is_the_conditions_point() -> None:
    r = run("y' = y", independent="t", conditions=["y(3) = 2"], t_end=4)
    assert r.status is Status.OK and r.t_start == 3
    assert abs(column(r, "y")[-1] - 2 * np.exp(1)) < 1e-6


def test_a_nonlinear_pendulum_agrees_with_a_taylor_integrator() -> None:
    r = run(
        r"\theta'' + \sin\theta = 0",
        independent="t",
        conditions=[r"\theta(0) = 1.5", r"\theta'(0) = 0"],
        t_end=15,
        rtol=1e-10,
        atol=1e-12,
    )
    assert r.status is Status.OK, r.message
    reference = mp.odefun(lambda t, s: [s[1], -mp.sin(s[0])], 0, [mp.mpf("1.5"), 0])
    t = np.array(r.t)
    for index in (50, 120, 199):
        assert abs(column(r, r"\theta")[index] - float(reference(t[index])[0])) < 1e-8
    assert checks(r)["first_integral"] == "passed"


def test_a_stiff_chemical_system_reproduces_published_values() -> None:
    # Robertson's problem; the reference values at t = 40 are the standard ones (about 7 digits).
    r = run(
        r"\dot{a} = -0.04 a + 10^4 b c",
        r"\dot{b} = 0.04 a - 10^4 b c - 3 \cdot 10^7 b^2",
        r"\dot{c} = 3 \cdot 10^7 b^2",
        conditions=["a(0) = 1", "b(0) = 0", "c(0) = 0"],
        t_end=40,
    )
    assert r.status is Status.OK, r.message
    assert r.method_used == "Radau" and "stiffness_indicated" in r.warnings
    assert "measured" in r.method_reason
    a, b, c = (column(r, n)[-1] for n in "abc")
    assert abs(a - 0.7158270687) < 1e-6 and abs(c - 0.2841637457) < 1e-6
    assert abs(b - 9.185534764e-6) < 1e-9
    assert abs(a + b + c - 1) < 1e-9  # mass is conserved


def test_a_stiff_linear_problem_has_an_exact_solution() -> None:
    # y' = -1000 (y - cos t) - sin t with y(0) = 1 has the exact solution cos t.
    r = run(
        r"y' = -1000 \cdot (y - \cos t) - \sin t",
        independent="t",
        conditions=["y(0) = 1"],
        t_end=5,
    )
    assert r.status is Status.OK, r.message
    assert np.max(np.abs(column(r, "y") - np.cos(np.array(r.t)))) < 1e-6


def test_a_coupled_system_conserves_the_quantity_it_should() -> None:
    # Lotka-Volterra: V = d x - c ln x + b y - a ln y is constant along solutions.
    a, b, c, d = 1.0, 0.5, 0.75, 0.25
    r = run(
        r"\dot{x} = a x - b x y",
        r"\dot{y} = -c y + d x y",
        conditions=["x(0) = 2", "y(0) = 1"],
        parameters={"a": a, "b": b, "c": c, "d": d},
        t_end=30,
        rtol=1e-10,
        atol=1e-12,
    )
    assert r.status is Status.OK, r.message
    x, y = column(r, "x"), column(r, "y")
    v = d * x - c * np.log(x) + b * y - a * np.log(y)
    assert np.max(np.abs(v - v[0])) < 1e-7


# --- convergence and metamorphic properties -------------------------------------------------


def test_the_error_falls_as_the_tolerance_is_refined() -> None:
    errors = []
    for rtol in (1e-6, 1e-8, 1e-10):
        r = run(
            r"y'' + y = 0",
            independent="t",
            conditions=["y(0) = 1", "y'(0) = 0"],
            t_end=20,
            rtol=rtol,
            atol=rtol / 100,
        )
        assert r.status is Status.OK, (rtol, r.message)
        errors.append(float(np.max(np.abs(column(r, "y") - np.cos(np.array(r.t))))))
    assert errors[0] > errors[1] > errors[2] or errors[2] < 1e-9
    assert errors[2] < errors[0] / 10


def test_a_higher_order_equation_and_its_first_order_system_give_one_trajectory() -> None:
    higher = run(
        r"y'' + 0.3 y' + y + 0.1 y^3 = 0",
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0.5"],
        t_end=15,
    )
    system = run(
        r"y' = v",
        r"v' = -0.3 v - y - 0.1 y^3",
        independent="t",
        conditions=["y(0) = 1", "v(0) = 0.5"],
        t_end=15,
    )
    assert higher.status is Status.OK and system.status is Status.OK
    assert np.max(np.abs(column(higher, "y") - column(system, "y"))) < 1e-6
    assert np.max(np.abs(column(higher, "y'") - column(system, "v"))) < 1e-6


def test_integrating_forwards_and_then_backwards_returns_to_the_start() -> None:
    forward = run(
        r"y'' + y + 0.5 y^3 = 0",
        independent="t",
        conditions=["y(0) = 0.8", "y'(0) = 0.3"],
        t_end=6,
        rtol=1e-10,
        atol=1e-12,
    )
    assert forward.status is Status.OK
    y_end, v_end = column(forward, "y")[-1], column(forward, "y'")[-1]
    back = run(
        r"y'' + y + 0.5 y^3 = 0",
        independent="t",
        conditions=[f"y(6) = {y_end:.15f}", f"y'(6) = {v_end:.15f}"],
        t_end=0,
        rtol=1e-10,
        atol=1e-12,
    )
    assert back.status is Status.OK, back.message
    assert abs(column(back, "y")[-1] - 0.8) < 1e-7 and abs(column(back, "y'")[-1] - 0.3) < 1e-7


def test_rescaling_time_rescales_the_trajectory() -> None:
    base = run("y'' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=6)
    fast = run("z'' + 4 z = 0", independent="s", conditions=["z(0) = 1", "z'(0) = 0"], t_end=3)
    # z(s) = y(2 s): the same samples, half the time.
    assert np.max(np.abs(column(base, "y") - column(fast, "z"))) < 1e-6


def test_the_scaled_unknown_scales_a_linear_trajectory() -> None:
    one = run(
        "y'' + 3 y' + 2 y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=4
    )
    five = run(
        "y'' + 3 y' + 2 y = 0", independent="t", conditions=["y(0) = 5", "y'(0) = 0"], t_end=4
    )
    assert np.max(np.abs(5 * column(one, "y") - column(five, "y"))) < 1e-6


def test_every_method_gives_the_same_answer_when_it_is_accepted() -> None:
    finals = {}
    for method in ("RK45", "DOP853", "Radau", "BDF", "LSODA"):
        r = run(
            r"y'' + 0.3 y' + y = 0",
            independent="t",
            conditions=["y(0) = 1", "y'(0) = 0"],
            t_end=10,
            method=method,
            rtol=1e-9,
            atol=1e-11,
        )
        assert r.status is Status.OK, (method, r.message)
        assert r.method_used == method and r.method_reason == "chosen by the user"
        finals[method] = column(r, "y")[-1]
    assert max(finals.values()) - min(finals.values()) < 1e-6


# --- the user supplies every number ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"equations": ["y' = a y"], "conditions": ["y(0) = 1"]}, "missing: a"),
        (
            {"equations": ["y' = a y"], "conditions": ["y(0) = 1"], "parameters": {"a": 1, "b": 2}},
            "Not parameters",
        ),
        ({"equations": ["y' = y"], "conditions": []}, "no initial values"),
        ({"equations": ["y'' = -y"], "conditions": ["y(0) = 1"]}, "incomplete"),
        (
            {"equations": ["y'' = -y"], "conditions": ["y(0) = 0", "y(1) = 1"]},
            "boundary-value",
        ),
        (
            {"equations": ["y' = y"], "conditions": ["y(0) = 1", "y(0) = 2"]},
            "too many",
        ),
        ({"equations": ["y' = a y"], "conditions": ["y(0) = y_0"], "parameters": {"a": 1}}, "y_0"),
    ],
)
def test_missing_or_invented_numbers_are_refused(kwargs: dict[str, Any], fragment: str) -> None:
    r = solve_numerical(OdeNumericInput(**{"independent": "t", "t_end": 1.0, **kwargs}), L)
    assert r.status is Status.PARSE_ERROR and fragment in r.message
    assert r.t == [] and r.series == []


def test_a_value_that_breaks_a_declared_assumption_is_refused() -> None:
    r = run(
        "y' = -k y",
        independent="t",
        conditions=["y(0) = 1"],
        parameters={"k": -1.0},
        assumptions={"k": ["positive"]},
        t_end=1,
    )
    assert r.status is Status.PARSE_ERROR and "positive" in r.message


def test_an_empty_interval_is_refused() -> None:
    r = run("y' = y", independent="t", conditions=["y(0) = 1"], t_end=0)
    assert r.status is Status.PARSE_ERROR and "empty" in r.message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_end": float("nan")},
        {"t_end": float("inf")},
        {"t_end": 1.0, "rtol": 0.0},
        {"t_end": 1.0, "rtol": 0.5},
        {"t_end": 1.0, "samples": 5},
        {"t_end": 1.0, "samples": 100000},
        {"t_end": 1.0, "method": "Euler"},
    ],
)
def test_out_of_range_settings_are_rejected_by_the_model(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        OdeNumericInput(equations=["y' = y"], independent="t", conditions=["y(0) = 1"], **kwargs)


def test_equations_that_cannot_be_put_in_first_order_form_are_unsupported() -> None:
    r = run(r"(y')^2 = y", independent="t", conditions=["y(0) = 1"], t_end=1)
    assert r.status is Status.UNSUPPORTED and "branch" in r.message
    r = run(
        "x' = y", "y = x", independent="t", unknowns=["x", "y"], conditions=["x(0) = 1"], t_end=1
    )
    assert r.status is Status.UNSUPPORTED
    names = [f"x{i}" for i in range(7)]
    big = [f"{n}'' = -{n}" for n in names]
    r = run(
        *big,
        fmt="text",
        independent="t",
        conditions=[f"{n}(0) = 1" for n in names] + [f"{n}'(0) = 0" for n in names],
        t_end=1,
    )
    assert r.status is Status.UNSUPPORTED and "states" in r.message


# --- trajectories that must not be shown ----------------------------------------------------


def test_a_finite_time_blow_up_shows_no_trajectory() -> None:
    r = run("y' = y^2", independent="t", conditions=["y(0) = 1"], t_end=2)
    assert r.status in (Status.NOT_VERIFIED, Status.RESOURCE_LIMIT)
    assert r.series == [] and r.t == []
    assert set(r.warnings) & {
        "step_collapse",
        "blow_up_suspected",
        "domain_left",
        "stiffness_indicated",
    }
    assert "no trajectory is shown" in r.message


def test_a_singularity_inside_the_interval_shows_no_trajectory() -> None:
    r = run(r"y' = \frac{1}{t - 1}", independent="t", conditions=["y(0) = 0"], t_end=2)
    assert r.status is not Status.OK and r.series == []


def test_leaving_the_real_domain_shows_no_trajectory() -> None:
    r = run(r"y' = \sqrt{y}", independent="t", conditions=["y(0) = 1"], t_end=1)
    assert r.status is Status.OK  # y' = sqrt(y) stays positive: a control
    r = run(r"y' = -\sqrt{y}", independent="t", conditions=["y(0) = 1"], t_end=5)
    assert r.status is not Status.OK and r.series == []  # y reaches 0 at t = 2 and cannot continue


def test_an_explicit_method_on_a_stiff_problem_is_flagged_or_stopped() -> None:
    r = run(
        r"y' = -2000 \cdot (y - \cos t)",
        independent="t",
        conditions=["y(0) = 0"],
        t_end=5,
        method="RK45",
    )
    if r.status is Status.OK:
        assert "stiffness_indicated" in r.warnings
    else:
        assert r.status in (Status.RESOURCE_LIMIT, Status.NOT_VERIFIED) and r.series == []


def test_the_evaluation_budget_is_enforced() -> None:
    r = run(
        r"y' = -1000000 \cdot (y - \cos t)",
        independent="t",
        conditions=["y(0) = 0"],
        t_end=50,
        method="RK45",
    )
    assert r.status is Status.RESOURCE_LIMIT and r.series == []
    assert "stiffness_indicated" in r.warnings


def test_the_automatic_choice_does_not_hide_a_failure() -> None:
    r = run("y' = y^2", independent="t", conditions=["y(0) = 1"], t_end=2, method="auto")
    assert r.status is not Status.OK and r.series == []
    assert r.diagnostics.methods_tried[:1] == ["DOP853"]


# --- the checks reject a wrong trajectory ----------------------------------------------------


def test_a_solver_that_is_wrong_by_one_part_in_ten_thousand_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = numerical.solve_ivp

    def wrong(*args: Any, **kwargs: Any) -> Any:
        sol = real(*args, **kwargs)
        sol.y = sol.y * (1 + 1e-4)
        return sol

    monkeypatch.setattr(numerical, "solve_ivp", wrong)
    r = run("y'' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=10)
    assert r.status is Status.NOT_VERIFIED and r.series == [] and r.t == []
    assert r.verification is not None and not r.verification.passed
    states = checks(r)
    assert states["independent_method"] == "failed" or states["initial_conditions"] == "failed"


def test_swapped_initial_values_are_caught_by_the_initial_condition_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = numerical.solve_ivp

    def swapped(fun: Any, span: Any, y0: Any, **kwargs: Any) -> Any:
        return real(fun, span, np.asarray(y0)[::-1].copy(), **kwargs)

    monkeypatch.setattr(numerical, "solve_ivp", swapped)
    r = run("y'' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=3)
    assert r.status is Status.NOT_VERIFIED and r.series == []
    assert checks(r)["initial_conditions"] == "failed"


def test_a_drifting_energy_fails_the_first_integral_check() -> None:
    problem = read_ode(OdeInput(equations=["y'' + y = 0"], independent="t"), L)
    solved, _ = solve_highest(problem)
    assert solved is not None
    system = build_system(problem, solved.rhs, {}, {("y", 0): 1.0, ("y", 1): 0.0}, 0.0, 10.0)
    t = np.linspace(0, 10, 100)
    exact = SimpleNamespace(sol=SimpleNamespace(y=np.array([np.cos(t), -np.sin(t)])))
    drifting = SimpleNamespace(
        sol=SimpleNamespace(y=np.array([np.cos(t) * (1 + 1e-3 * t), -np.sin(t)]))
    )
    good, latex = numerical._check_first_integral(system, exact, 1e-8)  # type: ignore[arg-type]
    bad, _ = numerical._check_first_integral(system, drifting, 1e-8)  # type: ignore[arg-type]
    assert good.status == "passed" and latex is not None
    assert bad.status == "failed" and (bad.value or 0) > 1e-4


def test_an_unverified_first_integral_is_not_used() -> None:
    # A velocity-dependent acceleration has no energy of this form: the check is skipped, not faked.
    r = run("y'' + 0.2 y' + y = 0", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=5)
    assert r.status is Status.OK and checks(r)["first_integral"] == "skipped"
    forced = run(
        "y'' + y = \\sin t", independent="t", conditions=["y(0) = 1", "y'(0) = 0"], t_end=5
    )
    assert forced.status is Status.OK and checks(forced)["first_integral"] == "skipped"


# --- limits, sizes and what is reported -------------------------------------------------------


def test_the_output_size_is_bounded() -> None:
    names = [f"x{i}" for i in range(6)]
    r = run(
        *[f"{n}'' = -{i + 1}*{n}" for i, n in enumerate(names)],
        fmt="text",
        independent="t",
        conditions=[f"{n}(0) = 1" for n in names] + [f"{n}'(0) = 0" for n in names],
        t_end=2,
        samples=400,
    )
    assert r.status is Status.OK, r.message
    assert sum(len(s.values) for s in r.series) <= numerical.MAX_NUMBERS
    assert len(r.t) == len(r.series[0].values) and len(r.series) == 12


def test_the_report_states_what_was_used() -> None:
    r = run(
        r"y'' + c y' + y = 0",
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
        parameters={"c": 0.5},
        t_end=5,
        rtol=1e-9,
        atol=1e-11,
    )
    assert r.status is Status.OK
    assert r.parameters == {"c": 0.5} and r.rtol == 1e-9 and r.atol == 1e-11
    assert [(i.label_latex, i.value) for i in r.initial] == [("y", 1.0), ("y'", 0.0)]
    assert r.steps and r.evaluations
    assert any("approximation" in n for n in r.notes)
    assert r.interpretation is not None and r.interpretation.independent == "t"


def test_diagnostics_hold_only_counts_and_words() -> None:
    secret = "Zqmark7919"
    r = run(
        f"{secret}' = -kappa_4242*{secret}",
        fmt="text",
        independent="s",
        conditions=[f"{secret}(31337) = 2"],
        parameters={"kappa_4242": 0.7},
        t_end=31340,
    )
    assert r.status is Status.OK, r.message
    dump = r.diagnostics.model_dump_json()
    for fragment in (secret, "4242", "31337", "kappa", "0.7"):
        assert fragment not in dump


# --- property: random linear systems against the matrix exponential --------------------------


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    entries=st.lists(st.integers(-3, 3), min_size=4, max_size=4),
    start=st.lists(st.integers(-3, 3), min_size=2, max_size=2),
    t_end=st.floats(0.5, 3.0, allow_nan=False),
)
def test_linear_systems_agree_with_the_matrix_exponential(
    entries: list[int], start: list[int], t_end: float
) -> None:
    a, b, c, d = entries
    r = run(
        f"x' = ({a})*x + ({b})*y",
        f"y' = ({c})*x + ({d})*y",
        fmt="text",
        independent="t",
        conditions=[f"x(0) = {start[0]}", f"y(0) = {start[1]}"],
        t_end=round(t_end, 3),
        rtol=1e-10,
        atol=1e-12,
    )
    assert r.status is Status.OK, r.message
    final = mp.expm(mp.matrix([[a, b], [c, d]]) * mp.mpf(r.t[-1])) * mp.matrix(start)
    scale = max(1.0, abs(float(final[0])), abs(float(final[1])))
    assert abs(column(r, "x")[-1] - float(final[0])) < 1e-7 * scale
    assert abs(column(r, "y")[-1] - float(final[1])) < 1e-7 * scale


# --- through the sandbox --------------------------------------------------------------------


def test_the_engine_runs_the_solver_in_the_sandbox(engine: Engine) -> None:
    request = OdeNumericInput(
        equations=["y'' + y = 0"],
        independent="t",
        conditions=["y(0) = 1", "y'(0) = 0"],
        t_end=5,
    )
    r = engine.ode_solve_numerical(request)
    assert r.status is Status.OK and r.duration_ms >= 0
    assert r.diagnostics.method == "DOP853" and r.diagnostics.steps
    refused = engine.ode_solve_numerical(request.model_copy(update={"conditions": []}))
    assert refused.status is Status.PARSE_ERROR
