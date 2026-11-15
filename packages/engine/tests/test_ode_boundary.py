"""Boundary-value problems: agreement with known solutions, checks that reject wrong ones, and every
way a solution must not be shown."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import numpy as np
import pytest
from minerva_engine import Engine, Limits, Status
from minerva_engine.ode import boundary
from minerva_engine.ode.boundary import solve_boundary
from minerva_engine.ode.models import OdeNumericInput, OdeNumericReport

L = Limits(timeout_seconds=60)


def bvp(*equations: str, fmt: str = "latex", **kw: Any) -> OdeNumericReport:
    request = OdeNumericInput(equations=list(equations), format=fmt, kind="bvp", **kw)  # type: ignore[arg-type]
    return solve_boundary(request, L)


def values(report: OdeNumericReport, label: str = "y") -> np.ndarray:
    return np.array(next(s.values for s in report.series if s.label_latex == label))


def checks(report: OdeNumericReport) -> dict[str, str]:
    assert report.verification is not None
    return {c.name: c.status for c in report.verification.checks}


def test_a_linear_problem_matches_its_closed_form() -> None:
    r = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert r.status is Status.OK, r.message
    x = np.array(r.t)
    assert np.max(np.abs(values(r) - np.sin(x) / np.sin(1))) < 1e-6
    assert (r.t_start, r.t_end, r.method_used) == (0, 1, "solve_bvp")
    assert checks(r) == {
        "boundary_conditions": "passed",
        "residual": "passed",
        "mesh_refinement": "passed",
        "shooting": "passed",
    }
    assert r.verification is not None and r.verification.passed
    assert any("nothing is claimed about uniqueness" in n for n in r.notes)
    assert "other solutions may exist" in r.method_reason


def test_a_mixed_condition_problem_matches_its_closed_form() -> None:
    # y'' = y, y(0) = 1, y'(1) = 0  ->  y = cosh(1 - x) / cosh(1)
    r = bvp("y'' = y", independent="x", conditions=["y(0) = 1", "y'(1) = 0"])
    assert r.status is Status.OK, r.message
    x = np.array(r.t)
    assert np.max(np.abs(values(r) - np.cosh(1 - x) / np.cosh(1))) < 1e-6
    assert np.max(np.abs(values(r, "y'") + np.sinh(1 - x) / np.cosh(1))) < 1e-5


def test_conditions_may_be_given_in_either_order_and_the_interval_is_the_span() -> None:
    r = bvp("y'' = y", independent="x", conditions=["y(2) = 0", "y(-1) = 1"])
    assert r.status is Status.OK, r.message
    assert (r.t_start, r.t_end) == (-1, 2)
    assert values(r)[0] == pytest.approx(1) and values(r)[-1] == pytest.approx(0, abs=1e-8)


def test_a_nonlinear_problem_matches_the_published_bratu_value() -> None:
    # y'' + exp(y) = 0, y(0) = y(1) = 0: the lower solution has y(1/2) = 0.140535... (Bratu, lambda = 1)
    r = bvp("y'' + exp(y) = 0", fmt="text", independent="x", conditions=["y(0) = 0", "y(1) = 0"])
    assert r.status is Status.OK, r.message
    middle = values(r)[len(r.t) // 2]
    x = r.t[len(r.t) // 2]
    # compare at x = 1/2 with a Taylor integrator that shares no code with the solver: shoot from the
    # slope that the solution has at 0 (read from the solution itself) and check y(1/2)
    slope = values(r, "y'")[0]
    reference = mp.odefun(lambda t, s: [s[1], -mp.exp(s[0])], 0, [0, slope])
    assert abs(float(reference(x)[0]) - middle) < 1e-6
    assert middle == pytest.approx(0.1405355816, abs=1e-6)


def test_an_airy_boundary_problem_matches_the_airy_functions() -> None:
    # y'' = x y, y(0) = 1, y(2) = 0.1 has the solution a Ai(x) + b Bi(x) fitted to the two values
    r = bvp("y'' = x*y", fmt="text", independent="x", conditions=["y(0) = 1", "y(2) = 0.1"])
    assert r.status is Status.OK, r.message
    system = mp.matrix([[mp.airyai(0), mp.airybi(0)], [mp.airyai(2), mp.airybi(2)]])
    a, b = mp.lu_solve(system, mp.matrix([1, 0.1]))
    x = np.array(r.t)
    exact = np.array([float(a * mp.airyai(t) + b * mp.airybi(t)) for t in x])
    assert np.max(np.abs(values(r) - exact)) < 1e-6


def test_a_system_is_solved_with_conditions_on_each_unknown() -> None:
    # x' = y, y' = -x with x(0) = 0 and x(pi/2) = 1  ->  x = sin t, y = cos t
    r = bvp(
        "x' = y",
        "y' = -x",
        fmt="text",
        independent="t",
        conditions=["x(0) = 0", "x(1.5707963267948966) = 1"],
    )
    assert r.status is Status.OK, r.message
    t = np.array(r.t)
    assert np.max(np.abs(values(r, "x") - np.sin(t))) < 1e-6
    assert np.max(np.abs(values(r, "y") - np.cos(t))) < 1e-6


def test_parameters_are_the_users_and_missing_ones_are_refused() -> None:
    ok = bvp(
        r"y'' = -k^2 y",
        independent="x",
        conditions=["y(0) = 0", "y(1) = 1"],
        parameters={"k": 2.0},
    )
    assert ok.status is Status.OK and ok.parameters == {"k": 2.0}
    assert np.max(np.abs(values(ok) - np.sin(2 * np.array(ok.t)) / np.sin(2))) < 1e-6
    missing = bvp(r"y'' = -k^2 y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert missing.status is Status.PARSE_ERROR and "missing: k" in missing.message
    assert missing.series == []


def test_the_interval_is_not_a_separate_number_to_invent() -> None:
    # t_end is not used: giving one changes nothing
    a = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    b = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"], t_end=99.0)
    assert (a.t_start, a.t_end) == (b.t_start, b.t_end) == (0, 1)


# --- what must not be shown ------------------------------------------------------------------


def test_a_problem_without_a_solution_is_not_shown() -> None:
    # y'' = -y, y(0) = 0, y(pi) = 1 has no solution: every solution of the equation is c sin(x)
    r = bvp(
        "y'' = -y",
        independent="x",
        conditions=["y(0) = 0", "y(3.14159265358979) = 1"],
    )
    assert r.status is Status.NOT_VERIFIED
    assert r.series == [] and r.t == []
    assert "No solution is shown" in r.message


@pytest.mark.parametrize(
    ("conditions", "fragment"),
    [
        ([], "no conditions"),
        (["y(0) = 0"], "incomplete"),
        (["y(0) = 0", "y'(0) = 1"], "initial-value problem"),
        (["y(0) = 0", "y(1) = 1", "y'(1) = 0"], "too many"),
    ],
)
def test_the_conditions_must_make_a_boundary_value_problem(
    conditions: list[str], fragment: str
) -> None:
    r = bvp("y'' = -y", independent="x", conditions=conditions)
    assert (
        r.status is Status.PARSE_ERROR and fragment in r.message and "does not choose" in r.message
    )
    assert r.series == []


def test_more_than_two_points_are_unsupported() -> None:
    r = bvp(
        "y''' = -y",
        independent="x",
        conditions=["y(0) = 0", "y(1) = 1", "y(2) = 0"],
    )
    assert r.status is Status.UNSUPPORTED and "two points" in r.message


def test_a_wrong_solver_is_caught_by_the_residual_and_boundary_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = boundary.solve_bvp

    class Wrong:
        """Claims success with an interpolant that is off by a little."""

        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.success = True
            self.x = inner.x

        def sol(self, x: Any, nu: int = 0) -> Any:
            out = self.inner.sol(x, nu)
            return out * (1 + 1e-3) if nu == 0 else out

    monkeypatch.setattr(boundary, "solve_bvp", lambda *a, **k: Wrong(real(*a, **k)))
    r = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert r.status is Status.NOT_VERIFIED and r.series == []
    assert r.verification is not None and not r.verification.passed
    assert "failed" in checks(r).values()


def test_a_solver_that_never_converges_is_reported_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failed:
        success = False

    monkeypatch.setattr(boundary, "solve_bvp", lambda *a, **k: Failed())
    r = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert r.status is Status.NOT_VERIFIED and "did not converge" in r.message
    assert r.warnings == ["not_converged"]
    assert r.diagnostics.methods_tried == ["zero", "linear between the boundary values", "one"]


def test_the_equation_must_be_solvable_for_its_highest_derivative() -> None:
    r = bvp("(y')^2 = y", independent="x", conditions=["y(0) = 1", "y(1) = 2"])
    assert r.status is Status.UNSUPPORTED and "highest derivative" in r.message


def test_the_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(boundary, "MAX_EVALUATIONS", 10)
    r = bvp("y'' = -y", independent="x", conditions=["y(0) = 0", "y(1) = 1"])
    assert r.status is Status.RESOURCE_LIMIT and r.series == []


def test_diagnostics_hold_only_counts_and_words() -> None:
    secret = "Zqmark7919"
    r = bvp(
        f"{secret}'' = -kappa_4242*{secret}",
        fmt="text",
        independent="s",
        conditions=[f"{secret}(31337) = 0", f"{secret}(31338) = 2"],
        parameters={"kappa_4242": 0.7},
    )
    assert r.status is Status.OK, r.message
    dump = r.diagnostics.model_dump_json()
    for fragment in (secret, "4242", "31337", "kappa"):
        assert fragment not in dump
    assert r.diagnostics.method == "bvp" and r.diagnostics.steps


def test_the_engine_runs_a_boundary_problem_in_the_sandbox(engine: Engine) -> None:
    request = OdeNumericInput(
        equations=["y'' = -y"],
        independent="x",
        kind="bvp",
        conditions=["y(0) = 0", "y(1) = 1"],
    )
    r = engine.ode_solve_numerical(request)
    assert r.status is Status.OK and r.method_used == "solve_bvp"
    assert engine.ode_solve_numerical(request.model_copy(update={"conditions": []})).status is (
        Status.PARSE_ERROR
    )
