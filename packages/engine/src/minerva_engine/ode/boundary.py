"""Two-point boundary-value problems, shown only when independent checks pass.

SciPy's `solve_bvp` proposes a solution from a deterministic ladder of starting guesses. It is shown only if

* every condition holds at its end (measured from the solution, not from the solver's report);
* the differential equation holds on a fine grid (the residual of the solution's own derivative against
  the equation, which the solver only controls in the root-mean-square);
* a solve on a refined mesh gives the same answer;
* integrating forward from the left end with a different integrator (Minerva's own Dormand-Prince)
  reproduces the solution at the right end, to within what the equation's sensitivity allows.

A boundary-value problem may have no solution, one, or several: what is shown is *a* solution, found
from the starting guess named in the report, and nothing is claimed about the others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.integrate import solve_bvp

from ..errors import MinervaMathError
from ..models import Limits, Status
from ..motion import _dopri
from ..motion import _Stop as DopriStop
from .analyze import _classify_conditions, interpretation, solve_highest
from .models import (
    InitialValue,
    NumericCheck,
    NumericVerification,
    NumericWarning,
    OdeDiagnostics,
    OdeNumericInput,
    OdeNumericReport,
    OdeSeries,
    OdeStage,
)
from .numerical import (
    MAX_NUMBERS,
    MAX_STATE,
    System,
    _number,
    _round,
    build_system,
    parameter_problem,
)
from .reading import _display, read_ode
from .verification import Unevaluable

START_NODES = 25
MAX_NODES = 20_000
MAX_EVALUATIONS = 5_000_000
RESIDUAL_FACTOR = 100
SHOOTING_LIMIT_GROWTH = 1e10


class _Halt(Exception):
    def __init__(self, reason: Literal["budget", "time", "domain"], message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass
class Boundary:
    index: int  # the state the condition is about
    left: bool  # at the left end of the interval
    value: float
    label: str


def _vector_field(system: System, stop_at: float, counter: list[int]) -> Any:
    def fun(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        counter[0] += int(x.size)
        if counter[0] > MAX_EVALUATIONS:
            raise _Halt("budget", "too many evaluations of the equation were needed")
        if time.monotonic() > stop_at:
            raise _Halt("time", "the solve used up its time budget")
        with np.errstate(all="ignore"):
            try:
                out = system.f(x, *y)
                result = np.array(
                    [np.broadcast_to(np.asarray(o, dtype=float), x.shape) for o in out]
                )
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                raise _Halt("domain", "the right-hand side left its real domain") from exc
        if not np.all(np.isfinite(result)):
            raise _Halt("domain", "the right-hand side became infinite or undefined")
        return result

    return fun


def _guesses(n: int, nodes: np.ndarray, boundaries: list[Boundary]) -> list[tuple[str, np.ndarray]]:
    shape = (n, nodes.size)
    left = {b.index: b.value for b in boundaries if b.left}
    right = {b.index: b.value for b in boundaries if not b.left}
    linear = np.zeros(shape)
    t = (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    for i in range(n):
        if i in left and i in right:
            linear[i] = left[i] + (right[i] - left[i]) * t
        elif i in left or i in right:
            linear[i] = left.get(i, right.get(i, 0.0))
    return [
        ("zero", np.zeros(shape)),
        ("linear between the boundary values", linear),
        ("one", np.ones(shape)),
    ]


def _scaled_difference(a: np.ndarray, b: np.ndarray) -> float:
    worst = 0.0
    for row_a, row_b in zip(a, b, strict=True):
        scale = max(float(np.max(np.abs(row_a))), 1e-9)
        worst = max(worst, float(np.max(np.abs(row_a - row_b))) / scale)
    return worst


def _boundary_check(
    sol: Any, boundaries: list[Boundary], a: float, b: float, rtol: float
) -> NumericCheck:
    worst = 0.0
    ya, yb = sol.sol(a), sol.sol(b)
    for c in boundaries:
        got = (ya if c.left else yb)[c.index]
        worst = max(worst, abs(float(got) - c.value) / max(1.0, abs(c.value)))
    limit = max(1e-9, RESIDUAL_FACTOR * rtol)
    return NumericCheck(
        name="boundary_conditions",
        status="passed" if worst <= limit else "failed",
        value=worst,
        tolerance=limit,
        note="measured on the solution's own interpolant at both ends",
    )


def _residual_check(sol: Any, fun: Any, a: float, b: float, rtol: float) -> NumericCheck:
    count = min(5000, 10 * sol.x.size + 1)
    xs = np.linspace(a, b, count)
    y = sol.sol(xs)
    slope = sol.sol(xs, 1)
    try:
        field = fun(xs, y)
    except _Halt:
        return NumericCheck(
            name="residual", status="skipped", note="could not evaluate the equation"
        )
    worst = float(np.max(np.abs(slope - field) / (1.0 + np.abs(field))))
    limit = max(1e-8, RESIDUAL_FACTOR * rtol)
    return NumericCheck(
        name="residual",
        status="passed" if np.isfinite(worst) and worst <= limit else "failed",
        value=worst,
        tolerance=limit,
        note=f"largest relative residual of the equation on {count} points",
    )


def _refinement_check(
    sol: Any, fun: Any, bc: Any, a: float, b: float, rtol: float, evaluation: np.ndarray
) -> NumericCheck:
    nodes = np.sort(np.concatenate([sol.x, (sol.x[:-1] + sol.x[1:]) / 2]))
    try:
        fine = solve_bvp(
            fun, bc, nodes, sol.sol(nodes), tol=rtol / 10, bc_tol=rtol / 10, max_nodes=MAX_NODES
        )
    except _Halt as exc:
        return NumericCheck(name="mesh_refinement", status="skipped", note=exc.message)
    if not fine.success:
        return NumericCheck(
            name="mesh_refinement",
            status="failed",
            note="the solve on the refined mesh did not converge",
        )
    difference = _scaled_difference(sol.sol(evaluation), fine.sol(evaluation))
    limit = max(1e-9, 1e3 * rtol)
    return NumericCheck(
        name="mesh_refinement",
        status="passed" if difference <= limit else "failed",
        value=difference,
        tolerance=limit,
        note="the change when the mesh is refined and the tolerance tightened",
    )


def _shooting_check(
    system: System, sol: Any, a: float, b: float, rtol: float, stop_at: float
) -> NumericCheck:
    start = np.asarray(sol.sol(a), dtype=float)
    times = [float(x) for x in np.linspace(a, b, 21)]
    try:
        states, _, _ = _dopri(system.f_math, list(start), times, stop_at, 150_000)
        nudge = start * (1 + 1e-8) + 1e-8
        other, _, _ = _dopri(system.f_math, list(nudge), times, stop_at, 150_000)
    except DopriStop as exc:
        return NumericCheck(name="shooting", status="skipped", note=exc.message)
    final = np.array(states[-1])
    growth = float(
        np.linalg.norm(np.array(other[-1]) - final) / max(np.linalg.norm(nudge - start), 1e-30)
    )
    if growth > SHOOTING_LIMIT_GROWTH:
        return NumericCheck(
            name="shooting",
            status="skipped",
            note="the equation amplifies errors too much for shooting to be informative",
        )
    reached = np.asarray(sol.sol(b), dtype=float)
    scale = max(float(np.max(np.abs(reached))), 1e-9)
    difference = float(np.max(np.abs(final - reached))) / scale
    limit = max(1e-6, 1e3 * rtol) * max(1.0, growth)
    return NumericCheck(
        name="shooting",
        status="passed" if difference <= limit else "failed",
        value=difference,
        tolerance=limit,
        note=f"integrating from the left end with Minerva's Dormand-Prince; the equation amplifies errors by {growth:.1e}",
    )


def solve_boundary(
    data: OdeNumericInput, limits: Limits, trace_id: str | None = None
) -> OdeNumericReport:
    started = time.perf_counter()
    stop_at = time.monotonic() + 0.6 * limits.timeout_seconds
    stages: list[OdeStage] = []
    diagnostics = OdeDiagnostics(trace_id=trace_id, method="bvp")

    def finish(report: OdeNumericReport) -> OdeNumericReport:
        stages.append(OdeStage(name="total", ms=round((time.perf_counter() - started) * 1000, 2)))
        diagnostics.stages = stages
        report.diagnostics = diagnostics
        return report

    def refuse(status: Status, message: str, reason: str, **extra: Any) -> OdeNumericReport:
        diagnostics.failure_reason = reason
        return finish(
            OdeNumericReport(status=status, message=message, method_requested=data.method, **extra)
        )

    try:
        problem = read_ode(data, limits)
    except MinervaMathError as exc:
        from ..ops import _failure

        status, message = _failure(exc)
        return refuse(status, message, status.value)
    stages.append(OdeStage(name="read", ms=round((time.perf_counter() - started) * 1000, 2)))
    meaning = interpretation(problem)
    diagnostics.order = max(problem.orders.values())
    diagnostics.state_dimension = sum(problem.orders.values())

    def stop(status: Status, message: str, reason: str, **extra: Any) -> OdeNumericReport:
        return refuse(status, message, reason, interpretation=meaning, **extra)

    solved, why = solve_highest(problem)
    if solved is None:
        return stop(
            Status.UNSUPPORTED, f"Not solved for the highest derivative: {why}.", "not_solved"
        )
    if sum(problem.orders.values()) > MAX_STATE:
        return stop(Status.UNSUPPORTED, f"At most {MAX_STATE} states are supported.", "too_large")
    kind, needed = _classify_conditions(problem)
    if kind != "boundary_value":
        text = {
            "general": "no conditions were given",
            "incomplete": "the conditions are incomplete",
            "overdetermined": "too many conditions were given",
            "initial_value": "all conditions are at one point: that is an initial-value problem",
        }[kind]
        return stop(
            Status.PARSE_ERROR,
            f"A boundary-value problem needs as many conditions as the order, at two points: {text}"
            + (f" (needed: {', '.join(needed)})" if needed else "")
            + ". Minerva does not choose them.",
            "conditions",
        )
    problem_with = parameter_problem(problem, data.parameters)
    if problem_with:
        return stop(Status.PARSE_ERROR, problem_with, "parameters")
    values = dict(data.parameters)
    points: list[float] = []
    for c in problem.conditions:
        point = _number(c.at, values)
        if point is None:
            return stop(
                Status.PARSE_ERROR, "The points of the conditions must be real numbers.", "interval"
            )
        if not any(abs(point - p) <= 1e-12 * max(1.0, abs(point)) for p in points):
            points.append(point)
    if len(points) != 2:
        return stop(
            Status.UNSUPPORTED,
            "Conditions at more than two points (multipoint problems) are not supported.",
            "multipoint",
        )
    a, b = sorted(points)
    labels = [(u, k) for u in problem.unknowns for k in range(problem.orders[u])]
    boundaries: list[Boundary] = []
    for c in problem.conditions:
        number = _number(c.value, values)
        point = _number(c.at, values)
        if number is None or point is None:
            return stop(
                Status.PARSE_ERROR,
                "The values of the conditions must be real numbers.",
                "conditions",
            )
        boundaries.append(
            Boundary(
                labels.index((c.unknown, c.order)),
                abs(point - a) <= abs(point - b),
                number,
                f"{_display(c.unknown, c.order)}({point:g})",
            )
        )
    try:
        system = build_system(problem, solved.rhs, values, dict.fromkeys(labels, 0.0), a, b)
    except Unevaluable:
        return stop(
            Status.UNSUPPORTED,
            "The equation uses a function that cannot be evaluated numerically here.",
            "unevaluable",
        )

    counter = [0]
    fun = _vector_field(system, stop_at, counter)

    def bc(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([(ya if c.left else yb)[c.index] - c.value for c in boundaries])

    nodes = np.linspace(a, b, START_NODES)
    sol: Any = None
    used = ""
    attempts: list[str] = []
    warnings: list[NumericWarning] = []
    try:
        for name, guess in _guesses(len(labels), nodes, boundaries):
            attempts.append(name)
            with np.errstate(all="ignore"):
                candidate = solve_bvp(
                    fun, bc, nodes, guess, tol=data.rtol, bc_tol=data.rtol, max_nodes=MAX_NODES
                )
            if candidate.success:
                sol, used = candidate, name
                break
    except _Halt as halt:
        status = Status.RESOURCE_LIMIT if halt.reason in ("budget", "time") else Status.NOT_VERIFIED
        diagnostics.methods_tried = attempts
        return stop(
            status,
            f"The solve did not complete ({halt.message}); no solution is shown.",
            halt.reason,
            t_start=a,
            t_end=b,
        )
    diagnostics.methods_tried = attempts
    if sol is None:
        warnings.append("not_converged")
        return stop(
            Status.NOT_VERIFIED,
            "The solver did not converge from any of the starting guesses ("
            + ", ".join(attempts)
            + "): there may be no solution, or it may need a better guess. No solution is shown.",
            "not_converged",
            warnings=warnings,
            t_start=a,
            t_end=b,
        )
    diagnostics.steps = int(sol.x.size)
    diagnostics.evaluations = counter[0]
    samples = max(20, min(data.samples, MAX_NUMBERS // (len(labels) + 1)))
    grid = np.linspace(a, b, samples)
    checks = [
        _boundary_check(sol, boundaries, a, b, data.rtol),
        _residual_check(sol, fun, a, b, data.rtol),
    ]
    try:
        checks.append(_refinement_check(sol, fun, bc, a, b, data.rtol, grid))
    except _Halt as halt:
        checks.append(NumericCheck(name="mesh_refinement", status="skipped", note=halt.message))
    checks.append(_shooting_check(system, sol, a, b, data.rtol, stop_at))
    required = ("boundary_conditions", "residual", "mesh_refinement")
    failed = [c for c in checks if c.status == "failed"]
    unrun = [c for c in checks if c.name in required and c.status == "skipped"]
    verification = NumericVerification(
        checks=checks,
        passed=not failed and not unrun,
        independent_method="Minerva's Dormand-Prince 5(4) (shooting)"
        if any(c.name == "shooting" and c.status == "passed" for c in checks)
        else None,
    )
    diagnostics.verified = int(verification.passed)
    diagnostics.max_residual = max((c.value or 0.0) for c in checks if c.status != "skipped")
    if failed or unrun:
        what = failed[0] if failed else unrun[0]
        return stop(
            Status.NOT_VERIFIED,
            f"The check '{what.name.replace('_', ' ')}' {'failed' if failed else 'could not be run'}"
            f"{' (' + what.note + ')' if what.note else ''}. No solution is shown.",
            "not_verified",
            verification=verification,
            method_used="solve_bvp",
            t_start=a,
            t_end=b,
        )
    values_on_grid = sol.sol(grid)
    series = [
        OdeSeries(
            label_latex=_display(u, k),
            unknown=u,
            order=k,
            values=[_round(float(x)) for x in values_on_grid[index]],
        )
        for index, (u, k) in enumerate(labels)
    ]
    from .compare import compare_closed_form

    comparison = compare_closed_form(problem, data, values, grid, series, limits)
    return finish(
        OdeNumericReport(
            status=Status.OK,
            interpretation=meaning,
            method_requested=data.method,
            method_used="solve_bvp",
            method_reason=f"a solution found from the starting guess '{used}'; other solutions may exist",
            t_start=a,
            t_end=b,
            parameters=values,
            initial=[InitialValue(label_latex=c.label, value=c.value) for c in boundaries],
            rtol=data.rtol,
            atol=data.atol,
            steps=int(sol.x.size),
            evaluations=counter[0],
            t=[_round(float(x)) for x in grid],
            series=series,
            verification=verification,
            comparison=comparison,
            warnings=warnings,
            notes=[
                "A boundary-value problem can have no solution, one or several. This is a numerical "
                "approximation of one of them, on the interval between the two points; nothing is claimed "
                "about uniqueness. The checks are measurements, not a bound on the error.",
            ],
        )
    )
