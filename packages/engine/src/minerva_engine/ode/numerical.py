"""Numerical initial-value solutions, shown only when they pass checks that do not rely on the run.

Every number comes from the user: initial values (the conditions), parameter values and the end of the
interval. The system is the equation solved for its highest derivatives and written as a first-order
system (`analyze.solve_highest`). SciPy integrates; the checks are

* the initial values are reproduced (this also catches a mix-up of unknowns and derivatives);
* an independent integrator agrees: Minerva's own Dormand-Prince code (`motion._dopri`) for a
  non-stiff run, the other implicit method for a stiff one;
* refining the tolerance makes the answer change less and less (convergence, not just agreement);
* the integral form of the equation holds over the solver's steps (a defect estimate that uses the
  solver's dense output but evaluates the equation itself);
* where the equation has a first integral that was itself verified (a single second-order equation
  with a position-only acceleration: the energy), it stays constant to the tolerance.

A numerical trajectory is an approximation on the interval given: never exact, never valid beyond it.
Stiffness is not predicted: the integrator is chosen from measured behaviour, and the report says so.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from ..errors import MinervaMathError
from ..latexprint import to_latex
from ..models import Limits, Status
from ..motion import _dopri, _respects
from ..motion import _Stop as DopriStop
from .analyze import _classify_conditions, interpretation, is_zero, solve_highest
from .models import (
    InitialValue,
    NumericCheck,
    NumericMethod,
    NumericVerification,
    NumericWarning,
    OdeDiagnostics,
    OdeNumericInput,
    OdeNumericReport,
    OdeSeries,
    OdeStage,
)
from .reading import OdeProblem, _display, read_ode
from .symbolic import SolverTimeout, deadline
from .verification import Unevaluable, lambdify_numeric

MAX_STATE = 12
MAX_EVALUATIONS = 400_000
MAX_NUMBERS = 20_000
EXPLICIT_PROBE_EVALUATIONS = 60_000
STIFFNESS_THRESHOLD = 3.0  # h * spectral radius at which an explicit method is stability-limited
REJECTED_RATIO_LIMIT = 0.4
FLOOR = 1e-11  # differences below this (scaled) are rounding, not discretisation
DEFECT_LIMIT = 1e3  # in units of the requested tolerance
BLOW_UP = 1e100
EXPLICIT = ("RK45", "DOP853")
STAGES = {"RK45": 6, "DOP853": 12}


class _Halt(Exception):
    def __init__(
        self, reason: Literal["budget", "time", "domain", "collapse", "blow_up"], message: str
    ):
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass
class _Budget:
    limit: int
    stop_at: float
    used: int = 0


@dataclass
class System:
    problem: OdeProblem
    labels: list[tuple[str, int]]
    states: list[sp.Symbol]
    f: Callable[..., Any]  # f(t, *states), as a numpy function
    f_math: Callable[[float, list[float]], list[float]]
    jacobian: Callable[[float, np.ndarray], np.ndarray]
    y0: np.ndarray
    t0: float
    t1: float
    rhs: list[sp.Expr]


@dataclass
class Run:
    method: str
    sol: Any
    steps: int
    evaluations: int
    indicator: float | None = None
    rejected: float | None = None


def _round(value: float) -> float:
    return float(f"{value:.10g}")


def _guard(f: Callable[..., Any], budget: _Budget) -> Callable[[float, np.ndarray], np.ndarray]:
    def wrapped(t: float, y: np.ndarray) -> np.ndarray:
        budget.used += 1
        if budget.used > budget.limit:
            raise _Halt("budget", "too many evaluations of the equation were needed")
        if time.monotonic() > budget.stop_at:
            raise _Halt("time", "the integration used up its time budget")
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(f(t, *y), dtype=float)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                raise _Halt("domain", "the right-hand side left its real domain") from exc
        if not np.all(np.isfinite(out)):
            raise _Halt("domain", "the right-hand side became infinite or undefined")
        if np.max(np.abs(y), initial=0.0) > BLOW_UP:
            raise _Halt("blow_up", "the solution grew beyond any bound")
        return out

    return wrapped


def _numeric_jacobian(
    f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray
) -> np.ndarray:
    base = f(t, y)
    columns = []
    for j in range(len(y)):
        step = 1e-7 * max(1.0, abs(y[j]))
        shifted = y.copy()
        shifted[j] += step
        columns.append((f(t, shifted) - base) / step)
    return np.array(columns).T


def build_system(
    problem: OdeProblem,
    solved_rhs: list[sp.Expr],
    values: dict[str, float],
    initial: dict[tuple[str, int], float],
    t0: float,
    t1: float,
) -> System:
    labels = [(u, k) for u in problem.unknowns for k in range(problem.orders[u])]
    states = [problem.jets[label] for label in labels]
    substitution = {p: sp.Float(values[p.name], 17) for p in problem.parameters}
    rhs: list[sp.Expr] = []
    for u in problem.unknowns:
        order = problem.orders[u]
        for k in range(order):
            if k < order - 1:
                rhs.append(problem.jets[(u, k + 1)])
            else:
                rhs.append(solved_rhs[problem.unknowns.index(u)].xreplace(substitution))
    t = problem.variable
    f = lambdify_numeric([t, *states], rhs, "numpy")
    f_math = lambdify_numeric([t, *states], rhs, "math")
    matrix = sp.Matrix(rhs).jacobian(states)
    symbolic = sp.count_ops(matrix) <= 3000
    j_np = lambdify_numeric([t, *states], matrix, "numpy") if symbolic else None

    def jacobian(time_: float, y: np.ndarray) -> np.ndarray:
        if j_np is not None:
            with np.errstate(all="ignore"):
                return np.array(j_np(time_, *y), dtype=float)
        return _numeric_jacobian(lambda tt, yy: np.asarray(f(tt, *yy), dtype=float), time_, y)

    return System(
        problem=problem,
        labels=labels,
        states=states,
        f=f,
        f_math=lambda tt, yy: [float(v) for v in f_math(tt, *yy)],
        jacobian=jacobian,
        y0=np.array([initial[label] for label in labels], dtype=float),
        t0=t0,
        t1=t1,
        rhs=rhs,
    )


# --- one run ---------------------------------------------------------------------------------


def _run(
    system: System, method: str, rtol: float, atol: float, times: np.ndarray, budget: _Budget
) -> Run:
    fun = _guard(system.f, budget)
    options: dict[str, Any] = {}
    if method in ("Radau", "BDF", "LSODA"):
        options["jac"] = lambda t, y: system.jacobian(t, y)
    try:
        with np.errstate(all="ignore"):
            sol = solve_ivp(
                fun,
                (system.t0, system.t1),
                system.y0,
                method=method,
                t_eval=times,
                dense_output=True,
                rtol=rtol,
                atol=atol,
                **options,
            )
    except _Halt:
        raise
    except (ValueError, TypeError, ArithmeticError, np.linalg.LinAlgError) as exc:
        raise _Halt("domain", "the integration failed on this right-hand side") from exc
    if sol.status != 0:
        text = str(sol.message)
        if "step size" in text or "spacing" in text:
            raise _Halt("collapse", "the step size collapsed: a singularity or a very stiff region")
        raise _Halt("domain", "the integrator stopped before the end of the interval")
    steps = max(0, len(sol.sol.ts) - 1)
    return Run(method, sol, steps, int(sol.nfev))


def _stiffness(system: System, run: Run) -> tuple[float, float | None]:
    """(the largest h * spectral radius of the Jacobian over some accepted steps, the rejected-step ratio)."""
    ts = run.sol.sol.ts
    count = len(ts) - 1
    if count < 1:
        return 0.0, None
    picks = sorted({int(i) for i in np.linspace(0, count - 1, min(count, 12))})
    worst = 0.0
    for i in picks:
        t = float(ts[i])
        h = abs(float(ts[i + 1] - ts[i]))
        try:
            j = system.jacobian(t, np.asarray(run.sol.sol(t), dtype=float))
            radius = float(np.max(np.abs(np.linalg.eigvals(j))))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            continue
        if math.isfinite(radius):
            worst = max(worst, h * radius)
    rejected = None
    stages = STAGES.get(run.method)
    if stages:
        attempted = max(1.0, (run.evaluations - 1) / stages)
        rejected = max(0.0, 1.0 - run.steps / attempted)
    return worst, rejected


def _scaled(a: np.ndarray, b: np.ndarray) -> float:
    """Largest difference of two trajectories (components in rows), each component scaled by its size."""
    worst = 0.0
    for row_a, row_b in zip(a, b, strict=True):
        scale = max(float(np.max(np.abs(row_a))), 1e-9)
        worst = max(worst, float(np.max(np.abs(row_a - row_b))) / scale)
    return worst


# --- the checks ------------------------------------------------------------------------------


def _check_initial(
    system: System, run: Run, conditions: dict[tuple[str, int], float]
) -> NumericCheck:
    worst = 0.0
    for index, label in enumerate(system.labels):
        want = conditions[label]
        got = float(run.sol.y[index][0])
        worst = max(worst, abs(got - want) / max(1.0, abs(want)))
    return NumericCheck(
        name="initial_conditions",
        status="passed" if worst <= 1e-12 else "failed",
        value=worst,
        tolerance=1e-12,
    )


def _check_independent(
    system: System, run: Run, times: np.ndarray, rtol: float, atol: float, stop_at: float
) -> tuple[NumericCheck, str | None]:
    tolerance = min(max(1e3 * rtol, 1e-7), 1e-2)
    explicit = run.method in EXPLICIT or (
        run.method == "LSODA" and (run.indicator or 0) < STIFFNESS_THRESHOLD
    )
    try:
        if explicit:
            name = "Minerva's Dormand-Prince 5(4)"
            forward = list(times)
            states, _, _ = _dopri(system.f_math, list(system.y0), forward, stop_at, 150_000)
            other = np.array(states).T
        else:
            method = "BDF" if run.method == "Radau" else "Radau"
            name = method
            budget = _Budget(MAX_EVALUATIONS, stop_at)
            ref = _run(system, method, min(rtol, 1e-8) / 10, min(atol, 1e-10) / 10, times, budget)
            other = ref.sol.y
    except DopriStop as exc:
        return NumericCheck(name="independent_method", status="skipped", note=exc.message), None
    except _Halt as exc:
        return NumericCheck(name="independent_method", status="skipped", note=exc.message), None
    difference = _scaled(run.sol.y, other)
    return (
        NumericCheck(
            name="independent_method",
            status="passed" if difference <= tolerance else "failed",
            value=difference,
            tolerance=tolerance,
            note=f"compared with {name}",
        ),
        name,
    )


def _check_refinement(
    system: System, run: Run, times: np.ndarray, rtol: float, atol: float, stop_at: float
) -> tuple[NumericCheck, list[float]]:
    levels = [rtol / 10, rtol / 100]
    trajectories = [run.sol.y]
    try:
        for level in levels:
            if level < 1e-13:
                break
            budget = _Budget(MAX_EVALUATIONS, stop_at)
            refined = _run(
                system, run.method, level, atol / 10 ** (len(trajectories)), times, budget
            )
            trajectories.append(refined.sol.y)
    except _Halt as exc:
        return NumericCheck(name="tolerance_refinement", status="skipped", note=exc.message), []
    errors = [_scaled(a, b) for a, b in itertools.pairwise(trajectories)]
    if not errors:
        return (
            NumericCheck(
                name="tolerance_refinement",
                status="skipped",
                note="the tolerance is already at the limit of double precision",
            ),
            [],
        )
    first = errors[0]
    converging = (len(errors) == 1 and first <= max(1e3 * rtol, FLOOR)) or (
        len(errors) >= 2 and (errors[1] <= 0.75 * errors[0] or max(errors) <= FLOOR)
    )
    if len(errors) == 1:
        converging = first <= max(10 * rtol, FLOOR)
    return (
        NumericCheck(
            name="tolerance_refinement",
            status="passed" if converging else "failed",
            value=first,
            tolerance=max(10 * rtol, FLOOR),
            note="the change between successive tolerances: "
            + ", ".join(f"{e:.2e}" for e in errors),
        ),
        errors,
    )


def _check_defect(system: System, run: Run, rtol: float, atol: float) -> NumericCheck:
    """z(b) - z(a) - integral of f along the dense output, over the solver's own steps, in units of the
    tolerance. Evaluates the equation itself with an 8-point Gauss rule; it does not differentiate."""
    ts = np.asarray(run.sol.sol.ts, dtype=float)
    if len(ts) < 2:
        return NumericCheck(name="defect", status="skipped", note="no steps to examine")
    nodes, weights = np.polynomial.legendre.leggauss(8)
    stride = max(1, (len(ts) - 1) // 150)
    worst = 0.0
    dense = run.sol.sol
    for i in range(0, len(ts) - 1, stride):
        a, b = float(ts[i]), float(ts[min(i + stride, len(ts) - 1)])
        mid, half = (a + b) / 2, (b - a) / 2
        total = np.zeros(len(system.y0))
        try:
            for node, weight in zip(nodes, weights, strict=True):
                s = mid + half * node
                with np.errstate(all="ignore"):
                    total += weight * np.asarray(system.f(s, *dense(s)), dtype=float)
        except (TypeError, ValueError, ArithmeticError):
            return NumericCheck(
                name="defect", status="skipped", note="could not evaluate the equation"
            )
        residual = np.asarray(dense(b)) - np.asarray(dense(a)) - half * total
        scale = atol + rtol * np.maximum(np.abs(dense(a)), np.abs(dense(b)))
        if not np.all(np.isfinite(residual)):
            return NumericCheck(name="defect", status="failed", note="not finite")
        worst = max(worst, float(np.max(np.abs(residual) / scale)))
    return NumericCheck(
        name="defect",
        status="passed" if worst <= DEFECT_LIMIT else "failed",
        value=worst,
        tolerance=DEFECT_LIMIT,
        note="in units of the requested tolerance",
    )


def _first_integral(system: System) -> tuple[Callable[..., Any], str] | None:
    """The energy of `y'' = a(y)`, if it can be integrated and the result is confirmed to be a first
    integral (its derivative along the equation is zero)."""
    problem = system.problem
    if len(problem.unknowns) != 1 or problem.orders[problem.unknowns[0]] != 2:
        return None
    y, v = system.states
    a = system.rhs[-1]  # the acceleration, with the parameters' numerical values in it
    if a.has(problem.variable) or a.has(v):
        return None
    try:
        with deadline(1.0):
            antiderivative = sp.integrate(a, y)
    except (SolverTimeout, NotImplementedError, ValueError, TypeError):
        return None
    if antiderivative.has(sp.Integral) or not is_zero(sp.diff(antiderivative, y) - a):
        return None
    energy = v**2 / 2 - antiderivative
    try:
        function = lambdify_numeric([y, v], energy, "numpy")
    except Unevaluable:
        return None
    return function, to_latex(energy, symbol_names=problem.display)


def _check_first_integral(system: System, run: Run, rtol: float) -> tuple[NumericCheck, str | None]:
    found = _first_integral(system)
    if found is None:
        return NumericCheck(
            name="first_integral",
            status="skipped",
            note="no verified first integral is known for this equation",
        ), None
    function, latex = found
    with np.errstate(all="ignore"):
        energy = np.asarray(function(run.sol.y[0], run.sol.y[1]), dtype=float)
    scale = max(float(np.max(np.abs(energy))), float(np.max(0.5 * run.sol.y[1] ** 2)), 1e-9)
    drift = float(np.max(np.abs(energy - energy[0]))) / scale
    tolerance = max(1e-6, 1e3 * rtol)
    return (
        NumericCheck(
            name="first_integral",
            status="passed" if np.isfinite(drift) and drift <= tolerance else "failed",
            value=drift,
            tolerance=tolerance,
            note="the energy, whose derivative along the equation is zero",
        ),
        latex,
    )


# --- the whole procedure -----------------------------------------------------------------------


def _number(expr: sp.Expr, values: dict[str, float]) -> float | None:
    try:
        value = complex(sp.N(expr.xreplace({sp.Symbol(k): v for k, v in values.items()}), 30))
    except (TypeError, ValueError):
        return None
    return value.real if value.imag == 0 and math.isfinite(value.real) else None


def parameter_problem(problem: OdeProblem, given: dict[str, float]) -> str | None:
    """Why the parameter values cannot be used (missing, not a parameter, against an assumption), or None.
    Every parameter needs a value from the user; none is ever supplied here."""
    names = {p.name for p in problem.parameters}
    missing = sorted(names - set(given))
    if missing:
        return "Values are required for every parameter; missing: " + ", ".join(missing) + "."
    extra = sorted(set(given) - names)
    if extra:
        return "Not parameters of this equation: " + ", ".join(extra) + "."
    for p in problem.parameters:
        violated = _respects(p, given[p.name])
        if violated:
            return f"The value given for {p.name} is not {violated}, which was declared."
    return None


def solve_numerical(
    data: OdeNumericInput, limits: Limits, trace_id: str | None = None
) -> OdeNumericReport:
    if data.kind == "bvp":
        from .boundary import solve_boundary

        return solve_boundary(data, limits, trace_id)
    started = time.perf_counter()
    stop_at = time.monotonic() + 0.6 * limits.timeout_seconds
    stages: list[OdeStage] = []
    diagnostics = OdeDiagnostics(trace_id=trace_id, method="ivp")
    state: dict[str, Any] = {"clock": started}

    def mark(name: Literal["read", "integrate", "verify", "refine", "total"]) -> None:
        now = time.perf_counter()
        stages.append(OdeStage(name=name, ms=round((now - state["clock"]) * 1000, 2)))
        state["clock"] = now

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
    mark("read")
    meaning = interpretation(problem)
    diagnostics.n_equations = len(problem.sides)
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

    # Every number must come from the user.
    kind, needed = _classify_conditions(problem)
    if kind != "initial_value":
        listing = ", ".join(needed) if needed else "initial values for every unknown and derivative"
        text = {
            "general": "no initial values were given",
            "incomplete": "the initial values are incomplete",
            "overdetermined": "too many conditions were given",
            "boundary_value": "conditions at two points make this a boundary-value problem",
        }[kind]
        return stop(
            Status.PARSE_ERROR,
            f"An initial-value problem needs every initial value, all at one point: {text}. "
            f"Needed: {listing}. Minerva does not choose them.",
            "conditions",
        )
    problem_with = parameter_problem(problem, data.parameters)
    if problem_with:
        return stop(Status.PARSE_ERROR, problem_with, "parameters")
    values = dict(data.parameters)
    t0 = _number(problem.conditions[0].at, values)
    if t0 is None:
        return stop(Status.PARSE_ERROR, "The starting point must be a real number.", "interval")
    initial: dict[tuple[str, int], float] = {}
    for c in problem.conditions:
        number = _number(c.value, values)
        if number is None:
            return stop(Status.PARSE_ERROR, "Initial values must be real numbers.", "conditions")
        initial[(c.unknown, c.order)] = number
    if data.t_end is None:
        return stop(
            Status.PARSE_ERROR,
            "Give the end of the interval (t_end): Minerva does not choose it.",
            "interval",
        )
    t1 = float(data.t_end)
    if t1 == t0:
        return stop(
            Status.PARSE_ERROR, "The interval is empty: t_end equals the start.", "interval"
        )
    try:
        system = build_system(problem, solved.rhs, values, initial, t0, t1)
    except Unevaluable:
        return stop(
            Status.UNSUPPORTED,
            "The equation uses a function that cannot be evaluated numerically here.",
            "unevaluable",
        )

    samples = max(20, min(data.samples, MAX_NUMBERS // (len(system.labels) + 1)))
    times = np.linspace(t0, t1, samples)
    warnings: list[NumericWarning] = []
    notes: list[str] = []
    tried: list[str] = []

    def fail(halt: _Halt, method: str) -> OdeNumericReport:
        if halt.reason in ("budget", "time"):
            if method in EXPLICIT:
                warnings.append("stiffness_indicated")
            status = Status.RESOURCE_LIMIT
        else:
            status = Status.NOT_VERIFIED
            warnings.append(
                {"collapse": "step_collapse", "blow_up": "blow_up_suspected"}.get(
                    halt.reason, "domain_left"
                )  # type: ignore[arg-type]
            )
        diagnostics.methods_tried = tried
        message = (
            f"The integration did not complete ({halt.message}); no trajectory is shown. "
            "This may be a singularity, a stiff problem or a solution that blows up."
        )
        return stop(status, message, halt.reason, warnings=warnings, t_start=t0, t_end=t1)

    # --- choose and run the primary integrator ---
    requested: NumericMethod = data.method
    run: Run | None = None
    reason = ""
    first = "DOP853" if requested == "auto" else requested
    budget = _Budget(
        EXPLICIT_PROBE_EVALUATIONS if requested == "auto" else MAX_EVALUATIONS, stop_at
    )
    tried.append(first)
    try:
        run = _run(system, first, data.rtol, data.atol, times, budget)
        if first in EXPLICIT or first == "LSODA":
            run.indicator, run.rejected = _stiffness(system, run)
        switch = requested == "auto" and (
            (run.indicator or 0) >= STIFFNESS_THRESHOLD
            or (run.rejected or 0) >= REJECTED_RATIO_LIMIT
        )
        if switch:
            raise _Halt("budget", "stiffness indicators")
        reason = (
            "chosen by the user"
            if requested != "auto"
            else "DOP853 completed and showed no sign of stiffness"
        )
        if requested != "auto" and (run.indicator or 0) >= STIFFNESS_THRESHOLD:
            warnings.append("stiffness_indicated")
            notes.append(
                "The step size is limited by stability rather than accuracy: the problem looks stiff, "
                "and Radau or BDF would be cheaper."
            )
    except _Halt as halt:
        # An explicit method on a stiff problem exhausts its budget, collapses its step or diverges:
        # in automatic mode that is a reason to try an implicit method once. Leaving the real domain
        # or running out of time is not.
        if requested != "auto" or halt.reason not in ("budget", "collapse", "blow_up"):
            return fail(halt, first)
        tried.append("Radau")
        budget = _Budget(MAX_EVALUATIONS, stop_at)
        try:
            run = _run(system, "Radau", data.rtol, data.atol, times, budget)
        except _Halt as again:
            return fail(again, "Radau")
        warnings.append("stiffness_indicated")
        reason = (
            "DOP853 showed signs of stiffness (a step size limited by stability, or too many "
            "evaluations), so Radau was used; this is judged from measured behaviour, not predicted"
        )
    mark("integrate")
    if run is None:
        return fail(_Halt("domain", "no result"), first)
    diagnostics.steps = run.steps
    diagnostics.evaluations = run.evaluations
    diagnostics.method = run.method

    # --- verification ---
    checks = [_check_initial(system, run, initial)]
    independent, name = _check_independent(system, run, times, data.rtol, data.atol, stop_at)
    checks.append(independent)
    mark("verify")
    refinement, errors = _check_refinement(system, run, times, data.rtol, data.atol, stop_at)
    checks.append(refinement)
    mark("refine")
    checks.append(_check_defect(system, run, data.rtol, data.atol))
    integral, integral_latex = _check_first_integral(system, run, data.rtol)
    checks.append(integral)
    mark("verify")

    required = ("initial_conditions", "independent_method", "tolerance_refinement")
    failed = [c for c in checks if c.status == "failed"]
    missing_required = [c for c in checks if c.name in required and c.status == "skipped"]
    verification = NumericVerification(
        checks=checks,
        passed=not failed and not missing_required,
        independent_method=name,
        refinement_errors=errors,
        first_integral_latex=integral_latex,
    )
    diagnostics.methods_tried = tried
    diagnostics.verified = int(verification.passed)
    diagnostics.max_residual = max((c.value or 0.0) for c in checks if c.status != "skipped")
    if failed or missing_required:
        if any(c.name == "tolerance_refinement" for c in failed):
            warnings.append("not_converged")
        what = failed[0] if failed else missing_required[0]
        word = "failed" if failed else "could not be run"
        return stop(
            Status.NOT_VERIFIED,
            f"The check '{what.name.replace('_', ' ')}' {word}"
            + (f" ({what.note})" if what.note else "")
            + ". No trajectory is shown.",
            "not_verified",
            verification=verification,
            warnings=warnings,
            method_used=run.method,
            t_start=t0,
            t_end=t1,
        )

    series = [
        OdeSeries(
            label_latex=_display(u, k),
            unknown=u,
            order=k,
            values=[_round(float(x)) for x in run.sol.y[index]],
        )
        for index, (u, k) in enumerate(system.labels)
    ]
    from .compare import compare_closed_form

    comparison = compare_closed_form(problem, data, values, times, series, limits)
    notes.append(
        "A numerical approximation on the interval given: it is not exact and says nothing beyond it. "
        "The checks above are measurements, not a bound on the error."
    )
    if requested == "auto":
        notes.append(
            "Stiffness is not predicted: the integrator was chosen from measured behaviour."
        )
    return finish(
        OdeNumericReport(
            status=Status.OK,
            interpretation=meaning,
            method_requested=requested,
            method_used=run.method,
            method_reason=reason,
            t_start=t0,
            t_end=t1,
            parameters=values,
            initial=[
                InitialValue(label_latex=_display(u, k), value=initial[(u, k)])
                for (u, k) in system.labels
            ],
            rtol=data.rtol,
            atol=data.atol,
            steps=run.steps,
            evaluations=run.evaluations,
            t=[_round(float(x)) for x in times],
            series=series,
            verification=verification,
            comparison=comparison,
            warnings=warnings,
            notes=notes,
        )
    )
