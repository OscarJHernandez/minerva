"""Numerical solution of the equations of motion of a Lagrangian, with every run verified.

The accelerations come from the symbolic derivation in `lagrangian.py` (itself confirmed by numerical
differentiation of L). They are turned into plain-float functions and integrated with an adaptive
Dormand-Prince 5(4) method. A trajectory is returned only if three independent checks pass:

* a second method (classical fixed-step Runge-Kutta 4 with a much finer step) agrees at every sample;
* integrating back from the final state recovers the initial state (time reversal);
* every constant of motion that the analyser verified symbolically stays constant along the trajectory.

Otherwise the run is `not_verified` and carries no data. Chaotic motion over a long span, stiff systems and
singularities fail these checks and are reported as such: this module does not claim to solve them.
A sweep integrates the same system for up to six values of one parameter, each run verified separately.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import MinervaMathError
from .lagrangian import Derivation, DerivationError, check_equations, derive, find_conserved
from .models import Limits, Status

MAX_RUNS = 6
MAX_NUMBERS = 20_000  # floats in all returned series, so the card and the stored message stay small
MIN_SAMPLES = 20
RTOL = 1e-12
ATOL = 1e-14
METHOD_TOL = 1e-6
REVERSAL_TOL = 1e-6
DRIFT_TOL = 1e-6
MAX_PRIMARY_STEPS = 200_000
MAX_RK4_STEPS = 600_000
RK4_STEP_FRACTION = 4  # the cross-check steps by 1/4 of the primary's smallest accepted step


# --- request and report models --------------------------------------------------------------------
class Sweep(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    name: str = Field(max_length=32, description="The parameter to vary, e.g. k")
    values: list[float] = Field(
        min_length=2, max_length=MAX_RUNS, description="Between 2 and 6 values of that parameter"
    )


class MotionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    lagrangian: str = Field(
        min_length=1, max_length=2_000, description=r"L(q, qdot, t) in LaTeX, using \dot{q}"
    )
    coordinates: list[str] = Field(min_length=1, max_length=8)
    time: str = Field(default="t", max_length=16)
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    parameters: dict[str, float] = Field(
        default_factory=dict,
        description="Numerical value of every parameter of L (m, k, g, ...), except a swept one",
    )
    initial_positions: list[float] = Field(description="q_i(t_start), in the order of coordinates")
    initial_velocities: list[float] = Field(description="dq_i/dt at t_start, same order")
    t_start: float = 0.0
    t_end: float = Field(description="End time; the span t_end - t_start must be positive")
    sweep: Sweep | None = None
    samples: int = Field(default=200, ge=MIN_SAMPLES, le=400)

    @field_validator("parameters")
    @classmethod
    def _few(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > 32:
            raise ValueError("too many parameters")
        return value

    @model_validator(mode="after")
    def _shapes(self) -> MotionRequest:
        n = len(self.coordinates)
        if len(self.initial_positions) != n or len(self.initial_velocities) != n:
            raise ValueError("give one initial position and one initial velocity per coordinate")
        return self


class MotionVerification(BaseModel):
    method_difference: float | None = None  # max relative difference to the RK4 cross-check
    reversal_error: float | None = None  # relative error after integrating back to t_start
    max_drift: float | None = None  # largest relative drift of a verified constant of motion
    drift_checked: int = 0  # how many constants of motion were followed
    steps: int = 0  # accepted steps of the primary integrator
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


class ConservedSeries(BaseModel):
    name: str
    kind: str
    expression_latex: str
    values: list[float]
    relative_drift: float


class MotionRun(BaseModel):
    status: Status
    parameter_value: float | None = None  # the swept parameter's value, when there is a sweep
    message: str = ""
    q: list[list[float]] = Field(default_factory=list)  # [coordinate][sample]
    v: list[list[float]] = Field(default_factory=list)
    conserved: list[ConservedSeries] = Field(default_factory=list)
    verification: MotionVerification = Field(default_factory=MotionVerification)


class MotionReport(BaseModel):
    status: Status
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    accelerations: list[str] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)  # as read
    initial_positions: list[float] = Field(default_factory=list)
    initial_velocities: list[float] = Field(default_factory=list)
    t_start: float = 0.0
    t_end: float = 0.0
    sweep_name: str | None = None
    t: list[float] = Field(default_factory=list)
    runs: list[MotionRun] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _round(values: list[float]) -> list[float]:
    """10 significant digits: far below the verified accuracy, and the stored result stays small."""
    return [float(f"{x:.10g}") for x in values]


def _fail(status: Status, message: str) -> MotionReport:
    return MotionReport(status=status, message=message)


# --- numerical integration ---------------------------------------------------------------------
class _Stop(Exception):
    """A run cannot continue; `status` and `message` say why."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


VectorField = Callable[[float, list[float]], list[float]]

_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_E = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)


def _evaluate(fn: VectorField, t: float, y: list[float]) -> list[float]:
    try:
        out = fn(t, y)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise _Stop(
            Status.NOT_VERIFIED,
            "The accelerations could not be evaluated along the motion (division by zero, or a "
            "value outside the real domain of a function): the motion reaches a singularity.",
        ) from exc
    if any(not math.isfinite(x) for x in out):
        raise _Stop(
            Status.NOT_VERIFIED, "The motion reaches a singularity (a value became infinite)."
        )
    return out


def _dopri(
    fn: VectorField, y0: list[float], times: list[float], deadline: float, budget: int
) -> tuple[list[list[float]], int, list[float]]:
    """Adaptive Dormand-Prince 5(4) through `times` (monotone).

    Returns the states, the accepted steps, and per segment the smallest accepted step (steps cut short
    to land on a sample time do not count): the cross-check needs to be finer than that.
    """
    n = len(y0)
    direction = 1.0 if times[-1] >= times[0] else -1.0
    t = times[0]
    y = list(y0)
    states = [list(y)]
    k1 = _evaluate(fn, t, y)
    h = abs(times[1] - times[0]) / 10
    steps = attempts = 0
    smallest: list[float] = []
    for target in times[1:]:
        smallest.append(abs(target - t))
        while (target - t) * direction > 0:
            attempts += 1
            if attempts > budget:
                raise _Stop(
                    Status.RESOURCE_LIMIT,
                    "Too many integration steps were needed (the motion is stiff or very fast).",
                )
            if time.monotonic() > deadline:
                raise _Stop(Status.RESOURCE_LIMIT, "The integration used up its time budget.")
            hit = h >= abs(target - t)
            step = (abs(target - t) if hit else h) * direction
            ks = [k1]
            for stage in range(1, 7):
                ys = [
                    y[i] + step * sum(a * k[i] for a, k in zip(_A[stage], ks, strict=False) if a)
                    for i in range(n)
                ]
                ks.append(_evaluate(fn, t + _C[stage] * step, ys))
            y_new = ys  # stage 7 argument is the fifth-order solution (FSAL)
            err = [step * sum(e * k[i] for e, k in zip(_E, ks, strict=True) if e) for i in range(n)]
            norm = math.sqrt(
                sum((err[i] / (ATOL + RTOL * max(abs(y[i]), abs(y_new[i])))) ** 2 for i in range(n))
                / n
            )
            factor = 5.0 if norm == 0 else min(5.0, max(0.2, 0.9 * norm ** (-0.2)))
            if norm <= 1.0:
                t = target if hit else t + step
                y = y_new
                k1 = ks[6]
                steps += 1
                if not hit:
                    smallest[-1] = min(smallest[-1], abs(step))
                h = max(h, abs(step) * factor) if hit and factor >= 1 else abs(step) * factor
            else:
                h = abs(step) * factor
            if h < 1e-13 * max(1.0, abs(t)):
                raise _Stop(
                    Status.NOT_VERIFIED,
                    "The step size collapsed: the motion reaches a singularity or the system is stiff.",
                )
        states.append(list(y))
    return states, steps, smallest


def _rk4(
    fn: VectorField, y0: list[float], times: list[float], counts: list[int], deadline: float
) -> list[list[float]]:
    n = len(y0)
    t = times[0]
    y = list(y0)
    states = [list(y)]
    for (a, b), count in zip(itertools.pairwise(times), counts, strict=True):
        h = (b - a) / count
        t = a
        for _ in range(count):
            k1 = _evaluate(fn, t, y)
            k2 = _evaluate(fn, t + h / 2, [y[i] + h / 2 * k1[i] for i in range(n)])
            k3 = _evaluate(fn, t + h / 2, [y[i] + h / 2 * k2[i] for i in range(n)])
            k4 = _evaluate(fn, t + h, [y[i] + h * k3[i] for i in range(n)])
            y = [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
            t += h
        if time.monotonic() > deadline:
            raise _Stop(Status.RESOURCE_LIMIT, "The cross-check used up its time budget.")
        states.append(list(y))
    return states


# --- one verified run ----------------------------------------------------------------------------
@dataclass
class _Prepared:
    d: Derivation
    accel: Callable[..., list[float]]
    conserved: list[tuple[str, str, str, Callable[..., float]]]  # name, kind, latex, function


def _respects(symbol: sp.Symbol, value: float) -> str | None:
    """None if `value` is allowed by the symbol's assumptions, else the violated one."""
    checks = (
        ("positive", value > 0),
        ("negative", value < 0),
        ("nonnegative", value >= 0),
        ("nonpositive", value <= 0),
        ("nonzero", value != 0),
        ("integer", float(value).is_integer()),
    )
    for name, holds in checks:
        if getattr(symbol, f"is_{name}", None) is True and not holds:
            return name
    return None


def _scaled_difference(a: list[list[float]], b: list[list[float]]) -> float:
    worst = 0.0
    for j in range(len(a[0])):
        scale = max(max(abs(row[j]) for row in a), 1e-9)
        worst = max(worst, max(abs(ra[j] - rb[j]) for ra, rb in zip(a, b, strict=True)) / scale)
    return worst


def _run(
    prep: _Prepared,
    values: dict[sp.Symbol, float],
    y0: list[float],
    times: list[float],
    deadline: float,
    value: float | None,
) -> MotionRun:
    n = prep.d.n
    pv = [values[s] for s in prep.d.params]

    def fn(t: float, y: list[float]) -> list[float]:
        return [*y[n:], *prep.accel(t, *y, *pv)]

    run = MotionRun(status=Status.NOT_VERIFIED, parameter_value=value)
    ver = run.verification
    try:
        states, steps, smallest = _dopri(fn, y0, times, deadline, MAX_PRIMARY_STEPS)
        ver.steps = steps
        counts = [
            max(8, math.ceil(RK4_STEP_FRACTION * abs(b - a) / h))
            for (a, b), h in zip(itertools.pairwise(times), smallest, strict=True)
        ]
        if sum(counts) > MAX_RK4_STEPS:
            raise _Stop(
                Status.RESOURCE_LIMIT, "The cross-check would need too many steps for this motion."
            )
        reference = _rk4(fn, y0, times, counts, deadline)
        ver.method_difference = _scaled_difference(states, reference)
        back, _, _ = _dopri(fn, states[-1], [times[-1], times[0]], deadline, MAX_PRIMARY_STEPS)
        scale = [max(max(abs(row[j]) for row in states), 1e-9) for j in range(2 * n)]
        ver.reversal_error = max(abs(back[-1][j] - y0[j]) / scale[j] for j in range(2 * n))
        series: list[ConservedSeries] = []
        for name, kind, latex, func in prep.conserved:
            vals = [func(t, *s, *pv) for t, s in zip(times, states, strict=True)]
            if any(not math.isfinite(x) for x in vals):
                raise _Stop(Status.NOT_VERIFIED, "A constant of motion became infinite.")
            top = max(max(abs(x) for x in vals), 1e-300)
            drift = max(abs(x - vals[0]) for x in vals) / top
            series.append(
                ConservedSeries(
                    name=name,
                    kind=kind,
                    expression_latex=latex,
                    values=_round(vals),
                    relative_drift=drift,
                )
            )
        ver.drift_checked = len(series)
        ver.max_drift = max((s.relative_drift for s in series), default=None)
    except _Stop as stop:
        run.status = stop.status
        run.message = stop.message
        return run

    problems: list[str] = []
    if (ver.method_difference or 0.0) > METHOD_TOL:
        problems.append(
            f"the Runge-Kutta cross-check differs by {ver.method_difference:.2e} (relative)"
        )
    if (ver.reversal_error or 0.0) > REVERSAL_TOL:
        problems.append(f"integrating back misses the initial state by {ver.reversal_error:.2e}")
    if (ver.max_drift or 0.0) > DRIFT_TOL:
        problems.append(f"a constant of motion drifts by {ver.max_drift:.2e} (relative)")
    if problems:
        run.message = (
            "Not verified: " + "; ".join(problems) + ". Chaotic or stiff motion over a long time "
            "span cannot be computed reliably this way; try a shorter span."
        )
        return run
    if not series:
        ver.notes.append("no verified constant of motion was available to follow")
    ver.passed = True
    run.status = Status.OK
    run.q = [_round([s[i] for s in states]) for i in range(n)]
    run.v = [_round([s[n + i] for s in states]) for i in range(n)]
    run.conserved = series
    run.message = "Verified: second method, time reversal and constants of motion agree."
    return run


# --- entry point -----------------------------------------------------------------------------------
def integrate_motion(request: MotionRequest, limits: Limits) -> MotionReport:
    started = time.monotonic()
    deadline = started + 0.6 * limits.timeout_seconds
    try:
        d = derive(
            request.lagrangian, request.coordinates, request.time, request.assumptions, limits
        )
    except DerivationError as exc:
        return _fail(exc.status, exc.message)
    if not d.solution:
        return _fail(
            Status.UNSUPPORTED,
            "The mass matrix is singular, so the accelerations are not determined and the system "
            "cannot be integrated as it stands.",
        )
    if not all(c.passed for c in check_equations(d)):
        return _fail(
            Status.NOT_VERIFIED,
            "The derived equations of motion could not be confirmed by numerical differentiation of L, "
            "so they are not integrated.",
        )

    # Values as read: every parameter present, none unknown, all allowed by the assumptions.
    sweep = request.sweep
    by_name = {s.name: s for s in d.params}
    if request.t_end <= request.t_start:
        return _fail(Status.PARSE_ERROR, "t_end must be greater than t_start.")
    if sweep is not None and sweep.name not in by_name:
        return _fail(
            Status.PARSE_ERROR,
            f"The swept parameter {sweep.name!r} does not occur in L "
            f"(parameters: {', '.join(by_name) or 'none'}).",
        )
    given = dict(request.parameters)
    if sweep is not None:
        given.pop(sweep.name, None)
    unknown = sorted(set(given) - set(by_name))
    if unknown:
        return _fail(
            Status.PARSE_ERROR, f"Unknown parameter(s) {unknown}: L has {sorted(by_name)}."
        )
    missing = sorted(set(by_name) - set(given) - ({sweep.name} if sweep else set()))
    if missing:
        return _fail(Status.PARSE_ERROR, f"Give a value for: {', '.join(missing)}.")
    for name, number in given.items():
        broken = _respects(by_name[name], number)
        if broken:
            return _fail(Status.PARSE_ERROR, f"{name} is declared {broken}, but {number} is not.")
    if sweep is not None:
        for number in sweep.values:
            broken = _respects(by_name[sweep.name], number)
            if broken:
                return _fail(
                    Status.PARSE_ERROR, f"{sweep.name} is declared {broken}, but {number} is not."
                )
    for i, symbol in enumerate(d.q):
        broken = _respects(symbol, request.initial_positions[i])
        if broken:
            return _fail(
                Status.PARSE_ERROR,
                f"{request.coordinates[i]} is declared {broken}, but its initial value "
                f"{request.initial_positions[i]} is not.",
            )

    args = [d.t, *d.q, *d.qd, *d.params]
    try:
        accel = sp.lambdify(args, [d.solution[a] for a in d.qdd], modules="math", cse=True)
        conserved: list[tuple[str, str, str, Callable[..., float]]] = []
        for item, expr in find_conserved(d).items:
            if item.verified:
                conserved.append(
                    (item.name, item.kind, item.expression_latex, sp.lambdify(args, expr, "math"))
                )
    except (MinervaMathError, NameError, TypeError, ValueError, SyntaxError):
        return _fail(
            Status.UNSUPPORTED, "These equations cannot be turned into a numerical function."
        )
    prep = _Prepared(d, accel, conserved)

    # The same grid for every run, small enough that all series together stay under MAX_NUMBERS.
    runs_wanted = len(sweep.values) if sweep else 1
    per_sample = runs_wanted * (2 * d.n + len(conserved)) + 1
    samples = max(MIN_SAMPLES, min(request.samples, MAX_NUMBERS // per_sample))
    span = request.t_end - request.t_start
    times = [request.t_start + span * k / samples for k in range(samples + 1)]
    times[-1] = request.t_end
    y0 = [*request.initial_positions, *request.initial_velocities]

    report = MotionReport(
        status=Status.NOT_VERIFIED,
        lagrangian_latex=d.show(d.lagrangian),
        coordinates=request.coordinates,
        accelerations=[f"{d.show(a)} = {d.show(d.solution[a])}" for a in d.qdd],
        parameters=dict(request.parameters),
        initial_positions=request.initial_positions,
        initial_velocities=request.initial_velocities,
        t_start=request.t_start,
        t_end=request.t_end,
        sweep_name=sweep.name if sweep else None,
        t=_round(times),
    )
    if samples < request.samples:
        report.notes.append(f"Sampled at {samples} points to keep the result small.")
    cases: list[tuple[dict[sp.Symbol, float], float | None]] = []
    base = {by_name[n]: v for n, v in given.items()}
    if sweep is not None:
        cases = [({**base, by_name[sweep.name]: v}, v) for v in sweep.values]
    else:
        cases = [(base, None)]
    for values, swept in cases:
        run = _run(prep, values, y0, times, deadline, swept)
        report.runs.append(run)

    verified = [r for r in report.runs if r.status is Status.OK]
    if verified:
        report.status = Status.OK
        report.message = f"{len(verified)} of {len(report.runs)} run(s) verified."
    elif all(r.status is Status.RESOURCE_LIMIT for r in report.runs):
        report.status = Status.RESOURCE_LIMIT
        report.message = report.runs[0].message
    else:
        report.status = Status.NOT_VERIFIED
        report.message = (
            "No run could be verified, so no trajectory is shown. " + report.runs[0].message
        )
    if len(verified) < len(report.runs) and verified:
        report.notes.append("Runs that failed verification carry no data; see their messages.")
    report.notes.append(
        "Numerical solution (adaptive Dormand-Prince 5(4), relative tolerance 1e-12), verified against a "
        "finer fixed-step Runge-Kutta 4 run, by integrating back to the start, and by the drift of the "
        "verified constants of motion. Stiff systems are not solved."
    )
    return report


def parse_request(payload: dict[str, object]) -> MotionRequest | MotionReport:
    """Validate a raw request; a bad one becomes a `parse_error` report instead of an exception."""
    try:
        return MotionRequest(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return _fail(Status.PARSE_ERROR, f"Invalid input ({where}): {first['msg']}")
