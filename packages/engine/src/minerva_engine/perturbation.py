"""Perturbation solution of a one-degree-of-freedom oscillator (Poincare-Lindstedt), with checks.

The system is a time-independent Lagrangian in one coordinate whose acceleration depends on the
position only: uddot = a(q). About an equilibrium q0 put u = q - q0 and write

    wddot + omega0^2 w = sum_{j=1..N} eps^j G_j(w) + O(eps^(N+1)),     G_j polynomial in w,

in one of two ways:

* `parameter`: eps is a parameter of L that the user names (L at eps = 0 must be a linear oscillator), the
  initial condition is u(0) = A, udot(0) = 0 with a symbolic amplitude A, and w = u;
* `amplitude`: eps is the size of the initial displacement, u = eps w, w(0) = 1 (the pendulum, any anharmonic
  well).

With the strained time tau = omega t, omega = omega0 + eps omega1 + ..., and w = w0 + eps w1 + ..., each order is
a linear equation solved exactly on cosine polynomials (cos(m tau)); omega_k is chosen to remove the cos(tau)
term that would grow without bound. SymPy supplies the expansion coefficients and the exact algebra; nothing
is shown unless all of the following hold:

1. The Euler-Lagrange equations and the equilibrium were confirmed (`expansion.prepare`), and the
   expansion of the acceleration in eps passed `series.check_series`.
2. The truncated solution, put into the *unexpanded* acceleration a(q) from the Lagrangian (numerically, in
   mpmath, at random parameters), leaves a residual that falls at least as eps^(N+1) when eps is halved twice.
3. The initial conditions hold (symbolic zero and numeric, reported separately), and every order is a finite
   cosine polynomial: bounded, no secular term.

The period ratio T/T0 = omega0/omega is a series in eps derived from omega and checked with `check_series`.
Optionally the solution is compared with the verified numerical integrator for numbers the user chose, at eps
and eps/2; that comparison is a measurement on that span, never a claim for all times. Velocity-dependent
accelerations, more than one coordinate, damping (no limit cycle from a Lagrangian) and non-polynomial
perturbations are `unsupported`: multiple-scales and averaging methods are not implemented.
"""

from __future__ import annotations

import math
import random
import time
from typing import Literal

import mpmath
import sympy as sp
from pydantic import BaseModel, ConfigDict, Field

from .expansion import NamedLatex, Refusal, _fresh, _zero_check, prepare
from .models import Limits, Status, Verification
from .motion import MotionRequest, integrate_motion
from .series import SeriesVerification, _eval, _falls, check_series
from .verify import MIN_VALID_POINTS, _sample

MAX_ORDER = 6
DPS = 60
EPS_VALUES = (mpmath.mpf("0.004"), mpmath.mpf("0.002"), mpmath.mpf("0.001"))
SAMPLE_ATTEMPTS = 20
TAUS = (0.37, 1.9, 4.4)
COMPARE_SAMPLES = 200

Trig = dict[int, sp.Expr]  # harmonic m -> coefficient of cos(m tau)


# --- models ----------------------------------------------------------------------------------------
class CompareRequest(BaseModel):
    """Numbers chosen by the user for a comparison with the verified numerical solution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    parameters: dict[str, float] = Field(
        description="Value of every parameter of L except the small parameter (when one is named)"
    )
    epsilon: float = Field(
        gt=0,
        description="Value of the small parameter (parameter mode) or the initial displacement (amplitude mode)",
    )
    amplitude: float | None = Field(
        default=None, description="Initial displacement u(0); parameter mode only"
    )
    t_end: float = Field(gt=0, description="End time of the comparison")


class OrderTerms(BaseModel):
    order: int
    frequency_latex: str | None = None  # omega_k (k >= 1)
    shape_latex: str


class PerturbationVerification(BaseModel):
    residual_method: str = (
        "the truncated solution put into the unexpanded acceleration from L; the residual must "
        "fall at least as eps^(order+1) when eps is halved twice"
    )
    residual_passed: bool = False
    parameter_sets_checked: int = 0
    initial_conditions: Verification | None = None
    initial_conditions_passed: bool = False
    bounded: bool = False  # every order is a finite cosine polynomial: no secular term
    expansion: SeriesVerification | None = None
    passed: bool = False
    note: str = ""


class Comparison(BaseModel):
    status: Status
    message: str = ""
    epsilon: float = 0.0
    predicted_order: int = 0
    t: list[float] = Field(default_factory=list)
    numerical: list[float] = Field(default_factory=list)  # u(t) from the verified integrator
    perturbative: list[float] = Field(default_factory=list)  # u(t) from the solution above
    max_error: float | None = None  # scaled as described in `scaling`
    max_error_half: float | None = None  # same, with eps/2
    observed_order: float | None = None
    scaling: str = ""
    notes: list[str] = Field(default_factory=list)


class PerturbationReport(BaseModel):
    status: Status
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    equilibrium: list[NamedLatex] = Field(default_factory=list)
    mode: Literal["parameter", "amplitude"] | None = None
    parameters: list[str] = Field(default_factory=list)  # parameters of L other than the small one
    small_latex: str | None = None
    amplitude_latex: str | None = None
    order: int = 0
    linear_frequency_latex: str | None = None
    frequency_latex: str | None = None
    period_ratio_latex: str | None = None
    solution_latex: str | None = None
    orders: list[OrderTerms] = Field(default_factory=list)
    verification: PerturbationVerification | None = None
    comparison: Comparison | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: object) -> PerturbationReport:
    return PerturbationReport(status=status, message=message, **extra)


# --- exact algebra on cosine polynomials -----------------------------------------------------------
def _t_add(a: Trig, b: Trig) -> Trig:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, sp.Integer(0)) + c
    return out


def _t_mul(a: Trig, b: Trig) -> Trig:
    """cos(m1 t) cos(m2 t) = (cos((m1+m2) t) + cos((m1-m2) t)) / 2; the two terms coincide if an index is 0."""
    out: Trig = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            half = ca * cb / 2
            for m in (ma + mb, abs(ma - mb)):
                out[m] = out.get(m, sp.Integer(0)) + half
    return out


def _t_pow(a: Trig, n: int) -> Trig:
    out: Trig = {0: sp.Integer(1)}
    for _ in range(n):
        out = _t_mul(out, a)
    return out


def _simplified(a: Trig) -> Trig:
    out = {m: sp.simplify(c) for m, c in a.items()}
    return {m: c for m, c in out.items() if c != 0}


# --- solving ---------------------------------------------------------------------------------------
class _Solved:
    def __init__(
        self,
        w: list[Trig],
        omegas: list[sp.Expr],
        s: list[sp.Expr],
        residual_resonance: list[sp.Expr],
    ) -> None:
        self.w = w  # w[k][m]: coefficient of eps^k cos(m tau)
        self.omegas = omegas  # omega_0 .. omega_N
        self.s = s
        self.residual_resonance = residual_resonance


def _solve_orders(
    cs: list[sp.Expr], w_sym: sp.Symbol, s0: sp.Expr, omega0: sp.Expr, amp: sp.Expr, eps: sp.Symbol
) -> _Solved:
    """Poincare-Lindstedt to order len(cs) - 1; `cs[j]` multiplies eps^j on the right-hand side."""
    n_orders = len(cs) - 1
    big_w = [sp.Symbol(f"W{i}") for i in range(max(n_orders, 1))]
    wsum = sum((eps**i * big_w[i] for i in range(n_orders)), sp.Integer(0))
    nonlinear = sp.Poly(
        sp.expand(sum((eps**j * cs[j].subs(w_sym, wsum) for j in range(1, n_orders + 1)), 0)), eps
    )
    w: list[Trig] = [{1: amp}]
    s: list[sp.Expr] = [s0]
    omegas: list[sp.Expr] = [omega0]
    resonance: list[sp.Expr] = []
    for k in range(1, n_orders + 1):
        coefficient_k = nonlinear.coeff_monomial(eps**k)
        rest: Trig = {}
        if coefficient_k != 0:
            for exps, c in sp.Poly(coefficient_k, *big_w[:k]).terms():
                term: Trig = {0: sp.Integer(1)}
                for i, e in enumerate(exps):
                    if e:
                        term = _t_mul(term, _t_pow(w[i], e))
                rest = _t_add(rest, {m: c * v for m, v in term.items()})
        for m_idx in range(1, k):  # - s_m w_{k-m}'' , and w'' = -m^2 w for cos(m tau)
            rest = _t_add(rest, {h: s[m_idx] * h * h * c for h, c in w[k - m_idx].items()})
        s_k = sp.simplify(-rest.get(1, sp.Integer(0)) / amp)
        s.append(s_k)
        resonance.append(sp.expand(rest.get(1, sp.Integer(0)) + s_k * amp))
        cross = sum((omegas[m_idx] * omegas[k - m_idx] for m_idx in range(1, k)), sp.Integer(0))
        omegas.append(sp.simplify((s_k - cross) / (2 * omega0)))
        new: Trig = {}
        for h, c in rest.items():
            if h != 1:
                new[h] = c / (s0 * (1 - h * h))
        new[1] = -sum(new.values(), sp.Integer(0))
        w.append(_simplified(new))
    return _Solved(w, omegas, s, resonance)


# --- numerical evaluation --------------------------------------------------------------------------
def _w_value(
    flat: list[list[tuple[int, mpmath.mpf]]],
    omegas: list[mpmath.mpf],
    eps: mpmath.mpf,
    tau: mpmath.mpf,
) -> tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf]:
    """w_N(tau), w_N''(tau) and omega_N for the truncated series at this eps."""
    w = mpmath.mpf(0)
    w2 = mpmath.mpf(0)
    for k, harmonics in enumerate(flat):
        for m, c in harmonics:
            w += eps**k * c * mpmath.cos(m * tau)
            w2 -= eps**k * c * m * m * mpmath.cos(m * tau)
    omega = sum((eps**k * o for k, o in enumerate(omegas)), mpmath.mpf(0))
    return w, w2, omega


def _check_residual(
    a_expr: sp.Expr,
    q: sp.Symbol,
    q0: sp.Expr,
    solved: _Solved,
    eps_sym: sp.Symbol,
    mode: Literal["parameter", "amplitude"],
    order: int,
    seed: int = 20260103,
) -> tuple[bool, int]:
    """Residual of the truncated solution in the unexpanded acceleration; returns (passed, sets)."""
    coeff_exprs = [harmonics[m] for harmonics in solved.w for m in sorted(harmonics)]
    layout = [sorted(harmonics) for harmonics in solved.w]
    symbols = sorted(
        (
            a_expr.free_symbols
            | q0.free_symbols
            | set().union(
                *(c.free_symbols for c in coeff_exprs), *(o.free_symbols for o in solved.omegas)
            )
        )
        - {q, eps_sym},
        key=lambda x: x.name,
    )
    needed = MIN_VALID_POINTS if symbols else 1
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    with mpmath.workdps(DPS):
        try:
            coefficients = sp.lambdify(symbols, coeff_exprs, "mpmath")
            omega_fn = sp.lambdify(symbols, solved.omegas, "mpmath")
            q0_fn = sp.lambdify(symbols, q0, "mpmath")
            a_args = [q, *([eps_sym] if mode == "parameter" else []), *symbols]
            a_fn = sp.lambdify(
                [x for x in a_args if x in a_expr.free_symbols or x == q], a_expr, "mpmath"
            )
        except (TypeError, ValueError, NameError):
            return False, 0
        a_uses_eps = eps_sym in a_expr.free_symbols
        a_symbols = [x for x in symbols if x in a_expr.free_symbols]
        passed_sets = 0
        for _ in range(SAMPLE_ATTEMPTS if symbols else 1):
            values = [mpmath.mpmathify(complex(_sample(s, rng))) for s in symbols]
            by_symbol = dict(zip(symbols, values, strict=True))
            try:
                flat_values = [mpmath.mpmathify(v) for v in coefficients(*values)]
                omegas = [mpmath.mpmathify(v) for v in omega_fn(*values)]
                q0_value = mpmath.mpmathify(q0_fn(*values))
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                continue
            flat: list[list[tuple[int, mpmath.mpf]]] = []
            at = 0
            for harmonics in layout:
                flat.append([(m, flat_values[at + i]) for i, m in enumerate(harmonics)])
                at += len(harmonics)
            ok = True
            for tau_value in TAUS:
                tau = mpmath.mpf(tau_value)
                remainders: list[mpmath.mpf] = []
                scale = mpmath.mpf(1)
                for eps in EPS_VALUES:
                    w, w2, omega = _w_value(flat, omegas, eps, tau)
                    lhs = omega**2 * w2
                    shift = eps * w if mode == "amplitude" else w
                    a_args_values = [q0_value + shift]
                    if a_uses_eps:
                        a_args_values.append(eps)
                    a_args_values += [by_symbol[x] for x in a_symbols]
                    a_value = _eval(a_fn, *a_args_values)
                    if a_value is None:
                        ok = False
                        break
                    rhs = a_value / eps if mode == "amplitude" else a_value
                    scale = max(scale, abs(lhs), abs(rhs))
                    remainders.append(abs(lhs - rhs))
                if not ok:
                    break
                if not (
                    _falls(remainders[0], remainders[1], order, scale)
                    and _falls(remainders[1], remainders[2], order, scale)
                ):
                    return False, passed_sets + 1
            if not ok:
                continue
            passed_sets += 1
            if passed_sets >= needed:
                break
    return passed_sets >= needed, passed_sets


# --- comparison with the verified numerical solution -------------------------------------------------
def _compare(
    request: CompareRequest,
    latex: str,
    coordinate: str,
    time_name: str,
    assumptions: dict[str, list[str]],
    mode: Literal["parameter", "amplitude"],
    small_name: str | None,
    q0: sp.Expr,
    solved: _Solved,
    amp: sp.Symbol | None,
    order: int,
    limits: Limits,
    budget: float,
) -> Comparison:
    scaling = (
        "max |u_numerical - u_perturbative| over the samples, divided by eps"
        if mode == "amplitude"
        else "max |u_numerical - u_perturbative| over the samples"
    )
    base = Comparison(
        status=Status.NOT_VERIFIED,
        epsilon=request.epsilon,
        predicted_order=order + 1,
        scaling=scaling,
    )
    if mode == "parameter" and request.amplitude is None:
        base.message = "Give the initial displacement (amplitude) for the comparison."
        base.status = Status.PARSE_ERROR
        return base
    if mode == "amplitude" and small_name is not None:  # pragma: no cover - guarded by caller
        return base
    run_limits = Limits(**{**limits.model_dump(), "timeout_seconds": max(budget / 2, 1.0) / 0.6})

    def run(eps_value: float) -> tuple[list[float], list[float], list[float]] | str:
        parameters = dict(request.parameters)
        if mode == "parameter" and small_name is not None:
            parameters[small_name] = eps_value
        displacement = eps_value if mode == "amplitude" else float(request.amplitude or 0.0)
        try:
            by_name = {s.name: s for s in q0.free_symbols}
            q0_value = float(
                q0.subs({by_name[n]: v for n, v in parameters.items() if n in by_name})
            )
        except (TypeError, ValueError, KeyError):
            return "The equilibrium position could not be evaluated at these parameter values."
        try:
            motion = MotionRequest(
                lagrangian=latex,
                coordinates=[coordinate],
                time=time_name,
                assumptions=assumptions,
                parameters=parameters,
                initial_positions=[q0_value + displacement],
                initial_velocities=[0.0],
                t_end=request.t_end,
                samples=COMPARE_SAMPLES,
            )
        except ValueError as exc:
            return str(exc)
        report = integrate_motion(motion, run_limits)
        if report.status in (Status.PARSE_ERROR, Status.UNSUPPORTED):
            return f"The numerical solution could not be set up: {report.message}"
        if (
            report.status is not Status.OK
            or not report.runs
            or report.runs[0].status is not Status.OK
        ):
            reason = report.message or (report.runs[0].message if report.runs else "")
            return f"The numerical solution did not pass its own checks ({reason or report.status.value})."
        numerical = report.runs[0].q[0]
        with mpmath.workdps(30):
            symbols = sorted(
                set().union(
                    *(c.free_symbols for h in solved.w for c in h.values()),
                    *(o.free_symbols for o in solved.omegas),
                ),
                key=lambda x: x.name,
            )
            values: list[float] = []
            for symbol in symbols:
                if amp is not None and symbol == amp:
                    values.append(float(request.amplitude or 0.0))
                elif symbol.name in parameters:
                    values.append(parameters[symbol.name])
                else:
                    return f"No value was given for the parameter {symbol.name}."
            try:
                coeff_exprs = [
                    (k, m, c) for k, h in enumerate(solved.w) for m, c in sorted(h.items())
                ]
                coeff_fn = sp.lambdify(symbols, [c for _, _, c in coeff_exprs], "mpmath")
                omega_fn = sp.lambdify(symbols, solved.omegas, "mpmath")
                coeffs = [mpmath.mpmathify(v) for v in coeff_fn(*values)]
                omegas = [mpmath.mpmathify(v) for v in omega_fn(*values)]
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                return "The solution could not be evaluated at these parameter values."
            flat: list[list[tuple[int, mpmath.mpf]]] = [[] for _ in solved.w]
            for (k, m, _), value in zip(coeff_exprs, coeffs, strict=True):
                flat[k].append((m, value))
            eps = mpmath.mpf(eps_value)
            omega = sum((eps**k * o for k, o in enumerate(omegas)), mpmath.mpf(0))
            perturbative: list[float] = []
            for t in report.t:
                w, _, _ = _w_value(flat, omegas, eps, omega * mpmath.mpf(t))
                perturbative.append(float(eps * w if mode == "amplitude" else w) + q0_value)
        return list(report.t), numerical, perturbative

    first = run(request.epsilon)
    if isinstance(first, str):
        base.message = first
        return base
    second = run(request.epsilon / 2)
    if isinstance(second, str):
        base.message = second
        return base
    t, numerical, perturbative = first
    divisor = request.epsilon if mode == "amplitude" else 1.0
    half_divisor = request.epsilon / 2 if mode == "amplitude" else 1.0
    error = max(abs(a - b) for a, b in zip(numerical, perturbative, strict=True)) / divisor
    error_half = max(abs(a - b) for a, b in zip(second[1], second[2], strict=True)) / half_divisor
    observed = math.log2(error / error_half) if error > 0 and error_half > 0 else None
    notes = [
        "These are measurements on this span for these numbers, not a claim for all times: "
        "the phase error of a truncated series grows with t."
    ]
    if observed is not None and observed < order:
        notes.append(
            f"The error fell at order {observed:.1f} when the small parameter was halved, below the "
            f"predicted {order + 1}: the span may be too long, or the small parameter too large, "
            "for this order."
        )
    return Comparison(
        status=Status.OK,
        message="Compared with the verified numerical solution at the given and at half the small parameter.",
        epsilon=request.epsilon,
        predicted_order=order + 1,
        t=t,
        numerical=[float(v) for v in numerical],
        perturbative=[float(v) for v in perturbative],
        max_error=error,
        max_error_half=error_half,
        observed_order=observed,
        scaling=scaling,
        notes=notes,
    )


# --- the operation -----------------------------------------------------------------------------------
def perturbation_solve(
    latex: str,
    coordinates: list[str],
    time_name: str,
    assumptions: dict[str, list[str]],
    small_parameter: str | None,
    equilibrium: str,
    order: int,
    compare: CompareRequest | None,
    limits: Limits,
) -> PerturbationReport:
    started = time.monotonic()
    deadline = started + 0.7 * limits.timeout_seconds
    if not 1 <= order <= MAX_ORDER:
        return _fail(Status.UNSUPPORTED, f"Give an order between 1 and {MAX_ORDER}.")
    if len(coordinates) != 1:
        return _fail(
            Status.UNSUPPORTED,
            "Perturbation solutions are implemented for one degree of freedom.",
        )
    prepared = prepare(
        latex, coordinates, time_name, assumptions, {coordinates[0]: equilibrium}, limits
    )
    if isinstance(prepared, Refusal):
        return _fail(prepared.status, prepared.message, order=order, **prepared.extra)
    d, q0 = prepared.d, prepared.q0[0]
    base: dict[str, object] = {**prepared.base, "order": order}
    q, qd, qdd = d.q[0], d.qd[0], d.qdd[0]
    acceleration = d.solution[qdd]
    if acceleration.has(qd):
        return _fail(
            Status.UNSUPPORTED,
            "The acceleration depends on the velocity (a position-dependent mass, damping or "
            "gyroscopic terms), and this method is implemented for accelerations that depend on the "
            "position only. Multiple scales and averaging are not implemented.",
            **base,
        )

    mode: Literal["parameter", "amplitude"] = (
        "amplitude" if small_parameter is None else "parameter"
    )
    taken = {s.name for s in d.params} | {q.name, qd.name, qdd.name, d.t.name}
    w_sym = _fresh("w", taken)
    amp_sym: sp.Symbol | None = None
    eps_sym: sp.Symbol
    if small_parameter is None:
        eps_sym = _fresh("epsilon", taken)
        amp: sp.Expr = sp.Integer(1)
    else:
        by_name = {s.name: s for s in d.params}
        if small_parameter not in by_name:
            return _fail(
                Status.PARSE_ERROR,
                f"{small_parameter} is not a parameter of L "
                f"(parameters: {', '.join(by_name) or 'none'}).",
                **base,
            )
        eps_sym = by_name[small_parameter]
        if eps_sym in q0.free_symbols:
            return _fail(
                Status.UNSUPPORTED,
                "The equilibrium may not depend on the small parameter.",
                **base,
            )
        amp_sym = sp.Symbol(_fresh("A", taken).name, positive=True)
        amp = amp_sym

    # Expansion coefficients c_0 .. c_N of  wddot + ... = sum eps^j c_j(w).
    substituted = acceleration.subs(q, q0 + (eps_sym * w_sym if mode == "amplitude" else w_sym))
    coefficients: list[sp.Expr] = []
    derivative = substituted
    for k in range(order + 2 if mode == "amplitude" else order + 1):
        if time.monotonic() > deadline:
            return _fail(Status.RESOURCE_LIMIT, "The time budget was used up.", **base)
        if k:
            derivative = sp.diff(derivative, eps_sym)
        value = derivative.subs(eps_sym, 0)
        if value.has(sp.zoo, sp.nan, sp.oo):
            return _fail(
                Status.UNSUPPORTED, "The acceleration is not analytic at the equilibrium.", **base
            )
        coefficients.append(sp.expand(sp.simplify(value / sp.factorial(k))))
    if mode == "amplitude":
        coefficients = coefficients[1:]  # b_0 = 0 at an equilibrium; c_j = b_{j+1}
        function = substituted / eps_sym
    else:
        function = substituted
    candidate = sum((c * eps_sym**k for k, c in enumerate(coefficients)), sp.Integer(0))
    expansion = check_series(function, eps_sym, sp.Integer(0), candidate, order)
    if not expansion.passed:
        return _fail(
            Status.NOT_VERIFIED,
            "The expansion of the equation of motion in the small parameter could not be verified, "
            f"so no solution is shown. {expansion.note}.",
            **base,
        )
    for c in coefficients:
        try:
            sp.Poly(c, w_sym)
        except sp.PolynomialError:
            return _fail(
                Status.UNSUPPORTED,
                "A perturbation term is not a polynomial in the displacement, so harmonic balance "
                "cannot be done exactly. For a smooth potential, choose the amplitude as the small "
                "parameter instead.",
                **base,
            )
    linear = sp.Poly(coefficients[0], w_sym)
    if linear.degree() > 1 or coefficients[0].subs(w_sym, 0) != 0:
        return _fail(
            Status.UNSUPPORTED,
            "With the small parameter set to zero the system is not a linear oscillator "
            "(the unperturbed acceleration is not proportional to the displacement).",
            **base,
        )
    s0 = sp.simplify(-linear.coeff_monomial(w_sym))
    if s0.is_positive is not True:
        return _fail(
            Status.UNSUPPORTED,
            "Cannot show that the unperturbed frequency squared "
            f"({sp.latex(s0)}) is positive from the declared assumptions, so an oscillation about "
            "this point is not established. Declare the parameters positive, or choose another point.",
            **base,
        )
    omega0 = sp.sqrt(s0)

    solved = _solve_orders(coefficients, w_sym, s0, omega0, amp, eps_sym)
    if time.monotonic() > deadline:
        return _fail(Status.RESOURCE_LIMIT, "The time budget was used up.", **base)

    # Checks.
    passed, sets = _check_residual(acceleration, q, q0, solved, eps_sym, mode, order)
    initial = _zero_check(
        sum(
            (sp.Abs(sum(h.values(), sp.Integer(0))) ** 2 for h in solved.w[1:]),
            sp.Integer(0),
        ),
        "w_k(0) = 0 for every correction",
    )
    verification = PerturbationVerification(
        residual_passed=passed,
        parameter_sets_checked=sets,
        initial_conditions=initial,
        initial_conditions_passed=initial.passed,
        bounded=True,
        expansion=expansion,
    )
    if not passed or not initial.passed:
        verification.note = (
            "the residual does not fall at the claimed order"
            if not passed
            else "the initial conditions do not hold"
        )
        return _fail(
            Status.NOT_VERIFIED,
            "A perturbation solution was produced but could not be verified, so it is not shown. "
            f"{verification.note}.",
            verification=verification,
            **base,
        )
    verification.passed = True
    verification.note = (
        "the residual in the unexpanded equation of motion falls at the claimed order, the initial "
        "conditions hold, and each order is a finite sum of cos(m tau) (no secular term)"
    )

    # Presentation.
    tau, t_sym = sp.Symbol(r"\tau"), sp.Symbol("t")
    omega_sym = sp.Symbol(r"\omega")
    small_latex = sp.latex(eps_sym)
    frequency = sum((eps_sym**k * o for k, o in enumerate(solved.omegas)), sp.Integer(0))
    ordered_frequency = sp.Add(
        *[eps_sym**k * o for k, o in enumerate(solved.omegas) if o != 0], evaluate=False
    )
    ratio_terms = [sp.simplify(o / omega0) for o in solved.omegas[1:]]
    period = [sp.Integer(1)]
    for n in range(1, order + 1):
        period.append(
            sp.simplify(-sum((ratio_terms[k - 1] * period[n - k] for k in range(1, n + 1)), 0))
        )
    period_poly = sum((eps_sym**k * p for k, p in enumerate(period)), sp.Integer(0))
    period_check = check_series(omega0 / frequency, eps_sym, sp.Integer(0), period_poly, order)
    orders: list[OrderTerms] = []
    for k, harmonics in enumerate(solved.w):
        shape = sum((c * sp.cos(m * tau) for m, c in sorted(harmonics.items())), sp.Integer(0))
        orders.append(
            OrderTerms(
                order=k,
                frequency_latex=(
                    rf"\omega_{{{k}}} = {sp.latex(solved.omegas[k])}" if k >= 1 else None
                ),
                shape_latex=rf"w_{{{k}}}(\tau) = {sp.latex(shape)}",
            )
        )
    shape_terms = []
    for k, harmonics in enumerate(solved.w):
        cosines = [
            c * sp.cos(m * omega_sym * t_sym) for m, c in sorted(harmonics.items()) if c != 0
        ]
        if cosines:
            shape_terms.append(eps_sym**k * sp.Add(*cosines, evaluate=False))
    full = sp.Add(*shape_terms, evaluate=False)
    displacement = eps_sym * full if mode == "amplitude" else full
    report = PerturbationReport(
        status=Status.OK,
        mode=mode,
        parameters=[s.name for s in d.params if s != eps_sym],
        small_latex=small_latex,
        amplitude_latex=sp.latex(amp_sym) if amp_sym is not None else None,
        linear_frequency_latex=rf"\omega_0 = {sp.latex(omega0)}",
        frequency_latex=rf"\omega = {sp.latex(ordered_frequency, order='none')}",
        period_ratio_latex=(
            rf"\frac{{T}}{{T_0}} = {sp.latex(period_poly, order='rev-lex')} + O\left({small_latex}^{{{order + 1}}}\right)"
            if period_check.passed
            else None
        ),
        solution_latex=rf"u(t) = {sp.latex(displacement, order='none')} + O\left({small_latex}^{{{order + 1}}}\right),\quad \omega t = \tau",
        orders=orders,
        verification=verification,
        **base,
    )
    if not period_check.passed:
        report.notes.append("The period series could not be verified, so it is not shown.")
    report.notes.append(
        "Initial conditions: the solution starts at rest at displacement "
        + (
            f"{sp.latex(amp_sym)} from the equilibrium."
            if amp_sym is not None
            else f"{small_latex} from the equilibrium (the small parameter is the amplitude)."
        )
    )
    if mode == "amplitude":
        report.notes.append(
            "The small parameter is the initial displacement itself; the series is local to small amplitudes."
        )
    report.message = (
        f"Poincare-Lindstedt solution to order {order}: the truncated solution leaves a residual in the "
        f"unexpanded equation of motion that falls at order {order + 1}."
    )

    if compare is not None:
        remaining = deadline - time.monotonic()
        if remaining < 1.0:
            report.notes.append("No time was left for the numerical comparison.")
        else:
            report.comparison = _compare(
                compare,
                latex,
                coordinates[0],
                time_name,
                assumptions,
                mode,
                small_parameter,
                q0,
                solved,
                amp_sym,
                order,
                limits,
                remaining,
            )
    return report
