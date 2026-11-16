"""Total time derivative along a motion (tool and operation `time_derivative`).

`differentiate` takes partial derivatives: by `t` of `\\mu r^2 \\dot\\theta` it answers 0, because the
coordinates and velocities are constants for a partial derivative. The derivative along a motion q(t)
is the chain rule

    d f/dt = ∂f/∂t + Σ_i ( ∂f/∂q_i q̇_i + ∂f/∂q̇_i q̈_i ).

The candidate is shown only if a central difference in t of the *input* expression, evaluated along
random polynomial motions (exact derivatives of the polynomial give q̇ and q̈, SymPy's `diff` is not
used), agrees with it at every valid sample point to 1e-12. With a Lagrangian the accelerations are
those of its Euler-Lagrange equations, confirmed numerically by `lagrangian.check_equations`; a zero
"on the equations of motion" is decided exactly as for a constant of motion
(`lagrangian.verify_zero_derivative`) and is worded that way, never as "conserved in general".
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import mpmath as mp
import sympy as sp

from .differentiate import (
    DerivativeStep,
    DerivativeVerification,
    DifferentiateResult,
    DiffVariable,
    _fail,
)
from .errors import LimitError, MinervaMathError, UnsupportedInput
from .lagrangian import (
    MAX_COORDINATES,
    DerivationError,
    check_equations,
    derive,
    reject_equation,
    verify_zero_derivative,
)
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status
from .variational import polypath
from .verify import _sample

DPS = 40
STEP = mp.mpf("1e-10")
REL_TOL = mp.mpf("1e-12")
MIN_POINTS = 6
MAX_POINTS = 10
ATTEMPTS = 60
DEGREE = 4
TIDY_MAX_OPS = 120


def _mp(value: sp.Expr) -> Any:
    number = sp.N(value, 30)
    re_part, im_part = number.as_real_imag()
    real = mp.mpf(str(re_part))
    return real if im_part == 0 else mp.mpc(real, mp.mpf(str(im_part)))


def _coordinate_start(symbol: sp.Symbol, rng: random.Random) -> mp.mpf:
    """The constant term of a polynomial motion, consistent with the symbol's declared sign."""
    if symbol.is_positive or symbol.is_nonnegative:
        return mp.mpf(rng.uniform(0.4, 2.0))
    if symbol.is_negative or symbol.is_nonpositive:
        return mp.mpf(rng.uniform(-2.0, -0.4))
    return mp.mpf(rng.uniform(-1.0, 1.0))


def check_total_derivative(
    f: sp.Expr,
    candidate: sp.Expr,
    q: Sequence[sp.Symbol],
    qd: Sequence[sp.Symbol],
    qdd: Sequence[sp.Symbol],
    t: sp.Symbol,
    *,
    seed: int = 20260401,
) -> DerivativeVerification:
    """Compare `candidate` with a central difference in t of `f` along random polynomial motions."""
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    motion = {*q, *qd, *qdd, t}
    others = sorted((f.free_symbols | candidate.free_symbols) - motion, key=lambda s: s.name)
    checked = agreed = 0
    with mp.workdps(DPS):
        f_fn = sp.lambdify([*q, *qd, t, *others], f, modules="mpmath")
        c_fn = sp.lambdify([*q, *qd, *qdd, t, *others], candidate, modules="mpmath")
        for _ in range(ATTEMPTS):
            if checked >= MAX_POINTS:
                break
            params = [_mp(_sample(s, rng)) for s in others]
            tables = []
            for symbol in q:
                coefficients = [_coordinate_start(symbol, rng)] + [
                    mp.mpf(rng.uniform(-1, 1)) * mp.mpf("0.15") / (1 + mp.mpf("0.5") * k)
                    for k in range(1, DEGREE + 1)
                ]
                tables.append(polypath.derivatives_table(coefficients, 2))
            s0 = mp.mpf(rng.uniform(0.1, 0.9))

            def motion_at(s: mp.mpf, order: int, tables: list[Any] = tables) -> list[Any]:
                return [polypath.evaluate(table[order], s) for table in tables]

            try:
                plus = f_fn(*motion_at(s0 + STEP, 0), *motion_at(s0 + STEP, 1), s0 + STEP, *params)
                minus = f_fn(*motion_at(s0 - STEP, 0), *motion_at(s0 - STEP, 1), s0 - STEP, *params)
                want = c_fn(*motion_at(s0, 0), *motion_at(s0, 1), *motion_at(s0, 2), s0, *params)
                fd = (plus - minus) / (2 * STEP)
                if not (mp.isfinite(fd) and mp.isfinite(want)):
                    continue
            except (ArithmeticError, ValueError, TypeError):
                continue
            checked += 1
            scale = max(mp.mpf(1), abs(fd), abs(want))
            agreed += int(abs(fd - want) <= REL_TOL * scale)
    passed = checked >= MIN_POINTS and agreed == checked
    if checked < MIN_POINTS:
        note = "too few valid sample motions to verify numerically"
    elif agreed != checked:
        note = "the numerical derivative along the motion disagrees with the symbolic one"
    else:
        note = "the numerical derivative along random motions agreed at every sampled point"
    return DerivativeVerification(
        method="central difference in t of the input along random polynomial motions, 40 digits",
        directions=["real"],
        steps_checked=1,
        points_checked=checked,
        points_agreed=agreed,
        passed=passed,
        note=note,
    )


def _chain_rule(
    f: sp.Expr,
    q: Sequence[sp.Symbol],
    qd: Sequence[sp.Symbol],
    qdd: Sequence[sp.Symbol],
    t: sp.Symbol,
) -> sp.Expr:
    out = sp.diff(f, t)
    for qi, qdi, qddi in zip(q, qd, qdd, strict=True):
        out += sp.diff(f, qi) * qdi + sp.diff(f, qdi) * qddi
    return out


def time_derivative(
    data: ExpressionInput,
    coordinates: list[str],
    time: str,
    lagrangian: str | None,
    limits: Limits,
) -> DifferentiateResult:
    from .ops import parse_expression  # local import: ops imports this module

    if not coordinates or len(coordinates) > MAX_COORDINATES:
        return _fail(Status.UNSUPPORTED, f"Give between 1 and {MAX_COORDINATES} coordinates.")
    if len(set(coordinates)) != len(coordinates):
        return _fail(Status.PARSE_ERROR, "Coordinates must be distinct.")
    variables = [DiffVariable(name=time, order=1)]
    try:
        if lagrangian is not None:
            reject_equation(lagrangian, "Lagrangian")
        expr = parse_expression(data, limits)
    except DerivationError as exc:
        return _fail(exc.status, exc.message, variables=variables, total_order=1)
    except LimitError as exc:
        return _fail(Status.RESOURCE_LIMIT, str(exc), variables=variables, total_order=1)
    except UnsupportedInput as exc:
        return _fail(Status.UNSUPPORTED, str(exc), variables=variables, total_order=1)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc), variables=variables, total_order=1)

    by_name = {s.name: s for s in expr.free_symbols}
    q = [by_name.get(c) or sp.Symbol(c) for c in coordinates]
    qd = [by_name.get(f"{c}_dot") or sp.Symbol(f"{c}_dot") for c in coordinates]
    qdd = [sp.Symbol(f"{c}_ddot") for c in coordinates]
    t = by_name.get(time) or sp.Symbol(time)
    names: dict[sp.Symbol, str] = {}
    for coordinate, velocity, acceleration in zip(q, qd, qdd, strict=True):
        base = to_latex(coordinate)
        names[velocity], names[acceleration] = rf"\dot{{{base}}}", rf"\ddot{{{base}}}"

    def show(e: sp.Expr) -> str:
        return str(to_latex(e, symbol_names=names))

    input_latex = show(expr)
    base_fields: dict[str, Any] = {
        "input_latex": input_latex,
        "variables": variables,
        "total_order": 1,
    }
    if any(s.name.endswith("_ddot") for s in expr.free_symbols):
        return _fail(
            Status.UNSUPPORTED,
            "The expression contains accelerations; its total time derivative needs third "
            "derivatives of the coordinates, which are not supported.",
            **base_fields,
        )
    known = {s.name for s in (*q, *qd, t)}
    for symbol in expr.free_symbols:
        if symbol.name.endswith("_dot") and symbol.name not in known:
            return _fail(
                Status.UNSUPPORTED,
                f"{symbol.name} is the velocity of {symbol.name[:-4]}, which is not among the "
                "coordinates; list it as a coordinate.",
                **base_fields,
            )
    if not (expr.free_symbols & {*q, *qd, t}):
        return _fail(
            Status.UNSUPPORTED,
            "The expression does not depend on the coordinates, their velocities or the time, so "
            "its derivative along a motion is 0 trivially.",
            **base_fields,
        )

    raw = _chain_rule(expr, q, qd, qdd, t)
    if raw.has(sp.Derivative, sp.Subs):
        return _fail(
            Status.UNSUPPORTED,
            "This expression has no complex derivative (for example |z|, re z, im z or conjugates).",
            **base_fields,
        )
    candidates = [raw]
    if sp.count_ops(raw) <= TIDY_MAX_OPS:
        candidates.insert(0, sp.simplify(raw))
    final = raw
    check: DerivativeVerification | None = None
    for candidate in dict.fromkeys(candidates):
        check = check_total_derivative(expr, candidate, q, qd, qdd, t)
        if check.passed:
            final = candidate
            break
    if check is None or not check.passed:
        note = check.note if check is not None else ""
        return _fail(
            Status.NOT_VERIFIED,
            f"A derivative was produced but could not be verified, so it is not shown. {note}.",
            verification=check,
            **base_fields,
        )

    notes = [
        "Total derivative along a motion q(t): velocities and accelerations are the first and "
        "second time derivatives of the coordinates (chain rule); this is not a partial derivative."
    ]
    steps = [DerivativeStep(variable=time, order_so_far=1, latex=show(final))]
    extra: dict[str, Any] = {}
    if lagrangian is not None:
        failure = _on_motion(
            final, lagrangian, coordinates, time, data, limits, q, qdd, show, steps, notes, extra
        )
        if failure is not None:
            return _fail(failure[0], failure[1], verification=check, **base_fields)
    return DifferentiateResult(
        status=Status.OK,
        notation_latex=rf"\frac{{d}}{{d{time}}}\left({input_latex}\right)",
        result_latex=show(final),
        result_text=str(final),
        steps=steps,
        verification=check,
        notes=notes,
        message="Total time derivative along a motion; verified against a central difference of the "
        "input along random motions.",
        **base_fields,
        **extra,
    )


def _on_motion(
    final: sp.Expr,
    lagrangian: str,
    coordinates: list[str],
    time: str,
    data: ExpressionInput,
    limits: Limits,
    q: Sequence[sp.Symbol],
    qdd: Sequence[sp.Symbol],
    show: Any,
    steps: list[DerivativeStep],
    notes: list[str],
    extra: dict[str, Any],
) -> tuple[Status, str] | None:
    """Substitute the Lagrangian's accelerations; fills `extra`, `steps`, `notes`; returns a failure."""
    try:
        d = derive(lagrangian, coordinates, time, data.assumptions, limits, data.format)
    except DerivationError as exc:
        return exc.status, exc.message
    if len(d.solution) != len(coordinates):
        return (
            Status.UNSUPPORTED,
            "The accelerations cannot be solved from this Lagrangian (singular mass matrix), so the "
            "derivative cannot be evaluated on its equations of motion.",
        )
    if not all(v.passed for v in check_equations(d)):
        return (
            Status.NOT_VERIFIED,
            "The Euler-Lagrange equations of this Lagrangian could not be confirmed numerically, so "
            "the derivative is not evaluated on them.",
        )
    mine = {s.name: s for s in final.free_symbols | set(q)}
    substitution = {
        acceleration: value.xreplace(
            {s: mine[s.name] for s in value.free_symbols if s.name in mine}
        )
        for acceleration, value in d.solution.items()
    }
    on_motion = final.xreplace(substitution)
    passed, ver = verify_zero_derivative(on_motion)
    checked, agreed = ver.numeric_points_checked, ver.numeric_points_agreed
    if passed:
        extra.update(on_motion_latex="0", zero_on_motion=True)
        extra["symbolic_zero_on_motion"] = ver.symbolic_zero
        notes.append(
            "Zero on the equations of motion of the Lagrangian given (confirmed numerically"
            + ("; also a symbolic zero" if ver.symbolic_zero else "; no symbolic zero found")
            + "). That is a statement about this Lagrangian's motions, not about other systems."
        )
        steps.append(DerivativeStep(variable=time, order_so_far=1, latex="0"))
    elif checked >= 4 and agreed < checked:
        shown = show(sp.simplify(on_motion))
        extra.update(on_motion_latex=shown, zero_on_motion=False)
        notes.append(
            "Not zero on the equations of motion: the expression below substitutes the accelerations "
            "of the confirmed Euler-Lagrange equations (an exact substitution, not separately "
            "sampled), so it is not a constant of motion."
        )
        steps.append(DerivativeStep(variable=time, order_so_far=1, latex=shown))
    else:
        extra.update(zero_on_motion=None)
        notes.append("Whether this vanishes on the equations of motion could not be decided here.")
    return None
