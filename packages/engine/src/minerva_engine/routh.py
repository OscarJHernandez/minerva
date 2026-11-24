"""Routh reduction of cyclic coordinates (operation and tool `routh_reduce`, issue #153).

For coordinates `c` that the Lagrangian does not contain (cyclic), the conjugate momenta
`p_c = dL/d(c_dot)` are constants of motion. Holding them fixed at their values, the Routhian

    R(q, q_dot, p_c, t) = L - sum_c p_c c_dot      (with c_dot written in terms of p_c)

is a Lagrangian for the remaining coordinates q: their Euler-Lagrange equations of R are those of L
on every motion with these momenta, and the cyclic velocities are recovered as `c_dot = -dR/dp_c`.
This is the convention in which R plays the role of a Lagrangian; Goldstein (§8.3) and Landau and
Lifshitz (§41) define the Routhian with the opposite sign (`sum p_c c_dot - L`).

What is established, in order (any failure is a refusal, never a partial result):

1. the Lagrangian's equations of motion are confirmed by numerical differentiation of L
   (`lagrangian.check_equations`) and the accelerations are solved for;
2. each requested coordinate is cyclic (`dL/dc` is a symbolic zero) and its momentum is verified to be
   conserved on the equations of motion (`lagrangian.verify_constant`);
3. the momenta are linear in the cyclic velocities with a nonsingular block `d^2L/(dc_dot dc'_dot)`,
   which is solved for the velocities exactly; the determinant of the block is reported as a restriction;
4. the Routhian, *as printed and read back*, is checked at random points by a route that shares no code
   with the solve: the momenta are central differences of the original L, the accelerations come from
   the confirmed derivation, the Euler-Lagrange residual of R along that motion (nested numerical
   differentiation of R) must vanish, `-dR/dp_c` (a central difference) and the printed
   reconstruction formula must return the sampled cyclic velocity, and the value of R must equal
   `L - sum p_c c_dot` at the sample (without it R + C, R + f(t) or R + dF(q, t)/dt would pass: they
   have the same equations and the same `-dR/dp_c`). A removed coordinate or velocity left in R fails.

The symbolic leg (the accelerations from R minus those of L with `c_dot` replaced) is reported separately
and never decides; "proved" is said only when it is a symbolic zero. The held-fixed symbol is `p_{c}`, the
name the Lagrangian card prints; a symbol of L with that name is refused, never renamed. The value of a
momentum is never chosen here: it stays the symbol `p_{c}`.
"""

from __future__ import annotations

import random
import time as time_module
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mpmath
import sympy as sp
from pydantic import BaseModel, Field

from .errors import MinervaMathError, ParseError
from .lagrangian import (
    Derivation,
    DerivationError,
    check_equations,
    derive,
    verify_constant,
)
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import Limits, Status, Verification
from .notation.derivatives import resolve_derivatives
from .notation.models import InputAware, InputCode, InputDiagnostic, InputReading
from .restrictions import determinant_conditions
from .verify import EL_DPS, EL_REL_TOL, MIN_VALID_POINTS, SAMPLE_ATTEMPTS, _real_sample, _sample

ROUTH_POINTS = 6  # sample motions; every valid one must agree
SYMBOLIC_MAX_OPS = 4000  # above this the symbolic leg is "not run", never a refutation

# Closed vocabulary of refusals (a log `reason_code` may carry it; never content).
Reason = Literal[
    "not_cyclic",
    "not_a_coordinate",
    "no_coordinate_left",
    "nonlinear_momentum",
    "degenerate_block",
    "mass_matrix_singular",
    "momentum_not_conserved",
    "momentum_name_unreadable",
    "equations_not_confirmed",
    "reduction_not_confirmed",
]

CONVENTION = (
    "R = L - Σ p_c ċ with each cyclic velocity ċ written in terms of its momentum p_c: the remaining "
    "coordinates obey the Euler-Lagrange equations of R, and ċ = -∂R/∂p_c. Goldstein §8.3 and "
    "Landau-Lifshitz §41 call the negative of this the Routhian."
)


class HeldMomentum(BaseModel):
    """A momentum held fixed: the symbol R uses and its value in the original variables."""

    coordinate: str
    symbol_latex: str  # p_{t}
    latex: str  # p_{t} = <dL/d(t_dot)>
    reason: str
    verification: Verification | None = None


class Reconstruction(BaseModel):
    coordinate: str
    latex: str  # \dot{t} = <solved velocity>


class RouthChecks(BaseModel):
    """The deciding numerical legs and the separate symbolic leg. Counts only, never content."""

    euler_lagrange: list[Verification] = Field(default_factory=list)  # one per remaining coordinate
    reconstruction: list[Verification] = Field(default_factory=list)  # one per cyclic coordinate
    # R itself equals L - sum p_c c_dot at the sample (the two other legs cannot see R + C or R + dF/dt)
    definition: Verification | None = None
    symbolic: bool | None = None  # accelerations agree symbolically; None: not run or undecided
    symbolic_note: str = ""


class RouthReport(InputAware):
    status: Status
    reason: Reason | None = None
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    cyclic: list[str] = Field(default_factory=list)
    remaining: list[str] = Field(default_factory=list)
    time: str = "t"
    # R as printed and read back: the single result of the card.
    result_latex: str | None = None
    held_fixed: list[HeldMomentum] = Field(default_factory=list)
    reconstruction: list[Reconstruction] = Field(default_factory=list)
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    verification: Verification | None = (
        None  # numeric legs decide `passed`; symbolic leg reported apart
    )
    checks: RouthChecks = Field(default_factory=RouthChecks)
    convention: str = CONVENTION
    stages: dict[str, int] = Field(default_factory=dict)
    message: str = ""
    duration_ms: int = 0


def _fail(
    status: Status,
    message: str,
    reason: Reason | None = None,
    diagnostic: InputDiagnostic | None = None,
    **extra: Any,
) -> RouthReport:
    return RouthReport(
        status=status, message=message, reason=reason, diagnostic=diagnostic, **extra
    )


@dataclass
class Reduction:
    """The exact reduction: R, the solved cyclic velocities and the block determinant."""

    routhian: sp.Expr
    velocities: dict[sp.Symbol, sp.Expr]  # c_dot -> expression in p_c and the remaining variables
    determinant: sp.Expr


class ReductionError(Exception):
    def __init__(self, reason: Reason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def momentum_symbol(coordinate: sp.Symbol) -> sp.Symbol:
    """`p_{c}`: the name the Lagrangian card prints for the momentum conjugate to `c`."""
    return sp.Symbol(f"p_{coordinate.name}")


def solve_reduction(
    d: Derivation, cyclic: Sequence[int], momenta: Sequence[sp.Symbol]
) -> Reduction:
    """Solve `p_c = dL/d(c_dot)` for the cyclic velocities (a linear system) and form R.

    Refused (never approximated) when a momentum is not linear in the cyclic velocities or the block
    of second derivatives is singular."""
    velocities = [d.qd[i] for i in cyclic]
    block = sp.Matrix(
        len(cyclic), len(cyclic), lambda a, b: sp.diff(d.momenta[cyclic[a]], velocities[b])
    )
    if any(sp.simplify(sp.diff(entry, v)) != 0 for entry in block for v in velocities):
        raise ReductionError(
            "nonlinear_momentum",
            "A momentum of a requested coordinate is not linear in the cyclic velocities, so it cannot be "
            "solved for them uniquely; Routh reduction is not available here.",
        )
    determinant = sp.simplify(block.det())
    if determinant == 0:
        raise ReductionError(
            "degenerate_block",
            "The momenta of the requested coordinates do not determine their velocities (the matrix of "
            "second derivatives of L with respect to those velocities is singular), so they cannot be "
            "held fixed in their place.",
        )
    at_zero = {v: 0 for v in velocities}
    offset = sp.Matrix([d.momenta[i].subs(at_zero) for i in cyclic])
    solved = block.LUsolve(sp.Matrix(momenta) - offset)
    replace = {v: sp.simplify(solved[k]) for k, v in enumerate(velocities)}
    routhian = d.lagrangian.subs(replace) - sum(
        (p * replace[v] for p, v in zip(momenta, velocities, strict=True)), sp.Integer(0)
    )
    rest = [d.qd[i] for i in range(d.n) if i not in cyclic]
    return Reduction(
        routhian=tidy(routhian, [*rest, *momenta]), velocities=replace, determinant=determinant
    )


def tidy(expr: sp.Expr, generators: Sequence[sp.Symbol]) -> sp.Expr:
    """A readable form: one term per monomial in the velocities and momenta, each coefficient factored
    (`R = a(r) r_dot^2 + b(r) p_t^2 + ...`). Equal to `expr` exactly; anything else is simplified whole."""
    expanded = sp.expand(expr)
    try:
        poly = sp.Poly(expanded, *generators)
    except sp.PolynomialError:
        return sp.simplify(expr)
    if any(not coefficient.free_symbols.isdisjoint(generators) for coefficient in poly.coeffs()):
        return sp.simplify(expr)
    terms = [
        sp.factor(sp.cancel(coefficient))
        * sp.Mul(*(g**e for g, e in zip(generators, monomial, strict=True)))
        for monomial, coefficient in poly.terms()
    ]
    return sp.Add(*terms)


def _mp(value: sp.Expr) -> Any:
    number = complex(value)
    return mpmath.mpf(number.real) if number.imag == 0 else mpmath.mpc(number.real, number.imag)


def check_reduction(
    d: Derivation,
    cyclic: Sequence[int],
    momenta: Sequence[sp.Symbol],
    routhian: sp.Expr,
    velocities: dict[sp.Symbol, sp.Expr],
    *,
    seed: int = 20261124,
) -> tuple[list[Verification], list[Verification], Verification]:
    """The deciding numerical legs (see the module docstring). Shares no code with `solve_reduction`.

    Returns one `Verification` per remaining coordinate (Euler-Lagrange residual of R on the motion of L),
    one per cyclic coordinate (`-dR/dp_c` and the printed velocity formula against the sample), and one for
    the value of R against its definition `L - sum p_c c_dot` at the sample: the first two cannot tell R
    from R + C, R + f(t) or R + dF(q, t)/dt, which have the same equations and the same `-dR/dp`."""
    remaining = [i for i in range(d.n) if i not in cyclic]
    q_rem = [d.q[i] for i in remaining]
    qd_rem = [d.qd[i] for i in remaining]
    symbols_in = d.lagrangian.free_symbols | routhian.free_symbols
    params = sorted(symbols_in - {*d.q, *d.qd, d.t, *momenta}, key=lambda s: s.name)
    el_note = "the Euler-Lagrange residual of R, by numerical differentiation along motions of L, vanishes"
    rec_note = "-dR/dp (central difference) and the velocity formula return the sampled velocity"
    unreadable = "R could not be evaluated numerically"
    failed = (
        [Verification(note=unreadable) for _ in remaining],
        [Verification(note=unreadable) for _ in cyclic],
        Verification(note=unreadable),
    )
    removed = {*(d.q[i] for i in cyclic), *(d.qd[i] for i in cyclic)}
    if not removed.isdisjoint(routhian.free_symbols) or any(
        not removed.isdisjoint(velocities[d.qd[i]].free_symbols) for i in cyclic
    ):
        # A removed coordinate or velocity left in R (or in a velocity formula) is not a reduction.
        return failed
    try:
        f_l = sp.lambdify([*d.q, *d.qd, d.t, *params], d.lagrangian, "mpmath")
        f_r = sp.lambdify([*q_rem, *qd_rem, *momenta, d.t, *params], routhian, "mpmath")
        accelerations = [
            sp.lambdify([*d.q, *d.qd, d.t, *params], d.solution[d.qdd[i]], "mpmath")
            for i in remaining
        ]
        formulas = [
            sp.lambdify([*q_rem, *qd_rem, *momenta, d.t, *params], velocities[d.qd[i]], "mpmath")
            for i in cyclic
        ]
    except Exception:
        return failed
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    el_checked = [0] * len(remaining)
    el_agreed = [0] * len(remaining)
    rec_checked = [0] * len(cyclic)
    rec_agreed = [0] * len(cyclic)
    def_checked = def_agreed = 0
    with mpmath.workdps(EL_DPS):
        for _ in range(SAMPLE_ATTEMPTS):
            if min([*el_checked, *rec_checked, def_checked]) >= ROUTH_POINTS:
                break
            a = [mpmath.mpf(_real_sample(s, rng)) for s in d.q]
            b = [mpmath.mpf(rng.uniform(-2.0, 2.0)) for _ in d.q]
            t0 = mpmath.mpf(_real_sample(d.t, rng))
            pmp = [_mp(_sample(s, rng)) for s in params]
            point = _sample_point(f_l, accelerations, d.n, cyclic, a, b, t0, pmp)
            if point is None:
                continue
            p_values, acc = point
            a_rem = [a[i] for i in remaining]
            b_rem = [b[i] for i in remaining]
            for k in range(len(remaining)):
                residual = _el_residual(f_r, k, a_rem, b_rem, acc, p_values, t0, pmp)
                if residual is None:
                    continue
                value, scale = residual
                el_checked[k] += 1
                if abs(value) <= EL_REL_TOL * max(1.0, scale):
                    el_agreed[k] += 1
            value_check = _definition(f_l, f_r, cyclic, a, b, a_rem, b_rem, p_values, t0, pmp)
            if value_check is not None:
                def_checked += 1
                gap, size = value_check
                if gap <= EL_REL_TOL * max(1.0, size):
                    def_agreed += 1
            for k, i in enumerate(cyclic):
                expected = b[i]
                got = _reconstructed(f_r, formulas[k], k, a_rem, b_rem, p_values, t0, pmp)
                if got is None:
                    continue
                if any(g != g for g in got):
                    continue
                rec_checked[k] += 1
                scale = max(1.0, abs(float(expected)))
                if all(abs(g - float(expected)) <= EL_REL_TOL * scale for g in got):
                    rec_agreed[k] += 1

    def verdict(checked: int, agreed: int, note: str) -> Verification:
        passed = checked >= MIN_VALID_POINTS and agreed == checked
        return Verification(
            symbolic_zero=False,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=passed,
            note=note if passed else "too few valid sample points or a disagreement was found",
        )

    return (
        [verdict(c, g, el_note) for c, g in zip(el_checked, el_agreed, strict=True)],
        [verdict(c, g, rec_note) for c, g in zip(rec_checked, rec_agreed, strict=True)],
        verdict(
            def_checked,
            def_agreed,
            "R equals L - sum p_c c_dot at the sample (p_c: dL/dc_dot numerically)",
        ),
    )


def _definition(
    f_l: Callable[..., Any],
    f_r: Callable[..., Any],
    cyclic: Sequence[int],
    a: list[Any],
    b: list[Any],
    a_rem: list[Any],
    b_rem: list[Any],
    p_values: list[Any],
    t0: Any,
    pmp: list[Any],
) -> tuple[float, float] | None:
    """|R - (L - sum p_c c_dot)| at the sample and the size of the terms it is measured against."""
    try:
        lagrangian = complex(f_l(*a, *b, t0, *pmp))
        held = [complex(p * b[i]) for p, i in zip(p_values, cyclic, strict=True)]
        routhian = complex(f_r(*a_rem, *b_rem, *p_values, t0, *pmp))
    except (ArithmeticError, ValueError, TypeError, NameError):
        return None
    gap = abs(routhian - (lagrangian - sum(held)))
    if gap != gap or gap == float("inf"):
        return None
    return gap, max(abs(lagrangian), abs(routhian), *(abs(h) for h in held))


def _reconstructed(
    f_r: Callable[..., Any],
    formula: Callable[..., Any],
    k: int,
    a: list[Any],
    b: list[Any],
    p_values: list[Any],
    t0: Any,
    pmp: list[Any],
) -> list[complex] | None:
    """The k-th cyclic velocity as `-dR/dp_k` (central difference of R) and from the printed formula."""
    try:
        from_r = -mpmath.diff(
            lambda x: f_r(*a, *b, *_replace(p_values, k, x), t0, *pmp), p_values[k]
        )
        from_formula = formula(*a, *b, *p_values, t0, *pmp)
        return [complex(from_r), complex(from_formula)]
    except (ArithmeticError, ValueError, TypeError, NameError):
        return None


def _replace(values: list[Any], k: int, x: Any) -> list[Any]:
    out = list(values)
    out[k] = x
    return out


def _sample_point(
    f_l: Callable[..., Any],
    accelerations: list[Callable[..., Any]],
    n: int,
    cyclic: Sequence[int],
    a: list[Any],
    b: list[Any],
    t0: Any,
    pmp: list[Any],
) -> tuple[list[Any], list[Any]] | None:
    """Momenta as central differences of the original L, and the remaining accelerations of L."""
    try:
        p_values = []
        for i in cyclic:
            p_values.append(mpmath.diff(lambda x, i=i: f_l(*a, *_replace(b, i, x), t0, *pmp), b[i]))
        acc = [g(*a, *b, t0, *pmp) for g in accelerations]
        values = [complex(v) for v in [*p_values, *acc]]
    except (ArithmeticError, ValueError, TypeError, NameError):
        return None
    if any(v != v or abs(v) == float("inf") for v in values):
        return None
    return p_values, acc


def _el_residual(
    f_r: Callable[..., Any],
    k: int,
    a: list[Any],
    b: list[Any],
    acc: list[Any],
    p_values: list[Any],
    t0: Any,
    pmp: list[Any],
) -> tuple[complex, float] | None:
    """`d/ds (dR/dqdot_k) - dR/dq_k` at s = 0 along `q(s) = a + b s + acc s^2 / 2`, momenta fixed,
    and the size of the two terms (the residual is measured against them)."""
    m = len(a)

    def momentum(s: Any) -> Any:
        qs = [a[j] + b[j] * s + acc[j] * s**2 / 2 for j in range(m)]
        qds = [b[j] + acc[j] * s for j in range(m)]
        return mpmath.diff(
            lambda x: f_r(*qs, *_replace(qds, k, x), *p_values, t0 + s, *pmp), qds[k]
        )

    try:
        rate = complex(mpmath.diff(momentum, 0))
        force = complex(
            mpmath.diff(lambda x: f_r(*_replace(a, k, x), *b, *p_values, t0, *pmp), a[k])
        )
    except (ArithmeticError, ValueError, TypeError, NameError):
        return None
    value = rate - force
    if value != value or abs(value) == float("inf"):
        return None
    return value, max(abs(rate), abs(force))


def _symbolic_leg(
    d: Derivation, cyclic: Sequence[int], routhian: sp.Expr, velocities: dict[sp.Symbol, sp.Expr]
) -> tuple[bool | None, str]:
    """The Euler-Lagrange expressions of R, evaluated on the accelerations of L with the cyclic velocities
    replaced, must vanish exactly. Reported apart; never decides, and "not reduced to 0" is undecided."""
    remaining = [i for i in range(d.n) if i not in cyclic]
    if sp.count_ops(routhian) > SYMBOLIC_MAX_OPS:
        return None, "not run: the Routhian is too large for the symbolic comparison"
    on_motion = {d.qdd[i]: d.solution[d.qdd[i]].subs(velocities) for i in remaining}
    for i in remaining:
        rate = sp.diff(routhian, d.qd[i])
        total = sp.diff(rate, d.t) + sum(
            (
                sp.diff(rate, d.q[j]) * d.qd[j] + sp.diff(rate, d.qd[j]) * d.qdd[j]
                for j in remaining
            ),
            sp.Integer(0),
        )
        residual = (total - sp.diff(routhian, d.q[i])).xreplace(on_motion)
        if sp.count_ops(residual) > SYMBOLIC_MAX_OPS:
            return None, "not run: the symbolic residual is too large"
        if sp.cancel(sp.together(residual)) == 0:
            continue
        if sp.simplify(residual) != 0:
            return None, "undecided: the symbolic residual was not reduced to 0"
    return True, (
        "the Euler-Lagrange expressions of R vanish symbolically on the accelerations of L with the "
        "cyclic velocities replaced"
    )


def _read_back(
    latex: str,
    d: Derivation,
    momenta: Sequence[sp.Symbol],
    remaining: list[str],
    assumptions: dict[str, list[str]],
    limits: Limits,
) -> sp.Expr:
    """R as a reader would read the printed text: the result shown is the one checked."""
    expr = parse_latex(latex, assumptions, limits)
    expr = resolve_derivatives(expr, remaining, d.t.name, assumptions).expr
    targets = {s.name: s for s in [*d.q, *d.qd, d.t, *d.params, *momenta]}
    return expr.xreplace({s: targets[s.name] for s in expr.free_symbols if s.name in targets})


def reduce(
    d: Derivation,
    cyclic: Sequence[str],
    *,
    assumptions: dict[str, list[str]] | None = None,
    limits: Limits | None = None,
) -> RouthReport:
    """Routh reduction of an already derived Lagrangian by the named cyclic coordinates.

    Reusable by callers that hold a `Derivation` (discovery of conserved quantities, issue #152). The
    equations of `d` are checked here (`check_equations`) whether or not the caller did."""
    assumptions = assumptions or {}
    limits = limits or Limits()
    show = d.show
    base = {
        "lagrangian_latex": show(d.lagrangian),
        "coordinates": list(d.coordinates),
        "cyclic": list(cyclic),
        "time": d.t.name,
    }
    unknown = [c for c in cyclic if c not in d.coordinates]
    if unknown or not cyclic or len(set(cyclic)) != len(cyclic):
        return _fail(
            Status.PARSE_ERROR,
            "Name each coordinate to remove once, from the declared coordinates"
            + (f" (not declared: {', '.join(unknown)})." if unknown else "."),
            "not_a_coordinate",
            **base,
        )
    index = [d.coordinates.index(c) for c in cyclic]
    remaining = [i for i in range(d.n) if i not in index]
    if not remaining:
        return _fail(
            Status.UNSUPPORTED,
            "At least one coordinate must remain after the reduction.",
            "no_coordinate_left",
            **base,
        )
    base["remaining"] = [d.coordinates[i] for i in remaining]

    momenta = [momentum_symbol(d.q[i]) for i in index]
    taken = {s.name for s in d.lagrangian.free_symbols} | {s.name for s in [*d.q, *d.qd, d.t]}
    clash = [p.name for p in momenta if p.name in taken]
    if clash:
        return _fail(
            Status.PARSE_ERROR,
            "The reduction holds each momentum fixed as the symbol "
            + ", ".join(f"`{to_latex(sp.Symbol(name))}`" for name in clash)
            + ", which the Lagrangian already uses for something else. Rename that symbol in the "
            "Lagrangian; nothing is renamed here.",
            None,
            InputDiagnostic(
                code=InputCode.MOMENTUM_SYMBOL_CLASH,
                field="lagrangian",
                symbols=clash,
                suggestions=[],
                recoverable=True,
            ),
            **base,
        )
    for p in momenta:
        try:
            back = parse_latex(to_latex(p), {}, limits)
        except (ParseError, MinervaMathError):
            back = None
        if not (isinstance(back, sp.Symbol) and back.name == p.name):
            return _fail(
                Status.UNSUPPORTED,
                f"The momentum of this coordinate would be written `{to_latex(p)}`, which cannot be read "
                "back as one symbol; use a coordinate name without a subscript.",
                "momentum_name_unreadable",
                **base,
            )

    started = time_module.perf_counter()
    checks = check_equations(d)
    stages = dict(d.stages)
    stages["check"] = round((time_module.perf_counter() - started) * 1000)
    if not all(c.passed for c in checks):
        return _fail(
            Status.NOT_VERIFIED,
            "The equations of motion of this Lagrangian could not be confirmed by numerical "
            "differentiation, so nothing is reduced.",
            "equations_not_confirmed",
            stages=stages,
            **base,
        )

    not_cyclic = [
        d.coordinates[i] for i in index if sp.simplify(sp.diff(d.lagrangian, d.q[i])) != 0
    ]
    if not_cyclic:
        return _fail(
            Status.UNSUPPORTED,
            f"{', '.join(not_cyclic)} {'is' if len(not_cyclic) == 1 else 'are'} not cyclic: the Lagrangian "
            "depends on it, so its momentum is not conserved and cannot be held fixed.",
            "not_cyclic",
            stages=stages,
            **base,
        )

    started = time_module.perf_counter()
    try:
        reduction = solve_reduction(d, index, momenta)
    except ReductionError as exc:
        return _fail(Status.UNSUPPORTED, exc.message, exc.reason, stages=stages, **base)
    stages["solve"] = round((time_module.perf_counter() - started) * 1000)

    if not d.solution:
        return _fail(
            Status.UNSUPPORTED,
            "The mass matrix of the Lagrangian is singular, so its accelerations (needed to confirm that "
            "the momenta are conserved and to check the reduction) cannot be solved for.",
            "mass_matrix_singular",
            stages=stages,
            **base,
        )
    started = time_module.perf_counter()
    held: list[HeldMomentum] = []
    for i, p in zip(index, momenta, strict=True):
        ok, verification = verify_constant(d, d.momenta[i])
        if not ok:
            return _fail(
                Status.NOT_VERIFIED,
                f"The momentum conjugate to {d.coordinates[i]} could not be verified to be conserved on the "
                "equations of motion, so it is not held fixed.",
                "momentum_not_conserved",
                stages=stages,
                **base,
            )
        held.append(
            HeldMomentum(
                coordinate=d.coordinates[i],
                symbol_latex=to_latex(p),
                latex=f"{to_latex(p)} = {show(d.momenta[i])}",
                reason=(
                    f"conserved because {d.coordinates[i]} is cyclic (L does not depend on it); verified "
                    "on the equations of motion"
                ),
                verification=verification,
            )
        )
    stages["conserved"] = round((time_module.perf_counter() - started) * 1000)

    result_latex = show(reduction.routhian)
    try:
        shown = _read_back(
            result_latex,
            d,
            momenta,
            base["remaining"],
            assumptions,
            limits,
        )
    except (ParseError, MinervaMathError):
        return _fail(
            Status.NOT_VERIFIED,
            "The Routhian could not be read back from its printed form, so it is not shown.",
            "reduction_not_confirmed",
            stages=stages,
            **base,
        )
    reconstruction = [
        Reconstruction(
            coordinate=d.coordinates[i],
            latex=f"{show(d.qd[i])} = {show(reduction.velocities[d.qd[i]])}",
        )
        for i in index
    ]

    started = time_module.perf_counter()
    el_checks, rec_checks, definition = check_reduction(
        d, index, momenta, shown, reduction.velocities
    )
    stages["verify"] = round((time_module.perf_counter() - started) * 1000)
    started = time_module.perf_counter()
    symbolic, symbolic_note = _symbolic_leg(d, index, shown, reduction.velocities)
    stages["symbolic"] = round((time_module.perf_counter() - started) * 1000)
    route = RouthChecks(
        euler_lagrange=el_checks,
        reconstruction=rec_checks,
        definition=definition,
        symbolic=symbolic,
        symbolic_note=symbolic_note,
    )
    legs = [*el_checks, *rec_checks, definition]
    passed = all(c.passed for c in legs)
    checked = sum(c.numeric_points_checked for c in legs)
    agreed = sum(c.numeric_points_agreed for c in legs)
    if not passed:
        return _fail(
            Status.NOT_VERIFIED,
            "The reduced Lagrangian (Routhian) did not pass the numerical check against the motion of the "
            "original Lagrangian, so it is not shown.",
            "reduction_not_confirmed",
            checks=route,
            stages=stages,
            **base,
        )
    restrictions, complete = determinant_conditions(reduction.determinant)
    report = RouthReport(
        status=Status.OK,
        reading=InputReading(format=d.format, normalizations=list(d.normalizations)),
        result_latex=result_latex,
        held_fixed=held,
        reconstruction=reconstruction,
        restrictions_latex=restrictions,
        restrictions_complete=complete,
        verification=Verification(
            symbolic_zero=symbolic is True,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=True,
            note="Euler-Lagrange residual of R on motions of L, and the reconstruction of the velocities",
        ),
        checks=route,
        stages=stages,
        message=(
            f"Reduced by {', '.join(cyclic)}: the Routhian R governs "
            f"{', '.join(base['remaining'])} with the momentum of "
            + ", ".join(h.coordinate for h in held)
            + " held fixed. "
            + (
                "Proved: the accelerations from R equal those of L symbolically, and checked numerically."
                if symbolic is True
                else "Checked numerically on motions of the original Lagrangian (not proved symbolically)."
            )
        ),
        **base,
    )
    return report


def routh_reduce(
    latex: str,
    coordinates: list[str],
    time: str,
    assumptions: dict[str, list[str]],
    cyclic: list[str],
    limits: Limits,
    format: Literal["latex", "text"] = "latex",
) -> RouthReport:
    """The operation: read the Lagrangian (as `analyze_lagrangian` does) and reduce it."""
    try:
        d = derive(latex, coordinates, time, assumptions, limits, format)
    except DerivationError as exc:
        return _fail(exc.status, exc.message, None, exc.diagnostic)
    return reduce(d, cyclic, assumptions=assumptions, limits=limits)
