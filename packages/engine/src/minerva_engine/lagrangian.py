"""Lagrangian mechanics: equations of motion and conserved quantities.

Given a Lagrangian L(q, q-dot, t) and the generalised coordinates, this derives the Euler-Lagrange
equations, the conjugate momenta and (when the mass matrix is invertible) the accelerations, and looks
for constants of motion:

* a cyclic coordinate (L does not depend on q_i) conserves its momentum p_i;
* no explicit time dependence conserves the energy function h = sum(q-dot_i p_i) - L;
* a rotation of a pair of coordinates that leaves L unchanged conserves q_i p_j - q_j p_i.

Every derived equation is checked against numerical differentiation of L (no SymPy differentiation).
Every constant is verified: its time derivative is evaluated on the equations of motion and must be
zero, symbolically and at random numerical points. The search covers these families only; symmetries
that hold up to a total time derivative (for example Galilean boosts) or hidden symmetries (for example
the Runge-Lenz vector) are not found, and the report says so.
"""

from __future__ import annotations

import itertools
import time as time_module
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from .clock import TimeUp, stop_at
from .errors import MinervaMathError, ParseError
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import Limits, Status, Verification
from .notation.derivatives import (
    join_spaced_keywords,
    refuse_latex_letter_runs,
    resolve_derivatives,
)
from .notation.models import (
    InputAware,
    InputCode,
    InputDiagnostic,
    InputReading,
    Normalization,
    NormalizationKind,
)
from .notation.names import alias_in_latex, clean_name, near_misses
from .notation.time_symbol import time_clash, time_clash_message
from .restrictions import determinant_conditions
from .textparse import parse_text
from .verify import clearly_nonzero, euler_lagrange_check, numeric_check

MAX_COORDINATES = 8


class EquationOfMotion(BaseModel):
    coordinate: str
    latex: str
    verification: Verification | None = None


class NamedExpression(BaseModel):
    coordinate: str
    latex: str


class ConservedQuantity(BaseModel):
    name: str
    kind: Literal["cyclic", "energy", "rotation"]
    expression_latex: str
    reason: str
    verified: bool
    verification: Verification | None = None


class LagrangianReport(InputAware):
    status: Status
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    time_dependent: bool = False
    equations: list[EquationOfMotion] = Field(default_factory=list)
    momenta: list[NamedExpression] = Field(default_factory=list)
    accelerations: list[NamedExpression] = Field(default_factory=list)
    conserved: list[ConservedQuantity] = Field(default_factory=list)
    tested: list[str] = Field(default_factory=list)
    # Candidate constants of motion the time budget did not reach: neither found nor ruled out.
    not_searched: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    # Where the solution is not defined: the mass matrix is divided by in solving for the accelerations,
    # so its determinant's factors must be nonzero (coordinate singularities such as the poles).
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    # Milliseconds per stage (parse, derive, solve, check, conserved): counters only, never content.
    stages: dict[str, int] = Field(default_factory=dict)
    message: str = ""
    duration_ms: int = 0


def _fail(
    status: Status, message: str, diagnostic: InputDiagnostic | None = None
) -> LagrangianReport:
    return LagrangianReport(status=status, message=message, diagnostic=diagnostic)


class DerivationError(Exception):
    """The input cannot be analysed; carries the explicit status to report."""

    def __init__(
        self, status: Status, message: str, diagnostic: InputDiagnostic | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.diagnostic = diagnostic


@dataclass
class Derivation:
    """The symbolic result of the Euler-Lagrange derivation, shared by the analyser and the integrator."""

    lagrangian: sp.Expr
    coordinates: list[str]
    q: list[sp.Symbol]
    qd: list[sp.Symbol]
    qdd: list[sp.Symbol]
    t: sp.Symbol
    params: list[sp.Symbol]
    momenta: list[sp.Expr]
    euler_raw: list[sp.Expr]  # expanded, used to solve for the accelerations
    euler: list[sp.Expr]  # simplified, shown to the user
    time_dependent: bool
    solution: dict[sp.Symbol, sp.Expr] = field(default_factory=dict)  # qdd_i -> expression
    names: dict[sp.Symbol, str] = field(default_factory=dict)
    mass_determinant: sp.Expr | None = None  # divided by when solving for the accelerations
    format: Literal["latex", "text"] = "latex"  # the syntax the source was read in
    normalizations: list[Normalization] = field(
        default_factory=list
    )  # spellings that were read as others
    stages: dict[str, int] = field(default_factory=dict)  # milliseconds per stage

    @property
    def n(self) -> int:
        return len(self.q)

    def show(self, expr: sp.Expr) -> str:
        return str(to_latex(expr, symbol_names=self.names))

    def total_derivative(self, f: sp.Expr) -> sp.Expr:
        """Total time derivative along a motion: q' = qd, qd' = qdd."""
        out = sp.diff(f, self.t)
        for qi, qdi, qddi in zip(self.q, self.qd, self.qdd, strict=True):
            out += sp.diff(f, qi) * qdi + sp.diff(f, qdi) * qddi
        return out


def derive(
    latex: str,
    coordinates: list[str],
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
    format: Literal["latex", "text"] = "latex",
) -> Derivation:
    """`latex` is the Lagrangian as typed, in the syntax `format` names (the argument keeps its old name).

    In text a velocity is written `q_dot` (LaTeX: `\\dot{q}`, which the parser turns into the same symbol);
    other spellings of a velocity of a declared coordinate (`qdot`, `dq_dt`, `q dot`) are read as that
    symbol by `notation.derivatives` and reported in `Derivation.normalizations`.
    """
    notes: list[Normalization] = [
        Normalization(
            kind=NormalizationKind.DECLARED_NAME,
            source=given,
            target=clean_name(given),
            note="the LaTeX spelling of the same name",
        )
        for given in coordinates
        if clean_name(given) != given
    ]
    coordinates = [clean_name(c) for c in coordinates]
    _check_coordinates(coordinates)
    clash = time_clash(coordinates, time, source=latex)
    if clash is not None:
        raise DerivationError(Status.PARSE_ERROR, time_clash_message(clash), clash)
    reject_equation(latex, "Lagrangian")
    started = time_module.perf_counter()
    try:
        if format == "text":
            latex, joined = join_spaced_keywords(latex, coordinates)
            notes.extend(joined)
            parsed = parse_text(latex, assumptions, limits)
        else:
            refuse_latex_letter_runs(latex, coordinates)
            parsed = parse_latex(latex, assumptions, limits)
        resolved = resolve_derivatives(parsed, coordinates, time, assumptions)
        names = {sym.name for sym in resolved.expr.free_symbols}
        if format == "latex" and any(n not in names for n in coordinates):
            # `x1` cannot be one symbol in LaTeX: read it as the `x_1` the input writes, if that is the only one.
            aliases, read_as = alias_in_latex(coordinates, names)
            if aliases:
                coordinates = [aliases.get(c, c) for c in coordinates]
                _check_coordinates(coordinates)
                notes.extend(read_as)
                resolved = resolve_derivatives(parsed, coordinates, time, assumptions)
    except ParseError as exc:
        raise DerivationError(Status.PARSE_ERROR, str(exc), exc.diagnostic("lagrangian")) from exc
    except MinervaMathError as exc:
        raise DerivationError(Status.PARSE_ERROR, str(exc)) from exc
    L = resolved.expr
    notes.extend(resolved.normalizations)
    parse_ms = round((time_module.perf_counter() - started) * 1000)
    names = {sym.name for sym in L.free_symbols}
    absent = [n for n in coordinates if not names & {n, f"{n}_dot", f"{n}_ddot"}]
    if absent:
        raise absent_coordinates_error(absent, names)
    derivation = derive_expr(L, coordinates, time)
    derivation.stages = {"parse": parse_ms, **derivation.stages}
    derivation.format = format
    derivation.normalizations = notes
    return derivation


def absent_coordinates_error(absent: list[str], names: set[str]) -> DerivationError:
    """Declared coordinates missing from the Lagrangian as read, with the read symbol each one most likely means.

    `x1` declared while the Lagrangian says `x_1` is a spelling mismatch: the suggestion is `x_1`, the name to
    declare. Only when nothing read is spelt alike is the velocity form suggested (the coordinate may appear in
    its velocity only, written another way).
    """
    suggestions: list[str] = []
    hints: list[str] = []
    for name in absent:
        close = near_misses(name, names)
        if close:
            suggestions.extend(close)
            hints.append(f"{name} → {' or '.join(close)}")
        else:
            suggestions.append(f"{name}_dot")
    message = (
        f"The declared coordinate(s) {', '.join(absent)} appear in neither the position nor the "
        f"velocity form in the Lagrangian as read (symbols: {', '.join(sorted(names)) or 'none'}). "
    )
    if hints:
        message += (
            f"Declare the coordinates with the names the Lagrangian uses ({'; '.join(hints)}): a declared name "
            "is matched exactly. "
        )
    message += "Velocities are \\dot{q} in LaTeX and q_dot in text."
    return DerivationError(
        Status.PARSE_ERROR,
        message,
        InputDiagnostic(
            code=InputCode.DECLARED_NAME_ABSENT,
            field="lagrangian",
            symbols=list(absent),
            suggestions=suggestions,
            recoverable=True,
        ),
    )


def reject_equation(source: str, what: str) -> None:
    """An equation (`L = T - V`, `L = T =`...) is not the expression asked for: say so, never pick a side."""
    if "=" in source:
        raise DerivationError(
            Status.PARSE_ERROR,
            f"Send only the {what} as an expression, not an equation: for `L = T = ...` give "
            "the part after the last `=`.",
            InputDiagnostic(
                code=InputCode.EQUATION_NOT_EXPRESSION,
                field=what.lower(),
                suggestions=["the part after the last `=`"],
                recoverable=True,
            ),
        )


def _check_coordinates(coordinates: list[str]) -> None:
    if not coordinates or len(coordinates) > MAX_COORDINATES:
        raise DerivationError(
            Status.UNSUPPORTED, f"Give between 1 and {MAX_COORDINATES} coordinates."
        )
    if len(set(coordinates)) != len(coordinates):
        raise DerivationError(Status.PARSE_ERROR, "Coordinates must be distinct.")


def derive_expr(
    L: sp.Expr,
    coordinates: list[str],
    time: str,
    *,
    q: list[sp.Symbol] | None = None,
    qd: list[sp.Symbol] | None = None,
    t: sp.Symbol | None = None,
) -> Derivation:
    """Euler-Lagrange derivation for an already built Lagrangian.

    `q`, `qd` and `t` may be given as symbols (with their assumptions); otherwise they are found in
    `L` by name (`<q>_dot` for velocities) or created plain. Geometry passes them explicitly, so the
    coordinates keep their declared assumptions and the curve parameter cannot clash with a coordinate.
    """
    _check_coordinates(coordinates)
    by_name = {s.name: s for s in L.free_symbols}
    if q is None:
        q = [by_name.get(n) or sp.Symbol(n) for n in coordinates]
    if qd is None:
        qd = [by_name.get(f"{n}_dot") or sp.Symbol(f"{n}_dot") for n in coordinates]
    if len(q) != len(coordinates) or len(qd) != len(coordinates):
        raise DerivationError(Status.PARSE_ERROR, "Coordinates and velocities do not match.")
    qdd = [sp.Symbol(f"{n}_ddot") for n in coordinates]
    if t is None:
        t = by_name.get(time) or sp.Symbol(time)
    if any(s.name.endswith("_ddot") for s in L.free_symbols):
        raise DerivationError(
            Status.UNSUPPORTED,
            "Lagrangians with accelerations (higher derivatives) are not supported.",
            InputDiagnostic(
                code=InputCode.ACCELERATION_IN_INPUT, field="lagrangian", recoverable=False
            ),
        )
    if not any(v in L.free_symbols for v in qd):
        raise DerivationError(
            Status.UNSUPPORTED,
            "The Lagrangian does not depend on any velocity (\\dot{q} in LaTeX, q_dot in text); "
            "it is not a dynamical Lagrangian.",
        )
    known = set(q) | set(qd) | {t}
    params = sorted(L.free_symbols - known, key=lambda s: s.name)

    names: dict[sp.Symbol, str] = {}
    for coordinate, velocity, acceleration in zip(q, qd, qdd, strict=True):
        base = to_latex(coordinate)
        names[velocity] = rf"\dot{{{base}}}"
        names[acceleration] = rf"\ddot{{{base}}}"

    d = Derivation(
        lagrangian=L,
        coordinates=coordinates,
        q=q,
        qd=qd,
        qdd=qdd,
        t=t,
        params=params,
        momenta=[sp.diff(L, v) for v in qd],
        euler_raw=[],
        euler=[],
        time_dependent=sp.simplify(sp.diff(L, t)) != 0,
        names=names,
    )
    n = d.n
    started = time_module.perf_counter()
    d.euler_raw = [sp.expand(d.total_derivative(d.momenta[i]) - sp.diff(L, q[i])) for i in range(n)]
    d.euler = [sp.simplify(e) for e in d.euler_raw]
    d.stages["derive"] = round((time_module.perf_counter() - started) * 1000)

    started = time_module.perf_counter()
    mass = sp.Matrix(n, n, lambda i, j: sp.diff(L, qd[i], qd[j]))
    determinant = sp.simplify(mass.det())
    if determinant != 0:
        d.mass_determinant = determinant
        rest = sp.Matrix([e.subs({a: 0 for a in qdd}) for e in d.euler_raw])
        accel = mass.LUsolve(-rest)
        d.solution = {qdd[i]: sp.simplify(accel[i]) for i in range(n)}
    d.stages["solve"] = round((time_module.perf_counter() - started) * 1000)
    return d


def check_equations(d: Derivation) -> list[Verification]:
    """Independent numerical check of every derived equation (see `verify.euler_lagrange_check`)."""
    return euler_lagrange_check(d.lagrangian, d.q, d.qd, d.qdd, d.t, d.euler)


@dataclass
class ConservedSearch:
    items: list[tuple[ConservedQuantity, sp.Expr]] = field(default_factory=list)
    tested: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    # Candidates the time budget did not reach: nothing is said about them, either way.
    not_searched: list[str] = field(default_factory=list)


# The share of the sandbox time limit an analysis may use before it stops searching for constants of motion
# and returns what it has (the equations of motion are already checked by then). The rest of the limit covers
# starting the sandbox child and the stage in flight when the budget runs out.
CONSERVED_SEARCH_SHARE = 0.6


def verify_zero_derivative(derivative: sp.Expr, digits: int = 30) -> tuple[bool, Verification]:
    """Is the time derivative of a candidate constant zero on the motion? Checked symbolically (exact
    cancellation first, then `simplify`) and at random numerical points; the numeric check decides `passed`.
    `digits` is the precision of the evaluation, never the tolerance: an expression built from several
    constants of motion has terms of size 1e20 at the wide sample points that cancel exactly."""
    symbolic_zero = sp.cancel(sp.together(derivative)) == 0 or sp.simplify(derivative) == 0
    checked, agreed = numeric_check(derivative, sp.Integer(0), digits=digits)
    passed = checked >= 4 and agreed == checked
    verification = Verification(
        symbolic_zero=symbolic_zero,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=passed,
        note="d/dt of the quantity, evaluated on the equations of motion, is zero",
    )
    return bool(passed), verification


def verify_constant(
    d: Derivation, quantity: sp.Expr, digits: int = 30
) -> tuple[bool, Verification | None]:
    """Is `quantity` constant on the motion? d/dt on the accelerations must be zero: symbolically and
    at random numerical points (the numeric check decides `passed`, as everywhere in the engine).
    `digits` is the precision of the evaluation, never the tolerance (see `verify_zero_derivative`)."""
    if not d.solution:
        return False, None
    return verify_zero_derivative(d.total_derivative(quantity).subs(d.solution), digits=digits)


def find_conserved(d: Derivation, deadline: float | None = None) -> ConservedSearch:
    """Cyclic coordinates, energy and rotations, each verified on the equations of motion.

    With a `deadline` (a `time.monotonic()` value) each candidate runs under `clock.stop_at`: one that the
    deadline interrupts, and every later one, is listed in `not_searched` and nothing is concluded about it.
    A candidate is ruled out by a numerical screen (`clearly_nonzero`) before its symbolic test when the
    quantity that must vanish is clearly nonzero at a sampled point; a candidate that survives the screen is
    tested exactly as before, so the screen can only save time, never add a constant.
    """
    L, q, qd, p, n, show = d.lagrangian, d.q, d.qd, d.momenta, d.n, d.show
    coordinates = d.coordinates
    out = ConservedSearch()

    def vanishes(expr: sp.Expr) -> bool:
        return not clearly_nonzero(expr) and sp.simplify(expr) == 0

    def verify_conserved(quantity: sp.Expr) -> tuple[bool, Verification | None]:
        return verify_constant(d, quantity)

    # 1. Cyclic coordinates.
    cyclic_reached = n
    for i in range(n):
        try:
            with stop_at(deadline):
                if not vanishes(sp.diff(L, q[i])):
                    continue
                ok, ver = verify_conserved(p[i])
        except TimeUp:
            cyclic_reached = i
            out.not_searched.append(
                "cyclic coordinates: " + ", ".join(coordinates[k] for k in range(i, n))
            )
            break
        out.items.append(
            (
                ConservedQuantity(
                    name=f"Momentum conjugate to {coordinates[i]}",
                    kind="cyclic",
                    expression_latex=f"p_{{{to_latex(q[i])}}} = {show(p[i])}",
                    reason=f"{coordinates[i]} is cyclic: L does not depend on it, so dp/dt = dL/dq = 0",
                    verified=ok,
                    verification=ver,
                ),
                p[i],
            )
        )
        out.features.append("lagrangian:cyclic")
    if "lagrangian:cyclic" not in out.features:
        if cyclic_reached == n:
            out.tested.append("cyclic coordinates: none (every coordinate appears in L)")
        elif cyclic_reached:
            out.tested.append(
                "cyclic coordinates: none among "
                + ", ".join(coordinates[k] for k in range(cyclic_reached))
            )

    # 2. Energy.
    if not d.time_dependent:
        try:
            with stop_at(deadline):
                energy = sp.expand(sum(qd[i] * p[i] for i in range(n)) - L)
                ok, ver = verify_conserved(energy)
        except TimeUp:
            out.not_searched.append("energy function")
        else:
            out.items.append(
                (
                    ConservedQuantity(
                        name="Energy function",
                        kind="energy",
                        expression_latex=f"h = {show(energy)}",
                        reason="L has no explicit time dependence, so h = sum(qdot dL/dqdot) - L is conserved",
                        verified=ok,
                        verification=ver,
                    ),
                    energy,
                )
            )
    else:
        out.tested.append("energy function: not conserved (L depends explicitly on time)")

    # 3. Rotations of coordinate pairs.
    found_rotation = False
    pairs = list(itertools.combinations(range(n), 2))
    reached = len(pairs)
    for index, (i, j) in enumerate(pairs):
        xi = [sp.Integer(0)] * n
        xi[i], xi[j] = -q[j], q[i]
        try:
            with stop_at(deadline):
                delta = sum(
                    sp.diff(L, q[k]) * xi[k]
                    + sp.diff(L, qd[k]) * sum(sp.diff(xi[k], q[m]) * qd[m] for m in range(n))
                    for k in (i, j)
                )
                if not vanishes(delta):
                    continue
                charge = sp.simplify(p[i] * xi[i] + p[j] * xi[j])
                ok, ver = verify_conserved(charge)
        except TimeUp:
            reached = index
            out.not_searched.append(
                "rotations of the pairs "
                + ", ".join(f"({coordinates[a]}, {coordinates[b]})" for a, b in pairs[index:])
            )
            break
        out.items.append(
            (
                ConservedQuantity(
                    name=f"Rotation symmetry in the ({coordinates[i]}, {coordinates[j]}) plane",
                    kind="rotation",
                    expression_latex=f"Q = {show(charge)}",
                    reason=(
                        f"L is unchanged by rotating ({coordinates[i]}, {coordinates[j]}) as Cartesian "
                        "coordinates, so by Noether's theorem Q = Σ p ξ is conserved"
                    ),
                    verified=ok,
                    verification=ver,
                ),
                charge,
            )
        )
        found_rotation = True
    if found_rotation:
        out.features.append("lagrangian:rotation")
    if n > 1 and not found_rotation and reached == len(pairs):
        out.tested.append(
            "rotations of each pair of coordinates (as Cartesian): none leaves L unchanged"
        )
    elif n > 1 and not found_rotation and reached:
        out.tested.append(
            f"rotations of the first {reached} pair(s) of coordinates (as Cartesian): none leaves L "
            "unchanged"
        )
    return out


def analyze_lagrangian(
    latex: str,
    coordinates: list[str],
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
    format: Literal["latex", "text"] = "latex",
) -> LagrangianReport:
    deadline = time_module.monotonic() + CONSERVED_SEARCH_SHARE * limits.timeout_seconds
    try:
        d = derive(latex, coordinates, time, assumptions, limits, format)
    except DerivationError as exc:
        return _fail(exc.status, exc.message, exc.diagnostic)
    q, qdd, p, n, show = d.q, d.qdd, d.momenta, d.n, d.show

    started = time_module.perf_counter()
    checks = check_equations(d)
    d.stages["check"] = round((time_module.perf_counter() - started) * 1000)
    if not all(c.passed for c in checks):
        failed = [coordinates[i] for i, c in enumerate(checks) if not c.passed]
        return LagrangianReport(
            status=Status.NOT_VERIFIED,
            lagrangian_latex=show(d.lagrangian),
            coordinates=coordinates,
            message=(
                "The derived equations of motion for "
                + ", ".join(failed)
                + " could not be confirmed by numerical differentiation of L, so they are not "
                "shown. Declaring assumptions (real, positive) on the symbols can help."
            ),
        )

    report = LagrangianReport(
        status=Status.OK,
        reading=InputReading(format=d.format, normalizations=list(d.normalizations)),
        lagrangian_latex=show(d.lagrangian),
        coordinates=coordinates,
        parameters=[s.name for s in d.params],
        time_dependent=d.time_dependent,
        equations=[
            EquationOfMotion(
                coordinate=coordinates[i], latex=f"{show(d.euler[i])} = 0", verification=checks[i]
            )
            for i in range(n)
        ],
        momenta=[
            NamedExpression(
                coordinate=coordinates[i], latex=f"p_{{{to_latex(q[i])}}} = {show(p[i])}"
            )
            for i in range(n)
        ],
        features=["lagrangian", f"lagrangian:dof:{min(n, 4)}", "ode:second-order"],
    )
    if n > 1:
        report.features.append("ode:system")
    if not d.time_dependent:
        report.features += ["lagrangian:time-independent", "ode:autonomous"]

    if d.solution:
        report.accelerations = [
            NamedExpression(
                coordinate=coordinates[i], latex=f"{show(qdd[i])} = {show(d.solution[qdd[i]])}"
            )
            for i in range(n)
        ]
    else:
        report.notes.append(
            "The mass matrix (second derivatives of L with respect to the velocities) is singular, so the "
            "accelerations cannot be solved for uniquely: the system is constrained or degenerate."
        )

    if d.mass_determinant is not None:
        report.restrictions_latex, report.restrictions_complete = determinant_conditions(
            d.mass_determinant
        )

    started = time_module.perf_counter()
    search = find_conserved(d, deadline)
    d.stages["conserved"] = round((time_module.perf_counter() - started) * 1000)
    report.stages = dict(d.stages)
    report.conserved = [item for item, _ in search.items]
    report.tested += search.tested
    report.features += search.features
    report.not_searched = list(search.not_searched)
    if not report.conserved:
        report.features.append("lagrangian:no-symmetry-found")
    report.notes.append(
        "Symmetries tested: cyclic coordinates, time translation, and rotations of coordinate pairs. "
        "Symmetries that hold only up to a total time derivative (for example Galilean boosts) and hidden "
        "symmetries (for example the Runge-Lenz vector of the Kepler problem) are not searched for here; "
        "the separate search for more constants of motion looks for them in a declared family. The symmetries of "
        "a metric (Killing vectors, and the conserved charges of its geodesics) are the geometry tool's "
        "killing_vectors and geodesic_constants."
    )
    unverified = [c for c in report.conserved if not c.verified]
    report.message = (
        f"{n} equation{'s' if n > 1 else ''} of motion, each confirmed by numerical differentiation; "
        f"{len(report.conserved) - len(unverified)} verified constant"
        f"{'' if len(report.conserved) - len(unverified) == 1 else 's'} of motion"
        + (f", {len(unverified)} not verified" if unverified else "")
        + "."
        + (
            " The time limit stopped the search for constants of motion; not searched, so nothing is "
            "claimed about them: " + "; ".join(report.not_searched) + "."
            if report.not_searched
            else ""
        )
    )
    return report
