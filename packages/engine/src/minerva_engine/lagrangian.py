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
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from .errors import MinervaMathError
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import Limits, Status, Verification
from .textparse import parse_text
from .verify import euler_lagrange_check, numeric_check

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


class LagrangianReport(BaseModel):
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
    features: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str) -> LagrangianReport:
    return LagrangianReport(status=status, message=message)


class DerivationError(Exception):
    """The input cannot be analysed; carries the explicit status to report."""

    def __init__(self, status: Status, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


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

    In text a velocity is written `q_dot` (LaTeX: `\\dot{q}`, which the parser turns into the same symbol).
    """
    _check_coordinates(coordinates)
    try:
        if format == "text":
            L = parse_text(latex, assumptions, limits)
        else:
            L = parse_latex(latex, assumptions, limits)
    except MinervaMathError as exc:
        raise DerivationError(Status.PARSE_ERROR, str(exc)) from exc
    return derive_expr(L, coordinates, time)


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
    d.euler_raw = [sp.expand(d.total_derivative(d.momenta[i]) - sp.diff(L, q[i])) for i in range(n)]
    d.euler = [sp.simplify(e) for e in d.euler_raw]

    mass = sp.Matrix(n, n, lambda i, j: sp.diff(L, qd[i], qd[j]))
    if sp.simplify(mass.det()) != 0:
        rest = sp.Matrix([e.subs({a: 0 for a in qdd}) for e in d.euler_raw])
        accel = mass.LUsolve(-rest)
        d.solution = {qdd[i]: sp.simplify(accel[i]) for i in range(n)}
    return d


def check_equations(d: Derivation) -> list[Verification]:
    """Independent numerical check of every derived equation (see `verify.euler_lagrange_check`)."""
    return euler_lagrange_check(d.lagrangian, d.q, d.qd, d.qdd, d.t, d.euler)


@dataclass
class ConservedSearch:
    items: list[tuple[ConservedQuantity, sp.Expr]] = field(default_factory=list)
    tested: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)


def verify_zero_derivative(derivative: sp.Expr) -> tuple[bool, Verification]:
    """Is the time derivative of a candidate constant zero on the motion? Checked symbolically (exact
    cancellation first, then `simplify`) and at random numerical points; the numeric check decides `passed`."""
    symbolic_zero = sp.cancel(sp.together(derivative)) == 0 or sp.simplify(derivative) == 0
    checked, agreed = numeric_check(derivative, sp.Integer(0))
    passed = checked >= 4 and agreed == checked
    verification = Verification(
        symbolic_zero=symbolic_zero,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=passed,
        note="d/dt of the quantity, evaluated on the equations of motion, is zero",
    )
    return bool(passed), verification


def verify_constant(d: Derivation, quantity: sp.Expr) -> tuple[bool, Verification | None]:
    """Is `quantity` constant on the motion? d/dt on the accelerations must be zero: symbolically and
    at random numerical points (the numeric check decides `passed`, as everywhere in the engine)."""
    if not d.solution:
        return False, None
    return verify_zero_derivative(d.total_derivative(quantity).subs(d.solution))


def find_conserved(d: Derivation) -> ConservedSearch:
    """Cyclic coordinates, energy and rotations, each verified on the equations of motion."""
    L, q, qd, p, n, show = d.lagrangian, d.q, d.qd, d.momenta, d.n, d.show
    coordinates = d.coordinates
    out = ConservedSearch()

    def verify_conserved(quantity: sp.Expr) -> tuple[bool, Verification | None]:
        return verify_constant(d, quantity)

    # 1. Cyclic coordinates.
    for i in range(n):
        if sp.simplify(sp.diff(L, q[i])) == 0:
            ok, ver = verify_conserved(p[i])
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
        out.tested.append("cyclic coordinates: none (every coordinate appears in L)")

    # 2. Energy.
    if not d.time_dependent:
        energy = sp.expand(sum(qd[i] * p[i] for i in range(n)) - L)
        ok, ver = verify_conserved(energy)
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
    for i, j in itertools.combinations(range(n), 2):
        xi = [sp.Integer(0)] * n
        xi[i], xi[j] = -q[j], q[i]
        delta = sum(
            sp.diff(L, q[k]) * xi[k]
            + sp.diff(L, qd[k]) * sum(sp.diff(xi[k], q[m]) * qd[m] for m in range(n))
            for k in (i, j)
        )
        if sp.simplify(delta) == 0:
            charge = sp.simplify(p[i] * xi[i] + p[j] * xi[j])
            ok, ver = verify_conserved(charge)
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
    if n > 1 and not found_rotation:
        out.tested.append(
            "rotations of each pair of coordinates (as Cartesian): none leaves L unchanged"
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
    try:
        d = derive(latex, coordinates, time, assumptions, limits, format)
    except DerivationError as exc:
        return _fail(exc.status, exc.message)
    q, qdd, p, n, show = d.q, d.qdd, d.momenta, d.n, d.show

    checks = check_equations(d)
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

    search = find_conserved(d)
    report.conserved = [item for item, _ in search.items]
    report.tested += search.tested
    report.features += search.features
    if not report.conserved:
        report.features.append("lagrangian:no-symmetry-found")
    report.notes.append(
        "Symmetries tested: cyclic coordinates, time translation, and rotations of coordinate pairs. "
        "Symmetries that hold only up to a total time derivative (for example Galilean boosts) and hidden "
        "symmetries (for example the Runge-Lenz vector of the Kepler problem) are not searched for here; "
        "the separate search for more constants of motion looks for them in a declared family."
    )
    unverified = [c for c in report.conserved if not c.verified]
    report.message = (
        f"{n} equation{'s' if n > 1 else ''} of motion, each confirmed by numerical differentiation; "
        f"{len(report.conserved) - len(unverified)} verified constant"
        f"{'' if len(report.conserved) - len(unverified) == 1 else 's'} of motion"
        + (f", {len(unverified)} not verified" if unverified else "")
        + "."
    )
    return report
