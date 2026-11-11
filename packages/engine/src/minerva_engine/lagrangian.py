"""Lagrangian mechanics: equations of motion and conserved quantities.

Given a Lagrangian L(q, q-dot, t) and the generalised coordinates, this derives the Euler-Lagrange
equations, the conjugate momenta and (when the mass matrix is invertible) the accelerations, and looks
for constants of motion:

* a cyclic coordinate (L does not depend on q_i) conserves its momentum p_i;
* no explicit time dependence conserves the energy function h = sum(q-dot_i p_i) - L;
* a rotation of a pair of coordinates that leaves L unchanged conserves q_i p_j - q_j p_i.

Every constant is verified: its time derivative is evaluated on the equations of motion and must be
zero, symbolically and at random numerical points. The search covers these families only; symmetries
that hold up to a total time derivative (for example Galilean boosts) or hidden symmetries (for example
the Runge-Lenz vector) are not found, and the report says so.
"""

from __future__ import annotations

import itertools
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from .errors import MinervaMathError
from .latexparse import parse_latex
from .models import Limits, Status, Verification
from .verify import numeric_check

MAX_COORDINATES = 8


class EquationOfMotion(BaseModel):
    coordinate: str
    latex: str


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


def analyze_lagrangian(
    latex: str,
    coordinates: list[str],
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
) -> LagrangianReport:
    if not coordinates or len(coordinates) > MAX_COORDINATES:
        return _fail(Status.UNSUPPORTED, f"Give between 1 and {MAX_COORDINATES} coordinates.")
    if len(set(coordinates)) != len(coordinates):
        return _fail(Status.PARSE_ERROR, "Coordinates must be distinct.")
    try:
        L = parse_latex(latex, assumptions, limits)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc))

    by_name = {s.name: s for s in L.free_symbols}
    q = [by_name.get(n) or sp.Symbol(n) for n in coordinates]
    qd = [by_name.get(f"{n}_dot") or sp.Symbol(f"{n}_dot") for n in coordinates]
    qdd = [sp.Symbol(f"{n}_ddot") for n in coordinates]
    t = by_name.get(time) or sp.Symbol(time)
    if any(s.name.endswith("_ddot") for s in L.free_symbols):
        return _fail(
            Status.UNSUPPORTED,
            "Lagrangians with accelerations (higher derivatives) are not supported.",
        )
    if not any(v in L.free_symbols for v in qd):
        return _fail(
            Status.UNSUPPORTED,
            "The Lagrangian does not depend on any velocity \\dot{q}; it is not a dynamical Lagrangian.",
        )
    known = set(q) | set(qd) | {t}
    params = sorted(L.free_symbols - known, key=lambda s: s.name)

    # Latex printing with dots over the velocities.
    names: dict[sp.Symbol, str] = {}
    for coordinate, velocity, acceleration in zip(q, qd, qdd, strict=True):
        base = sp.latex(coordinate)
        names[velocity] = rf"\dot{{{base}}}"
        names[acceleration] = rf"\ddot{{{base}}}"

    def show(expr: sp.Expr) -> str:
        return str(sp.latex(expr, symbol_names=names))

    def D(f: sp.Expr) -> sp.Expr:
        """Total time derivative along a motion: q' = qd, qd' = qdd."""
        out = sp.diff(f, t)
        for qi, qdi, qddi in zip(q, qd, qdd, strict=True):
            out += sp.diff(f, qi) * qdi + sp.diff(f, qdi) * qddi
        return out

    n = len(q)
    p = [sp.diff(L, v) for v in qd]
    euler = [sp.expand(D(p[i]) - sp.diff(L, q[i])) for i in range(n)]
    time_dependent = sp.simplify(sp.diff(L, t)) != 0

    report = LagrangianReport(
        status=Status.OK,
        lagrangian_latex=show(L),
        coordinates=coordinates,
        parameters=[s.name for s in params],
        time_dependent=time_dependent,
        equations=[
            EquationOfMotion(coordinate=coordinates[i], latex=f"{show(sp.simplify(euler[i]))} = 0")
            for i in range(n)
        ],
        momenta=[
            NamedExpression(
                coordinate=coordinates[i], latex=f"p_{{{sp.latex(q[i])}}} = {show(p[i])}"
            )
            for i in range(n)
        ],
        features=["lagrangian", f"lagrangian:dof:{min(n, 4)}", "ode:second-order"],
    )
    if n > 1:
        report.features.append("ode:system")
    if not time_dependent:
        report.features += ["lagrangian:time-independent", "ode:autonomous"]

    # Accelerations from the Euler-Lagrange equations, when the mass matrix is invertible.
    mass = sp.Matrix(n, n, lambda i, j: sp.diff(L, qd[i], qd[j]))
    solution: dict[sp.Symbol, sp.Expr] = {}
    if sp.simplify(mass.det()) != 0:
        rest = sp.Matrix([e.subs({a: 0 for a in qdd}) for e in euler])
        accel = mass.LUsolve(-rest)
        solution = {qdd[i]: sp.simplify(accel[i]) for i in range(n)}
        report.accelerations = [
            NamedExpression(
                coordinate=coordinates[i], latex=f"{show(qdd[i])} = {show(solution[qdd[i]])}"
            )
            for i in range(n)
        ]
    else:
        report.notes.append(
            "The mass matrix (second derivatives of L with respect to the velocities) is singular, so the "
            "accelerations cannot be solved for uniquely: the system is constrained or degenerate."
        )

    def verify_conserved(quantity: sp.Expr) -> tuple[bool, Verification | None]:
        if not solution:
            return False, None
        derivative = D(quantity).subs(solution)
        symbolic_zero = sp.simplify(derivative) == 0
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

    # 1. Cyclic coordinates.
    for i in range(n):
        if sp.simplify(sp.diff(L, q[i])) == 0:
            ok, ver = verify_conserved(p[i])
            report.conserved.append(
                ConservedQuantity(
                    name=f"Momentum conjugate to {coordinates[i]}",
                    kind="cyclic",
                    expression_latex=f"p_{{{sp.latex(q[i])}}} = {show(p[i])}",
                    reason=f"{coordinates[i]} is cyclic: L does not depend on it, so dp/dt = dL/dq = 0",
                    verified=ok,
                    verification=ver,
                )
            )
            report.features.append("lagrangian:cyclic")
    if "lagrangian:cyclic" not in report.features:
        report.tested.append("cyclic coordinates: none (every coordinate appears in L)")

    # 2. Energy.
    if not time_dependent:
        energy = sp.expand(sum(qd[i] * p[i] for i in range(n)) - L)
        ok, ver = verify_conserved(energy)
        report.conserved.append(
            ConservedQuantity(
                name="Energy function",
                kind="energy",
                expression_latex=f"h = {show(energy)}",
                reason="L has no explicit time dependence, so h = sum(qdot dL/dqdot) - L is conserved",
                verified=ok,
                verification=ver,
            )
        )
    else:
        report.tested.append("energy function: not conserved (L depends explicitly on time)")

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
            report.conserved.append(
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
                )
            )
            found_rotation = True
    if found_rotation:
        report.features.append("lagrangian:rotation")
    if n > 1 and not found_rotation:
        report.tested.append(
            "rotations of each pair of coordinates (as Cartesian): none leaves L unchanged"
        )

    if not report.conserved:
        report.features.append("lagrangian:no-symmetry-found")
    report.notes.append(
        "Symmetries tested: cyclic coordinates, time translation, and rotations of coordinate pairs. "
        "Symmetries that hold only up to a total time derivative (for example Galilean boosts) and hidden "
        "symmetries (for example the Runge-Lenz vector of the Kepler problem) are not searched for."
    )
    unverified = [c for c in report.conserved if not c.verified]
    report.message = (
        f"{n} equation{'s' if n > 1 else ''} of motion; "
        f"{len(report.conserved) - len(unverified)} verified constant"
        f"{'' if len(report.conserved) - len(unverified) == 1 else 's'} of motion"
        + (f", {len(unverified)} not verified" if unverified else "")
        + "."
    )
    return report
