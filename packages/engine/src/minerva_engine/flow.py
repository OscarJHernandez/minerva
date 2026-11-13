"""The equations of motion of a Lagrangian or Hamiltonian system as one first-order flow.

`discovery.py` searches for conserved quantities of a *flow*: phase variables `(q, v)` with
`dq/dt = q_rate` and `dv/dt = v_rate`. For a Lagrangian `v` is the velocity and the rates come from the
Euler-Lagrange equations solved for the accelerations (`lagrangian.derive`, unchanged). For a Hamiltonian
`v` is the canonical momentum and `dq/dt = dH/dp`, `dp/dt = -dH/dq`. A conserved quantity `I(q, v, t)`
satisfies `dI/dt = dI/dt|explicit + sum(dI/dq * q_rate + dI/dv * v_rate) = 0`, which for a Hamiltonian is
`{I, H} + dI/dt = 0`.

Before anything is searched the equations are confirmed independently of the symbolic derivative: the
Lagrangian's by numerical differentiation of L (`verify.euler_lagrange_check`), the Hamiltonian's by
numerical differentiation of H (`hamilton_check`, below). Known constants (cyclic momenta, energy,
rotations) are each verified on the flow before they are used.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import mpmath
import sympy as sp

from .errors import MinervaMathError
from .lagrangian import (
    ConservedQuantity,
    DerivationError,
    check_equations,
    derive,
    find_conserved,
    verify_constant,
    verify_zero_derivative,
)
from .latexparse import parse_latex
from .latexprint import to_latex
from .models import Limits, Status, Verification

MAX_COORDINATES = 3


@dataclass
class Flow:
    system: Literal["lagrangian", "hamiltonian"]
    function_latex: str  # L or H as shown to the user
    coordinates: list[str]
    q: list[sp.Symbol]
    v: list[sp.Symbol]  # velocities (Lagrangian) or momenta (Hamiltonian)
    t: sp.Symbol
    params: list[sp.Symbol]
    q_rate: list[sp.Expr]
    v_rate: list[sp.Expr]
    time_dependent: bool
    names: dict[sp.Symbol, str] = field(default_factory=dict)
    known: list[tuple[ConservedQuantity, sp.Expr]] = field(default_factory=list)
    tested: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _verify: Callable[[sp.Expr], tuple[bool, Verification | None]] | None = None

    @property
    def n(self) -> int:
        return len(self.q)

    def show(self, expr: sp.Expr) -> str:
        return str(to_latex(expr, symbol_names=self.names))

    def total_derivative(self, f: sp.Expr) -> sp.Expr:
        """d/dt of f(q, v, t) along the flow."""
        out = sp.diff(f, self.t)
        for qi, vi, dq, dv in zip(self.q, self.v, self.q_rate, self.v_rate, strict=True):
            out += sp.diff(f, qi) * dq + sp.diff(f, vi) * dv
        return out

    def verify(self, quantity: sp.Expr) -> tuple[bool, Verification | None]:
        """Is `quantity` constant on the flow? Symbolic zero and random numerical points (see
        `lagrangian.verify_zero_derivative`)."""
        if self._verify is not None:
            return self._verify(quantity)
        return verify_zero_derivative(self.total_derivative(quantity))


def lagrangian_flow(
    latex: str,
    coordinates: list[str],
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
) -> Flow:
    d = derive(latex, coordinates, time, assumptions, limits)
    if len(coordinates) > MAX_COORDINATES:
        raise DerivationError(
            Status.UNSUPPORTED, f"The search handles between 1 and {MAX_COORDINATES} coordinates."
        )
    if not d.solution:
        raise DerivationError(
            Status.UNSUPPORTED,
            "The accelerations cannot be solved for (singular mass matrix), so the condition "
            "dC/dt = 0 on the motion cannot be written.",
        )
    if not all(c.passed for c in check_equations(d)):
        raise DerivationError(
            Status.NOT_VERIFIED,
            "The equations of motion could not be confirmed by numerical differentiation of L, so "
            "nothing was searched.",
        )
    search = find_conserved(d)
    return Flow(
        system="lagrangian",
        function_latex=d.show(d.lagrangian),
        coordinates=coordinates,
        q=d.q,
        v=d.qd,
        t=d.t,
        params=d.params,
        q_rate=list(d.qd),
        v_rate=[d.solution[a] for a in d.qdd],
        time_dependent=d.time_dependent,
        names=d.names,
        known=[(item, expr) for item, expr in search.items if item.verified],
        tested=search.tested,
        _verify=lambda quantity: verify_constant(d, quantity),
    )


def hamilton_check(
    hamiltonian: sp.Expr,
    q: list[sp.Symbol],
    p: list[sp.Symbol],
    t: sp.Symbol,
    q_rate: list[sp.Expr],
    p_rate: list[sp.Expr],
    *,
    seed: int = 20260103,
) -> Verification:
    """Compare `dH/dp` and `-dH/dq` with numerical derivatives of the lambdified H (mpmath), at random
    positive points. Needs at least 4 valid points and every one to agree."""
    note = "numerical differentiation of H agrees with the derived Hamilton equations"
    params = sorted(
        hamiltonian.free_symbols.union(*(e.free_symbols for e in q_rate + p_rate)) - {*q, *p, t},
        key=lambda s: s.name,
    )
    args = [*q, *p, t, *params]
    try:
        h = sp.lambdify(args, hamiltonian, "mpmath")
        rates = [sp.lambdify(args, e, "mpmath") for e in [*q_rate, *p_rate]]
    except Exception:
        return Verification(note="the Hamiltonian could not be evaluated numerically")
    n = len(q)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    with mpmath.workdps(30):
        for _ in range(40):
            point = [mpmath.mpf(rng.uniform(0.6, 1.9)) for _ in args]
            try:
                ok = True
                for k in range(2 * n):
                    if k < n:  # dq_k/dt = dH/dp_k
                        index, sign = n + k, 1
                    else:  # dp_k/dt = -dH/dq_k
                        index, sign = k - n, -1

                    def along(
                        x: mpmath.mpf, index: int = index, point: list[mpmath.mpf] = point
                    ) -> mpmath.mpf:
                        shifted = list(point)
                        shifted[index] = x
                        return h(*shifted)

                    numeric = sign * mpmath.diff(along, point[index])
                    derived = rates[k](*point)
                    if abs(mpmath.im(derived)) > 1e-20 or abs(mpmath.im(numeric)) > 1e-20:
                        ok = False
                        break
                    a, b = mpmath.re(numeric), mpmath.re(derived)
                    if abs(a - b) > mpmath.mpf(10) ** -12 * max(1, abs(a), abs(b)):
                        checked += 1
                        ok = False
                        break
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                continue
            if ok:
                checked += 1
                agreed += 1
            if checked >= 8:
                break
    return Verification(
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=checked >= 4 and agreed == checked,
        note=note,
    )


def hamiltonian_flow(
    latex: str,
    coordinates: list[str],
    momenta: list[str] | None,
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
) -> Flow:
    n = len(coordinates)
    if not coordinates or n > MAX_COORDINATES:
        raise DerivationError(
            Status.UNSUPPORTED, f"The search handles between 1 and {MAX_COORDINATES} coordinates."
        )
    if len(set(coordinates)) != n:
        raise DerivationError(Status.PARSE_ERROR, "Coordinates must be distinct.")
    p_names = momenta if momenta else [f"p_{c}" for c in coordinates]
    if len(p_names) != n or len(set(p_names)) != n or set(p_names) & set(coordinates):
        raise DerivationError(
            Status.PARSE_ERROR, "Give one distinct momentum name per coordinate (default p_<q>)."
        )
    try:
        H = parse_latex(latex, assumptions, limits)
    except MinervaMathError as exc:
        raise DerivationError(Status.PARSE_ERROR, str(exc)) from exc
    by_name = {s.name: s for s in H.free_symbols}
    q = [by_name.get(c) or sp.Symbol(c) for c in coordinates]
    p = [by_name.get(c) or sp.Symbol(c) for c in p_names]
    t = by_name.get(time) or sp.Symbol(time)
    missing = [name for name, s in zip(p_names, p, strict=True) if s not in H.free_symbols]
    if missing:
        raise DerivationError(
            Status.UNSUPPORTED,
            "The Hamiltonian does not depend on the momentum "
            + ", ".join(missing)
            + "; it is not a regular Hamiltonian (check the momentum names).",
        )
    params = sorted(H.free_symbols - {*q, *p, t}, key=lambda s: s.name)
    q_rate = [sp.diff(H, pi) for pi in p]
    p_rate = [-sp.diff(H, qi) for qi in q]
    if not sp.simplify(sp.Matrix(n, n, lambda i, j: sp.diff(H, p[i], p[j])).det()):
        raise DerivationError(
            Status.UNSUPPORTED,
            "The Hessian of H in the momenta is singular, so H is not a regular Hamiltonian.",
        )
    check = hamilton_check(H, q, p, t, q_rate, p_rate)
    if not check.passed:
        raise DerivationError(
            Status.NOT_VERIFIED,
            "Hamilton's equations could not be confirmed by numerical differentiation of H, so "
            "nothing was searched. Declaring assumptions (real, positive) on the symbols can help.",
        )
    time_dependent = sp.simplify(sp.diff(H, t)) != 0
    names = {pi: to_latex(pi) for pi in p}
    flow = Flow(
        system="hamiltonian",
        function_latex=str(to_latex(H)),
        coordinates=coordinates,
        q=q,
        v=p,
        t=t,
        params=params,
        q_rate=q_rate,
        v_rate=p_rate,
        time_dependent=time_dependent,
        names=names,
        notes=[f"momenta: {', '.join(p_names)}"],
    )
    _hamiltonian_known(flow, H)
    return flow


def _hamiltonian_known(flow: Flow, H: sp.Expr) -> None:
    """Cyclic momenta, energy and rotations of coordinate pairs, each verified on the flow."""
    q, p, n = flow.q, flow.v, flow.n

    def add(kind: str, name: str, reason: str, expr: sp.Expr) -> None:
        ok, ver = flow.verify(expr)
        if ok:
            flow.known.append(
                (
                    ConservedQuantity(
                        name=name,
                        kind=kind,
                        expression_latex=flow.show(expr),
                        reason=reason,
                        verified=True,
                        verification=ver,
                    ),
                    expr,
                )
            )

    for i in range(n):
        if sp.simplify(sp.diff(H, q[i])) == 0:
            add(
                "cyclic",
                f"Momentum conjugate to {flow.coordinates[i]}",
                f"{flow.coordinates[i]} is cyclic: H does not depend on it",
                p[i],
            )
    if not any(item.kind == "cyclic" for item, _ in flow.known):
        flow.tested.append("cyclic coordinates: none (every coordinate appears in H)")
    if not flow.time_dependent:
        add("energy", "Energy function", "H has no explicit time dependence", H)
    else:
        flow.tested.append("energy: not conserved (H depends explicitly on time)")
    found_rotation = False
    for i, j in itertools.combinations(range(n), 2):
        generator = q[i] * p[j] - q[j] * p[i]
        bracket = sum(
            sp.diff(generator, q[k]) * sp.diff(H, p[k])
            - sp.diff(generator, p[k]) * sp.diff(H, q[k])
            for k in range(n)
        )
        if sp.simplify(bracket) == 0:
            before = len(flow.known)
            add(
                "rotation",
                f"Rotation symmetry in the ({flow.coordinates[i]}, {flow.coordinates[j]}) plane",
                "the bracket of the rotation generator with H is zero",
                generator,
            )
            found_rotation = found_rotation or len(flow.known) > before
    if n > 1 and not found_rotation:
        flow.tested.append(
            "rotations of each pair of coordinates (as Cartesian): none commutes with H"
        )


__all__ = ["Flow", "hamilton_check", "hamiltonian_flow", "lagrangian_flow"]
