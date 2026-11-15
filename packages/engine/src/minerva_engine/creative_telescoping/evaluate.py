"""Initial values from the existing verified integrator, constants, and the final value.

I(a0) and its first derivatives at a rational a0 are *independent* evaluations: each is the
verified definite integral of `d^k f / da^k` at `a = a0`, obtained from `integrate`, which checks
its answer by quadrature. The solution of the ODE is accepted only if it satisfies the ODE
exactly, reproduces those values, and agrees with quadrature of the original integrand at sampled
parameter values (`check_definite`); the last is numerical evidence, never a proof.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import sympy as sp

from ..integrate import check_definite, definite_value
from ..latexprint import to_latex
from .models import ValueVerification
from .ode import constants_of, satisfies
from .operators import DifferentialOperator

POINTS = (1, 2, 3, sp.Rational(1, 2), 4, sp.Rational(1, 3), 5)


@dataclass
class Initial:
    point: sp.Expr
    values: list[sp.Expr]


def candidate_points(domain: tuple[sp.Expr, sp.Expr]) -> list[sp.Expr]:
    lo, hi = domain
    points = [sp.Rational(p) for p in POINTS]
    if hi == 0:
        points = [-p for p in points]
    return [p for p in points if lo < p < hi]


def _evaluate(
    f: sp.Expr, var: sp.Symbol, lower: sp.Expr, upper: sp.Expr, deadline: float
) -> sp.Expr | None:
    """The verified definite integral as an expression (None: not available)."""
    value, _ = definite_value(
        f,
        var,
        lower,
        upper,
        notation="",
        base={"input_latex": to_latex(f), "variable": var.name},
        deadline=deadline,
    )
    return value


def initial_values(
    f: sp.Expr,
    var: sp.Symbol,
    a: sp.Symbol,
    count: int,
    points: list[sp.Expr],
    lower: sp.Expr,
    upper: sp.Expr,
    deadline: float,
) -> Initial | None:
    """I^(k)(a0) for k < count at the first point where every one is available."""
    for point in points[:3]:
        values: list[sp.Expr] = []
        for k in range(count):
            if time.monotonic() > deadline:
                return None
            value = _evaluate(sp.diff(f, a, k).subs(a, point), var, lower, upper, deadline)
            if value is None:
                break
            values.append(value)
        else:
            return Initial(point, values)
    return None


def fix_constants(
    solution: sp.Expr, a: sp.Symbol, initial: Initial
) -> tuple[sp.Expr, list[sp.Eq]] | None:
    constants = constants_of(solution)
    if len(constants) != len(initial.values):
        return None
    equations = [
        sp.Eq(sp.diff(solution, a, k).subs(a, initial.point), v)
        for k, v in enumerate(initial.values)
    ]
    try:
        answer = sp.solve(equations, constants, dict=True)
    except Exception:
        return None
    if len(answer) != 1 or set(answer[0]) != set(constants):
        return None
    return sp.simplify(solution.subs(answer[0])), [sp.Eq(c, answer[0][c]) for c in constants]


def accept(
    f: sp.Expr,
    var: sp.Symbol,
    a: sp.Symbol,
    operator: DifferentialOperator,
    rhs: sp.Expr,
    value: sp.Expr,
    initial: Initial,
    lower: sp.Expr,
    upper: sp.Expr,
    deadline: float,
) -> ValueVerification:
    """Every condition for showing `value`; `passed` only if all of them hold."""
    ode_ok = satisfies(operator, rhs, value)
    held = True
    for k, v in enumerate(initial.values):
        got = sp.diff(value, a, k).subs(a, initial.point)
        try:
            held = held and bool(sp.simplify(got - v) == 0)
        except Exception:
            held = False
    quad = check_definite(f, var, lower, upper, value, deadline=deadline)
    passed = ode_ok and held and quad.passed
    if not ode_ok:
        note = "the candidate does not satisfy the differential equation"
    elif not held:
        note = "the candidate does not reproduce the initial values"
    elif not quad.passed:
        note = f"quadrature of the integrand does not confirm it ({quad.note})"
    else:
        note = (
            "satisfies the equation exactly and the initial values; confirmed by numerical "
            "quadrature at sampled parameter values (evidence, not a proof)"
        )
    return ValueVerification(
        ode_symbolic_zero=ode_ok,
        initial_values_hold=held,
        quadrature_points_checked=quad.points_checked,
        quadrature_points_agreed=quad.points_agreed,
        passed=passed,
        note=note,
    )
