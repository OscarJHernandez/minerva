"""Independent checks of a first variation. Candidates are plain data, so tests can forge them.

The deciding leg uses only the original integrand: for a polynomial path `x` and polynomial
perturbations `η` it compares the finite-difference directional derivative of the *action*,

    (S[x + hη] - S[x - hη]) / 2h,

with the claimed value `∫ Σ EL_i η_i dt + Σ c_{end,i,j} η_i^(j)(end)`. Perturbations that vanish
together with their lower derivatives at both ends isolate the Euler-Lagrange expressions;
perturbations that do not vanish test every boundary coefficient. Polynomial arithmetic gives the
path derivatives exactly, so neither the symbolic total derivative nor `sp.diff` of a path enters.

Weaker legs are reported separately and never decide: a pointwise evaluation of the Euler-Lagrange
expression with numerical differentiation (`check_el_pointwise`), each integration-by-parts step
(`check_ibp_step`), and SymPy's own `euler_equations` (`sympy_euler`).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import mpmath as mp
import sympy as sp
from sympy.calculus.euler import euler_equations

from ..errors import LimitError
from . import polypath
from .firstvar import FirstVariationData, Stage
from .jets import apply_standins
from .models import End
from .numerics import Numerics, Poly
from .quadrature import FDTable, fd_table, integrate
from .setup import Problem

STEPS = (mp.mpf("1e-3"), mp.mpf("5e-4"), mp.mpf("2.5e-4"))


@dataclass(frozen=True)
class VariationForm:
    """What is claimed: EL_i per function and the coefficient of each end perturbation in δS."""

    interior: tuple[sp.Expr, ...]
    boundary: Mapping[tuple[End, int, int], sp.Expr]

    @staticmethod
    def of(data: FirstVariationData) -> VariationForm:
        return VariationForm(tuple(data.euler), dict(data.coefficients))


@dataclass
class Comparison:
    kind: str  # "bubble" | "generic"
    function: int | None
    analytic: Any
    scale: mp.mpf
    table: FDTable
    quadrature_error: mp.mpf


@dataclass
class VariationCheck:
    verdict: Literal["passed", "failed", "inconclusive"]
    comparisons: list[Comparison] = field(default_factory=list)
    seed: int = 0
    note: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    @property
    def points_checked(self) -> int:
        return len(self.comparisons)

    @property
    def points_agreed(self) -> int:
        return sum(1 for c in self.comparisons if c.table.verdict == "agree")

    @property
    def worst_rel_error(self) -> mp.mpf:
        return max((c.table.rows[-1].rel_error for c in self.comparisons), default=mp.mpf(0))


def perturbation(
    num: Numerics, kind: str, *, function: int | None = None, degree: int | None = None
) -> Poly:
    jets = num.jets
    top = max(jets.orders)
    degree = degree if degree is not None else 2 * top + 3
    coefficients: list[polypath.Coeffs] = []
    for i in range(jets.count):
        if function is not None and i != function:
            coefficients.append([mp.mpf(0)] * (degree + 1))
            continue
        vanishing = (
            [(end, j) for end in ("lower", "upper") for j in range(jets.orders[i])]
            if kind == "bubble"
            else []
        )
        coefficients.append(polypath.admissible(num.rng, degree, num.lower, num.upper, vanishing))
    return Poly(coefficients)


def action_value(num: Numerics, path: Poly) -> tuple[Any, mp.mpf]:
    """S[path] = ∫ L dt + G_upper(b) + G_lower(a) by two agreeing Gauss-Legendre rules."""
    problem = num.problem
    table = num.with_depth(path)
    lagrangian = num.compile(problem.lagrangian)
    integral = integrate(
        lambda s: lagrangian(num.jet_values(s, path, table)), num.lower, num.upper, fast=True
    )
    total = integral.value
    for term, point in ((problem.lower_term, num.lower), (problem.upper_term, num.upper)):
        if term != 0:
            total += num.compile(term)(num.jet_values(point, path, table))
    return total, integral.error


def claimed_variation(
    num: Numerics, candidate: VariationForm, path: Poly, eta: Poly
) -> tuple[Any, mp.mpf, mp.mpf]:
    """∫ Σ EL_i η_i dt + Σ c η^(j)(end): (value, scale = sum of absolute sizes, quadrature error)."""
    jets = num.jets
    table, eta_table = num.with_depth(path), num.with_depth(eta)
    euler = [num.compile(e) for e in candidate.interior]

    def terms(s: mp.mpf) -> tuple[Any, mp.mpf]:
        values = num.jet_values(s, path, table, eta, eta_table)
        total: Any = mp.mpf(0)
        size = mp.mpf(0)
        for i, fn in enumerate(euler):
            piece = fn(values) * values[jets.eta[i][0]]
            total += piece
            size += abs(piece)
        return total, size

    integral = integrate(lambda s: terms(s)[0], num.lower, num.upper, fast=True)
    scale = integrate(lambda s: terms(s)[1], num.lower, num.upper, fast=True).value
    value = integral.value
    for (end, i, j), coefficient in candidate.boundary.items():
        point = num.lower if end == "lower" else num.upper
        values = num.jet_values(point, path, table, eta, eta_table)
        piece = num.compile(coefficient)(values) * values[jets.eta[i][j]]
        value += piece
        scale += abs(piece)
    return value, mp.mpf(scale), integral.error


def check_first_variation(
    problem: Problem,
    candidate: VariationForm,
    *,
    seed: int = 20260301,
    deadline: float | None = None,
    comparisons: int = 4,
) -> VariationCheck:
    """Compare a claimed first variation with finite differences of the action itself."""
    num = Numerics(problem, seed)
    jets = num.jets
    path = num.random_path(2 * max(jets.orders) + 2)
    plan: list[tuple[str, int | None]] = [("bubble", None), ("generic", None)]
    plan += [("bubble", None), ("generic", None)][: max(0, comparisons - 2)]
    if jets.count > 1:
        plan += [("bubble", i) for i in range(jets.count)]
    result = VariationCheck("passed", seed=seed)
    verdicts: list[str] = []
    for kind, function in plan:
        if deadline is not None and time.monotonic() > deadline:
            raise LimitError("the verification ran out of time")
        eta = perturbation(num, kind, function=function)
        analytic, scale, quad_error = claimed_variation(num, candidate, path, eta)
        errors: list[mp.mpf] = [quad_error]

        def action(e: mp.mpf, eta: Poly = eta, errors: list[mp.mpf] = errors) -> Any:
            value, err = action_value(num, path.plus(eta, e))
            errors.append(err)
            return value

        table = fd_table(action, analytic, scale, STEPS, order=1, noise=max(errors))
        result.comparisons.append(Comparison(kind, function, analytic, scale, table, max(errors)))
        verdicts.append(table.verdict)
    if "disagree" in verdicts:
        result.verdict = "failed"
    elif any(v != "agree" for v in verdicts):
        result.verdict, result.note = (
            "inconclusive",
            "a finite-difference comparison was inconclusive",
        )
    return result


# -- weaker legs ---------------------------------------------------------------------------------
@dataclass
class PointwiseCheck:
    passed: bool
    points: int
    worst_rel_error: float


def check_el_pointwise(
    problem: Problem, interior: tuple[sp.Expr, ...], *, seed: int = 20260302, points: int = 4
) -> PointwiseCheck:
    """EL_i against Σ(-1)^k d^k/ds^k [∂L/∂u^(k)] with the outer derivative taken numerically.

    The partial derivatives are SymPy's, the total derivative is numerical: this catches `∂` used
    where `D` was meant, but it does not check the theory, so it never decides.
    """
    num = Numerics(problem, seed)
    jets = num.jets
    path = num.random_path(2 * max(jets.orders) + 2)
    table = num.with_depth(path)
    worst = 0.0
    with mp.workdps(60):
        for _ in range(points):
            s0 = num.lower + (num.upper - num.lower) * mp.mpf(num.rng.uniform(0.2, 0.8))
            for i, claimed in enumerate(interior):
                reference: Any = mp.mpf(0)
                for k in range(jets.orders[i] + 1):
                    partial = num.compile(jets.partial(problem.lagrangian, i, k))

                    def g(s: mp.mpf, partial: Any = partial) -> Any:
                        return partial(num.jet_values(s, path, table))

                    reference += (-1) ** k * (mp.diff(g, s0, k) if k else g(s0))
                value = num.compile(claimed)(num.jet_values(s0, path, table))
                scale = max(abs(reference), abs(value), mp.mpf(1))
                worst = max(worst, float(abs(value - reference) / scale))
    return PointwiseCheck(worst < 1e-12, points * len(interior), worst)


@dataclass
class StageCheck:
    index: int
    passed: bool
    worst_rel_error: float


def check_ibp_step(
    problem: Problem, before: Stage, after: Stage, *, seed: int = 20260303, points: int = 3
) -> StageCheck:
    """`before.interior - after.interior = d/ds (after.boundary - before.boundary)` at sample points.

    The derivative on the right is numerical, on polynomial paths and perturbations.
    """
    num = Numerics(problem, seed)
    jets = num.jets
    path = num.random_path(2 * max(jets.orders) + 2)
    eta = perturbation(num, "generic")
    table, eta_table = num.with_depth(path), num.with_depth(eta)
    flux = num.compile(after.boundary - before.boundary)
    drop = num.compile(before.interior - after.interior)
    worst = 0.0
    with mp.workdps(50):
        for _ in range(points):
            s0 = num.lower + (num.upper - num.lower) * mp.mpf(num.rng.uniform(0.2, 0.8))
            numeric = mp.diff(lambda s: flux(num.jet_values(s, path, table, eta, eta_table)), s0)
            direct = drop(num.jet_values(s0, path, table, eta, eta_table))
            scale = max(abs(numeric), abs(direct), mp.mpf(1))
            worst = max(worst, float(abs(numeric - direct) / scale))
    return StageCheck(after.index, worst < 1e-12, worst)


def sympy_euler(problem: Problem, interior: tuple[sp.Expr, ...]) -> bool | None:
    """SymPy's `euler_equations` on a function-of-t copy; None when the comparison cannot be made."""
    jets = problem.jets
    try:
        t = jets.variable
        functions = [sp.Function(name)(t) for name in jets.names]
        mapping: dict[sp.Symbol, sp.Expr] = {}
        for i in range(jets.count):
            for k in range(jets.depth + 1):
                mapping[jets.u[i][k]] = (
                    functions[i] if k == 0 else sp.Derivative(functions[i], (t, k))
                )
        replacements = standins(problem.lagrangian)
        explicit = apply_standins(problem.lagrangian, replacements)
        theirs = euler_equations(explicit.xreplace(mapping), functions, t)
        ordered = [
            (mapping[jets.u[i][k]], jets.u[i][k])
            for k in range(jets.depth, -1, -1)
            for i in range(jets.count)
        ]
        for eq, mine in zip(theirs, interior, strict=True):
            lhs = eq.lhs.doit()
            for source, target in ordered:
                lhs = lhs.xreplace({source: target})
            difference = sp.simplify(lhs - apply_standins(mine, replacements))
            if difference != 0:
                return False
        return True
    except (TypeError, ValueError, NotImplementedError, AttributeError, RecursionError):
        return None


def standins(expr: sp.Expr) -> dict[Any, sp.Lambda]:
    import random

    from .jets import standins as make

    return make(expr, random.Random(7), 4)  # noqa: S311 - deterministic stand-ins
