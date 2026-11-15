"""Bounded Almkvist-Zeilberger search for a telescoper of a hyperexponential integrand.

For `f` with `f_x / f = r1` and `f_a / f = r2` rational, `d^k f / da^k = R_k f` with
`R_0 = 1`, `R_(k+1) = dR_k/da + R_k r2`. A certificate is sought of the form `G = y f` with
`y = Y / D` for a polynomial `Y` of degree at most `d` and `D` the lcm of the denominators of
`R_0 ... R_N`. Then `d/dx (y f) = (y' + r1 y) f`, and the telescoping equation

    sum_k c_k(a) R_k = y' + r1 y

is linear in the coefficients of `Y` and in the `c_k`, with `a`-dependent (rational function)
coefficients. It is solved exactly by a nullspace computation over the field of rational
functions in the parameter and the other symbols. Families are searched in a fixed order (order
N = 0, 1, ...; degree d = 0, 1, ...), cheapest first, so the first solution has minimal order for
the families tried. No candidate is trusted: `certificate.verify` decides.

A different backend (a holonomic closure, an Ore-algebra system run out of process) can be
supplied through the `Telescoper` protocol; its output goes through the same verification.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import sympy as sp
from sympy.polys.matrices import DomainMatrix

from .integrand import Hyperexponential
from .models import MAX_UNKNOWNS, Family, Refusal
from .operators import DifferentialOperator, normalise

MAX_DENOMINATOR_DEGREE = 40


@dataclass(frozen=True)
class Candidate:
    operator: DifferentialOperator
    y: sp.Expr  # G / f, a rational function of x
    order: int
    degree: int


class Telescoper(Protocol):
    """Anything that proposes (operator, certificate) pairs, one family at a time."""

    def families(
        self, h: Hyperexponential, *, max_order: int, max_degree: int, deadline: float
    ) -> Iterator[tuple[Family, Candidate | None]]: ...


def _generic(h: Hyperexponential) -> tuple[dict[sp.Basic, sp.Dummy], bool]:
    """Constants such as pi or sqrt(2) are made generic symbols; I selects a Gaussian field."""
    atoms: dict[sp.Basic, sp.Dummy] = {}
    expr = h.r1 + h.r2
    for atom in expr.atoms(sp.NumberSymbol):
        atoms[atom] = sp.Dummy("c", positive=True)
    for p in expr.atoms(sp.Pow):
        base, power = p.as_base_exp()
        if base.is_Rational and power.is_Rational and not power.is_Integer:
            atoms[p] = sp.Dummy("c", positive=True)
    return atoms, bool(expr.has(sp.I))


def _log_derivative_powers(h: Hyperexponential, order: int) -> list[sp.Expr]:
    r: list[sp.Expr] = [sp.Integer(1)]
    for _ in range(order):
        r.append(sp.cancel(sp.together(sp.diff(r[-1], h.a) + r[-1] * h.r2)))
    return r


class AlmkvistZeilberger:
    """The ansatz search described in the module docstring."""

    def families(
        self, h: Hyperexponential, *, max_order: int, max_degree: int, deadline: float
    ) -> Iterator[tuple[Family, Candidate | None]]:
        x, a = h.x, h.a
        generic, gaussian = _generic(h)
        forward = {k: v for k, v in generic.items()}
        r1 = h.r1.xreplace(forward)
        r2 = h.r2.xreplace(forward)
        symbols = sorted((r1.free_symbols | r2.free_symbols | {a}) - {x}, key=lambda s: str(s))
        base = sp.QQ_I if gaussian else sp.QQ
        field = base.frac_field(*symbols)
        back = {v: k for k, v in generic.items()}

        R = [sp.Integer(1)]
        for order in range(max_order + 1):
            if order:
                R.append(sp.cancel(sp.together(sp.diff(R[-1], a) + R[-1] * r2)))
            D = sp.lcm_list([sp.denom(r) for r in R])
            B = sp.denom(r1)
            if sp.degree(D * B, x) > MAX_DENOMINATOR_DEGREE:
                raise Refusal("resource_limit", "The denominators grew too large to search.")
            for degree in range(max_degree + 1):
                if time.monotonic() > deadline:
                    raise Refusal("resource_limit", "The search ran out of time.")
                unknowns = (degree + 1) + (order + 1)
                if unknowns > MAX_UNKNOWNS:
                    raise Refusal("resource_limit", "The search family became too large.")
                found = self._solve(r1, R, D, B, x, order, degree, field, symbols)
                family = Family(
                    order=order,
                    degree=degree,
                    unknowns=unknowns,
                    outcome="found" if found else "none",
                )
                if found is None:
                    yield family, None
                    continue
                cs, y = found
                cs = [c.xreplace(back) for c in cs]
                y = y.xreplace(back)
                params = [s for s in symbols if s in cs_free(cs)]
                coefficients = normalise(a, cs, [s for s in params if s != a] + [a])
                # The certificate is scaled with the operator: L f = dG/dx is linear in both.
                k = next(i for i, c in enumerate(cs) if c != 0)
                y = sp.cancel(y * sp.cancel(coefficients[k] / cs[k]))
                op = DifferentialOperator(a, tuple(coefficients))
                yield family, Candidate(op, y, order, degree)

    @staticmethod
    def _solve(
        r1: sp.Expr,
        R: list[sp.Expr],
        D: sp.Expr,
        B: sp.Expr,
        x: sp.Symbol,
        order: int,
        degree: int,
        field: Any,
        symbols: list[sp.Symbol],
    ) -> tuple[list[sp.Expr], sp.Expr] | None:
        """One family: the nonzero-operator solution of the linear system, or None."""
        columns: list[sp.Expr] = []
        for i in range(degree + 1):  # y = Y / D with Y = sum y_i x^i
            xi = x**i
            columns.append(
                sp.expand((sp.diff(xi, x) * D - xi * sp.diff(D, x)) * B + sp.numer(r1) * xi * D)
            )
        for k in range(order + 1):
            columns.append(sp.expand(-sp.cancel(R[k] * D * D * B)))
        polys = [sp.Poly(c, x) for c in columns]
        top = max(p.degree() for p in polys)
        rows = [
            [field.from_sympy(sp.sympify(p.coeff_monomial(x**r))) for p in polys]
            for r in range(top + 1)
        ]
        matrix = DomainMatrix(rows, (len(rows), len(columns)), field)
        basis = matrix.nullspace().to_Matrix()
        for vector in (basis.row(i) for i in range(basis.rows)):
            cs = [sp.sympify(v) for v in vector[degree + 1 :]]
            if all(c == 0 for c in cs):
                continue  # y f is constant: a degenerate solution that says nothing about I(a)
            ys = [sp.sympify(v) for v in vector[: degree + 1]]
            Y = sum(c * x**i for i, c in enumerate(ys))
            return cs, sp.cancel(Y / D)
        return None


def cs_free(cs: list[sp.Expr]) -> set[sp.Basic]:
    out: set[sp.Basic] = set()
    for c in cs:
        out |= c.free_symbols
    return out


def discover(
    h: Hyperexponential,
    *,
    max_order: int,
    max_degree: int,
    deadline: float,
    accept: Callable[[Candidate], bool],
    backend: Telescoper | None = None,
) -> tuple[Candidate | None, list[Family], int]:
    """The first candidate in the fixed family order that `accept` verifies.

    Returns (candidate or None, the families tried, the number of candidates rejected).
    """
    backend = backend or AlmkvistZeilberger()
    tried: list[Family] = []
    rejected = 0
    for family, candidate in backend.families(
        h, max_order=max_order, max_degree=max_degree, deadline=deadline
    ):
        if candidate is None:
            tried.append(family)
            continue
        if accept(candidate):
            tried.append(family)
            return candidate, tried, rejected
        rejected += 1
        tried.append(family.model_copy(update={"outcome": "rejected"}))
    return None, tried, rejected
