"""The second variation: the quadratic form, its finite-difference check, and Ritz inertia.

Convention: `S[x + εη] = S[x] + ε δS + ½ ε² δ²S + ...`, so

    δ²S = ∫ Σ_{p,q} H_pq η_p η_q dt + Σ_ends Σ_{p,q} G_pq η_p η_q,   H_pq = ∂²L/∂u_p ∂u_q,

with `p = (i, k)` a function and a derivative order (and `G_pq` the second derivatives of the end
terms). The form is verified by second differences of the action along polynomial perturbations
(independent of the symbolic Hessian), with deliberately wrong candidates (a missing cross term, a
factor of two) rejected by the same check.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mpmath as mp
import sympy as sp

from ..errors import LimitError
from . import polypath
from .jets import JetSpace
from .models import End
from .numerics import JetField, Numerics, Poly, ShiftedField
from .quadrature import Integral, _legendre_rule, fd_table, integrate
from .setup import Problem
from .verify import Comparison, VariationCheck, action_value, perturbation

Pair = tuple[int, int]  # (function index, derivative order)
STEPS2 = (mp.mpf("5e-3"), mp.mpf("2.5e-3"), mp.mpf("1.25e-3"))


@dataclass(frozen=True)
class QuadraticForm:
    """δ²S as data: Hessian entries of the integrand and of the end terms (ordered pairs)."""

    interior: dict[tuple[Pair, Pair], sp.Expr]
    endpoint: dict[End, dict[tuple[Pair, Pair], sp.Expr]]


def second_form(problem: Problem) -> QuadraticForm:
    jets = problem.jets
    pairs = jets.jets_in(problem.lagrangian)
    interior: dict[tuple[Pair, Pair], sp.Expr] = {}
    for p in pairs:
        for q in pairs:
            h = sp.expand(sp.diff(problem.lagrangian, jets.u[p[0]][p[1]], jets.u[q[0]][q[1]]))
            if h != 0:
                interior[(p, q)] = h
    endpoint: dict[End, dict[tuple[Pair, Pair], sp.Expr]] = {}
    for end, term in (("lower", problem.lower_term), ("upper", problem.upper_term)):
        entries: dict[tuple[Pair, Pair], sp.Expr] = {}
        if term != 0:
            ends = jets.jets_in(term)
            for p in ends:
                for q in ends:
                    h = sp.expand(sp.diff(term, jets.u[p[0]][p[1]], jets.u[q[0]][q[1]]))
                    if h != 0:
                        entries[(p, q)] = h
        endpoint[end] = entries  # type: ignore[index]
    return QuadraticForm(interior, endpoint)


def quadratic_expression(jets: JetSpace, entries: dict[tuple[Pair, Pair], sp.Expr]) -> sp.Expr:
    return sum(
        (h * jets.eta[p[0]][p[1]] * jets.eta[q[0]][q[1]] for (p, q), h in entries.items()),
        sp.Integer(0),
    )


def quadratic_value(
    num: Numerics, form: QuadraticForm, path: JetField, eta: JetField
) -> tuple[Any, mp.mpf, mp.mpf]:
    """(δ²S, size of the terms, quadrature error) from the claimed form along `path`, `eta`."""
    jets = num.jets
    interior = num.compile(quadratic_expression(jets, form.interior))

    def f(s: mp.mpf) -> Any:
        return interior(num.jet_values(s, path, eta))

    integral = integrate(f, num.lower, num.upper, fast=True)
    scale = integrate(lambda s: abs(f(s)), num.lower, num.upper, fast=True).value
    value = integral.value
    for end, entries in form.endpoint.items():
        if not entries:
            continue
        point = num.lower if end == "lower" else num.upper
        piece = num.compile(quadratic_expression(jets, entries))(num.jet_values(point, path, eta))
        value += piece
        scale += abs(piece)
    return value, mp.mpf(scale), integral.error


def check_second_variation(
    problem: Problem,
    form: QuadraticForm,
    *,
    seed: int = 20260501,
    deadline: float | None = None,
    comparisons: int = 3,
) -> VariationCheck:
    """Second differences of the action against the claimed quadratic form (polynomial paths)."""
    num = Numerics(problem, seed)
    jets = num.jets
    path = num.field(num.random_path(2 * max(jets.orders) + 2))
    floor = mp.mpf("1e-12") * (1 + abs(action_value(num, path)[0]))
    result = VariationCheck("passed", seed=seed)
    verdicts: list[str] = []
    for n in range(comparisons):
        if deadline is not None and time.monotonic() > deadline:
            raise LimitError("the verification ran out of time")
        eta = num.field(perturbation(num, "generic" if n else "bubble"))
        claimed, scale, quad_error = quadratic_value(num, form, path, eta)
        errors: list[mp.mpf] = [quad_error]

        def action(e: mp.mpf, eta: JetField = eta, errors: list[mp.mpf] = errors) -> Any:
            value, err = action_value(num, ShiftedField(path, eta, e))
            errors.append(err)
            return value

        table = fd_table(action, claimed, scale, STEPS2, order=2, noise=max(errors), floor=floor)
        result.comparisons.append(
            Comparison("bubble" if n == 0 else "generic", None, claimed, scale, table, max(errors))
        )
        verdicts.append(table.verdict)
    if "disagree" in verdicts:
        result.verdict = "failed"
    elif any(v != "agree" for v in verdicts):
        result.verdict, result.note = "inconclusive", "a second difference was inconclusive"
    return result


# -- Ritz inertia ----------------------------------------------------------------------------------
@dataclass
class Ritz:
    eigenvalues: list[mp.mpf]
    negative: int
    zero: int
    positive: int
    directions: list[tuple[mp.mpf, list[Poly]]]  # (eigenvalue, perturbation) for negative ones


def admissible_basis(
    num: Numerics, per_function: int, vanishing: dict[int, list[tuple[str, int]]]
) -> list[Poly]:
    """Polynomial perturbations (one function moved at a time) satisfying the fixed conditions."""
    jets = num.jets
    basis: list[Poly] = []
    for i in range(jets.count):
        rows = sorted(set(vanishing.get(i, [])))
        degree = len(rows) + per_function - 1
        constraint = mp.matrix(max(len(rows), 1), degree + 1)
        for r, (end, j) in enumerate(rows):
            point = num.lower if end == "lower" else num.upper
            for k in range(j, degree + 1):
                constraint[r, k] = mp.factorial(k) / mp.factorial(k - j) * point ** (k - j)
        if rows:
            _, sigma, vt = mp.svd_r(constraint, full_matrices=True)
            rank = sum(1 for v in sigma if v > mp.mpf("1e-18"))
            null = [[vt[r, c] for c in range(degree + 1)] for r in range(rank, degree + 1)]
        else:
            null = [
                [mp.mpf(1) if c == m else mp.mpf(0) for c in range(degree + 1)]
                for m in range(degree + 1)
            ]
        for vector in null[:per_function]:
            coefficients = [[mp.mpf(0)] * (degree + 1) for _ in range(jets.count)]
            coefficients[i] = list(vector)
            basis.append(Poly(coefficients))
    return basis


def ritz_inertia(num: Numerics, form: QuadraticForm, path: JetField, basis: Sequence[Poly]) -> Ritz:
    """Eigenvalues of the quadratic form on span(basis), relative to the L2 inner product."""
    jets = num.jets
    fields = [num.field(b) for b in basis]
    size = len(fields)
    nodes, weights = _legendre_rule(16, mp.mp.dps)
    panels = 6
    width = (num.upper - num.lower) / panels
    compiled = {pq: num.compile(h) for pq, h in form.interior.items()}
    gram = mp.zeros(size, size)
    mass = mp.zeros(size, size)
    for panel in range(panels):
        mid = num.lower + (panel + mp.mpf(1) / 2) * width
        for x, w in zip(nodes, weights, strict=True):
            s = mid + width / 2 * x
            weight = width / 2 * w
            at_path = num.jet_values(s, path)
            h = {pq: fn(at_path) for pq, fn in compiled.items()}
            e = [f.at(s) for f in fields]
            for a in range(size):
                for b in range(a, size):
                    total = sum(
                        (hv * e[a][p[0]][p[1]] * e[b][q[0]][q[1]] for (p, q), hv in h.items()),
                        mp.mpf(0),
                    )
                    gram[a, b] += weight * total
                    mass[a, b] += weight * sum(
                        (e[a][i][0] * e[b][i][0] for i in range(jets.count)), mp.mpf(0)
                    )
    for a in range(size):
        for b in range(a):
            gram[a, b], mass[a, b] = gram[b, a], mass[b, a]
    for end, entries in form.endpoint.items():
        if not entries:
            continue
        point = num.lower if end == "lower" else num.upper
        at_end = num.jet_values(point, path)
        values = {pq: num.compile(h)(at_end) for pq, h in entries.items()}
        e = [f.at(point) for f in fields]
        for a in range(size):
            for b in range(size):
                gram[a, b] += sum(
                    (hv * e[a][p[0]][p[1]] * e[b][q[0]][q[1]] for (p, q), hv in values.items()),
                    mp.mpf(0),
                )
    chol = mp.cholesky(mass)
    inverse = mp.inverse(chol)
    reduced = inverse * gram * inverse.T
    reduced = (reduced + reduced.T) / 2
    values, vectors = mp.eigsy(reduced)
    ordered = sorted(range(size), key=lambda n: values[n])
    scale = max(abs(values[n]) for n in range(size)) or mp.mpf(1)
    tol = mp.mpf("1e-9") * scale
    eig = [values[n] for n in ordered]
    negative = sum(1 for v in eig if v < -tol)
    zero = sum(1 for v in eig if abs(v) <= tol)
    positive = size - negative - zero
    directions: list[tuple[mp.mpf, list[Poly]]] = []
    for n in ordered:
        if values[n] < -tol:
            coefficient = inverse.T * vectors[:, n]
            combined = _combine(basis, [coefficient[a] for a in range(size)])
            directions.append((values[n], [combined]))
    return Ritz(eig, negative, zero, positive, directions)


def _combine(basis: Sequence[Poly], weights: Sequence[mp.mpf]) -> Poly:
    count = len(basis[0].coefficients)
    out: list[polypath.Coeffs] = [[mp.mpf(0)] for _ in range(count)]
    for b, w in zip(basis, weights, strict=True):
        for i in range(count):
            out[i] = polypath.add(out[i], b.coefficients[i], w)
    return Poly(out)


def integral_of(f: Any, num: Numerics) -> Integral:
    return integrate(f, num.lower, num.upper)
