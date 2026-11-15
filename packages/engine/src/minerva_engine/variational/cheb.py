"""Method A: solve the derived Euler-Lagrange equations by global Chebyshev collocation.

This is a strong-form method: it uses the *derived* Euler-Lagrange expressions and the natural
boundary conditions (`coefficient = 0` for every free end derivative), not the action. It shares no
discretisation with method B (local Hermite elements on the action). The unknowns are the function
values at Chebyshev-Lobatto nodes; derivatives come from the differentiation matrix, boundary
conditions replace the first and last `n_i` equation rows of each function, and Newton's method runs
in extended precision (`mpmath`, 30 digits) because the high-order differentiation matrices are
ill-conditioned in double precision.
"""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from .firstvar import FirstVariationData
from .hermite import NotConverged
from .models import End
from .numerics import Compiled, JetField, Numerics


def _lobatto(M: int) -> list[Any]:
    return [mp.cos(mp.pi * j / (M - 1)) for j in range(M)]


def differentiation_matrix(M: int) -> mp.matrix:
    x = _lobatto(M)
    c = [2 if j in (0, M - 1) else 1 for j in range(M)]
    D = mp.zeros(M, M)
    for i in range(M):
        for j in range(M):
            if i != j:
                D[i, j] = (mp.mpf(c[i]) / c[j]) * (-1) ** (i + j) / (x[i] - x[j])
        D[i, i] = -sum((D[i, j] for j in range(M) if j != i), mp.mpf(0))
    return D


class ChebField:
    """The solution: nodal values and their derivatives, evaluated by barycentric interpolation."""

    def __init__(self, nodes: list[Any], derivs: list[list[list[Any]]]) -> None:
        self.nodes = nodes
        self.derivs = derivs  # [function][k] -> nodal values of the k-th derivative
        M = len(nodes)
        self.weights = [(-1) ** j * (mp.mpf(1) / 2 if j in (0, M - 1) else 1) for j in range(M)]

    def at(self, s: Any) -> list[list[Any]]:
        for j, node in enumerate(self.nodes):
            if abs(s - node) < mp.mpf("1e-25"):
                return [[row[j] for row in per] for per in self.derivs]
        terms = [w / (s - node) for w, node in zip(self.weights, self.nodes, strict=True)]
        total = mp.fsum(terms)
        return [
            [mp.fsum(t * v for t, v in zip(terms, row, strict=True)) / total for row in per]
            for per in self.derivs
        ]


def solve_chebyshev(
    num: Numerics,
    derived: FirstVariationData,
    fixed: dict[tuple[End, int, int], Any],
    guess: JetField,
    M: int,
    *,
    max_iterations: int = 40,
) -> ChebField:
    jets = num.jets
    count, orders = jets.count, jets.orders
    a, b = num.lower, num.upper
    half, mid = (b - a) / 2, (a + b) / 2
    xs = _lobatto(M)
    nodes = [mid - half * v for v in xs]
    D0 = differentiation_matrix(M)
    Ds = -D0 / half  # d/ds on the ascending nodes
    top = max(2 * n for n in orders)
    powers = [mp.eye(M)]
    for _ in range(top):
        powers.append(Ds * powers[-1])

    equations: list[tuple[Compiled, dict[tuple[int, int], Compiled]]] = []
    for i in range(count):
        e = derived.euler[i]
        partials = {
            (j, k): num.compile(sp.diff(e, jets.u[j][k]))
            for j, k in jets.jets_in(e)
            if sp.diff(e, jets.u[j][k]) != 0
        }
        equations.append((num.compile(e), partials))
    naturals: dict[tuple[End, int, int], tuple[Compiled, dict[tuple[int, int], Compiled]]] = {}
    for (end, i, j), c in derived.coefficients.items():
        if j >= orders[i] or (end, i, j) in fixed:
            continue
        partials = {
            (p, k): num.compile(sp.diff(c, jets.u[p][k]))
            for p, k in jets.jets_in(c)
            if sp.diff(c, jets.u[p][k]) != 0
        }
        naturals[(end, i, j)] = (num.compile(c), partials)
    for _, parts in naturals.values():
        if not parts:
            raise NotConverged("a natural boundary condition does not involve the unknowns")

    start = [guess.at(s) for s in nodes]
    x = mp.matrix(count * M, 1)
    for i in range(count):
        for j in range(M):
            x[i * M + j] = mp.re(start[j][i][0])

    def jet_at(xv: mp.matrix, row: int) -> dict[sp.Symbol, Any]:
        values: dict[sp.Symbol, Any] = {jets.variable: nodes[row], **num.params}
        for i in range(count):
            block = xv[i * M : (i + 1) * M, 0]
            for k in range(2 * orders[i] + 1):
                if k <= top:
                    values[jets.u[i][k]] = sum(
                        (powers[k][row, c] * block[c] for c in range(M)), mp.mpf(0)
                    )
        return values

    def residual_and_jacobian(xv: mp.matrix) -> tuple[mp.matrix, mp.matrix]:
        n = count * M
        R = mp.matrix(n, 1)
        J = mp.zeros(n, n)
        cache = {row: jet_at(xv, row) for row in range(M)}
        for i in range(count):
            ni = orders[i]
            for row in range(M):
                r = i * M + row
                end: End | None = None
                slot = 0
                if row < ni:
                    end, slot, node_row = "lower", row, 0
                elif row >= M - ni:
                    end, slot, node_row = "upper", M - 1 - row, M - 1
                if end is None:
                    fn, parts = equations[i]
                    values = cache[row]
                    R[r] = fn(values)
                    for (jj, k), pf in parts.items():
                        coefficient = pf(values)
                        for col in range(M):
                            J[r, jj * M + col] += coefficient * powers[k][row, col]
                    continue
                key = (end, i, slot)
                if key in fixed:
                    block = xv[i * M : (i + 1) * M, 0]
                    value = sum((powers[slot][node_row, c] * block[c] for c in range(M)), mp.mpf(0))
                    R[r] = value - fixed[key]
                    for col in range(M):
                        J[r, i * M + col] = powers[slot][node_row, col]
                else:
                    fn, parts = naturals[key]
                    values = cache[node_row]
                    R[r] = fn(values)
                    for (jj, k), pf in parts.items():
                        coefficient = pf(values)
                        for col in range(M):
                            J[r, jj * M + col] += coefficient * powers[k][node_row, col]
        return R, J

    R, J = residual_and_jacobian(x)
    norm = mp.norm(R, mp.inf)
    for _ in range(max_iterations):
        try:
            step = mp.lu_solve(J, -R)
        except ZeroDivisionError as exc:
            raise NotConverged("the collocation Jacobian is singular") from exc
        scale = mp.mpf(1)
        for _ in range(40):
            trial = x + scale * step
            try:
                new_R, new_J = residual_and_jacobian(trial)
            except (ZeroDivisionError, ValueError, OverflowError) as exc:
                raise NotConverged("the equations could not be evaluated") from exc
            new_norm = mp.norm(new_R, mp.inf)
            if new_norm <= (1 - mp.mpf("1e-4") * scale) * norm or new_norm < mp.mpf("1e-24"):
                break
            scale /= 2
        else:
            raise NotConverged("the line search could not reduce the residual")
        x, R, J, norm = trial, new_R, new_J, new_norm
        if scale * mp.norm(step, mp.inf) <= mp.mpf("1e-18") * (1 + mp.norm(x, mp.inf)):
            derivs = []
            for i in range(count):
                block = x[i * M : (i + 1) * M, 0]
                derivs.append(
                    [
                        [
                            sum((powers[k][row, c] * block[c] for c in range(M)), mp.mpf(0))
                            for row in range(M)
                        ]
                        for k in range(2 * orders[i] + 1)
                    ]
                )
            return ChebField(nodes, derivs)
    raise NotConverged("Newton's method did not converge within the iteration limit")
