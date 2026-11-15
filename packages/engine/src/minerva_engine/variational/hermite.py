"""Method B: discretise the action with Hermite elements and solve ∇S_h = 0.

For a function whose integrand has highest derivative `n`, the unknown is a piecewise polynomial of
degree `2n+1` with continuous derivatives up to order `n` (dofs `h^j x^(j)` at the nodes, `j <= n`).
The discrete action `S_h(z)` is the Gauss-quadrature integral of the *original* integrand plus the
end terms; fixed values are set, and free ends are natural. Newton's method is applied to the
gradient (not to a minimisation), with a backtracking line search on the gradient norm, so saddle
points and maxima are found as readily as minima.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import mpmath as mp
import sympy as sp

from . import banded
from .jets import JetSpace, apply_standins
from .models import End
from .numerics import Numerics
from .quadrature import _legendre_rule

QUAD_POINTS = {1: 6, 2: 8}


class NotConverged(Exception):
    """Newton did not converge from this starting guess (never a mathematical claim)."""


@dataclass
class FloatFn:
    symbols: tuple[sp.Symbol, ...]
    fn: Callable[..., float]

    def __call__(self, values: dict[sp.Symbol, float]) -> float:
        return float(self.fn(*[values[s] for s in self.symbols]))


def compile_float(num: Numerics, expr: sp.Expr) -> FloatFn:
    explicit = (
        apply_standins(sp.sympify(expr), num.replacements) if num.replacements else sp.sympify(expr)
    )
    fixed = {s: sp.Float(float(mp.re(v)), 17) for s, v in num.params.items()}
    explicit = explicit.xreplace(fixed)
    symbols = tuple(sorted(explicit.free_symbols, key=lambda s: s.name))
    return FloatFn(symbols, sp.lambdify(symbols, explicit, modules="math"))


@cache
def reference_basis(n: int) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Coefficients (in ξ) of the Hermite basis with dofs j = 0..n at ξ = 0 and 1: [e][j] -> coeffs."""
    degree = 2 * n + 1
    xi = sp.Symbol("xi")
    rows, rhs_keys = [], []
    for end in (0, 1):
        for j in range(n + 1):
            rows.append([sp.diff(xi**c, xi, j).subs(xi, end) for c in range(degree + 1)])
            rhs_keys.append((end, j))
    matrix = sp.Matrix(rows)
    inverse = matrix.inv()
    out = []
    for e in (0, 1):
        per_j = []
        for j in range(n + 1):
            column = inverse[:, rhs_keys.index((e, j))]
            per_j.append(tuple(float(c) for c in column))
        out.append(tuple(per_j))
    return tuple(out)


def _poly_derivative(coeffs: Sequence[float], k: int, x: float) -> float:
    total = 0.0
    for power, c in enumerate(coeffs):
        if power >= k and c != 0.0:
            factor = 1.0
            for q in range(k):
                factor *= power - q
            total += c * factor * x ** (power - k)
    return total


@dataclass
class Layout:
    jets: JetSpace
    orders: tuple[int, ...]
    nd: int
    offsets: tuple[int, ...]

    @staticmethod
    def of(jets: JetSpace) -> Layout:
        orders = jets.orders
        offsets, total = [], 0
        for n in orders:
            offsets.append(total)
            total += n + 1
        return Layout(jets, orders, total, tuple(offsets))

    def index(self, node: int, i: int, j: int) -> int:
        return node * self.nd + self.offsets[i] + j


@dataclass
class Discrete:
    """The compiled pieces of S_h for one problem (independent of the mesh)."""

    layout: Layout
    lower: float
    upper: float
    grad: dict[tuple[int, int], FloatFn]
    hess: dict[tuple[tuple[int, int], tuple[int, int]], FloatFn]
    end_value: dict[End, FloatFn | None]
    end_grad: dict[End, dict[tuple[int, int], FloatFn]]
    end_hess: dict[End, dict[tuple[tuple[int, int], tuple[int, int]], FloatFn]]
    lagrangian: FloatFn


def discretise(num: Numerics) -> Discrete:
    problem, jets = num.problem, num.jets
    layout = Layout.of(jets)
    L = problem.lagrangian
    pairs = jets.jets_in(L)
    grad = {p: compile_float(num, sp.diff(L, jets.u[p[0]][p[1]])) for p in pairs}
    hess = {}
    for p in pairs:
        for q in pairs:
            h = sp.diff(L, jets.u[p[0]][p[1]], jets.u[q[0]][q[1]])
            if h != 0:
                hess[(p, q)] = compile_float(num, h)
    end_value: dict[End, FloatFn | None] = {}
    end_grad: dict[End, dict[tuple[int, int], FloatFn]] = {}
    end_hess: dict[End, dict[tuple[tuple[int, int], tuple[int, int]], FloatFn]] = {}
    for end, term in (("lower", problem.lower_term), ("upper", problem.upper_term)):
        key: End = end  # type: ignore[assignment]
        if term == 0:
            end_value[key], end_grad[key], end_hess[key] = None, {}, {}
            continue
        ep = jets.jets_in(term)
        end_value[key] = compile_float(num, term)
        end_grad[key] = {p: compile_float(num, sp.diff(term, jets.u[p[0]][p[1]])) for p in ep}
        end_hess[key] = {
            (p, q): compile_float(num, h)
            for p in ep
            for q in ep
            if (h := sp.diff(term, jets.u[p[0]][p[1]], jets.u[q[0]][q[1]])) != 0
        }
    return Discrete(
        layout, float(num.lower), float(num.upper), grad, hess, end_value, end_grad, end_hess,
        compile_float(num, L),
    )  # fmt: skip


@dataclass
class Mesh:
    discrete: Discrete
    elements: int
    h: float = field(init=False)
    nodes: list[float] = field(init=False)

    def __post_init__(self) -> None:
        a, b = self.discrete.lower, self.discrete.upper
        self.h = (b - a) / self.elements
        self.nodes = [a + m * self.h for m in range(self.elements + 1)]

    @property
    def size(self) -> int:
        return (self.elements + 1) * self.discrete.layout.nd


def _gauss(order: int) -> tuple[list[float], list[float]]:
    nodes, weights = _legendre_rule(order, 20)
    return [float((x + 1) / 2) for x in nodes], [float(w / 2) for w in weights]


def _basis_table(n: int, points: list[float], kmax: int) -> list[list[list[float]]]:
    """table[k][d][g]: k-th ξ-derivative of local basis d (d = e*(n+1)+j) at the Gauss point g."""
    coeffs = reference_basis(n)
    table = []
    for k in range(kmax + 1):
        per_d = []
        for e in (0, 1):
            for j in range(n + 1):
                per_d.append([_poly_derivative(coeffs[e][j], k, x) for x in points])
        table.append(per_d)
    return table


def assemble(
    mesh: Mesh, z: list[float], free: list[int] | None = None
) -> tuple[list[float], list[dict[int, float]]]:
    """Gradient and Hessian of S_h at z, restricted to the free dofs (all if `free` is None)."""
    d = mesh.discrete
    layout = d.layout
    jets = layout.jets
    h = mesh.h
    size = mesh.size
    free_index = {g: n for n, g in enumerate(free)} if free is not None else None
    grad = [0.0] * (len(free) if free is not None else size)
    rows: list[dict[int, float]] = [dict() for _ in range(len(grad))]

    def add_g(gi: int, v: float) -> None:
        n = gi if free_index is None else free_index.get(gi, -1)
        if n >= 0:
            grad[n] += v

    def add_k(gi: int, gj: int, v: float) -> None:
        if free_index is None:
            rows[gi][gj] = rows[gi].get(gj, 0.0) + v
            return
        a, b = free_index.get(gi, -1), free_index.get(gj, -1)
        if a >= 0 and b >= 0:
            rows[a][b] = rows[a].get(b, 0.0) + v

    big = max(layout.orders)
    points, weights = _gauss(QUAD_POINTS.get(big, 8))
    full = {n: _basis_table(n, points, n) for n in set(layout.orders)}

    pairs = list(d.grad)
    for m in range(mesh.elements):
        t0 = mesh.nodes[m]
        local = {
            i: [z[layout.index(m + e, i, j)] for e in (0, 1) for j in range(layout.orders[i] + 1)]
            for i in range(jets.count)
        }
        for g_idx, (xi, w) in enumerate(zip(points, weights, strict=True)):
            values: dict[sp.Symbol, float] = {jets.variable: t0 + xi * h}
            for i in range(jets.count):
                n = layout.orders[i]
                for k in range(n + 1):
                    values[jets.u[i][k]] = (
                        sum(q * full[n][k][dd][g_idx] for dd, q in enumerate(local[i])) / h**k
                    )
            weight = w * h
            nl = {
                p: [
                    full[layout.orders[p[0]]][p[1]][dd][g_idx] / h ** p[1]
                    for dd in range(2 * (layout.orders[p[0]] + 1))
                ]
                for p in pairs
            }
            for p, fn in d.grad.items():
                value = fn(values) * weight
                for dd, phi in enumerate(nl[p]):
                    e, j = divmod(dd, layout.orders[p[0]] + 1)
                    add_g(layout.index(m + e, p[0], j), value * phi)
            for (p, q), fn in d.hess.items():
                value = fn(values) * weight
                for dd, phi_p in enumerate(nl[p]):
                    ep, jp = divmod(dd, layout.orders[p[0]] + 1)
                    gi = layout.index(m + ep, p[0], jp)
                    for ee, phi_q in enumerate(nl[q]):
                        eq, jq = divmod(ee, layout.orders[q[0]] + 1)
                        add_k(gi, layout.index(m + eq, q[0], jq), value * phi_p * phi_q)
    # end terms: the jets at a node are the scaled dofs z / h^k
    for end, node, bound in (("lower", 0, d.lower), ("upper", mesh.elements, d.upper)):
        if d.end_value[end] is None:  # type: ignore[index]
            continue
        values = {jets.variable: bound}
        for i in range(jets.count):
            for k in range(layout.orders[i] + 1):
                values[jets.u[i][k]] = z[layout.index(node, i, k)] / h**k
        for p, fn in d.end_grad[end].items():  # type: ignore[index]
            add_g(layout.index(node, p[0], p[1]), fn(values) / h ** p[1])
        for (p, q), fn in d.end_hess[end].items():  # type: ignore[index]
            add_k(
                layout.index(node, p[0], p[1]),
                layout.index(node, q[0], q[1]),
                fn(values) / h ** (p[1] + q[1]),
            )
    return grad, rows


def action_value(mesh: Mesh, z: list[float]) -> float:
    """S_h(z): Gauss quadrature of the integrand plus the end terms."""
    d = mesh.discrete
    layout, jets, h = d.layout, d.layout.jets, mesh.h
    big = max(layout.orders)
    points, weights = _gauss(QUAD_POINTS.get(big, 8))
    tables = {n: _basis_table(n, points, n) for n in set(layout.orders)}
    total = 0.0
    for m in range(mesh.elements):
        local = {
            i: [z[layout.index(m + e, i, j)] for e in (0, 1) for j in range(layout.orders[i] + 1)]
            for i in range(jets.count)
        }
        for g, (xi, w) in enumerate(zip(points, weights, strict=True)):
            values: dict[sp.Symbol, float] = {jets.variable: mesh.nodes[m] + xi * h}
            for i in range(jets.count):
                n = layout.orders[i]
                for k in range(n + 1):
                    values[jets.u[i][k]] = (
                        sum(q * tables[n][k][dd][g] for dd, q in enumerate(local[i])) / h**k
                    )
            total += w * h * d.lagrangian(values)
    return total


@dataclass
class Boundary:
    """Fixed dofs: global index -> scaled value; everything else is free."""

    fixed: dict[int, float]

    def free(self, size: int) -> list[int]:
        return [g for g in range(size) if g not in self.fixed]


def boundary_of(mesh: Mesh, fixed: dict[tuple[End, int, int], float]) -> Boundary:
    layout = mesh.discrete.layout
    out: dict[int, float] = {}
    for (end, i, j), value in fixed.items():
        node = 0 if end == "lower" else mesh.elements
        out[layout.index(node, i, j)] = value * mesh.h**j
    return Boundary(out)


def newton(
    mesh: Mesh,
    z0: list[float],
    boundary: Boundary,
    *,
    max_iterations: int = 40,
    tol: float = 1e-11,
) -> tuple[list[float], int]:
    z = list(z0)
    for g, v in boundary.fixed.items():
        z[g] = v
    free = boundary.free(mesh.size)
    bandwidth = 2 * mesh.discrete.layout.nd
    grad, rows = assemble(mesh, z, free)
    norm = max((abs(v) for v in grad), default=0.0)
    for iteration in range(1, max_iterations + 1):
        try:
            step = banded.solve(rows, [-v for v in grad], bandwidth)
        except banded.Singular as exc:
            raise NotConverged(
                "the Jacobian is singular: not isolated, or no solution near here"
            ) from exc
        size = max((abs(v) for v in step), default=0.0)
        scale = 1.0
        accepted = False
        for _ in range(30):
            trial = list(z)
            for n, g in enumerate(free):
                trial[g] += scale * step[n]
            new_grad, new_rows = assemble(mesh, trial, free)
            new_norm = max((abs(v) for v in new_grad), default=0.0)
            if new_norm <= (1 - 1e-4 * scale) * norm or new_norm < 1e-14:
                accepted = True
                break
            scale /= 2
        if not accepted:
            raise NotConverged("the line search could not reduce the gradient")
        z, grad, rows, norm = trial, new_grad, new_rows, new_norm
        if scale * size <= tol * (1 + max(abs(v) for v in z)):
            return z, iteration
    raise NotConverged("Newton's method did not converge within the iteration limit")


class HermiteField:
    """The solution as a JetField: derivatives up to each function's order (higher ones are not smooth)."""

    def __init__(self, mesh: Mesh, z: list[float]) -> None:
        self.mesh, self.z = mesh, z
        self.layout = mesh.discrete.layout
        self.coeffs = {n: reference_basis(n) for n in set(self.layout.orders)}

    def at(self, s: Any) -> list[list[Any]]:
        mesh, layout = self.mesh, self.layout
        a = mesh.discrete.lower
        m = min(mesh.elements - 1, max(0, int((float(s) - a) / mesh.h)))
        xi = (float(s) - mesh.nodes[m]) / mesh.h
        out = []
        for i, n in enumerate(layout.orders):
            row = []
            for k in range(n + 1):
                total = 0.0
                for e in (0, 1):
                    for j in range(n + 1):
                        total += self.z[layout.index(m + e, i, j)] * _poly_derivative(
                            self.coeffs[n][e][j], k, xi
                        )
                row.append(mp.mpf(total / mesh.h**k))
            out.append(row)
        return out


def nodal_guess(mesh: Mesh, source: Any, depth: int) -> list[float]:
    """Scaled dofs at the mesh nodes of a JetField (values and derivatives up to each order)."""
    layout = mesh.discrete.layout
    z = [0.0] * mesh.size
    for m, t in enumerate(mesh.nodes):
        rows = source.at(mp.mpf(t))
        for i, n in enumerate(layout.orders):
            for j in range(n + 1):
                value = rows[i][j] if j < len(rows[i]) else mp.mpf(0)
                z[layout.index(m, i, j)] = float(mp.re(value)) * mesh.h**j
    return z


def check_discretisation(
    mesh: Mesh,
    z: list[float],
    *,
    assembler: Callable[[Mesh, list[float]], tuple[list[float], list[dict[int, float]]]] = assemble,
    step: float = 1e-6,
    tolerance: float = 1e-5,
) -> bool:
    """The gradient of S_h agrees with central differences of S_h, and the Hessian with those of the gradient.

    `action_value` shares no assembly code with `assemble` (it only evaluates the integrand), so a
    wrong stencil, a missing scale factor or a dropped term in the assembler is caught here.
    """
    grad, rows = assembler(mesh, z)
    for index in range(0, len(z), max(1, len(z) // 12)):
        up, down = list(z), list(z)
        up[index] += step
        down[index] -= step
        numeric = (action_value(mesh, up) - action_value(mesh, down)) / (2 * step)
        if abs(grad[index] - numeric) > tolerance * max(1.0, abs(grad[index]), abs(numeric)):
            return False
        for j, value in rows[index].items():
            up_j, down_j = list(z), list(z)
            up_j[j] += step
            down_j[j] -= step
            numeric_h = (assembler(mesh, up_j)[0][index] - assembler(mesh, down_j)[0][index]) / (
                2 * step
            )
            if abs(value - numeric_h) > tolerance * max(1.0, abs(value), abs(numeric_h)):
                return False
    return True
