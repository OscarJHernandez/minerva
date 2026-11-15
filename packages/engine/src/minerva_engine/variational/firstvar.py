"""The first variation by explicit integration by parts.

For `S[x] = ∫_a^b L dt + G_b + G_a` and `x_i → x_i + ε η_i`:

    δL = Σ_{i,k} P_{i,k} η_i^(k),               P_{i,k} = ∂L/∂u_i^(k)      (a partial derivative)

One derivative is moved per step, `Q η^(k) = D(Q η^(k-1)) - (D Q) η^(k-1)`, which adds `Q η^(k-1)`
to the boundary function `B` and `-D Q` to the coefficient of `η^(k-1)` (`D` is the *total*
derivative). When every derivative has been moved, the integrand is `Σ EL_i η_i` and

    δS = ∫_a^b Σ_i EL_i η_i dt + [B]_a^b + (δG_b) + (δG_a),
    EL_i = Σ_k (-1)^k D^k P_{i,k},  B = Σ_{i,j} Π_{i,j} η_i^(j),
    Π_{i,j} = Σ_{k>j} (-1)^{k-j-1} D^{k-j-1} P_{i,k}.

Nothing is discarded: the boundary form is kept whole, and `coefficients` lists, for every end and
every derivative of every function, what multiplies the value of that perturbation there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp

from ..errors import MinervaMathError
from .jets import JetSpace
from .models import End
from .setup import Problem


class VariationError(MinervaMathError):
    """The derivation disagreed with its own closed form: a bug, never shown as a result."""


def tidy(expr: sp.Expr) -> sp.Expr:
    """Expand; simplify only when the expression is small and has quotients or fractional powers."""
    out = sp.expand(expr)
    if (
        out.has(sp.Pow)
        and sp.count_ops(out) < 250
        and any(
            (p.exp.is_Rational and not p.exp.is_Integer) or p.exp.is_negative
            for p in out.atoms(sp.Pow)
        )
    ):
        out = sp.simplify(out)
    return out


@dataclass(frozen=True)
class Stage:
    index: int
    function: int  # index of the function whose derivative was moved (-1 for stage 0)
    order: int  # the derivative that was moved: η^(order) -> η^(order-1)
    interior: sp.Expr  # the integrand after the step, linear in the η jets
    boundary: sp.Expr  # the boundary function B after the step
    needs_derivatives: int  # highest derivative of any function the integrand now contains


@dataclass
class FirstVariationData:
    delta_integrand: sp.Expr
    stages: list[Stage]
    euler: list[sp.Expr]  # EL_i
    pi: dict[tuple[int, int], sp.Expr]  # Π_{i,j}, j < n_i
    boundary: sp.Expr  # B
    # What multiplies η_i^(j) at an end in δS, with t replaced by the end:
    coefficients: dict[tuple[End, int, int], sp.Expr] = field(default_factory=dict)
    null: bool = False
    el_orders: list[int] = field(
        default_factory=list
    )  # highest derivative of x_j in each EL (max over j)


def closed_euler(jets: JetSpace, L: sp.Expr, i: int) -> sp.Expr:
    return sum(
        ((-1) ** k * jets.Dn(jets.partial(L, i, k), k) for k in range(jets.orders[i] + 1)),
        sp.Integer(0),
    )


def closed_pi(jets: JetSpace, L: sp.Expr, i: int, j: int) -> sp.Expr:
    return sum(
        (
            (-1) ** (k - j - 1) * jets.Dn(jets.partial(L, i, k), k - j - 1)
            for k in range(j + 1, jets.orders[i] + 1)
        ),
        sp.Integer(0),
    )


def _interior(jets: JetSpace, q: dict[tuple[int, int], sp.Expr]) -> sp.Expr:
    return sum((c * jets.eta[i][k] for (i, k), c in q.items() if c != 0), sp.Integer(0))


def first_variation(problem: Problem) -> FirstVariationData:
    jets, L = problem.jets, problem.lagrangian
    top = {i: jets.orders[i] for i in range(jets.count)}
    q: dict[tuple[int, int], sp.Expr] = {}
    for i in range(jets.count):
        for k in range(top[i] + 1):
            q[(i, k)] = jets.partial(L, i, k)
    delta = _interior(jets, q)
    boundary = sp.Integer(0)
    stages = [Stage(0, -1, 0, delta, boundary, _needs(jets, delta))]
    pi: dict[tuple[int, int], sp.Expr] = {}
    for i in range(jets.count):
        for k in range(top[i], 0, -1):
            moved = q[(i, k)]
            pi[(i, k - 1)] = moved
            boundary = boundary + moved * jets.eta[i][k - 1]
            q[(i, k - 1)] = q[(i, k - 1)] - jets.D(moved)
            q[(i, k)] = sp.Integer(0)
            interior = _interior(jets, q)
            stages.append(Stage(len(stages), i, k, interior, boundary, _needs(jets, interior)))

    euler = [tidy(q[(i, 0)]) for i in range(jets.count)]
    pi = {key: tidy(value) for key, value in pi.items()}

    # The stepwise result must equal the closed forms (a different route to the same objects).
    for i in range(jets.count):
        if sp.simplify(euler[i] - closed_euler(jets, L, i)) != 0:
            raise VariationError(
                "the stepwise Euler-Lagrange expression disagrees with its closed form"
            )
        for j in range(top[i]):
            if sp.simplify(pi[(i, j)] - closed_pi(jets, L, i, j)) != 0:
                raise VariationError(
                    "the stepwise boundary coefficient disagrees with its closed form"
                )

    data = FirstVariationData(
        delta_integrand=delta,
        stages=stages,
        euler=euler,
        pi=pi,
        boundary=sum((p * jets.eta[i][j] for (i, j), p in pi.items()), sp.Integer(0)),
        null=all(e == 0 for e in euler),
        el_orders=[
            max((jets.order_in(euler[i], j) for j in range(jets.count)), default=-1)
            for i in range(jets.count)
        ],
    )
    data.coefficients = boundary_coefficients(problem, pi)
    return data


def _needs(jets: JetSpace, expr: sp.Expr) -> int:
    return max((k for _, k in jets.jets_in(expr)), default=0)


def boundary_coefficients(
    problem: Problem, pi: dict[tuple[int, int], sp.Expr]
) -> dict[tuple[End, int, int], sp.Expr]:
    """The coefficient of η_i^(j)(end) in δS, `t` replaced by the end.

    Upper end: `+Π_{i,j} + ∂G_b/∂u_i^(j)`. Lower end: `-Π_{i,j} + ∂G_a/∂u_i^(j)` (the lower end of
    `[B]_a^b` enters with a minus sign). Derivatives of order `j >= n_i` carry only the endpoint term.
    """
    jets = problem.jets
    out: dict[tuple[End, int, int], sp.Expr] = {}
    ends: tuple[tuple[End, sp.Expr, sp.Expr, int], ...] = (
        ("lower", problem.lower, problem.lower_term, -1),
        ("upper", problem.upper, problem.upper_term, 1),
    )
    for end, bound, term, sign in ends:
        for i in range(jets.count):
            reach = max(jets.orders[i], jets.order_in(term, i) + 1) if term != 0 else jets.orders[i]
            for j in range(reach):
                value = sign * pi.get((i, j), sp.Integer(0)) + jets.partial(term, i, j)
                value = tidy(value.subs(jets.variable, bound)) if value != 0 else sp.Integer(0)
                out[(end, i, j)] = value
    return out
