"""Functional independence of a set of constants of motion, and the relations among them.

Several conserved expressions need not be independent: the Laplace-Runge-Lenz components, the angular
momentum and the energy of the Kepler problem are four constants of which only three are functionally
independent. This module says what can be *proved* about that and what is only evidence.

* Functional rank. The Jacobian of the constants with respect to the phase variables, evaluated exactly
  (rational arithmetic) at rational points, gives a **proved lower bound** of the generic rank: the rank
  can only drop under specialisation of the parameters and of the point. Angles are handled through the
  rational parametrisation `cos = (1-u^2)/(1+u^2)`, `sin = 2u/(1+u^2)`, a local change of variable that
  leaves the rank unchanged.
* Algebraic relations. A polynomial relation `R(C_1, .., C_k) = 0` is found by an exact nullspace over the
  monomials of the constants up to a stated number of factors; it is an identity in the phase variables
  (not a numerical coincidence). Only relations that are not consequences of lower ones are kept.
* Upper bound. A relation whose gradient with respect to the constants does not vanish at a point of the
  image lowers the dimension of the image there (implicit function theorem), so the rank is at most `k`
  minus the rank of the gradients of the relations at that point. When lower and upper bound meet, the
  generic rank is **proved**; otherwise the report says which part is proved and which is not.

Nothing here samples numbers to conclude independence: a drop in rank at an unlucky point cannot produce a
false "independent", because the rank at any point is only ever used as a lower bound. Points where the
rank falls (singular regions) are not classified; the rank is the generic one.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from pydantic import BaseModel, Field
from sympy.polys.matrices import DomainMatrix

from .latexprint import to_latex

# Rational sample values: parameters, phase variables and the angle parameter u. Several points are
# tried because a point may be singular for one of the constants.
_SAMPLES = (
    (sp.Rational(3, 2), sp.Rational(5, 3), sp.Rational(7, 4), sp.Rational(2, 5), sp.Rational(9, 7)),
    (
        sp.Rational(5, 4),
        sp.Rational(7, 5),
        sp.Rational(11, 6),
        sp.Rational(3, 7),
        sp.Rational(13, 9),
    ),
    (
        sp.Rational(7, 3),
        sp.Rational(4, 3),
        sp.Rational(13, 8),
        sp.Rational(5, 9),
        sp.Rational(8, 11),
    ),
    (
        sp.Rational(11, 5),
        sp.Rational(9, 4),
        sp.Rational(3, 2),
        sp.Rational(7, 8),
        sp.Rational(17, 13),
    ),
)
MAX_RELATION_FACTORS = 4
MAX_RELATION_COLUMNS = 150


class Relation(BaseModel):
    latex: str
    factors: int  # highest number of constants multiplied together in the relation
    exact: bool = True  # an identity in the phase variables, found by an exact nullspace


class IndependenceReport(BaseModel):
    constants: int = 0
    rank_lower: int = 0  # proved: the generic rank is at least this
    rank_upper: int = 0  # proved: the generic rank is at most this
    rank_proved: bool = False
    max_possible: int | None = None
    relations: list[Relation] = Field(default_factory=list)
    # Per constant, in discovery order: "independent" (rank grew, proved), "dependent" (a relation
    # expresses it through the earlier ones, proved) or "unresolved".
    status: list[str] = Field(default_factory=list)
    method: str = ""
    notes: list[str] = Field(default_factory=list)
    duration_ms: int = 0


@dataclass
class PolyContext:
    """The polynomial setting of the constants (see `discovery.py`)."""

    variables: list[sp.Symbol]  # every polynomial variable, including cos/sin symbols
    free: list[sp.Symbol]  # independent phase variables (not cos/sin), including t if used
    angles: list[tuple[sp.Symbol, sp.Symbol]]
    shift: sp.Expr  # multiplies a Laurent monomial to make it a polynomial
    params: list[sp.Symbol]
    reduce_angles: Any = None  # callable: polynomial -> polynomial modulo cos^2 + sin^2 = 1
    mixed: list[str] = field(default_factory=list)  # coordinates used both as angle and as power


def _rational_point(
    ctx: PolyContext, index: int
) -> tuple[dict[sp.Symbol, sp.Expr], list[sp.Symbol]]:
    values = _SAMPLES[index % len(_SAMPLES)]
    out: dict[sp.Symbol, sp.Expr] = {}
    k = 0
    for s in ctx.params:
        out[s] = values[k % len(values)] + sp.Rational(k, 7)
        k += 1
    for s in ctx.free:
        out[s] = values[k % len(values)] + sp.Rational(k, 11)
        k += 1
    us: list[sp.Symbol] = []
    for c, _s in ctx.angles:
        u = sp.Symbol(f"u_{c.name}")
        us.append(u)
        out[u] = values[k % len(values)]
        k += 1
    return out, us


def _jacobian(exprs: list[sp.Expr], ctx: PolyContext) -> tuple[sp.Matrix, list[sp.Symbol]]:
    sub: dict[sp.Symbol, sp.Expr] = {}
    us: list[sp.Symbol] = []
    for c, s in ctx.angles:
        u = sp.Symbol(f"u_{c.name}")
        us.append(u)
        sub[c] = (1 - u**2) / (1 + u**2)
        sub[s] = 2 * u / (1 + u**2)
    variables = [*ctx.free, *us]
    rows = [[sp.diff(e.subs(sub), x) for x in variables] for e in exprs]
    return sp.Matrix(rows), variables


def point_rank(
    exprs: list[sp.Expr],
    ctx: PolyContext,
    jacobian: tuple[sp.Matrix, list[sp.Symbol]] | None = None,
) -> tuple[int, dict[sp.Symbol, sp.Expr] | None]:
    """Largest exact rank of the Jacobian over the sample points (a proved lower bound of the generic
    rank), with the point where it was attained."""
    if not exprs:
        return 0, None
    matrix, _ = jacobian or _jacobian(exprs, ctx)
    best, where = 0, None
    for i in range(len(_SAMPLES)):
        point, _us = _rational_point(ctx, i)
        try:
            value = matrix.subs(point)
            if value.has(sp.zoo, sp.nan, sp.oo):
                continue
            rank = int(value.rank())
        except (ZeroDivisionError, TypeError, ValueError):
            continue
        if rank > best:
            best, where = rank, point
    return best, where


def _label_symbols(k: int) -> list[sp.Symbol]:
    return [sp.Symbol(f"C{j + 1}") for j in range(k)]


def _find_relations(
    exprs: list[sp.Expr],
    ctx: PolyContext,
    field_: Any,
    deadline: float,
) -> tuple[list[sp.Expr], bool, list[int]]:
    """Polynomial relations among the constants, lowest degree first, each not a multiple of an earlier
    one. Returns (relations as expressions in C1..Ck, complete within the caps, degrees)."""
    k = len(exprs)
    cs = _label_symbols(k)
    relations: list[dict[tuple[int, ...], sp.Expr]] = []
    degrees: list[int] = []
    complete = True
    z = ctx.shift**MAX_RELATION_FACTORS
    cache: dict[tuple[int, ...], dict[tuple[int, ...], sp.Expr]] = {}

    def monomials(depth: int) -> list[tuple[int, ...]]:
        out = [
            alpha for alpha in itertools.product(range(depth + 1), repeat=k) if sum(alpha) <= depth
        ]
        return sorted(out, key=lambda a: (sum(a), a))

    def terms_of(alpha: tuple[int, ...]) -> dict[tuple[int, ...], sp.Expr]:
        if alpha in cache:
            return cache[alpha]
        product = sp.Mul(*[exprs[j] ** alpha[j] for j in range(k)]) if any(alpha) else sp.Integer(1)
        poly_expr = sp.expand(product * z)
        if ctx.reduce_angles is not None:
            poly_expr = ctx.reduce_angles(poly_expr)
        poly = sp.Poly(poly_expr, *ctx.variables)
        cache[alpha] = {tuple(m): c for m, c in poly.terms()}
        return cache[alpha]

    for depth in range(1, MAX_RELATION_FACTORS + 1):
        cols = monomials(depth)
        if len(cols) > MAX_RELATION_COLUMNS or time.monotonic() > deadline:
            complete = False
            break
        try:
            rows: dict[tuple[int, ...], dict[int, sp.Expr]] = {}
            for j, alpha in enumerate(cols):
                for monom, coeff in terms_of(alpha).items():
                    rows.setdefault(monom, {})[j] = coeff
        except (sp.PolynomialError, TypeError):
            complete = False
            break
        row_keys = list(rows)
        sparse = {
            r: {j: field_.from_sympy(sp.sympify(v)) for j, v in rows[key].items()}
            for r, key in enumerate(row_keys)
        }
        null = DomainMatrix(sparse, (len(row_keys), len(cols)), field_).to_dense().nullspace()
        if null.shape[0] == 0:
            continue
        col_index = {alpha: j for j, alpha in enumerate(cols)}
        # Multiples of earlier relations by monomials: not new.
        multiples: list[dict[int, sp.Expr]] = []
        for rel, rel_degree in zip(relations, degrees, strict=True):
            for mu in monomials(depth - rel_degree):
                vec = {}
                for alpha, coeff in rel.items():
                    beta = tuple(a + m for a, m in zip(alpha, mu, strict=True))
                    vec[col_index[beta]] = coeff
                multiples.append(vec)
        from .invariants import _new_directions

        def to_row(vector: dict[int, sp.Expr]) -> dict[int, Any]:
            return {j: field_.from_sympy(sp.sympify(v)) for j, v in vector.items()}

        fresh = _new_directions(null, multiples, to_row, field_, len(cols))
        for vector in fresh:
            clean = {cols[j]: sp.together(sp.sympify(v)) for j, v in vector.items()}
            lcm = sp.Integer(1)
            for v in clean.values():
                lcm = sp.lcm(lcm, sp.fraction(v)[1])
            clean = {a: sp.cancel(v * lcm) for a, v in clean.items()}
            gcd = sp.gcd_list(list(clean.values()))
            if gcd != 0:
                clean = {a: sp.cancel(v / gcd) for a, v in clean.items()}
            relations.append(clean)
            degrees.append(max(sum(a) for a in clean))
    out = [
        sp.Add(*[coeff * sp.Mul(*[cs[j] ** a[j] for j in range(k)]) for a, coeff in rel.items()])
        for rel in relations
    ]
    return out, complete, degrees


def analyze_independence(
    exprs: list[sp.Expr],
    ctx: PolyContext,
    max_possible: int | None,
    deadline: float,
    *,
    with_relations: bool = True,
) -> IndependenceReport:
    """Rank and relations of constants given in polynomial form (see `PolyContext`), in discovery order."""
    started = time.monotonic()
    k = len(exprs)
    report = IndependenceReport(constants=k, max_possible=max_possible)
    if k == 0:
        report.rank_proved = True
        report.method = "no constants"
        return report
    if ctx.mixed:
        report.rank_upper = min(k, max_possible or k)
        report.status = ["unresolved"] * k
        report.method = "not analysed"
        report.notes.append(
            "A coordinate is used both as an angle and as a power in different constants "
            f"({', '.join(ctx.mixed)}); rank is not analysed."
        )
        return report

    jac = _jacobian(exprs, ctx)
    rank, point = point_rank(exprs, ctx, jac)
    report.rank_lower = rank
    cap = k if max_possible is None else min(k, max_possible)
    upper = cap
    report.method = "exact Jacobian rank at rational points (proved lower bound)"

    relations: list[sp.Expr] = []
    complete = True
    if with_relations and rank < k:
        domain = sp.QQ.frac_field(*ctx.params) if ctx.params else sp.QQ
        try:
            relations, complete, degrees = _find_relations(exprs, ctx, domain, deadline)
        except (sp.PolynomialError, sp.polys.polyerrors.BasePolynomialError):
            relations, complete, degrees = [], False, []
        cs = _label_symbols(k)
        names = {c: f"C_{{{j + 1}}}" for j, c in enumerate(cs)}
        for rel, deg in zip(relations, degrees, strict=True):
            report.relations.append(
                Relation(latex=f"{to_latex(rel, symbol_names=names)} = 0", factors=deg)
            )
        if relations and point is not None:
            g0 = [e.subs(point) for e in exprs]
            grads = sp.Matrix(
                [
                    [sp.diff(rel, c).subs(dict(zip(cs, g0, strict=True))).subs(point) for c in cs]
                    for rel in relations
                ]
            )
            try:
                upper = min(upper, k - int(grads.rank()))
                report.method += "; relations bound the rank from above (implicit function theorem)"
            except (ZeroDivisionError, TypeError, ValueError):
                pass
    report.rank_upper = upper
    report.rank_proved = rank == upper
    if rank < upper:
        report.notes.append(
            f"The generic rank is at least {rank} (proved) and at most {upper}; the exact value "
            "is not established. Numerical evidence is not used to close the gap."
        )
    if not complete and rank < k:
        report.notes.append(
            "The search for algebraic relations stopped at its size or time cap; absence of a "
            "relation here is not a proof of independence."
        )

    # Status of each constant, in discovery order.
    prefix_rank = 0
    for j in range(k):
        r_j, _ = point_rank(exprs[: j + 1], ctx)
        if r_j > prefix_rank:
            report.status.append("independent")
            prefix_rank = r_j
        elif prefix_rank == upper:
            # The proved upper bound for all of them equals the proved rank of the earlier ones, so
            # this one adds nothing: its gradient lies in the span of the earlier gradients.
            report.status.append("dependent")
        else:
            dependent = False
            if relations and point is not None:
                cs = _label_symbols(k)
                at_point = dict(zip(cs, [e.subs(point) for e in exprs], strict=True))
                for rel in relations:
                    involved = [i for i, c in enumerate(cs) if rel.has(c)]
                    if involved and max(involved) == j:
                        slope = sp.diff(rel, cs[j]).subs(at_point).subs(point)
                        if slope != 0:
                            dependent = True
                            break
            report.status.append("dependent" if dependent else "unresolved")
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report


__all__ = ["IndependenceReport", "PolyContext", "Relation", "analyze_independence", "point_rank"]
