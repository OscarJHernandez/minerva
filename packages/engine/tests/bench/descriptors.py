"""How hard a reference invariant is, measured on the invariant itself and checked, not assumed.

For each invariant: momentum degree, the range of power of each coordinate, the number of terms, and the
set of *minimal families* `(d, P, C)`: degree in the momenta, bound on the exponent of every ordinary
coordinate, and highest power of a cyclic coordinate. The minimum is not read off the expression: an
equivalent invariant may be simpler. The check asks whether the invariant is the sum of an element of the
family and a multiple of `1` and `H` (the energy is always known to Minerva), where the family element is
any combination of `1`, `H`, the reference invariants and their pairwise products that lies in the family.
That is exact linear algebra over the rationals. It is a minimum *within that span*, and says so.

Independent of `minerva_engine`: the ladder it will be compared with is only read back from a report.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import sympy as sp

Monomial = tuple[int, ...]  # exponents of (q_1..q_n, p_1..p_n)

GRID_DEGREES = range(1, 5)
GRID_POWERS = range(0, 5)
GRID_CYCLIC = range(0, 3)


@dataclass
class Descriptor:
    representable: bool  # Laurent polynomial in single coordinates, polynomial in the momenta
    momentum_degree: int = 0
    power_range: tuple[int, int] = (0, 0)
    terms: int = 0
    ops: int = 0
    frontier: list[tuple[int, int, int]] | None = None  # minimal (d, P, C) within the checked span
    scope: str = ""
    reason: str = ""


def laurent_terms(
    expr: sp.Expr, q: list[sp.Symbol], p: list[sp.Symbol]
) -> dict[Monomial, sp.Rational] | None:
    """Terms of `expr` as exponent vectors, or None if it is not a Laurent polynomial in each coordinate
    separately and a polynomial in the momenta (a square root, or a denominator mixing coordinates)."""
    n = len(q)
    try:
        num, den = sp.fraction(sp.together(sp.expand(expr)))
        den_poly = sp.Poly(den, *q, *p)
        if len(den_poly.terms()) != 1:
            return None
        (den_exp, den_coeff), *_ = den_poly.terms()
        if any(den_exp[n:]):
            return None
        out: dict[Monomial, sp.Rational] = {}
        for exp, coeff in sp.Poly(num, *q, *p).terms():
            key = tuple(e - d for e, d in zip(exp, den_exp, strict=True))
            out[key] = sp.Rational(coeff) / sp.Rational(den_coeff)
        return out
    except (sp.PolynomialError, TypeError, ValueError):
        return None


def _allowed(mono: Monomial, n: int, d: int, power: int, cyc: int, cyclic: set[int]) -> bool:
    """Is the monomial in the family (d, P, C)? A cyclic coordinate appears with powers 0..C only (a
    negative or large power of it is outside every family); the others with |exponent| <= P."""
    qs, ps = mono[:n], mono[n:]
    if sum(ps) > d:
        return False
    for i, e in enumerate(qs):
        if i in cyclic:
            if not 0 <= e <= cyc:
                return False
        elif abs(e) > power:
            return False
    return True


def describe(expr: sp.Expr, q: list[sp.Symbol], p: list[sp.Symbol], cyclic: set[int]) -> Descriptor:
    n = len(q)
    terms = laurent_terms(expr, q, p)
    if terms is None:
        return Descriptor(False, ops=int(sp.count_ops(expr)), reason="not a Laurent polynomial")
    qexp = [m[:n] for m in terms]
    flat = [e for m in qexp for e in m] or [0]
    d = max((sum(m[n:]) for m in terms), default=0)
    descriptor = Descriptor(
        True,
        momentum_degree=d,
        power_range=(min(flat), max(flat)),
        terms=len(terms),
        ops=int(sp.count_ops(expr)),
    )
    if any(not 0 <= m[i] <= max(GRID_CYCLIC) for m in qexp for i in cyclic):
        descriptor.representable = False
        descriptor.reason = "a cyclic coordinate appears with a negative or large power"
    return descriptor


def frontiers(
    H: sp.Expr,
    invariants: list[tuple[str, sp.Expr]],
    q: list[sp.Symbol],
    p: list[sp.Symbol],
    cyclic: set[int],
) -> dict[str, list[tuple[int, int, int]] | None]:
    """Minimal families per invariant, checked by linear algebra over the span of 1, H, the invariants
    and their pairwise products (see the module docstring)."""
    n = len(q)
    base = [("H", H), *invariants]
    gens: list[tuple[str, sp.Expr]] = [("1", sp.Integer(1)), *base]
    for (na, ea), (nb, eb) in itertools.combinations_with_replacement(base, 2):
        gens.append((f"{na}*{nb}", ea * eb))
    table: list[tuple[str, dict[Monomial, sp.Rational]]] = []
    for name, e in gens:
        t = laurent_terms(e, q, p)
        if t is not None:
            table.append((name, t))
    index = {name: j for j, (name, _) in enumerate(table)}
    out: dict[str, list[tuple[int, int, int]] | None] = {}
    for label, _ in invariants:
        if label not in index:
            out[label] = None
    monomials = sorted({m for _, t in table for m in t})
    row = {m: i for i, m in enumerate(monomials)}
    full = sp.zeros(len(monomials), len(table))
    for j, (_, t) in enumerate(table):
        for m, c in t.items():
            full[row[m], j] = c
    column_of = {name: j for j, (name, _) in enumerate(table)}
    always = [full[:, column_of[name]] for name in ("1", "H") if name in column_of]
    supported: dict[str, list[tuple[int, int, int]]] = {label: [] for label, _ in invariants}
    for d, power, cyc in itertools.product(GRID_DEGREES, GRID_POWERS, GRID_CYCLIC):
        outside = [r for m, r in row.items() if not _allowed(m, n, d, power, cyc, cyclic)]
        if outside:
            null = full[outside, :].nullspace()
        else:
            null = [sp.eye(len(table))[:, c] for c in range(len(table))]
        inside = sp.Matrix.hstack(*[full * v for v in null], *always)
        base_rank = int(inside.rank())
        for label, _ in invariants:
            j = column_of.get(label)
            if j is None:
                continue
            # The invariant is the sum of an element of the family and a multiple of 1 and H.
            if int(sp.Matrix.hstack(inside, full[:, j]).rank()) == base_rank:
                supported[label].append((d, power, cyc))
    for label, points in supported.items():
        if label in out:
            continue
        minimal = [
            a
            for a in points
            if not any(b != a and all(x <= y for x, y in zip(b, a, strict=True)) for b in points)
        ]
        out[label] = sorted(minimal) or None
    return out


def dominated(
    frontier: list[tuple[int, int, int]] | None, tiers: list[tuple[int, int, int]]
) -> bool:
    """Is some searched tier (d, P, C) at least as large as a minimal family of the invariant?"""
    if not frontier:
        return False
    return any(
        all(t >= f for t, f in zip(tier, minimal, strict=True))
        for minimal in frontier
        for tier in tiers
    )
