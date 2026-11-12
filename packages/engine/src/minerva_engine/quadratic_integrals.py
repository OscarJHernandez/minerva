"""Integrals quadratic in the velocities: the Runge-Lenz vector and its relatives.

For L = (m/2) |qdot|^2 - V(q) in Cartesian coordinates (a constant identity mass matrix) look for

    I = m K_ij(q) qdot_i qdot_j + U(q),

with K a symmetric tensor whose entries are polynomials of degree <= 2 in q. Then

    dI/dt = m (d_k K_ij) qdot_i qdot_j qdot_k - 2 K_ij (d_i V) qdot_j + (d_j U) qdot_j ,

so I is conserved exactly when

    (1) d_(k K_ij) = 0                          K is a Killing tensor of flat space (linear algebra, exact),
    (2) d_j U = 2 K_ij d_i V                    the 1-form 2 K_ij d_i V is closed, and U is its potential.

Condition (2) is linear in K, so the allowed K form a vector space, found by a numerical null space at random
points and random parameter values (mpmath, 50 digits) and made exact by rationalising its reduced echelon
form; K that depend on the parameters are not found. The trivial part is removed: the energy (K = identity)
and the products of the linear integrals that the symmetries of V already give (translations and rotations
with xi . grad V = 0), since those are not new. For each remaining K the potential U is integrated by SymPy.

Nothing found here is trusted. Each candidate is checked by the same test as every other constant of motion
(`lagrangian.verify_quantity`: d/dt along the equations of motion is zero, symbolically and at random points,
reported separately); a candidate that cannot be rationalised or integrated, or fails that test, is dropped.
So the search may be heuristic without risk of a false result. It is also bounded: at most three coordinates,
quadratic in the velocities, K of degree <= 2 in the positions, no explicit time, potentials built from powers
and roots only. Integrals of higher degree in the velocities (products of the ones found are not listed),
integrals that need other functions of the positions, and integrals in non-Cartesian coordinates are not found,
and the report says so.
"""

from __future__ import annotations

import itertools
import random
import time
from fractions import Fraction

import mpmath
import sympy as sp

from .lagrangian import ConservedQuantity, Derivation, verify_quantity
from .verify import _real_sample

MAX_COORDINATES = 3
DPS = 50
NULL_TOL = mpmath.mpf("1e-25")
PIVOT_TOL = mpmath.mpf("1e-18")
RATIONAL_TOL = mpmath.mpf("1e-18")
MAX_OPS = 80
SAMPLE_ATTEMPTS = 400


class QuadraticSearch:
    def __init__(self) -> None:
        self.items: list[tuple[ConservedQuantity, sp.Expr]] = []
        self.tested: list[str] = []
        self.notes: list[str] = []


# --- small linear algebra helpers ---------------------------------------------------------------
def _rref(rows: list[list[mpmath.mpf]]) -> list[list[mpmath.mpf]]:
    m = [row[:] for row in rows]
    pivot_row = 0
    for c in range(len(m[0]) if m else 0):
        if pivot_row >= len(m):
            break
        best = max(range(pivot_row, len(m)), key=lambda i: abs(m[i][c]))
        if abs(m[best][c]) < PIVOT_TOL:
            continue
        m[pivot_row], m[best] = m[best], m[pivot_row]
        pivot = m[pivot_row][c]
        m[pivot_row] = [x / pivot for x in m[pivot_row]]
        for i in range(len(m)):
            if i != pivot_row and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[pivot_row], strict=True)]
        pivot_row += 1
    return m[:pivot_row]


def _null_space(rows: list[list[mpmath.mpf]], ncols: int) -> list[list[sp.Rational]] | None:
    """Rational basis (reduced echelon form) of the numerical null space of `rows`; None if not rational."""
    if len(rows) < ncols:
        return None
    matrix = mpmath.matrix(rows)
    _, s, v = mpmath.svd_r(matrix, full_matrices=False)
    scale = max(abs(x) for x in s)
    if scale == 0:
        return None
    null = [[v[i, j] for j in range(ncols)] for i in range(len(s)) if abs(s[i]) < NULL_TOL * scale]
    if not null:
        return []
    out: list[list[sp.Rational]] = []
    for row in _rref(null):
        rational_row: list[sp.Rational] = []
        for x in row:
            fraction = Fraction(mpmath.nstr(x, 40)).limit_denominator(10**6)
            if abs(x - mpmath.mpf(fraction.numerator) / fraction.denominator) > RATIONAL_TOL:
                return None
            rational_row.append(sp.Rational(fraction.numerator, fraction.denominator))
        out.append(rational_row)
    return out


# --- Killing tensors and vectors of flat space ------------------------------------------------------
def _killing_tensors(q: list[sp.Symbol]) -> list[sp.Matrix]:
    """Symmetric tensors with entries of degree <= 2 in q and d_(k K_ij) = 0 (an exact rational basis)."""
    n = len(q)
    monomials = [sp.Integer(1), *q, *[q[a] * q[b] for a in range(n) for b in range(a, n)]]
    k_matrix = sp.zeros(n, n)
    unknowns: list[sp.Symbol] = []
    for i in range(n):
        for j in range(i, n):
            expr = sp.Integer(0)
            for mu in monomials:
                c = sp.Symbol(f"c{len(unknowns)}")
                unknowns.append(c)
                expr += c * mu
            k_matrix[i, j] = expr
            k_matrix[j, i] = expr
    equations: list[sp.Expr] = []
    for i, j, k in itertools.product(range(n), repeat=3):
        e = sp.expand(
            sp.diff(k_matrix[i, j], q[k])
            + sp.diff(k_matrix[j, k], q[i])
            + sp.diff(k_matrix[k, i], q[j])
        )
        if e != 0:
            equations.extend(sp.Poly(e, *q).coeffs())
    a_matrix, _ = sp.linear_eq_to_matrix(equations, unknowns)
    return [k_matrix.subs(dict(zip(unknowns, v, strict=True))) for v in a_matrix.nullspace()]


def _killing_vectors(q: list[sp.Symbol]) -> list[list[sp.Expr]]:
    n = len(q)
    vectors: list[list[sp.Expr]] = []
    for i in range(n):
        vectors.append([sp.Integer(1) if j == i else sp.Integer(0) for j in range(n)])
    for i, j in itertools.combinations(range(n), 2):
        v = [sp.Integer(0)] * n
        v[i], v[j] = -q[j], q[i]
        vectors.append(v)
    return vectors


def _coordinates(
    tensor: sp.Matrix, q: list[sp.Symbol]
) -> dict[tuple[int, int, tuple[int, ...]], sp.Expr]:
    out: dict[tuple[int, int, tuple[int, ...]], sp.Expr] = {}
    n = len(q)
    for i in range(n):
        for j in range(i, n):
            for exps, c in sp.Poly(tensor[i, j], *q).terms():
                out[(i, j, exps)] = c
    return out


# --- the search -----------------------------------------------------------------------------------
def _flat_form(d: Derivation) -> tuple[sp.Expr, sp.Expr] | None:
    """(m, V) if L = (m/2) sum qdot^2 - V(q) with m constant; else None."""
    n = d.n
    mass = sp.Matrix(n, n, lambda i, j: sp.diff(d.lagrangian, d.qd[i], d.qd[j]))
    m = mass[0, 0]
    if m.free_symbols & (set(d.q) | set(d.qd) | {d.t}):
        return None
    for i, j in itertools.product(range(n), repeat=2):
        if sp.simplify(mass[i, j] - (m if i == j else 0)) != 0:
            return None
    rest = sp.expand(
        d.lagrangian - sp.Rational(1, 2) * m * sum((v**2 for v in d.qd), sp.Integer(0))
    )
    if rest.free_symbols & set(d.qd):
        return None
    return m, -rest


def find_quadratic(d: Derivation, deadline: float) -> QuadraticSearch:
    out = QuadraticSearch()
    skipped = "integrals quadratic in the velocities: not searched"
    if d.time_dependent or not d.solution:
        return out
    if d.n < 2:
        out.notes.append(f"{skipped} (needs at least two coordinates).")
        return out
    if d.n > MAX_COORDINATES:
        out.notes.append(f"{skipped} (more than {MAX_COORDINATES} coordinates).")
        return out
    flat = _flat_form(d)
    if flat is None:
        out.notes.append(
            f"{skipped}: it needs L = (m/2)|qdot|^2 - V(q) in Cartesian coordinates, and these "
            "coordinates are not of that form (so, for example, a Kepler problem in polar "
            "coordinates is not searched for its Runge-Lenz vector)."
        )
        return out
    mass, potential = flat
    if potential.atoms(sp.Function) or sp.count_ops(potential) > MAX_OPS:
        out.notes.append(
            f"{skipped}: the potential must be built from powers and roots of the coordinates "
            f"(at most {MAX_OPS} operations)."
        )
        return out

    q = d.q
    n = d.n
    params = sorted(potential.free_symbols - set(q), key=lambda s: s.name)
    grad = [sp.diff(potential, x) for x in q]
    basis = _killing_tensors(q)
    if not basis:
        return out

    # Closedness of w_j = 2 K_ij d_i V, for each basis tensor: F_jl = d_l w_j - d_j w_l.
    pairs = list(itertools.combinations(range(n), 2))
    flat_f: list[sp.Expr] = []
    for k_tensor in basis:
        w = [2 * sum((k_tensor[i, j] * grad[i] for i in range(n)), sp.Integer(0)) for j in range(n)]
        flat_f.extend(sp.diff(w[j], q[lq]) - sp.diff(w[lq], q[j]) for j, lq in pairs)
    killing_vectors = _killing_vectors(q)
    flat_g = [sum((v[i] * grad[i] for i in range(n)), sp.Integer(0)) for v in killing_vectors]
    if time.monotonic() > deadline:
        out.notes.append(f"{skipped} (time budget used up).")
        return out

    rng = random.Random(20260104)  # noqa: S311 - deterministic sampling, not cryptographic
    rows_f: list[list[mpmath.mpf]] = []
    rows_g: list[list[mpmath.mpf]] = []
    valid_points = 0
    with mpmath.workdps(DPS):
        try:
            f_fn = sp.lambdify([*q, *params], flat_f, "mpmath")
            g_fn = sp.lambdify([*q, *params], flat_g, "mpmath")
        except (TypeError, ValueError, NameError):
            out.notes.append(f"{skipped} (the potential could not be evaluated numerically).")
            return out
        need_f = len(basis) + 8
        for _ in range(SAMPLE_ATTEMPTS):
            if len(rows_f) >= len(pairs) * need_f and len(rows_g) >= len(killing_vectors) + 8:
                break
            point = [mpmath.mpf(_real_sample(s, rng)) for s in q]
            values = [mpmath.mpf(_real_sample(s, rng)) for s in params]
            try:
                fv = [mpmath.mpmathify(x) for x in f_fn(*point, *values)]
                gv = [mpmath.mpmathify(x) for x in g_fn(*point, *values)]
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                continue
            if not all(mpmath.isfinite(x) and abs(x) < mpmath.mpf(10) ** 12 for x in [*fv, *gv]):
                continue
            valid_points += 1
            for p_index in range(len(pairs)):
                rows_f.append([fv[a * len(pairs) + p_index] for a in range(len(basis))])
            rows_g.append(gv)
        # Rows are normalised so that the tolerances mean the same for every potential.
        rows_f = [r for r in (_normalised(row) for row in rows_f) if r is not None]
        rows_g = [r for r in (_normalised(row) for row in rows_g) if r is not None]
        # No equation at all (every row zero, e.g. a free particle) means every unknown is allowed.
        closed = (
            _unit_vectors(len(basis))
            if not rows_f and valid_points >= 6
            else _null_space(rows_f, len(basis))
        )
        symmetry = (
            _unit_vectors(len(killing_vectors))
            if not rows_g and valid_points >= 6
            else _null_space(rows_g, len(killing_vectors))
        )
    if closed is None:
        out.notes.append(
            f"{skipped} (the integrals found numerically do not have rational coefficients, "
            "or too few sample points were valid)."
        )
        return out
    if time.monotonic() > deadline:
        out.notes.append(f"{skipped} (time budget used up).")
        return out

    # Exact tensors from the rational combinations.
    def combine(vector: list[sp.Rational]) -> sp.Matrix:
        total = sp.zeros(n, n)
        for coefficient, k_tensor in zip(vector, basis, strict=True):
            total += coefficient * k_tensor
        return total

    candidates = [combine(v) for v in closed]

    # Symmetries of V (exact): vector fields xi with xi . grad V = 0.
    symmetric_fields: list[list[sp.Expr]] = []
    for vector in symmetry or []:
        field = [
            sum((c * kv[i] for c, kv in zip(vector, killing_vectors, strict=True)), sp.Integer(0))
            for i in range(n)
        ]
        if sp.simplify(sum((field[i] * grad[i] for i in range(n)), sp.Integer(0))) == 0:
            symmetric_fields.append(field)
    trivial = [sp.eye(n)]
    for a, b in itertools.combinations_with_replacement(range(len(symmetric_fields)), 2):
        xa, xb = sp.Matrix(symmetric_fields[a]), sp.Matrix(symmetric_fields[b])
        trivial.append((xa * xb.T + xb * xa.T) / 2)

    keys = sorted(
        {key for t in [*trivial, *candidates] for key in _coordinates(t, q)},
        key=lambda k: (k[0], k[1], k[2]),
    )

    def vectorise(tensor: sp.Matrix) -> sp.Matrix:
        coords = _coordinates(tensor, q)
        return sp.Matrix([[coords.get(key, sp.Integer(0)) for key in keys]])

    trivial_rows = [vectorise(t) for t in trivial]
    base = sp.Matrix.vstack(*trivial_rows)
    if candidates:
        all_closed = sp.Matrix.vstack(*[vectorise(c) for c in candidates])
        if all_closed.rank() != sp.Matrix.vstack(base, all_closed).rank():
            out.notes.append(
                "Some known trivial integrals were not among the closed tensors found numerically; "
                "the integrals listed below may include combinations of known ones."
            )
    chosen: list[sp.Matrix] = []
    current = base
    for tensor in candidates:
        row = vectorise(tensor)
        stacked = sp.Matrix.vstack(current, row)
        if stacked.rank() > current.rank():
            chosen.append(tensor)
            current = stacked
    if not chosen:
        out.tested.append(
            "integrals quadratic in the velocities (K of degree <= 2 in the positions): none beyond the "
            "energy and the products of the linear integrals"
        )
        return out

    show = d.show
    index = 0
    for tensor in chosen:
        if time.monotonic() > deadline:
            out.notes.append("The search for quadratic integrals stopped: time budget used up.")
            break
        found = _potential(tensor, potential, q, params)
        if found is None:
            out.notes.append(
                "A quadratic integral exists but its potential U could not be built in closed form "
                "(the potential must be a sum of terms homogeneous in the coordinates, and no logarithm "
                "may be needed)."
            )
            continue
        w, u = found
        if any(sp.simplify(sp.diff(u, q[j]) - w[j]) != 0 for j in range(n)):
            out.notes.append("A candidate was not exactly integrable and was dropped.")
            continue
        quantity = sp.expand(
            mass
            * sum(
                (tensor[i, j] * d.qd[i] * d.qd[j] for i in range(n) for j in range(n)),
                sp.Integer(0),
            )
            + u
        )
        ok, verification = verify_quantity(d, quantity)
        if not ok:
            out.notes.append("A candidate quadratic integral failed verification and was dropped.")
            continue
        index += 1
        out.items.append(
            (
                ConservedQuantity(
                    name=f"Quadratic integral {index}",
                    kind="quadratic",
                    expression_latex=f"I_{{{index}}} = {show(quantity)}",
                    reason=(
                        "quadratic in the velocities, I = m K_ij(q) qdot_i qdot_j + U(q) with K a Killing tensor "
                        "of flat space; independent of the energy and of products of the linear integrals; "
                        "verified by d/dt = 0 on the equations of motion"
                    ),
                    verified=ok,
                    verification=verification,
                ),
                quantity,
            )
        )
    if out.items:
        out.notes.append(
            f"{len(out.items)} quadratic integral{'s' if len(out.items) != 1 else ''} found beyond the energy "
            "and the products of the linear integrals. The search is bounded: Cartesian coordinates, quadratic "
            "in the velocities, K of degree <= 2 in the positions, no explicit time. Other integrals may exist."
        )
    return out


def _unit_vectors(n: int) -> list[list[sp.Rational]]:
    return [[sp.Integer(1) if i == j else sp.Integer(0) for j in range(n)] for i in range(n)]


def _normalised(row: list[mpmath.mpf]) -> list[mpmath.mpf] | None:
    norm = mpmath.sqrt(sum((x * x for x in row), mpmath.mpf(0)))
    if norm < mpmath.mpf("1e-30"):
        return None  # a row of zeros says nothing about the unknowns
    return [x / norm for x in row]


def _homogeneous_terms(
    potential: sp.Expr, q: list[sp.Symbol]
) -> list[tuple[sp.Expr, sp.Rational]] | None:
    """The terms of V with their degrees of homogeneity in q; None if some term is not homogeneous."""
    s = sp.Symbol("s", positive=True)
    scaling = {x: s * x for x in q}
    out: list[tuple[sp.Expr, sp.Rational]] = []
    for term in sp.Add.make_args(sp.expand(potential)):
        if not (term.free_symbols & set(q)):
            continue  # a constant does not enter the force
        ratio = sp.simplify(term.subs(scaling, simultaneous=True) / term)
        if ratio == 1:
            out.append((term, sp.Rational(0)))
            continue
        base, exponent = ratio.as_base_exp()
        if base != s or not exponent.is_Rational:
            return None
        out.append((term, sp.Rational(exponent)))
    return out


def _potential(
    tensor: sp.Matrix, potential: sp.Expr, q: list[sp.Symbol], params: list[sp.Symbol]
) -> tuple[list[sp.Expr], sp.Expr] | None:
    """(w, U) with w_j = 2 K_ij d_i V and grad U = w, without symbolic integration.

    If V is a sum of terms V_a homogeneous of degree p_a in q and K has parts K^(e) of degree e, the part
    2 K^(e)_ij d_i V_a of w is homogeneous of degree p_a + e - 1. Each degree class of a closed form is
    closed. For a class of degree d != -1, Euler's formula gives U = q . W / (d + 1). For d = -1 (the
    Runge-Lenz vector of the Kepler problem is of this kind) U is homogeneous of degree 0 and Euler's formula
    says nothing; there U is looked for as a combination of nu(q) V_a with nu monomials of degree e, with
    coefficients found numerically and rationalised. The caller checks grad U = w exactly.
    """
    n = len(q)
    terms = _homogeneous_terms(potential, q)
    if terms is None:
        return None
    w_total = [sp.Integer(0)] * n
    u = sp.Integer(0)
    degenerate: list[tuple[sp.Expr, int, list[sp.Expr]]] = []  # (V_a, e, w piece)
    for e in range(3):
        k_e = sp.zeros(n, n)
        for i in range(n):
            for j in range(n):
                for exps, c in sp.Poly(tensor[i, j], *q).terms():
                    if sum(exps) == e:
                        k_e[i, j] += c * sp.prod([x**k for x, k in zip(q, exps, strict=True)])
        if k_e == sp.zeros(n, n):
            continue
        for term, p in terms:
            grad = [sp.diff(term, x) for x in q]
            w_piece = [
                2 * sum((k_e[i, j] * grad[i] for i in range(n)), sp.Integer(0)) for j in range(n)
            ]
            w_total = [w_total[j] + w_piece[j] for j in range(n)]
            degree = p + e - 1
            if degree == -1:
                degenerate.append((term, e, w_piece))
            else:
                u += sum((q[j] * w_piece[j] for j in range(n)), sp.Integer(0)) / (degree + 1)
    if degenerate:
        extra = _degree_zero_potential(degenerate, q, params)
        if extra is None:
            return None
        u += extra
    return w_total, u


def _degree_zero_potential(
    pieces: list[tuple[sp.Expr, int, list[sp.Expr]]], q: list[sp.Symbol], params: list[sp.Symbol]
) -> sp.Expr | None:
    n = len(q)
    candidates: list[sp.Expr] = []
    for term, e, _ in pieces:
        for exps in itertools.combinations_with_replacement(range(n), e):
            candidates.append(term * sp.prod([q[i] for i in exps]))
    target = [sum((piece[j] for _, _, piece in pieces), sp.Integer(0)) for j in range(n)]
    rng = random.Random(20260105)  # noqa: S311 - deterministic sampling, not cryptographic
    rows: list[list[mpmath.mpf]] = []
    rhs: list[mpmath.mpf] = []
    with mpmath.workdps(DPS):
        try:
            matrix_fn = sp.lambdify(
                [*q, *params], [sp.diff(c, x) for x in q for c in candidates], "mpmath"
            )
            target_fn = sp.lambdify([*q, *params], target, "mpmath")
        except (TypeError, ValueError, NameError):
            return None
        for _ in range(SAMPLE_ATTEMPTS):
            if len(rows) >= n * (len(candidates) + 6):
                break
            point = [mpmath.mpf(_real_sample(s, rng)) for s in q]
            values = [mpmath.mpf(_real_sample(s, rng)) for s in params]
            try:
                derivatives = [mpmath.mpmathify(x) for x in matrix_fn(*point, *values)]
                wanted = [mpmath.mpmathify(x) for x in target_fn(*point, *values)]
            except (ArithmeticError, ValueError, TypeError, ZeroDivisionError):
                continue
            if not all(
                mpmath.isfinite(x) and abs(x) < mpmath.mpf(10) ** 12
                for x in [*derivatives, *wanted]
            ):
                continue
            for j in range(n):
                rows.append([derivatives[j * len(candidates) + c] for c in range(len(candidates))])
                rhs.append(wanted[j])
        if len(rows) < len(candidates) + 2:
            return None
        try:
            solution, residual = mpmath.qr_solve(mpmath.matrix(rows), mpmath.matrix(rhs))
        except (ArithmeticError, ValueError, ZeroDivisionError):
            return None
        scale = max(abs(x) for x in rhs) or mpmath.mpf(1)
        if residual > mpmath.mpf("1e-18") * scale * len(rows):
            return None
        total = sp.Integer(0)
        for c, candidate in zip(solution, candidates, strict=True):
            fraction = Fraction(mpmath.nstr(c, 40)).limit_denominator(10**6)
            if abs(c - mpmath.mpf(fraction.numerator) / fraction.denominator) > RATIONAL_TOL:
                return None
            total += sp.Rational(fraction.numerator, fraction.denominator) * candidate
    return total
