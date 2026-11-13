"""A search for constants of motion beyond the Noether families of `lagrangian.py`.

`find_conserved` looks for cyclic coordinates, energy and rotations. This module asks a different
question: does the system have a constant of motion of the form

    C(q, q-dot) = sum over |alpha| <= D of  f_alpha(q) * q-dot^alpha

where each f_alpha is a combination of a finite, declared set of functions of the coordinates (powers
q^a with |a| <= P for an ordinary coordinate; 1, cos, sin, ... up to a chosen harmonic for a coordinate
declared to be an angle)? The condition dC/dt = 0 on the Euler-Lagrange equations is linear in the
unknown coefficients, so it is solved exactly (rational arithmetic over the field of the parameters,
no floating point). The Runge-Lenz vector of the Kepler problem and the Fradkin tensor of the
isotropic oscillator are in this family; nothing is looked up or assumed about them.

What is reported:

* only constants whose time derivative, evaluated on the equations of motion, is zero symbolically
  and at random numerical points (the same check as for Noether constants); a candidate that fails
  is dropped, never shown;
* only constants that are new: the cyclic momenta, the energy, the rotation charges, 1 and their
  products that lie in the ansatz are factored out before counting;
* the exact ansatz that was searched. A search that finds nothing proves nothing outside that ansatz.

Supported: a Lagrangian without explicit time dependence whose accelerations are rational in the
coordinates, the velocities and (for declared angles) cos and sin. Anything else is `unsupported`.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import sympy as sp
from pydantic import BaseModel, Field
from sympy.polys.matrices import DomainMatrix

from .lagrangian import (
    Derivation,
    DerivationError,
    check_equations,
    derive,
    find_conserved,
    verify_constant,
)
from .models import Limits, Status, Verification

MAX_COORDINATES = 3
MAX_UNKNOWNS = 500
MAX_DEGREE = 2
MAX_POWER = 4
MAX_HARMONICS = 2


class FoundConstant(BaseModel):
    name: str
    expression_latex: str
    verified: bool
    verification: Verification | None = None


class InvariantReport(BaseModel):
    status: Status
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    angles: list[str] = Field(default_factory=list)
    degree: int = 0
    power_bound: int = 0
    harmonics: int = 0
    unknowns: int = 0
    ansatz: str = ""
    known: list[str] = Field(default_factory=list)
    found: list[FoundConstant] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: Any) -> InvariantReport:
    return InvariantReport(status=status, message=message, **extra)


def _fresh(name: str, taken: set[str]) -> sp.Symbol:
    while name in taken:
        name += "_"
    taken.add(name)
    return sp.Symbol(name)


def search_invariants(
    latex: str,
    coordinates: list[str],
    angles: list[str],
    degree: int,
    power_bound: int,
    harmonics: int,
    time: str,
    assumptions: dict[str, list[str]],
    limits: Limits,
    format: Literal["latex", "text"] = "latex",
) -> InvariantReport:
    if not coordinates or len(coordinates) > MAX_COORDINATES:
        return _fail(
            Status.UNSUPPORTED,
            f"The search handles between 1 and {MAX_COORDINATES} coordinates.",
        )
    if not set(angles) <= set(coordinates):
        return _fail(Status.PARSE_ERROR, "Every angle must be one of the coordinates.")
    if not 1 <= degree <= MAX_DEGREE:
        return _fail(Status.UNSUPPORTED, f"The degree in the velocities is 1 to {MAX_DEGREE}.")
    if not 0 <= power_bound <= MAX_POWER:
        return _fail(Status.UNSUPPORTED, f"The power bound is 0 to {MAX_POWER}.")
    if not 1 <= harmonics <= MAX_HARMONICS:
        return _fail(Status.UNSUPPORTED, f"The number of harmonics is 1 to {MAX_HARMONICS}.")
    try:
        d = derive(latex, coordinates, time, assumptions, limits, format)
    except DerivationError as exc:
        return _fail(exc.status, exc.message)
    show = d.show
    base: dict[str, Any] = {
        "lagrangian_latex": show(d.lagrangian),
        "coordinates": coordinates,
        "angles": angles,
        "degree": degree,
        "power_bound": power_bound,
        "harmonics": harmonics,
    }
    if d.time_dependent:
        return _fail(
            Status.UNSUPPORTED,
            "The Lagrangian depends explicitly on time; the search covers autonomous systems only.",
            **base,
        )
    if not d.solution:
        return _fail(
            Status.UNSUPPORTED,
            "The accelerations cannot be solved for (singular mass matrix), so the condition "
            "dC/dt = 0 on the motion cannot be written.",
            **base,
        )
    checks = check_equations(d)
    if not all(c.passed for c in checks):
        return _fail(
            Status.NOT_VERIFIED,
            "The equations of motion could not be confirmed by numerical differentiation of L, so "
            "nothing was searched.",
            **base,
        )
    try:
        return _search(d, angles, degree, power_bound, harmonics, base)
    except (sp.PolynomialError, sp.polys.polyerrors.BasePolynomialError):
        return _fail(
            Status.UNSUPPORTED, "The equations of motion are outside the family searched.", **base
        )


def _search(
    d: Derivation,
    angles: list[str],
    degree: int,
    power_bound: int,
    harmonics: int,
    base: dict[str, Any],
) -> InvariantReport:
    q, qd, n = d.q, d.qd, d.n
    taken = {s.name for s in d.lagrangian.free_symbols} | {s.name for s in d.q + d.qd}
    angle_idx = [i for i, name in enumerate(d.coordinates) if name in angles]
    plain_idx = [i for i in range(n) if i not in angle_idx]
    cos_s = {i: _fresh(f"c_{d.coordinates[i]}", taken) for i in angle_idx}
    sin_s = {i: _fresh(f"s_{d.coordinates[i]}", taken) for i in angle_idx}

    trig = {}
    for i in angle_idx:
        trig[sp.cos(q[i])] = cos_s[i]
        trig[sp.sin(q[i])] = sin_s[i]
    acc = [sp.together(d.solution[a].subs(trig)) for a in d.qdd]
    variables = [q[i] for i in plain_idx] + list(qd) + [cos_s[i] for i in angle_idx]
    variables += [sin_s[i] for i in angle_idx]
    for expr in acc:
        if any(q[i] in expr.free_symbols for i in angle_idx) or expr.atoms(sp.Function):
            return _fail(
                Status.UNSUPPORTED,
                "A declared angle appears other than through cos and sin, or a function other than "
                "those appears in the equations of motion.",
                **base,
            )
        if not expr.is_rational_function(*variables):
            return _fail(
                Status.UNSUPPORTED,
                "The accelerations are not rational in the coordinates, the velocities and the "
                "cos and sin of the declared angles (square roots and the like are not searched).",
                **base,
            )

    # The ansatz: functions of q times monomials of degree <= D in the velocities.
    per_coordinate: list[list[sp.Expr]] = []
    for i in range(n):
        if i in angle_idx:
            c, s = cos_s[i], sin_s[i]
            funcs = [sp.Integer(1)] + [
                c**e * s**g
                for g in (0, 1)
                for e in range(0, harmonics + 1)
                if 1 <= e + g <= harmonics
            ]
        else:
            funcs = [q[i] ** a for a in range(-power_bound, power_bound + 1)]
        per_coordinate.append(funcs)
    velocity_monomials = [
        alpha
        for alpha in itertools.product(range(degree + 1), repeat=n)
        if 1 <= sum(alpha) <= degree
    ]
    velocity_monomials.append(tuple(0 for _ in range(n)))
    columns: list[sp.Expr] = []
    for combo in itertools.product(*per_coordinate):
        f = sp.Mul(*combo)
        for alpha in velocity_monomials:
            columns.append(f * sp.Mul(*[qd[i] ** alpha[i] for i in range(n)]))
    unknowns = len(columns)
    base["unknowns"] = unknowns
    if unknowns > MAX_UNKNOWNS:
        return _fail(
            Status.RESOURCE_LIMIT,
            f"The ansatz has {unknowns} unknown coefficients (the limit is {MAX_UNKNOWNS}); lower the "
            "degree, the power bound or the number of harmonics.",
            **base,
        )

    shift = sp.Mul(*[q[i] ** power_bound for i in plain_idx])

    def monomial_key(term: sp.Expr) -> tuple[int, ...]:
        poly = sp.Poly(sp.expand(term * shift), *variables)
        terms = poly.terms()
        if len(terms) != 1:
            raise sp.PolynomialError("a column is a single monomial")
        return tuple(terms[0][0])

    def order(col: sp.Expr) -> tuple[int, int, tuple[int, ...]]:
        key = monomial_key(col)
        k_plain = len(plain_idx)
        velocity_degree = sum(key[k_plain : k_plain + n])
        harmonic_degree = sum(key[k_plain + n :])
        return (-velocity_degree, harmonic_degree, key)

    columns.sort(key=order)
    column_index = {monomial_key(col): j for j, col in enumerate(columns)}

    def total_derivative(expr: sp.Expr) -> sp.Expr:
        out = sp.Integer(0)
        for i in plain_idx:
            out += sp.diff(expr, q[i]) * qd[i]
        for i in angle_idx:
            out += sp.diff(expr, cos_s[i]) * (-sin_s[i] * qd[i])
            out += sp.diff(expr, sin_s[i]) * (cos_s[i] * qd[i])
        for i in range(n):
            out += sp.diff(expr, qd[i]) * acc[i]
        return out

    denominator = sp.Integer(1)
    for expr in acc:
        denominator = sp.lcm(denominator, sp.fraction(sp.together(expr))[1])
    # d/dt lowers a power of a coordinate by one, hence one more factor of each.
    clear = shift * denominator * sp.Mul(*[q[i] for i in plain_idx])

    def reduce_angles(poly_expr: sp.Expr) -> sp.Expr:
        out = sp.expand(poly_expr)
        for i in angle_idx:
            s, c = sin_s[i], cos_s[i]
            out = sp.expand(sp.rem(out, s**2 + c**2 - 1, s))
        return out

    rows: dict[tuple[int, ...], dict[int, sp.Expr]] = {}
    for j, col in enumerate(columns):
        raw = sp.cancel(sp.together(total_derivative(col) * clear))
        if not raw.is_polynomial(*variables):
            return _fail(
                Status.UNSUPPORTED,
                "The equations of motion are outside the family searched.",
                **base,
            )
        poly = sp.Poly(reduce_angles(raw), *variables)
        for monom, coeff in poly.terms():
            rows.setdefault(tuple(monom), {})[j] = coeff

    field = sp.QQ.frac_field(*sorted(d.params, key=lambda s: s.name)) if d.params else sp.QQ
    row_keys = list(rows)
    sparse = {
        r: {j: field.from_sympy(sp.sympify(v)) for j, v in rows[key].items()}
        for r, key in enumerate(row_keys)
    }
    matrix = DomainMatrix(sparse, (len(row_keys), unknowns), field)
    null = matrix.to_dense().nullspace()

    # Known constants and their products that lie in the ansatz are factored out.
    search = find_conserved(d)
    known = [(item, expr) for item, expr in search.items if item.verified]
    base["known"] = [item.name for item, _ in known]
    basis_exprs = [sp.Integer(1)] + [expr for _, expr in known]
    basis_exprs += [a * b for a, b in itertools.combinations_with_replacement(basis_exprs[1:], 2)]
    known_vectors: list[dict[int, sp.Expr]] = []
    for expr in basis_exprs:
        converted = sp.expand(sp.sympify(expr).subs(trig))
        if any(q[i] in converted.free_symbols for i in angle_idx):
            continue
        shifted = sp.expand(converted * shift)
        if not shifted.is_polynomial(*variables):
            continue
        poly = sp.Poly(reduce_angles(shifted), *variables)
        vector: dict[int, sp.Expr] = {}
        for monom, coeff in poly.terms():
            column = column_index.get(tuple(monom))
            if column is None:
                vector = {}
                break
            vector[column] = coeff
        if vector:
            known_vectors.append(vector)

    def to_domain_row(vector: dict[int, sp.Expr]) -> dict[int, Any]:
        return {j: field.from_sympy(sp.sympify(v)) for j, v in vector.items()}

    new_rows = _new_directions(null, known_vectors, to_domain_row, field, unknowns)

    found: list[FoundConstant] = []
    dropped = 0
    back = {cos_s[i]: sp.cos(q[i]) for i in angle_idx} | {sin_s[i]: sp.sin(q[i]) for i in angle_idx}
    for vector in new_rows:
        expr = _primitive(vector, columns, field)
        shown = sp.expand(expr.subs(back))
        ok, ver = verify_constant(d, shown)
        if not ok or ver is None or not ver.symbolic_zero:
            dropped += 1
            continue
        found.append(
            FoundConstant(
                name=f"Constant of motion {len(found) + 1}",
                expression_latex=f"C_{{{len(found) + 1}}} = {d.show(shown)}",
                verified=True,
                verification=ver,
            )
        )

    ansatz = _describe(d, angles, degree, power_bound, harmonics, plain_idx, angle_idx, unknowns)
    notes = [
        "Nothing outside this family was searched; finding none proves nothing beyond it.",
        "Each constant is determined up to a constant factor and up to adding the known constants "
        "(and their products) that were factored out.",
    ]
    if dropped:
        notes.append(
            f"{dropped} candidate{'s' if dropped != 1 else ''} did not pass verification and "
            f"{'are' if dropped != 1 else 'is'} not shown."
        )
    report = InvariantReport(
        status=Status.OK,
        ansatz=ansatz,
        found=found,
        notes=notes,
        features=["lagrangian", "lagrangian:invariant-search"]
        + (["lagrangian:hidden-constant"] if found else ["lagrangian:no-hidden-constant"]),
        message=(
            f"{len(found)} new verified constant{'s' if len(found) != 1 else ''} of motion in the "
            f"searched family ({unknowns} unknown coefficients)."
        ),
        **base,
    )
    return report


def _describe(
    d: Derivation,
    angles: list[str],
    degree: int,
    power_bound: int,
    harmonics: int,
    plain_idx: list[int],
    angle_idx: list[int],
    unknowns: int,
) -> str:
    parts = [f"polynomials of degree at most {degree} in the velocities"]
    if plain_idx:
        names = ", ".join(d.coordinates[i] for i in plain_idx)
        parts.append(f"powers of {names} from {-power_bound} to {power_bound}")
    if angle_idx:
        names = ", ".join(angles)
        parts.append(f"cos and sin of {names} up to harmonic {harmonics}")
    return (
        "Constants of the form sum f(q) times a velocity monomial: "
        + "; ".join(parts)
        + f". {unknowns} unknown coefficients, solved exactly."
    )


def _new_directions(
    null: DomainMatrix,
    known_vectors: list[dict[int, sp.Expr]],
    to_domain_row: Any,
    field: Any,
    unknowns: int,
) -> list[dict[int, sp.Expr]]:
    """Basis of the nullspace modulo the span of the known constants, as sparse sympy vectors."""
    if null.shape[0] == 0:
        return []
    if known_vectors:
        known_matrix = DomainMatrix(
            {r: to_domain_row(v) for r, v in enumerate(known_vectors)},
            (len(known_vectors), unknowns),
            field,
        ).to_dense()
        reduced, pivots = known_matrix.rref()
        eliminated = []
        for r in range(len(pivots)):
            eliminated.append(reduced.extract([r], list(range(unknowns))))
        residual = null
        for p, row in zip(pivots, eliminated, strict=True):
            factor = residual.extract(list(range(residual.shape[0])), [p])
            residual = residual - factor * row
    else:
        residual = null
    final, final_pivots = residual.rref()
    out: list[dict[int, sp.Expr]] = []
    dense = final.to_Matrix()
    for r in range(len(final_pivots)):
        out.append({j: dense[r, j] for j in range(unknowns) if dense[r, j] != 0})
    return out


def _primitive(vector: dict[int, sp.Expr], columns: list[sp.Expr], field: Any) -> sp.Expr:
    """sum of v_j * column_j with the coefficients scaled to coprime polynomials in the parameters."""
    coefficients = {j: sp.together(sp.sympify(v)) for j, v in vector.items()}
    lcm = sp.Integer(1)
    for v in coefficients.values():
        lcm = sp.lcm(lcm, sp.fraction(v)[1])
    scaled = {j: sp.cancel(v * lcm) for j, v in coefficients.items()}
    gcd = sp.gcd_list(list(scaled.values()))
    if gcd != 0:
        scaled = {j: sp.cancel(v / gcd) for j, v in scaled.items()}
    return sp.Add(*[v * columns[j] for j, v in scaled.items()])


__all__ = ["FoundConstant", "InvariantReport", "search_invariants"]
