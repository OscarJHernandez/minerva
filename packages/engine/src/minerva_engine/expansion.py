"""Equations of motion expanded about an equilibrium, and the linear normal modes.

Starting from the accelerations derived by `lagrangian.derive`, put q_i = q0_i + eps u_i and
qdot_i = eps v_i (u are the deviations from the equilibrium, v their rates) and expand each
acceleration in eps:

    uddot_i = F1_i(u, v) + F2_i(u, v) + ... + FN_i(u, v) + O(|(u, v)|^(N+1)),

where Fk is homogeneous of degree k. The terms come from repeated differentiation in eps (SymPy
proposes them); they are shown only if everything below holds, and a failure anywhere is `not_verified`
with no formula:

* the Euler-Lagrange equations were confirmed by numerical differentiation of L (as in
  `analyze_lagrangian`), and the solved accelerations satisfy them;
* q0 is an equilibrium: every acceleration vanishes there at zero velocity (symbolic and numeric);
* each expansion passes `series.check_series` (coefficients against a Cauchy-integral estimate, and the
  remainder falling as eps^(N+1)), at random values of u, v and the parameters.

With no velocity-linear terms (no damping, no gyroscopic terms) and at most two coordinates, the normal
modes are given: each omega^2 is a root of det(A + omega^2 I), checked by substitution, with the eigenvector
checked by (A + omega^2 I) w = 0. Its sign is reported only when the declared assumptions decide it; a
negative omega^2 is a statement about the *linearised* equations, not about the full motion. Larger systems
and systems with velocity-linear terms get the characteristic polynomial only.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Literal

import mpmath
import sympy as sp
from pydantic import BaseModel, Field

from .errors import LimitError, MinervaMathError, UnsupportedInput
from .lagrangian import Derivation, DerivationError, check_equations, derive
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status, Verification
from .notation.models import InputAware
from .series import MAX_ORDER, SeriesVerification, check_series
from .verify import (
    MIN_VALID_POINTS,
    NARROW_POINTS,
    SAMPLE_ATTEMPTS,
    WIDE_POINTS,
    _sample,
    numeric_check,
)

MAX_POLY_COORDINATES = 4
NODE_FACTOR = 5
# The eigenvector check evaluates (A + omega^2 I) w at sampled parameter values with this many digits and
# accepts a residual only if it is below RESIDUAL_TOLERANCE of the size of the terms that cancel. Exact
# arithmetic leaves ~1e-55 at 60 digits, a vector wrong by 1e-9 leaves ~1e-9.
RESIDUAL_DIGITS = 60
RESIDUAL_TOLERANCE = mpmath.mpf(10) ** -25
RESIDUAL_SEED = 20260101


class ExpansionTerm(BaseModel):
    degree: int
    latex: str


class ExpandedEquation(BaseModel):
    coordinate: str
    full_latex: str  # \ddot{\delta q} = F1 + F2 + ... + O(...)
    terms: list[ExpansionTerm] = Field(default_factory=list)
    verification: SeriesVerification | None = None


class NormalMode(BaseModel):
    omega_squared_latex: str
    sign: Literal["positive", "negative", "zero", "unknown"]
    omega_latex: str | None = None  # only when omega^2 is provably positive
    multiplicity: int = 1
    eigenvector_latex: str | None = None
    verified: bool = False
    verification: Verification | None = None


class NamedLatex(BaseModel):
    name: str
    latex: str
    name_latex: str = ""  # the name as LaTeX, e.g. \\theta for "theta"


class ExpansionReport(InputAware):
    status: Status
    lagrangian_latex: str | None = None
    coordinates: list[str] = Field(default_factory=list)
    equilibrium: list[NamedLatex] = Field(default_factory=list)
    order: int = 0
    equilibrium_verification: Verification | None = None
    # When the point is refused as not an equilibrium: each coordinate's acceleration there at zero velocity,
    # from the equations of motion that were checked (so the reader sees which force does not balance).
    accelerations_at_point: list[NamedLatex] = Field(default_factory=list)
    equations: list[ExpandedEquation] = Field(default_factory=list)
    stiffness_latex: str | None = None  # A in  uddot = A u + B v + ...
    velocity_latex: str | None = None  # B, only when it is not zero
    characteristic_latex: str | None = None
    modes: list[NormalMode] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0
    # Milliseconds per stage (prepare, expand, linear, modes): names and numbers only.
    stages: dict[str, int] = Field(default_factory=dict)


def _elapsed_ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _fail(status: Status, message: str, **extra: object) -> ExpansionReport:
    return ExpansionReport(status=status, message=message, **extra)


def _zero_check(expr: sp.Expr, note: str, failed_note: str | None = None) -> Verification:
    """Symbolic and numeric test that `expr` is zero (reported separately).

    `note` says what was established when the check passes; `failed_note`, when given, what was found when it
    does not (a note that reads "every acceleration vanishes" next to `passed: false` contradicts itself).
    """
    symbolic = sp.simplify(expr) == 0
    checked, agreed = numeric_check(expr, sp.Integer(0))
    passed = checked >= MIN_VALID_POINTS and agreed == checked
    return Verification(
        symbolic_zero=bool(symbolic),
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=passed,
        note=note if passed or failed_note is None else failed_note,
    )


def _row_residuals_vanish(matrix: mpmath.matrix, vec: mpmath.matrix, n: int) -> bool:
    """Every row of matrix * vec is below RESIDUAL_TOLERANCE of the size of the products in it."""
    if all(vec[j] == 0 for j in range(n)):
        raise ValueError("the zero vector satisfies every residual")
    held = True
    for i in range(n):
        residual = sum((matrix[i, j] * vec[j] for j in range(n)), mpmath.mpf(0))
        scale = sum((abs(matrix[i, j]) * abs(vec[j]) for j in range(n)), mpmath.mpf(0))
        if not (mpmath.isfinite(residual) and mpmath.isfinite(scale)):
            raise ValueError("not finite")
        held = held and abs(residual) <= RESIDUAL_TOLERANCE * scale
    return held


def _vector_residual_check(
    a_matrix: sp.Matrix, root: sp.Expr, vector: sp.Matrix, note: str
) -> Verification:
    """Test (A + root I) vector = 0 at sampled parameter values, with 60-digit mpmath arithmetic.

    The sum of |r_i|^2 of nested radicals that `_zero_check` used to evaluate is exactly zero, so SymPy's
    `evalf` kept raising its precision to resolve the cancellation (0.1-0.3 s per point, 5 s per mode for a
    double pendulum with no assumptions). Here the entries are lambdified, the points are the same kind of
    assumption-respecting samples as `numeric_check`, and each row's residual is compared with the size of
    the products that cancel in it. A point where something is undefined is skipped; the vector must be
    nonzero there (the zero vector satisfies everything). There is no symbolic leg.
    """
    n = a_matrix.shape[0]
    symbols = sorted(
        a_matrix.free_symbols | root.free_symbols | vector.free_symbols, key=lambda s: s.name
    )
    shifted = a_matrix + root * sp.eye(n)
    entries = sp.lambdify(symbols, shifted, "mpmath")
    components = sp.lambdify(symbols, vector, "mpmath")
    rng = random.Random(RESIDUAL_SEED)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    with mpmath.workdps(RESIDUAL_DIGITS):
        for wide, limit in ((False, NARROW_POINTS), (True, WIDE_POINTS)):
            got = 0
            for _ in range(SAMPLE_ATTEMPTS):
                point = [
                    mpmath.mpmathify(
                        complex(_sample(s, rng, wide=wide, spread=max(1, len(symbols))))
                    )
                    for s in symbols
                ]
                try:
                    held = _row_residuals_vanish(entries(*point), components(*point), n)
                except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                    continue  # undefined here (or the zero vector): not a point of the check
                checked += 1
                got += 1
                agreed += 1 if held else 0
                if got >= limit:
                    break
    return Verification(
        symbolic_zero=False,
        numeric_points_checked=checked,
        numeric_points_agreed=agreed,
        passed=checked >= MIN_VALID_POINTS and agreed == checked,
        note=note,
    )


def _fresh(name: str, taken: set[str]) -> sp.Symbol:
    while name in taken:
        name += "_"
    taken.add(name)
    return sp.Symbol(name)


@dataclass
class Prepared:
    """A confirmed derivation about a confirmed equilibrium, shared by the expansion and perturbation."""

    d: Derivation
    q0: list[sp.Expr]
    shown_coordinates: list[str]
    base: dict[str, object]  # lagrangian, coordinates, equilibrium and its verification


@dataclass
class Refusal:
    status: Status
    message: str
    extra: dict[str, object] = field(default_factory=dict)


def prepare(
    latex: str,
    coordinates: list[str],
    time_name: str,
    assumptions: dict[str, list[str]],
    equilibrium: dict[str, str],
    limits: Limits,
) -> Prepared | Refusal:
    """Derive the equations of motion, confirm them, and confirm the equilibrium point."""
    from .ops import parse_expression  # local import: ops imports this module

    unknown = set(equilibrium) - set(coordinates)
    if unknown:
        return Refusal(
            Status.PARSE_ERROR,
            f"Equilibrium given for {', '.join(sorted(unknown))}, which are not coordinates.",
        )
    try:
        d = derive(latex, coordinates, time_name, assumptions, limits)
    except DerivationError as exc:
        return Refusal(exc.status, str(exc), {"diagnostic": exc.diagnostic})

    shown_coordinates = [to_latex(s) for s in d.q]
    base: dict[str, object] = {
        "lagrangian_latex": d.show(d.lagrangian),
        "coordinates": coordinates,
    }
    if d.time_dependent:
        return Refusal(
            Status.UNSUPPORTED,
            "L depends explicitly on time, so there is no equilibrium point to expand about.",
            base,
        )
    if not d.solution:
        return Refusal(
            Status.UNSUPPORTED,
            "The mass matrix is singular, so the accelerations cannot be solved for; "
            "the system is constrained or degenerate.",
            base,
        )

    # 1. The equations of motion and the solved accelerations, confirmed independently of SymPy.
    checks = check_equations(d)
    if not all(c.passed for c in checks):
        return Refusal(
            Status.NOT_VERIFIED,
            "The derived equations of motion could not be confirmed by numerical differentiation "
            "of L, so nothing is computed. Declaring assumptions (real, positive) can help.",
            base,
        )
    for i in range(d.n):
        if not _zero_check(d.euler_raw[i].subs(d.solution), "").passed:
            return Refusal(
                Status.NOT_VERIFIED,
                "The solved accelerations do not satisfy the Euler-Lagrange equations.",
                base,
            )

    # 2. The equilibrium point.
    q0: list[sp.Expr] = []
    param_set = set(d.params)
    try:
        for name in coordinates:
            value = parse_expression(
                ExpressionInput(
                    input=equilibrium.get(name, "0"), format="text", assumptions=assumptions
                ),
                limits,
            )
            if not value.free_symbols <= param_set:
                return Refusal(
                    Status.UNSUPPORTED,
                    f"The equilibrium value of {name} may use only numbers and the parameters of L "
                    f"({', '.join(s.name for s in d.params) or 'none'}).",
                    base,
                )
            q0.append(value)
    except LimitError as exc:
        return Refusal(Status.RESOURCE_LIMIT, str(exc), base)
    except UnsupportedInput as exc:
        return Refusal(Status.UNSUPPORTED, str(exc), base)
    except MinervaMathError as exc:
        return Refusal(Status.PARSE_ERROR, str(exc), base)
    base["equilibrium"] = [
        NamedLatex(name=n, latex=to_latex(v), name_latex=shown_coordinates[i])
        for i, (n, v) in enumerate(zip(coordinates, q0, strict=True))
    ]

    at_rest = {**dict(zip(d.q, q0, strict=True)), **{v: sp.Integer(0) for v in d.qd}}
    residuals = [d.solution[a].subs(at_rest) for a in d.qdd]
    bad = [coordinates[i] for i, r in enumerate(residuals) if r.has(sp.zoo, sp.nan, sp.oo)]
    if bad:
        return Refusal(
            Status.UNSUPPORTED,
            f"The accelerations are singular at this point (for {', '.join(bad)}).",
            base,
        )
    equilibrium_check = _zero_check(
        sum((sp.Abs(r) ** 2 for r in residuals), sp.Integer(0)),
        "every acceleration vanishes at the point, at zero velocity",
        "not every acceleration vanishes at the point at zero velocity",
    )
    if not equilibrium_check.passed:
        accelerations = [
            NamedLatex(name=n, latex=to_latex(sp.simplify(r)), name_latex=shown_coordinates[i])
            for i, (n, r) in enumerate(zip(coordinates, residuals, strict=True))
        ]
        return Refusal(
            Status.UNSUPPORTED,
            "This is not an equilibrium of the system: the accelerations do not all vanish there "
            "at zero velocity. Expansion about a non-equilibrium point is not supported.",
            {
                **base,
                "equilibrium_verification": equilibrium_check,
                "accelerations_at_point": accelerations,
            },
        )
    base["equilibrium_verification"] = equilibrium_check
    return Prepared(d, q0, shown_coordinates, base)


def expand_motion(
    latex: str,
    coordinates: list[str],
    time_name: str,
    assumptions: dict[str, list[str]],
    equilibrium: dict[str, str],
    order: int,
    limits: Limits,
) -> ExpansionReport:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    if not 1 <= order <= MAX_ORDER:
        return _fail(Status.UNSUPPORTED, f"Give an order between 1 and {MAX_ORDER}.")
    stages: dict[str, int] = {}
    prepared = prepare(latex, coordinates, time_name, assumptions, equilibrium, limits)
    stages["prepare"] = _elapsed_ms(started)
    if isinstance(prepared, Refusal):
        return _fail(prepared.status, prepared.message, order=order, **prepared.extra)
    d, q0, shown_coordinates = prepared.d, prepared.q0, prepared.shown_coordinates
    base: dict[str, object] = {**prepared.base, "order": order}

    # 3. Expand each acceleration in eps.
    taken = {s.name for s in d.params} | {s.name for s in d.q} | {s.name for s in d.qd} | {"eps"}
    taken |= {s.name for s in d.qdd} | {d.t.name}
    eps = _fresh("eps", taken)
    u = [_fresh(f"d_{name}", taken) for name in coordinates]
    v = [_fresh(f"d_{name}_dot", taken) for name in coordinates]
    names: dict[sp.Symbol, str] = {}
    for i in range(d.n):
        names[u[i]] = rf"\delta {shown_coordinates[i]}"
        names[v[i]] = rf"\dot{{\delta {shown_coordinates[i]}}}"

    def show(expr: sp.Expr) -> str:
        return str(to_latex(expr, symbol_names=names))

    substitution = {
        **{d.q[i]: q0[i] + eps * u[i] for i in range(d.n)},
        **{d.qd[i]: eps * v[i] for i in range(d.n)},
    }
    equations: list[ExpandedEquation] = []
    first_order: list[sp.Expr] = []
    for i in range(d.n):
        if time.monotonic() > deadline:
            return _fail(Status.RESOURCE_LIMIT, "The time budget was used up.", **base)
        f = d.solution[d.qdd[i]].subs(substitution, simultaneous=True)
        coefficients: list[sp.Expr] = []
        derivative = f
        for k in range(order + 1):
            if k:
                derivative = sp.diff(derivative, eps)
            value = derivative.subs(eps, 0)
            if value.has(sp.zoo, sp.nan, sp.oo):
                return _fail(
                    Status.UNSUPPORTED,
                    f"The acceleration of {coordinates[i]} is not analytic at the equilibrium.",
                    **base,
                )
            coefficients.append(sp.expand(sp.simplify(value / sp.factorial(k))))
        if sum(sp.count_ops(c) for c in coefficients) > limits.max_tree_nodes:
            return _fail(Status.RESOURCE_LIMIT, "The expansion became too large to check.", **base)
        candidate = sum((c * eps**k for k, c in enumerate(coefficients)), sp.Integer(0))
        check = check_series(f, eps, sp.Integer(0), candidate, order)
        if not check.passed:
            return _fail(
                Status.NOT_VERIFIED,
                f"The expansion of the acceleration of {coordinates[i]} could not be verified, so "
                f"no expansion is shown. {check.note}.",
                **base,
            )
        terms = [
            ExpansionTerm(degree=k, latex=show(c))
            for k, c in enumerate(coefficients)
            if k >= 1 and c != 0
        ]
        left = rf"\ddot{{\delta {shown_coordinates[i]}}}"
        nonzero = [c for c in coefficients[1:] if c != 0]
        right = (
            str(to_latex(sp.Add(*nonzero, evaluate=False), symbol_names=names, order="none"))
            if nonzero
            else "0"
        )
        equations.append(
            ExpandedEquation(
                coordinate=coordinates[i],
                full_latex=rf"{left} = {right} + O\left(\delta^{{{order + 1}}}\right)",
                terms=terms,
                verification=check,
            )
        )
        first_order.append(coefficients[1])

    stages["expand"] = _elapsed_ms(started) - stages["prepare"]
    report = ExpansionReport(status=Status.OK, equations=equations, stages=stages, **base)
    stages = report.stages  # the report keeps its own copy: later stages are added to that one

    # 4. The linear system: uddot = A u + B v.
    a_matrix = sp.Matrix(d.n, d.n, lambda i, j: sp.diff(first_order[i], u[j]))
    b_matrix = sp.Matrix(d.n, d.n, lambda i, j: sp.diff(first_order[i], v[j]))
    rebuilt = a_matrix * sp.Matrix(u) + b_matrix * sp.Matrix(v) - sp.Matrix(first_order)
    if any(sp.expand(r) != 0 for r in rebuilt):
        report.notes.append("The linear part could not be separated into A u + B v.")
        report.message = _message(report)
        return report
    a_matrix = a_matrix.applyfunc(sp.simplify)
    b_matrix = b_matrix.applyfunc(sp.simplify)
    report.stiffness_latex = to_latex(a_matrix)
    velocity_free = all(e == 0 for e in b_matrix)
    if not velocity_free:
        report.velocity_latex = to_latex(b_matrix)

    lam = _fresh("lambda", taken)
    if d.n <= MAX_POLY_COORDINATES:
        poly = sp.factor((lam**2 * sp.eye(d.n) - lam * b_matrix - a_matrix).det(method="berkowitz"))
        report.characteristic_latex = rf"{to_latex(poly, symbol_names={lam: r'\lambda'})} = 0"
        report.notes.append(
            "Characteristic polynomial of the linearised system: each root λ gives solutions "
            "proportional to exp(λ t)."
        )
    stages["linear"] = _elapsed_ms(started) - stages["prepare"] - stages["expand"]
    if velocity_free and d.n <= 2:
        modes_started = time.monotonic()
        _modes(report, a_matrix, d.n, taken, deadline)
        stages["modes"] = _elapsed_ms(modes_started)
    elif not velocity_free:
        report.notes.append(
            "The linear part has velocity terms (damping or gyroscopic), so normal-mode "
            "frequencies are not given."
        )
    else:
        report.notes.append(
            f"Normal modes are given for up to 2 coordinates; this system has {d.n}."
        )
    report.message = _message(report)
    return report


def _message(report: ExpansionReport) -> str:
    n = len(report.equations)
    return (
        f"{n} acceleration{'s' if n != 1 else ''} expanded to order {report.order} about the "
        "equilibrium; the Euler-Lagrange equations, the equilibrium and each expansion were "
        "checked independently of SymPy's series."
    )


def _modes(
    report: ExpansionReport, a_matrix: sp.Matrix, n: int, taken: set[str], deadline: float
) -> None:
    w = _fresh("w", taken)
    poly = sp.expand((a_matrix + w * sp.eye(n)).det())
    roots = sp.roots(sp.Poly(poly, w))
    if sum(roots.values()) != n:
        report.notes.append("The normal-mode frequencies could not be found in closed form.")
        return
    for root, multiplicity in roots.items():
        if time.monotonic() > deadline:
            report.modes = []
            report.notes.append(
                "The normal modes were not computed: the time budget was used up. The expansion above "
                "is verified; the modes are not shown."
            )
            return
        substitution = sp.expand(poly.subs(w, root))
        ver = _zero_check(substitution, "det(A + omega^2 I) is zero at this omega^2")
        sign: Literal["positive", "negative", "zero", "unknown"] = "unknown"
        if root.is_positive:
            sign = "positive"
        elif root.is_negative:
            sign = "negative"
        elif root.is_zero:
            sign = "zero"
        eigenvector: str | None = None
        vectors = (a_matrix + root * sp.eye(n)).nullspace(simplify=True)
        if len(vectors) == 1:
            vec = vectors[0]
            if _vector_residual_check(a_matrix, root, vec, "").passed:
                eigenvector = to_latex(vec.applyfunc(sp.simplify))
        report.modes.append(
            NormalMode(
                omega_squared_latex=rf"\omega^2 = {to_latex(sp.simplify(root))}",
                sign=sign,
                omega_latex=(
                    rf"\omega = {to_latex(sp.sqrt(sp.simplify(root)))}"
                    if sign == "positive"
                    else None
                ),
                multiplicity=multiplicity,
                eigenvector_latex=eigenvector,
                verified=ver.passed,
                verification=ver,
            )
        )
    if any(not m.verified for m in report.modes):
        report.modes = []
        report.notes.append("A normal-mode frequency could not be verified, so none is shown.")
