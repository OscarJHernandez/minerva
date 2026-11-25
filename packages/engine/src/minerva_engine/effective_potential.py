"""Radial effective potential of geodesics as one verified result (operation and tool `effective_potential`,
issue #207).

For a metric in coordinates split into cyclic ones (for a static, spherically symmetric metric `t` and `phi`),
one radial coordinate and coordinates fixed on a plane (`theta = pi/2`), the geodesics are the motions of the
Lagrangian `L = g(v, v) / 2`. What is established, in order (any failure is a refusal, never a partial result):

1. the geodesic equations are confirmed by numerical differentiation of `L` (`lagrangian.check_equations`);
2. the restriction to the plane is **consistent**: the acceleration of every fixed coordinate, taken from the
   confirmed equations of the whole system and evaluated at the fixed value with a zero velocity, is a symbolic
   zero, and a numerical zero at random values of everything else. A restriction that is not shown consistent is
   refused (`equatorial_not_consistent`, `consistency_undecided`), never assumed;
3. the restricted Lagrangian is reduced by the cyclic coordinates (`routh.reduce`, with its own verification:
   each momentum is verified conserved and held fixed as `p_{c}`; the Euler-Lagrange residual of the Routhian R on
   motions of the restricted Lagrangian vanishes);
4. the normalisation the USER states, `g(v, v) = kappa`, is written with `g(v, v) = 2 L = 2 (R - sum p_c dR/dp_c)`
   (the velocities of the removed coordinates are `-dR/dp_c`, which the reduction verified), solved for
   `r_dot^2`, and split at the momentum of the first cyclic coordinate: `r_dot^2 + V_eff(r) = alpha p_1^2` with
   `alpha` free of the coordinates. A different shape is refused (`radial_not_quadratic`, `not_separable`).

The result is shown only if the numerical legs, which share no code with the route that built it, pass: at points
of the whole system with the fixed coordinates at their values, `r_dot` taken as a root of `g(v, v) = kappa` and
the momenta as central differences of the full Lagrangian, (a) `r_dot^2 + V_eff - alpha p_1^2` vanishes and
(b) the acceleration of the radial coordinate from the full geodesic equations equals `-V_eff'(r) / 2` at fixed
momenta. The symbolic leg (the radial equation differentiated along the Routhian's own motion) is reported apart;
"proved" is said only when it and the consistency check are symbolic zeros. Restrictions are about the
coordinates (zeros of the factors the formulas divide by), never statements about singularities. The momenta keep
their names `p_{t}`, `p_{\\phi}`: `E = -p_t` and `L = p_phi` are the user's own step. The normalisation has no
default: it is the user's number.
"""

from __future__ import annotations

import random
import time as time_module
from typing import Any, Literal

import mpmath
import sympy as sp
from pydantic import BaseModel, Field

from .errors import MinervaMathError
from .geometry.compute import geodesic_lagrangian
from .geometry.models import GeometryInput
from .geometry.setup import Geometry, GeometryError, build_geometry
from .lagrangian import DerivationError, check_equations, derive_expr
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status, Verification
from .notation.models import InputAware, InputCode, InputDiagnostic, InputReading
from .restrictions import determinant_conditions
from .routh import (
    HeldMomentum,
    RouthChecks,
    _read_back,
    momentum_symbol,
    reduce,
)
from .verify import MIN_VALID_POINTS, SAMPLE_ATTEMPTS, _real_sample, _sample

POINTS = 6  # sample points; every valid one must agree
VALUE_REL_TOL = 1e-15  # exact algebra at 40 digits: a 1e-6 relative error in V_eff is far above it
ACCELERATION_REL_TOL = 1e-12  # one numerical derivative of V_eff against the full equations
CONSISTENCY_TOL = 1e-12
DPS = 40
SYMBOLIC_MAX_OPS = 4000
MAX_COORDINATES = 4

Reason = Literal[
    "unassigned_coordinates",
    "normalisation_unreadable",
    "metric_not_supported",
    "equations_not_confirmed",
    "equatorial_not_consistent",
    "consistency_undecided",
    "reduction_not_confirmed",
    "radial_not_quadratic",
    "not_separable",
    "not_verified",
]

CONVENTION = (
    "Geodesics are the motions of L = g(v, v)/2 with an affine parameter. R = L - Σ p_c ċ is the Routhian of the "
    "restricted Lagrangian (the sign in which it acts as a Lagrangian). The normalisation g(v, v) = κ is the "
    "user's: κ = -1 is timelike and κ = 0 null in a signature (-, +, +, +). The momenta keep the names p_c shown; "
    "naming one E or L is the user's own step."
)


class EffectivePotentialInput(BaseModel):
    coordinates: list[str] = Field(min_length=2, max_length=MAX_COORDINATES)
    metric: list[list[str]] = Field(min_length=2, max_length=MAX_COORDINATES)
    format: Literal["latex", "text"] = "latex"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    cyclic: list[str] = Field(min_length=1, max_length=3)
    radial: str = Field(min_length=1, max_length=16)
    fixed: dict[str, str] = Field(default_factory=dict)
    normalisation: str = Field(min_length=1, max_length=24)


class Consistency(BaseModel):
    """Whether the fixed coordinates stay fixed: their accelerations at the fixed values, computed."""

    coordinate: str
    acceleration_latex: str  # the acceleration at the fixed value and zero velocity, as simplified
    symbolic: bool | None = None  # True: exactly zero
    numeric: Verification | None = None


class EffectivePotentialChecks(BaseModel):
    """The deciding numerical legs and the separate symbolic leg. Counts only, never content."""

    routhian: RouthChecks | None = (
        None  # the Euler-Lagrange residual of R on motions of the restricted L
    )
    normalisation: Verification | None = (
        None  # r_dot^2 + V_eff - alpha p_1^2 at points of the whole system
    )
    radial_acceleration: Verification | None = (
        None  # r_ddot of the full geodesic equations = -V_eff'/2
    )
    symbolic: bool | None = None  # the radial equation differentiated along the Routhian's motion
    symbolic_note: str = ""


class EffectivePotentialReport(InputAware):
    status: Status
    reason: Reason | None = None
    coordinates: list[str] = Field(default_factory=list)
    cyclic: list[str] = Field(default_factory=list)
    radial: str = ""
    fixed: list[str] = Field(default_factory=list)  # `\theta = \frac{\pi}{2}`
    time: str = "s"
    metric_latex: str | None = None
    lagrangian_latex: str | None = None  # the restricted geodesic Lagrangian
    normalisation_latex: str | None = None  # g(v, v) = kappa, as the user stated it
    held_fixed: list[HeldMomentum] = Field(default_factory=list)
    routhian_latex: str | None = None
    normalisation_equation_latex: str | None = None  # g(v, v) in the held momenta, = kappa
    veff_latex: str | None = None  # V_{eff}(r) = ...
    # The single result: r_dot^2 + V_eff(r) = alpha p_1^2, written out.
    result_latex: str | None = None
    restrictions_latex: list[str] = Field(default_factory=list)
    restrictions_complete: bool = True
    consistency: list[Consistency] = Field(default_factory=list)
    checks: EffectivePotentialChecks = Field(default_factory=EffectivePotentialChecks)
    verification: Verification | None = None
    convention: str = CONVENTION
    stages: dict[str, int] = Field(default_factory=dict)
    message: str = ""
    duration_ms: int = 0


class EffectivePotentialError(Exception):
    def __init__(
        self,
        status: Status,
        reason: Reason,
        message: str,
        diagnostic: InputDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message
        self.diagnostic = diagnostic


def _fail(error: EffectivePotentialError, **extra: Any) -> EffectivePotentialReport:
    return EffectivePotentialReport(
        status=error.status,
        reason=error.reason,
        message=error.message,
        diagnostic=error.diagnostic,
        **extra,
    )


def read_normalisation(text: str) -> sp.Rational:
    """The user's `kappa` of `g(v, v) = kappa`: an exact rational number (`-1`, `0`, `1`, `-1/2`)."""
    try:
        value = sp.Rational(text.strip().replace(" ", ""))
    except (ValueError, TypeError) as exc:
        raise EffectivePotentialError(
            Status.PARSE_ERROR,
            "normalisation_unreadable",
            "The normalisation g(v, v) = κ must be a rational number the user states (-1 for a timelike geodesic, "
            "0 for a null one). Ask the user; nothing is assumed.",
            InputDiagnostic(
                code=InputCode.NORMALISATION_UNREADABLE,
                field="normalisation",
                recoverable=True,
            ),
        ) from exc
    return value


def _parameter_symbol(geo: Geometry) -> sp.Symbol:
    taken = {s.name for s in [*geo.coords, *geo.params, *geo.velocities]}
    for name in ("s", "tau", "lambda_"):
        if name not in taken:
            return sp.Symbol(name, real=True)
    return sp.Symbol("affine_parameter", real=True)


def _mp(value: Any) -> Any:
    number = complex(value)
    return mpmath.mpf(number.real) if number.imag == 0 else mpmath.mpc(number.real, number.imag)


def _parse_value(text: str, data: EffectivePotentialInput, limits: Limits) -> sp.Expr:
    from .ops import parse_expression

    try:
        return parse_expression(
            ExpressionInput(input=text, format=data.format, assumptions=data.assumptions),
            limits,
            strict_calls=True,
        )
    except MinervaMathError as exc:
        raise EffectivePotentialError(
            Status.PARSE_ERROR, "metric_not_supported", f"The fixed value could not be read: {exc}"
        ) from exc


def _consistency(
    d: Any,
    fixed: dict[sp.Symbol, sp.Expr],
    free_indices: list[int],
    cyclic_names: list[str],
) -> list[Consistency]:
    """The accelerations of the fixed coordinates at the fixed values and zero velocity: exactly zero, and
    numerically zero at random values of everything else (the equations being the confirmed ones)."""
    out: list[Consistency] = []
    index = {s: i for i, s in enumerate(d.q)}
    at_fixed: dict[sp.Symbol, sp.Expr] = dict(fixed)
    for c in fixed:
        at_fixed[d.qd[index[c]]] = sp.Integer(0)
    params = sorted(
        set().union(*(e.free_symbols for e in d.solution.values())) - {*d.q, *d.qd, d.t},
        key=lambda s: s.name,
    )
    rng = random.Random(20261125)  # noqa: S311 - deterministic sampling, not cryptographic
    for c in fixed:
        acc = d.solution[d.qdd[index[c]]]
        restricted = acc.subs(at_fixed)
        symbolic: bool | None
        if sp.count_ops(restricted) > SYMBOLIC_MAX_OPS:
            symbolic = None
        else:
            simplified = sp.simplify(restricted)
            symbolic = True if simplified == 0 else None
            restricted = simplified
        checked = agreed = 0
        try:
            f = sp.lambdify([*d.q, *d.qd, d.t, *params], acc, "mpmath")
        except Exception:
            f = None
        with mpmath.workdps(DPS):
            for _ in range(SAMPLE_ATTEMPTS):
                if f is None or checked >= POINTS:
                    break
                a = [_mp(fixed[s]) if s in fixed else mpmath.mpf(_real_sample(s, rng)) for s in d.q]
                b = [
                    mpmath.mpf(0) if d.q[i] in fixed else mpmath.mpf(rng.uniform(-2.0, 2.0))
                    for i in range(d.n)
                ]
                t0 = mpmath.mpf(_real_sample(d.t, rng))
                pmp = [_mp(_sample(s, rng)) for s in params]
                try:
                    value = complex(f(*a, *b, t0, *pmp))
                except (ArithmeticError, ValueError, TypeError, NameError):
                    continue
                if value != value or abs(value) == float("inf"):
                    continue
                checked += 1
                if abs(value) <= CONSISTENCY_TOL:
                    agreed += 1
        verification = Verification(
            symbolic_zero=symbolic is True,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=checked >= MIN_VALID_POINTS and agreed == checked,
            note="the acceleration of the fixed coordinate at its value with zero velocity",
        )
        out.append(
            Consistency(
                coordinate=c.name,
                acceleration_latex=str(to_latex(restricted)) if symbolic else "",
                symbolic=symbolic,
                numeric=verification,
            )
        )
    del free_indices, cyclic_names
    return out


def split_radial(
    routhian: sp.Expr,
    momenta: list[sp.Symbol],
    rd: sp.Symbol,
    kappa: sp.Rational,
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """`(N, alpha, V, a)`: `N = g(v, v)` written in the momenta, `r_dot^2 + V = alpha p_1^2` with `alpha` free of
    the coordinates, and `a`, the coefficient of `r_dot^2` in `N` (the formulas divide by it).

    `N = 2 (R - sum p_c dR/dp_c)` (the velocity of a removed coordinate is `-dR/dp_c`, verified by the reduction).
    Refused when `N` is not `a r_dot^2 + b` or the equation does not split at the first momentum as
    `alpha p_1^2 + (terms without p_1)`."""
    norm = sp.expand(2 * (routhian - sum((p * sp.diff(routhian, p) for p in momenta), sp.S.Zero)))
    try:
        poly = sp.Poly(norm, rd)
    except sp.PolynomialError as exc:
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "radial_not_quadratic",
            "The normalisation is not a polynomial in the radial velocity, so it cannot be solved for it.",
        ) from exc
    if poly.degree() != 2 or poly.coeff_monomial(rd) != 0:
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "radial_not_quadratic",
            "The normalisation is not of the form a(r) r_dot^2 + b(r, p) (a term linear in r_dot, or another "
            "degree, appears), so the radial equation is not written here.",
        )
    a = sp.cancel(poly.coeff_monomial(rd**2))
    b = poly.coeff_monomial(1)
    if a == 0:
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "radial_not_quadratic",
            "The radial velocity does not appear in the normalisation.",
        )
    w = sp.cancel(sp.together((kappa - b) / a))  # r_dot^2 = w
    p1 = momenta[0]
    try:
        in_p = sp.Poly(sp.expand(sp.together(w)), p1)
    except sp.PolynomialError as exc:
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "not_separable",
            "r_dot^2 is not a polynomial in the first momentum.",
        ) from exc
    if in_p.degree() > 2 or in_p.coeff_monomial(p1) != 0:
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "not_separable",
            f"r_dot^2 is not alpha {p1.name}^2 plus terms without {p1.name}, so the equation "
            "r_dot^2 + V_eff = alpha p^2 cannot be written with it. Name the coordinate whose momentum plays the "
            "energy first.",
        )
    alpha = sp.cancel(in_p.coeff_monomial(p1**2))
    rest = sp.cancel(in_p.coeff_monomial(1))
    radial_name = rd.name.removesuffix("_dot")
    coordinate_symbols = {s for s in w.free_symbols if s.name in (radial_name, rd.name)} | {rd}
    coordinate_symbols |= {s for s in alpha.free_symbols if s.name.endswith("_dot")}
    if alpha == 0 or not alpha.free_symbols.isdisjoint(coordinate_symbols):
        raise EffectivePotentialError(
            Status.UNSUPPORTED,
            "not_separable",
            "The coefficient of the first momentum squared vanishes.",
        )
    return norm, alpha, sp.cancel(-rest), a


def _factor(expr: sp.Expr) -> sp.Expr:
    return sp.factor(sp.cancel(expr))


def _symbolic_leg(
    d0: Any,
    radial: sp.Symbol,
    radial_velocity: sp.Symbol,
    velocities: dict[sp.Symbol, sp.Expr],
    alpha: sp.Expr,
    veff: sp.Expr,
    p1: sp.Symbol,
) -> tuple[bool | None, str]:
    """`2 r_ddot + V_eff'` along the Routhian's own motion (the acceleration of the restricted Lagrangian with the
    removed velocities replaced), reduced modulo the radial equation `r_dot^2 = alpha p^2 - V`, must be zero.
    Reported apart; never decides; "not reduced to 0" is undecided."""
    if not d0.solution:
        return None, "not run: the accelerations of the restricted Lagrangian were not solved"
    acc = d0.solution[d0.qdd[d0.q.index(radial)]].xreplace(velocities)
    residual = 2 * acc + sp.diff(veff, radial)
    if sp.count_ops(residual) > SYMBOLIC_MAX_OPS:
        return None, "not run: the symbolic residual is too large"
    numerator, _ = sp.fraction(sp.together(residual))
    try:
        poly = sp.Poly(sp.expand(numerator), radial_velocity)
    except sp.PolynomialError:
        return None, "not run: the residual is not a polynomial in the radial velocity"
    w = alpha * p1**2 - veff
    reduced: dict[int, sp.Expr] = {0: sp.S.Zero, 1: sp.S.Zero}
    for (k,), coefficient in poly.terms():
        reduced[k % 2] += coefficient * w ** (k // 2)
    if all(sp.simplify(sp.cancel(sp.together(e))) == 0 for e in reduced.values()):
        return True, (
            "2 r̈ + V_eff' vanishes symbolically along the motion of the Routhian once r_dot² is replaced by the "
            "radial equation"
        )
    return None, "undecided: the symbolic residual was not reduced to 0"


def check_radial_equation(
    d: Any,
    radial: sp.Symbol,
    cyclic: list[int],
    fixed: dict[sp.Symbol, sp.Expr],
    momenta: list[sp.Symbol],
    alpha: sp.Expr,
    veff: sp.Expr,
    kappa: sp.Rational,
    *,
    seed: int = 20261125,
) -> tuple[Verification, Verification]:
    """The two deciding legs; they use the full Lagrangian and its confirmed accelerations only.

    At a point of the whole system (fixed coordinates at their values with zero velocity, random cyclic velocities,
    `r_dot` a root of `2 L = kappa` found numerically, the momenta central differences of `L`):
    `r_dot^2 + V_eff - alpha p_1^2` must vanish, and the radial acceleration of the full geodesic equations must
    equal `-V_eff'(r) / 2` at fixed momenta."""
    unreadable = "the Lagrangian or the result could not be evaluated numerically"
    failed = (Verification(note=unreadable), Verification(note=unreadable))
    r_index = d.q.index(radial)
    params = sorted(
        (d.lagrangian.free_symbols | veff.free_symbols | alpha.free_symbols)
        - {*d.q, *d.qd, d.t, *momenta},
        key=lambda s: s.name,
    )
    try:
        f_l = sp.lambdify([*d.q, *d.qd, d.t, *params], d.lagrangian, "mpmath")
        f_v = sp.lambdify([radial, *momenta, *params], veff, "mpmath")
        f_a = sp.lambdify([radial, *params], alpha, "mpmath")
        f_acc = sp.lambdify([*d.q, *d.qd, d.t, *params], d.solution[d.qdd[r_index]], "mpmath")
    except Exception:
        return failed
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    kap = mpmath.mpf(kappa.p) / mpmath.mpf(kappa.q)
    value_checked = value_agreed = acc_checked = acc_agreed = 0
    with mpmath.workdps(DPS):
        for _ in range(SAMPLE_ATTEMPTS * 4):
            if min(value_checked, acc_checked) >= POINTS:
                break
            a = [_mp(fixed[s]) if s in fixed else mpmath.mpf(_real_sample(s, rng)) for s in d.q]
            b = [mpmath.mpf(0) if s in fixed else mpmath.mpf(rng.uniform(-2.0, 2.0)) for s in d.q]
            t0 = mpmath.mpf(_real_sample(d.t, rng))
            pmp = [_mp(_sample(s, rng)) for s in params]

            def norm_at(
                x: Any, a: list[Any] = a, b: list[Any] = b, t0: Any = t0, pmp: list[Any] = pmp
            ) -> Any:
                bb = list(b)
                bb[r_index] = x
                return 2 * f_l(*a, *bb, t0, *pmp)

            try:
                c0, plus, minus = norm_at(0), norm_at(1), norm_at(-1)
                quad = (plus + minus) / 2 - c0
                lin = (plus - minus) / 2
                disc = lin**2 - 4 * quad * (c0 - kap)
                if abs(quad) < mpmath.mpf(10) ** -20 or disc < 0:
                    continue
                root = (-lin + rng.choice([-1, 1]) * mpmath.sqrt(disc)) / (2 * quad)
                b[r_index] = root
                p_values = [
                    mpmath.diff(
                        lambda x, i=i, a=a, b=b, t0=t0, pmp=pmp: f_l(
                            *a, *[x if j == i else b[j] for j in range(d.n)], t0, *pmp
                        ),
                        b[i],
                    )
                    for i in cyclic
                ]
                v = f_v(a[r_index], *p_values, *pmp)
                al = f_a(a[r_index], *pmp)
                lhs = root**2 + v - al * p_values[0] ** 2
                scale = max(abs(root**2), abs(v), abs(al * p_values[0] ** 2))
                acceleration = f_acc(*a, *b, t0, *pmp)
                slope = mpmath.diff(lambda x, p=p_values, pm=pmp: f_v(x, *p, *pm), a[r_index])
                gap = acceleration + slope / 2
                gap_scale = max(abs(acceleration), abs(slope) / 2)
            except (ArithmeticError, ValueError, TypeError, NameError, ZeroDivisionError):
                continue
            numbers = [complex(x) for x in (lhs, scale, gap, gap_scale)]
            if any(x != x or abs(x) == float("inf") for x in numbers):
                continue
            value_checked += 1
            if abs(numbers[0]) <= VALUE_REL_TOL * max(1.0, abs(numbers[1])):
                value_agreed += 1
            acc_checked += 1
            if abs(numbers[2]) <= ACCELERATION_REL_TOL * max(1.0, abs(numbers[3])):
                acc_agreed += 1

    def verdict(checked: int, agreed: int, note: str) -> Verification:
        passed = checked >= MIN_VALID_POINTS and agreed == checked
        return Verification(
            symbolic_zero=False,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=passed,
            note=note if passed else "too few valid sample points or a disagreement was found",
        )

    return (
        verdict(
            value_checked,
            value_agreed,
            "r_dot² + V_eff - alpha p² vanishes at points of the whole system on g(v, v) = κ (momenta by "
            "differences of the full Lagrangian)",
        ),
        verdict(
            acc_checked,
            acc_agreed,
            "the radial acceleration of the full geodesic equations equals -V_eff'/2 at fixed momenta",
        ),
    )


def effective_potential(data: EffectivePotentialInput, limits: Limits) -> EffectivePotentialReport:
    started = time_module.perf_counter()
    stages: dict[str, int] = {}
    base: dict[str, Any] = {
        "coordinates": list(data.coordinates),
        "cyclic": list(data.cyclic),
        "radial": data.radial,
    }
    try:
        report = _run(data, limits, stages, base)
    except EffectivePotentialError as exc:
        return _fail(exc, stages=stages, **base)
    except GeometryError as exc:
        return EffectivePotentialReport(
            status=exc.status,
            reason="metric_not_supported",
            message=exc.message,
            diagnostic=exc.diagnostic,
            stages=stages,
            **base,
        )
    except MinervaMathError as exc:
        return EffectivePotentialReport(
            status=Status.PARSE_ERROR,
            reason="metric_not_supported",
            message=str(exc),
            stages=stages,
            **base,
        )
    report.stages = stages
    report.duration_ms = round((time_module.perf_counter() - started) * 1000)
    return report


def _run(
    data: EffectivePotentialInput,
    limits: Limits,
    stages: dict[str, int],
    base: dict[str, Any],
) -> EffectivePotentialReport:
    names = list(data.coordinates)
    assigned = [*data.cyclic, data.radial, *data.fixed]
    if (
        len(set(assigned)) != len(assigned)
        or set(assigned) != set(names)
        or len(names) != len(set(names))
    ):
        raise EffectivePotentialError(
            Status.PARSE_ERROR,
            "unassigned_coordinates",
            "Every coordinate must be exactly one of: a cyclic coordinate to hold (`cyclic`), the radial "
            "coordinate (`radial`) or a coordinate fixed on a plane (`fixed`, with its value).",
        )
    kappa = read_normalisation(data.normalisation)

    started = time_module.perf_counter()
    geo_input = GeometryInput(
        coordinates=names,
        metric=data.metric,
        format=data.format,
        assumptions=data.assumptions,
        functions=[],
        quantities=["determinant"],
    )
    geo = build_geometry(geo_input, limits, MAX_COORDINATES)
    fixed_values = {
        geo.coords[names.index(c)]: _parse_value(text, data, limits)
        for c, text in data.fixed.items()
    }
    parameter = _parameter_symbol(geo)
    try:
        d = derive_expr(
            geodesic_lagrangian(geo),
            names,
            parameter.name,
            q=geo.coords,
            qd=geo.velocities,
            t=parameter,
        )
    except DerivationError as exc:
        raise EffectivePotentialError(exc.status, "equations_not_confirmed", exc.message) from exc
    base["time"] = parameter.name
    if not d.solution or not all(c.passed for c in check_equations(d)):
        raise EffectivePotentialError(
            Status.NOT_VERIFIED,
            "equations_not_confirmed",
            "The geodesic equations of this metric could not be confirmed by numerical differentiation of "
            "g(v, v)/2, so nothing is reduced.",
        )
    stages["geodesics"] = round((time_module.perf_counter() - started) * 1000)

    base["fixed"] = [f"{to_latex(c)} = {to_latex(v)}" for c, v in fixed_values.items()]
    metric_latex = str(to_latex(geo.g))
    # 2. the restriction is consistent, by computation
    started = time_module.perf_counter()
    consistency = _consistency(d, fixed_values, [], list(data.cyclic))
    stages["consistency"] = round((time_module.perf_counter() - started) * 1000)
    context = {"metric_latex": metric_latex, "consistency": consistency}
    for item in consistency:
        if (
            item.numeric is not None
            and not item.numeric.passed
            and (item.numeric.numeric_points_agreed < item.numeric.numeric_points_checked)
        ):
            raise EffectivePotentialError(
                Status.UNSUPPORTED,
                "equatorial_not_consistent",
                f"Fixing {item.coordinate} at the stated value is not consistent with the geodesic equations: its "
                "acceleration does not vanish there, so a geodesic does not stay on that plane. Nothing is "
                "restricted.",
                InputDiagnostic(
                    code=InputCode.RESTRICTION_NOT_CONSISTENT,
                    field="fixed",
                    symbols=[item.coordinate],
                    recoverable=False,
                ),
            )
        if not (item.symbolic is True and item.numeric is not None and item.numeric.passed):
            raise EffectivePotentialError(
                Status.NOT_VERIFIED,
                "consistency_undecided",
                f"Whether {item.coordinate} stays at the stated value could not be decided (its acceleration there "
                "was not reduced to exactly 0, or too few sample points were valid), so nothing is restricted.",
                InputDiagnostic(
                    code=InputCode.RESTRICTION_NOT_CONSISTENT,
                    field="fixed",
                    symbols=[item.coordinate],
                    recoverable=False,
                ),
            )

    # 3. restrict and reduce
    free = [i for i, c in enumerate(d.q) if c not in fixed_values]
    at_fixed: dict[sp.Symbol, sp.Expr] = dict(fixed_values)
    for c in fixed_values:
        at_fixed[d.qd[d.q.index(c)]] = sp.Integer(0)
    restricted = d.lagrangian.xreplace(at_fixed)
    names0 = [names[i] for i in free]
    started = time_module.perf_counter()
    try:
        d0 = derive_expr(
            restricted,
            names0,
            parameter.name,
            q=[d.q[i] for i in free],
            qd=[d.qd[i] for i in free],
            t=parameter,
        )
    except DerivationError as exc:
        raise EffectivePotentialError(exc.status, "reduction_not_confirmed", exc.message) from exc
    base_report = {
        **context,
        "lagrangian_latex": d0.show(d0.lagrangian),
        "normalisation_latex": f"g(v, v) = {to_latex(kappa)}",
    }
    reduction = reduce(d0, list(data.cyclic), assumptions=data.assumptions, limits=limits)
    stages["reduction"] = round((time_module.perf_counter() - started) * 1000)
    if reduction.status is not Status.OK or reduction.result_latex is None:
        raise EffectivePotentialError(
            reduction.status if reduction.status is not Status.OK else Status.NOT_VERIFIED,
            "reduction_not_confirmed",
            "The Routh reduction of the restricted Lagrangian by the cyclic coordinates did not pass its own "
            f"verification, so no effective potential is shown. {reduction.message}",
            reduction.diagnostic,
        )
    cyclic_index = [d0.coordinates.index(c) for c in data.cyclic]
    momenta = [momentum_symbol(d0.q[i]) for i in cyclic_index]
    routhian = _read_back(
        reduction.result_latex, d0, momenta, reduction.remaining, data.assumptions, limits
    )
    radial = d0.q[d0.coordinates.index(data.radial)]
    radial_velocity = d0.qd[d0.coordinates.index(data.radial)]

    # 4. the user's normalisation, solved for r_dot^2
    norm, alpha, veff, lead = _split(routhian, momenta, radial_velocity, kappa)
    veff = _factor(veff)
    p1 = momenta[0]
    show = d0.show
    veff_latex = f"V_{{\\mathrm{{eff}}}}\\left({show(radial)}\\right) = {show(veff)}"
    alpha_latex = "" if alpha == 1 else f"{show(alpha)} "
    result_latex = f"{show(radial_velocity)}^{{2}} + {show(veff)} = {alpha_latex}{show(p1)}^{{2}}"

    result_latex_plain = (
        f"r_dot^2 + V_eff({radial.name}) = {'' if alpha == 1 else 'alpha '}{p1.name}^2"
    )

    # 5. independent legs
    started = time_module.perf_counter()
    normalisation, acceleration = check_radial_equation(
        d,
        d.q[d.coordinates.index(data.radial)],
        cyclic_index_full(d, data.cyclic),
        fixed_values,
        momenta,
        alpha,
        veff,
        kappa,
    )
    stages["verify"] = round((time_module.perf_counter() - started) * 1000)
    velocities = _removed_velocities(d0, cyclic_index, momenta, routhian)
    started = time_module.perf_counter()
    symbolic, note = _symbolic_leg(d0, radial, radial_velocity, velocities, alpha, veff, p1)
    stages["symbolic"] = round((time_module.perf_counter() - started) * 1000)
    checks = EffectivePotentialChecks(
        routhian=reduction.checks,
        normalisation=normalisation,
        radial_acceleration=acceleration,
        symbolic=symbolic,
        symbolic_note=note,
    )
    if not (normalisation.passed and acceleration.passed):
        raise_not_verified(checks)
    restrictions, complete = _restrictions(
        reduction.restrictions_latex, reduction.restrictions_complete, lead
    )
    proved = symbolic is True and all(c.symbolic is True for c in consistency)
    checked = normalisation.numeric_points_checked + acceleration.numeric_points_checked
    agreed = normalisation.numeric_points_agreed + acceleration.numeric_points_agreed
    return EffectivePotentialReport(
        status=Status.OK,
        reading=InputReading(format=data.format, normalizations=[]),
        **{k: v for k, v in base.items() if k != "time"},
        time=parameter.name,
        metric_latex=metric_latex,
        lagrangian_latex=base_report["lagrangian_latex"],
        normalisation_latex=base_report["normalisation_latex"],
        held_fixed=reduction.held_fixed,
        routhian_latex=reduction.result_latex,
        normalisation_equation_latex=f"{show(norm)} = {to_latex(kappa)}",
        veff_latex=veff_latex,
        result_latex=result_latex,
        restrictions_latex=restrictions,
        restrictions_complete=complete,
        consistency=consistency,
        checks=checks,
        verification=Verification(
            symbolic_zero=proved,
            numeric_points_checked=checked,
            numeric_points_agreed=agreed,
            passed=True,
            note="r_dot² + V_eff = alpha p² on g(v, v) = κ at points of the whole system, and the radial acceleration",
        ),
        message=(
            f"The radial equation of the geodesics with g(v, v) = {kappa}: {result_latex_plain}. "
            + (
                "Proved: the equatorial restriction and the radial equation are symbolic zeros, and checked "
                "numerically against the full geodesic equations."
                if proved
                else "Checked numerically against the full geodesic equations (not proved symbolically)."
            )
        ),
    )


def cyclic_index_full(d: Any, cyclic: list[str]) -> list[int]:
    return [d.coordinates.index(c) for c in cyclic]


def raise_not_verified(checks: EffectivePotentialChecks) -> None:
    raise EffectivePotentialError(
        Status.NOT_VERIFIED,
        "not_verified",
        "The radial equation did not pass the numerical check against the full geodesic equations, so it is not "
        "shown. (Parameters with no declared assumption are sampled as complex numbers, for which there is no "
        "real radial velocity: declare the real ones, for example positive.)",
    )


def _restrictions(
    from_reduction: list[str], complete: bool, lead: sp.Expr
) -> tuple[list[str], bool]:
    """The reduction's restrictions and the zeros of what the radial equation divides by (coordinates only)."""
    extra, extra_complete = determinant_conditions(
        sp.numer(sp.together(lead)) * sp.denom(sp.together(lead))
    )
    out = list(from_reduction)
    for text in extra:
        if text not in out:
            out.append(text)
    return out, complete and extra_complete


def _removed_velocities(
    d0: Any, cyclic_index: list[int], momenta: list[sp.Symbol], routhian: sp.Expr
) -> dict[sp.Symbol, sp.Expr]:
    """The velocities of the removed coordinates as the reduction defines them: `-dR/dp_c`."""
    return {d0.qd[i]: -sp.diff(routhian, p) for i, p in zip(cyclic_index, momenta, strict=True)}


def _split(
    routhian: sp.Expr, momenta: list[sp.Symbol], rd: sp.Symbol, kappa: sp.Rational
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """The seam the tests use to put a wrong candidate between the construction and the checks."""
    return split_radial(routhian, momenta, rd, kappa)
