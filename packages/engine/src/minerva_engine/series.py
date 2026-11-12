"""Taylor expansion about a finite point, with two checks that do not use SymPy's series.

`sympy.series` produces the candidate. It is shown only if both of these agree with it, computed
from the *original function* with mpmath at 80 digits, at random values of every other symbol that
respect the declared assumptions:

1. Coefficients. The Taylor coefficients are c_k = (1/2 pi i) oint f(z) / (z - x0)^(k+1) dz. A trapezoid
   rule on circles around the point (64 nodes) estimates each c_k with no differentiation at all.
   The estimate is trusted only at a radius where it is stable (the estimates at r and r/2 agree), and
   every candidate coefficient up to the requested order must match it.
2. Remainder order. |f(x0 + h u) - P(h u)| is measured at h, h/2 and h/4 along real and, for a
   variable that may be complex, imaginary directions; it must shrink at least like h^(order+1).

The two legs are reported together; if either fails the status is `not_verified` and no formula is
shown. Poles, branch points, logarithms, and functions with no complex derivative (|x|, re, im,
conjugate, sign, steps) are `unsupported`: Laurent, Puiseux and asymptotic series are not produced.
The measured errors at a few distances are measurements of the truncation error, not bounds.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any, Literal

import mpmath
import sympy as sp
from pydantic import BaseModel, Field

from .build import Builder
from .errors import LimitError, MinervaMathError
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status
from .verify import MIN_VALID_POINTS, _sample

MAX_ORDER = 12
DPS = 80
NODES = 64
RADII = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125)
TOL = mpmath.mpf("1e-12")
FLOOR = mpmath.mpf("1e-30")
TINY = mpmath.mpf("1e-40")
RATIO_SLACK = mpmath.mpf("0.7")
SAMPLE_ATTEMPTS = 20
RESULT_NODE_FACTOR = 5
NON_ANALYTIC = (
    sp.Abs,
    sp.re,
    sp.im,
    sp.conjugate,
    sp.arg,
    sp.sign,
    sp.Heaviside,
    sp.DiracDelta,
    sp.floor,
    sp.ceiling,
)

Leg = Literal["agree", "disagree", "unavailable"]


class SeriesTerm(BaseModel):
    power: int
    coefficient_latex: str


class SeriesVerification(BaseModel):
    coefficient_method: str = (
        "Cauchy integral of the original function on circles around the point "
        "(trapezoid rule, 64 nodes, 80 digits), trusted where the estimates at r and r/2 agree"
    )
    remainder_method: str = (
        "|f - P| at h, h/2, h/4 must fall at least as h^(order+1) along each direction"
    )
    directions: list[Literal["real", "imaginary"]] = Field(default_factory=list)
    parameter_sets_checked: int = 0
    coefficients_passed: bool = False
    remainder_passed: bool = False
    radius: float | None = None  # smallest circle radius that was needed
    passed: bool = False
    note: str = ""


class MeasuredError(BaseModel):
    distance: float
    absolute_error: str
    relative_error: str | None = None


class SeriesResult(BaseModel):
    status: Status
    input_latex: str | None = None
    variable: str | None = None
    point_latex: str | None = None
    order: int = 0
    result_latex: str | None = None
    result_text: str | None = None
    remainder_latex: str | None = None
    terms: list[SeriesTerm] = Field(default_factory=list)
    verification: SeriesVerification | None = None
    measured_errors: list[MeasuredError] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: Any) -> SeriesResult:
    return SeriesResult(status=status, message=message, **extra)


# --- numerical legs ----------------------------------------------------------------------------
Fn = Callable[..., Any]


def _eval(fn: Fn, *args: Any) -> Any:
    try:
        value = mpmath.mpmathify(fn(*args))
    except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, NameError):
        return None
    return value if mpmath.isfinite(value) else None


def _circle(fn: Fn, params: tuple[Any, ...], x0: Any, radius: Any, order: int) -> list[Any] | None:
    """Scaled coefficient estimates s_k = c_k r^k, k = 0..order, from the trapezoid rule."""
    values = []
    for j in range(NODES):
        node = x0 + radius * mpmath.expjpi(mpmath.mpf(2 * j) / NODES)
        value = _eval(fn, node, *params)
        if value is None:
            return None
        values.append(value)
    out = []
    for k in range(order + 1):
        total = mpmath.mpc(0)
        for j, value in enumerate(values):
            total += value * mpmath.expjpi(-mpmath.mpf(2 * ((j * k) % NODES)) / NODES)
        out.append(total / NODES)
    return out


def _coefficient_leg(
    fn: Fn, params: tuple[Any, ...], x0: Any, coeffs: list[Any], order: int
) -> tuple[Leg, float | None]:
    disagreed: float | None = None
    for radius in RADII:
        r = mpmath.mpf(radius)
        big = _circle(fn, params, x0, r, order)
        small = _circle(fn, params, x0, r / 2, order)
        if big is None or small is None:
            continue
        target = [coeffs[k] * r**k for k in range(order + 1)]
        scale = max(abs(v) for v in [*big, *target])
        # Per coefficient: a relative tolerance, plus a floor for coefficients that are zero.
        # (A tolerance relative to the largest term alone would let a wrong high-order
        # coefficient pass on a small circle, where its scaled size c_k r^k is tiny.)
        limits = [TOL * max(abs(big[k]), abs(target[k])) + FLOOR * scale for k in range(order + 1)]
        stable = all(abs(big[k] - small[k] * 2**k) <= limits[k] for k in range(order + 1))
        matches = all(abs(big[k] - target[k]) <= limits[k] for k in range(order + 1))
        if stable and matches:
            return "agree", radius
        if stable and disagreed is None:
            # A pole inside the circle also gives stable estimates, so try smaller circles first.
            disagreed = radius
    if disagreed is not None:
        return "disagree", disagreed
    return "unavailable", None


def _polynomial(coeffs: list[Any], h: Any) -> Any:
    total = mpmath.mpc(0)
    for c in reversed(coeffs):
        total = total * h + c
    return total


def _falls(a: Any, b: Any, order: int, scale: Any) -> bool:
    if a <= TINY * scale or b <= TINY * scale:
        return True
    return bool(a / b >= RATIO_SLACK * 2 ** (order + 1))


def _remainder_leg(
    fn: Fn,
    params: tuple[Any, ...],
    x0: Any,
    coeffs: list[Any],
    order: int,
    radius: float,
    var: sp.Symbol,
    directions: list[Any],
) -> bool:
    h0 = mpmath.mpf(min(0.02, radius / 8))
    for unit in directions:
        remainders = []
        scale = mpmath.mpf(1)
        for step in (h0, h0 / 2, h0 / 4):
            h = unit * step
            value = _eval(fn, x0 + h, *params)
            if value is None:
                return False
            scale = max(scale, abs(value))
            remainders.append(abs(value - _polynomial(coeffs, h)))
        if not (
            _falls(remainders[0], remainders[1], order, scale)
            and _falls(remainders[1], remainders[2], order, scale)
        ):
            return False
    return True


def _allowed_steps(var: sp.Symbol, x0: sp.Expr) -> list[tuple[Any, str]]:
    """Directions (as mpmath units) in which a small step stays in the variable's declared domain."""
    start = float(x0)
    out: list[tuple[Any, str]] = []
    for sign in (1, -1):
        moved = start + 0.001 * sign
        if var.is_positive and moved <= 0:
            continue
        if var.is_nonnegative and moved < 0:
            continue
        if var.is_negative and moved >= 0:
            continue
        if var.is_nonpositive and moved > 0:
            continue
        out.append((mpmath.mpf(sign), "real"))
    if var.is_real is not True:
        out.append((mpmath.mpc(0, 1), "imaginary"))
        out.append((mpmath.mpc(0, -1), "imaginary"))
    return out


def check_series(
    f: sp.Expr,
    var: sp.Symbol,
    x0: sp.Expr,
    candidate: sp.Expr,
    order: int,
    *,
    seed: int = 20260101,
) -> SeriesVerification:
    """Check that `candidate` (a polynomial in `var - x0`) is the order-`order` Taylor polynomial of `f`."""
    h = sp.Dummy("h")
    verification = SeriesVerification()
    try:
        poly = sp.Poly(sp.expand(candidate.subs(var, x0 + h)), h)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        verification.note = "the candidate is not a polynomial in the distance from the point"
        return verification
    if poly.degree() > order:
        verification.note = "the candidate has terms beyond the requested order"
        return verification
    raw = list(reversed(poly.all_coeffs()))
    raw += [sp.Integer(0)] * (order + 1 - len(raw))
    others = sorted(f.free_symbols - {var}, key=lambda s: s.name)
    needed = MIN_VALID_POINTS if others else 1
    attempts = SAMPLE_ATTEMPTS if others else 1
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    steps = _allowed_steps(var, x0)
    verification.directions = [
        name for name in ("real", "imaginary") if any(d == name for _, d in steps)
    ]
    units = [u for u, _ in steps]
    if not units:
        verification.note = "no direction around the point stays inside the declared domain"
        return verification

    with mpmath.workdps(DPS):
        try:
            f_fn = sp.lambdify([var, *others], f, modules="mpmath")
            c_fn = sp.lambdify(others, raw, modules="mpmath")
        except (TypeError, ValueError, NameError):
            verification.note = "the expression cannot be evaluated numerically"
            return verification
        x0_mp = mpmath.mpmathify(complex(x0))
        checked = 0
        smallest = 1.0
        for _ in range(attempts):
            params = tuple(mpmath.mpmathify(complex(_sample(s, rng))) for s in others)
            coeffs = _eval_list(c_fn, params)
            if coeffs is None:
                continue
            leg, radius = _coefficient_leg(f_fn, params, x0_mp, coeffs, order)
            if leg == "unavailable" or radius is None:
                continue
            if leg == "disagree":
                verification.parameter_sets_checked = checked + 1
                verification.radius = radius
                verification.note = (
                    "the Taylor coefficients computed from the function disagree with the candidate"
                )
                return verification
            verification.coefficients_passed = True
            if not _remainder_leg(f_fn, params, x0_mp, coeffs, order, radius, var, units):
                verification.parameter_sets_checked = checked + 1
                verification.radius = radius
                verification.note = "the remainder does not shrink at the order the series claims"
                verification.coefficients_passed = True
                return verification
            checked += 1
            smallest = min(smallest, radius)
            if checked >= needed:
                break
    verification.parameter_sets_checked = checked
    if checked < needed:
        verification.coefficients_passed = False
        verification.note = (
            "too few valid sample points to verify (singularities or unsupported functions)"
        )
        return verification
    verification.coefficients_passed = True
    verification.remainder_passed = True
    verification.radius = smallest
    verification.passed = True
    verification.note = "coefficients agree with a Cauchy-integral estimate and the remainder shrinks at the claimed order"
    return verification


def _eval_list(fn: Fn, params: tuple[Any, ...]) -> list[Any] | None:
    try:
        values = [mpmath.mpmathify(v) for v in fn(*params)]
    except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, NameError):
        return None
    return values if all(mpmath.isfinite(v) for v in values) else None


# --- the operation -----------------------------------------------------------------------------
def _measured(f: sp.Expr, var: sp.Symbol, x0: sp.Expr, candidate: sp.Expr) -> list[MeasuredError]:
    """Truncation error at a few distances, for an expression with no other symbols."""
    out: list[MeasuredError] = []
    steps = [u for u, d in _allowed_steps(var, x0) if d == "real"]
    if not steps:
        return out
    unit = steps[0]
    with mpmath.workdps(DPS):
        try:
            f_fn = sp.lambdify(var, f, modules="mpmath")
            p_fn = sp.lambdify(var, candidate, modules="mpmath")
        except (TypeError, ValueError, NameError):
            return out
        x0_mp = mpmath.mpmathify(complex(x0))
        for distance in (0.5, 0.1, 0.01):
            point = x0_mp + unit * mpmath.mpf(distance)
            exact = _eval(f_fn, point)
            approx = _eval(p_fn, point)
            if exact is None or approx is None:
                continue
            error = abs(exact - approx)
            relative = mpmath.nstr(error / abs(exact), 3) if abs(exact) > TINY else None
            out.append(
                MeasuredError(
                    distance=distance, absolute_error=mpmath.nstr(error, 3), relative_error=relative
                )
            )
    return out


def series(
    data: ExpressionInput,
    variable: str,
    point: str,
    order: int,
    limits: Limits,
) -> SeriesResult:
    from .ops import parse_expression  # local import: ops imports this module

    started = time.monotonic()
    if not 1 <= order <= MAX_ORDER:
        return _fail(Status.UNSUPPORTED, f"Give an order between 1 and {MAX_ORDER}.")
    try:
        expr = parse_expression(data, limits)
        builder = Builder(data.assumptions, limits)
        var = builder.symbol(variable)
        x0 = parse_expression(ExpressionInput(input=point, format="text"), limits)
    except LimitError as exc:
        return _fail(Status.RESOURCE_LIMIT, str(exc))
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc))

    base: dict[str, Any] = {
        "input_latex": to_latex(expr),
        "variable": var.name,
        "point_latex": to_latex(x0),
        "order": order,
    }
    if x0.free_symbols or x0.is_finite is not True or x0.is_real is not True:
        return _fail(
            Status.UNSUPPORTED,
            "The expansion point must be a finite real number (infinity and symbolic points "
            "are not supported).",
            **base,
        )
    if var.is_integer or var.is_rational:
        return _fail(
            Status.UNSUPPORTED,
            f"{var.name} is declared integer or rational, so there is no Taylor series in it.",
            **base,
        )
    for atom in expr.atoms(*NON_ANALYTIC):
        if var in atom.free_symbols:
            return _fail(
                Status.UNSUPPORTED,
                "This expression contains a function with no complex derivative "
                "(|x|, re, im, conjugate, sign, a step); it has no Taylor series.",
                **base,
            )
    if expr.atoms(sp.Derivative, sp.Integral, sp.Sum, sp.Product):
        return _fail(
            Status.UNSUPPORTED,
            "Unevaluated derivatives, sums and integrals are not supported.",
            **base,
        )

    if var not in expr.free_symbols:
        verification = SeriesVerification(
            passed=True,
            coefficients_passed=True,
            remainder_passed=True,
            note="the expression does not depend on the variable, so it is its own series",
        )
        return SeriesResult(
            status=Status.OK,
            result_latex=to_latex(expr),
            result_text=str(expr),
            terms=[SeriesTerm(power=0, coefficient_latex=to_latex(expr))],
            verification=verification,
            message="The expression does not depend on the variable.",
            **base,
        )

    try:
        candidate = sp.series(expr, var, x0, order + 1).removeO()
    except (NotImplementedError, ValueError, TypeError, RecursionError) as exc:
        return _fail(
            Status.UNSUPPORTED,
            "SymPy could not expand this at the point: there may be a pole, a branch point or "
            f"an essential singularity ({type(exc).__name__}).",
            **base,
        )
    if candidate.has(sp.Order) or candidate.has(sp.zoo, sp.oo, sp.nan):
        return _fail(
            Status.UNSUPPORTED, "The expansion at this point is not a Taylor series.", **base
        )
    nodes = sum(1 for _ in sp.preorder_traversal(candidate))
    if nodes > limits.max_tree_nodes * RESULT_NODE_FACTOR:
        return _fail(Status.RESOURCE_LIMIT, "The series became too large to check.", **base)

    h = sp.Dummy("h")
    try:
        poly = sp.Poly(sp.expand(candidate.subs(var, x0 + h)), h)
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        return _fail(
            Status.UNSUPPORTED,
            "The expansion has negative or fractional powers or logarithms at this point "
            "(a pole or branch point). Laurent, Puiseux and asymptotic series are not supported.",
            **base,
        )
    coefficients = list(reversed(poly.all_coeffs()))

    if time.monotonic() - started > 0.8 * limits.timeout_seconds:
        return _fail(Status.RESOURCE_LIMIT, "The time budget was used up before checking.", **base)
    check = check_series(expr, var, x0, candidate, order)
    if not check.passed:
        return _fail(
            Status.NOT_VERIFIED,
            f"A series was produced but could not be verified, so it is not shown. {check.note}.",
            verification=check,
            **base,
        )

    simple = [sp.simplify(c) for c in coefficients]
    terms = [
        SeriesTerm(power=k, coefficient_latex=to_latex(c)) for k, c in enumerate(simple) if c != 0
    ]
    # Shown in ascending powers of (x - point). The distance is a symbol whose name is its own
    # LaTeX, so SymPy does not expand (x - 1) back into x - 1 across terms.
    distance_latex = to_latex(var) if x0 == 0 else rf"\left({to_latex(var - x0)}\right)"
    distance_symbol = sp.Symbol(distance_latex)
    shown_latex = to_latex(
        sum((c * distance_symbol**k for k, c in enumerate(simple)), sp.Integer(0)), order="rev-lex"
    )
    shown = sum((c * (var - x0) ** k for k, c in enumerate(simple)), sp.Integer(0))
    if sp.expand(shown - candidate) != 0:  # never expected: same coefficients, same polynomial
        return _fail(
            Status.NOT_VERIFIED, "The displayed series differs from the verified one.", **base
        )
    remainder = rf"O\left({distance_latex}^{{{order + 1}}}\right)"
    notes: list[str] = []
    if candidate == expr:
        notes.append("The expression is a polynomial of this degree: the series is exact.")
    measured = [] if expr.free_symbols - {var} else _measured(expr, var, x0, candidate)
    return SeriesResult(
        status=Status.OK,
        result_latex=shown_latex,
        result_text=str(shown),
        remainder_latex=remainder,
        terms=terms,
        verification=check,
        measured_errors=measured,
        notes=notes,
        message="Expanded with SymPy; coefficients and remainder order verified numerically.",
        **base,
    )
