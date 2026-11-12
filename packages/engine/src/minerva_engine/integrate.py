"""Antiderivatives, each checked by differentiating the answer numerically.

`sympy.integrate` proposes a candidate `F` for `integral of f dx`; it never vouches for itself. The
candidate is accepted only if an independent numerical derivative of `F` (the same 40-digit central
difference as `differentiate.check_derivative`, in the real and, for a variable that may be complex,
the imaginary direction) agrees with `f` at every sampled point. The symbolic test
`simplify(F' - f) == 0` is reported separately as extra evidence and never decides the outcome.

A `Piecewise` candidate (for example `x**n` with the case `n = -1`) is checked branch by branch: an
equality condition is substituted into both `f` and the branch, the generic branch is sampled as is. A
branch whose condition cannot be sampled (an inequality on a parameter or on the variable) makes the
whole result `not_verified`.

Definite integrals (`lower` and `upper` given) are checked differently, on purpose: the fundamental
theorem of calculus fails for an integrand with a singularity inside the interval (`1/x**2` on
[-1, 1]), so differentiating an antiderivative would "verify" a wrong value. Instead SymPy's value is
compared with independent adaptive quadrature (mpmath, 30 digits) at assumption-respecting parameter
values, and a quadrature whose own error estimate is large (a pole, divergence) never counts as
agreement. Convergence conditions that SymPy reports and the declared assumptions do not imply give
`conditions_not_satisfied`. A missing closed form is `unsupported`: it does not prove none exists.
"""

from __future__ import annotations

import random
import time

import mpmath
import sympy as sp
from pydantic import BaseModel, Field

from .build import Builder
from .differentiate import (
    RESULT_NODE_FACTOR,
    TIDY_MAX_OPS,
    DerivativeVerification,
    _nodes,
    check_derivative,
)
from .errors import LimitError, MinervaMathError, ParseError
from .latexprint import to_latex
from .models import ExpressionInput, Limits, Status
from .verify import MIN_VALID_POINTS, SAMPLE_ATTEMPTS, _sample

SPECIAL_FUNCTIONS: tuple[type[sp.Basic], ...] = (
    sp.erf,
    sp.erfc,
    sp.erfi,
    sp.Ei,
    sp.expint,
    sp.li,
    sp.Si,
    sp.Ci,
    sp.Shi,
    sp.Chi,
    sp.fresnels,
    sp.fresnelc,
    sp.polylog,
    sp.gamma,
    sp.uppergamma,
    sp.lowergamma,
    sp.besselj,
    sp.bessely,
    sp.besseli,
    sp.besselk,
    sp.hyper,
    sp.meijerg,
)


class IntegrateVerification(DerivativeVerification):
    method: str = "central finite difference of the antiderivative, 40 digits"
    symbolic_zero: bool | None = None  # simplify(F' - f) == 0, extra evidence only
    branches_checked: int = 1


class QuadratureVerification(BaseModel):
    method: str = "adaptive quadrature (mpmath, 30 digits) at sampled parameter values"
    points_checked: int = 0
    points_agreed: int = 0
    passed: bool = False
    note: str = ""


class IntegrateResult(BaseModel):
    status: Status
    input_latex: str | None = None
    variable: str | None = None
    definite: bool = False
    lower_latex: str | None = None
    upper_latex: str | None = None
    numeric: str | None = None  # the value to 15 digits, when it has no free parameters
    conditions_latex: list[str] = Field(default_factory=list)
    quadrature: QuadratureVerification | None = None
    notation_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    verification: IntegrateVerification | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: object) -> IntegrateResult:
    return IntegrateResult(status=status, message=message, **extra)


def _branches(candidate: sp.Expr) -> list[tuple[sp.Expr, object]]:
    if isinstance(candidate, sp.Piecewise):
        return [(e, c) for e, c in candidate.args]
    return [(candidate, sp.true)]


def _fixing(a: sp.Expr, b: sp.Expr, var: sp.Symbol) -> dict[sp.Symbol, sp.Expr] | None:
    """The substitution that makes `a == b`, if one parameter determines it uniquely."""
    free = (a - b).free_symbols - {var}
    if len(free) != 1 or var in (a - b).free_symbols:
        return None
    (symbol,) = free
    try:
        solutions = sp.solve(sp.Eq(a, b), symbol)
    except Exception:
        return None
    if len(solutions) != 1 or not solutions[0].is_number:
        return None
    return {symbol: solutions[0]}


def check_antiderivative(f: sp.Expr, var: sp.Symbol, candidate: sp.Expr) -> IntegrateVerification:
    """Numerically differentiate `candidate` and compare with `f`, branch by branch."""
    checked = agreed = 0
    directions: set[str] = set()
    branches = _branches(candidate)
    notes: list[str] = []
    passed = True
    excluded: list[sp.Basic] = []  # earlier "a != b" conditions: later branches hold where a == b
    for expr, cond in branches:
        target, source = f, expr
        substitution: dict[sp.Symbol, sp.Expr] | None = None
        if isinstance(cond, sp.Equality):
            substitution = _fixing(cond.lhs, cond.rhs, var)
        elif isinstance(cond, sp.Unequality):
            excluded.append(cond)  # sampled generically: a random sample avoids a == b
        elif cond is sp.true or cond is True:
            if len(excluded) == 1:
                substitution = _fixing(excluded[0].lhs, excluded[0].rhs, var)
            elif excluded:
                substitution = None
                passed = False
        else:
            passed = False
        needs_substitution = isinstance(cond, sp.Equality) or (
            (cond is sp.true or cond is True) and len(excluded) == 1
        )
        if needs_substitution:
            if substitution is None:
                passed = False
            else:
                target, source = f.subs(substitution), expr.subs(substitution)
        if not passed:
            notes.append(f"the case '{to_latex(cond)}' cannot be sampled")
            break
        verdict = check_derivative(source, var, target)
        checked += verdict.points_checked
        agreed += verdict.points_agreed
        directions |= set(verdict.directions)
        if not verdict.passed:
            passed = False
            notes.append(verdict.note)
            break
    if passed:
        note = "the numerical derivative of the antiderivative agreed with the integrand everywhere"
    else:
        note = "; ".join(notes) or "verification failed"
    return IntegrateVerification(
        directions=[d for d in ("real", "imaginary") if d in directions] or ["real"],
        steps_checked=len(branches),
        branches_checked=len(branches),
        points_checked=checked,
        points_agreed=agreed,
        passed=passed,
        note=note,
    )


_INFINITY = {
    "oo": sp.oo,
    "+oo": sp.oo,
    "-oo": -sp.oo,
    r"\infty": sp.oo,
    r"+\infty": sp.oo,
    r"-\infty": -sp.oo,
    "infinity": sp.oo,
    "-infinity": -sp.oo,
}
QUAD_DPS = 30
QUAD_REL_TOL = sp.Float("1e-8")
QUAD_ERROR_TOL = 1e-10
GRID_POINTS = 401
BLOW_UP = 1e12


def _bound(text: str, data: ExpressionInput, limits: Limits) -> sp.Expr:
    from .ops import parse_expression

    key = text.strip().replace(" ", "")
    if key in _INFINITY:
        return _INFINITY[key]
    value = parse_expression(
        ExpressionInput(input=text, format=data.format, assumptions={}), limits
    )
    if value.free_symbols or value.is_real is not True:
        raise ParseError(
            "A bound must be a real number or infinity (write oo or \\infty); in LaTeX use "
            "\\exponentialE and \\pi for the constants."
        )
    return value


def _mp(bound: sp.Expr) -> mpmath.mpf:
    if bound is sp.oo:
        return mpmath.inf
    if bound is -sp.oo:
        return -mpmath.inf
    return mpmath.mpf(str(sp.N(bound, QUAD_DPS + 5)))


def check_definite(
    f: sp.Expr,
    var: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    value: sp.Expr,
    *,
    deadline: float,
    seed: int = 20260101,
) -> QuadratureVerification:
    """Compare `value` with adaptive quadrature of `f`, at sampled parameter values."""
    params = sorted((f.free_symbols | value.free_symbols) - {var}, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    a, b = _mp(lower), _mp(upper)
    finite = lower.is_finite and upper.is_finite
    checked = agreed = 0
    attempts = 1 if not params else SAMPLE_ATTEMPTS
    with mpmath.workdps(QUAD_DPS):
        for _ in range(attempts):
            if time.monotonic() > deadline:
                break
            values = {p: _sample(p, rng) for p in params}
            try:
                want = complex(value.xreplace(values).evalf(QUAD_DPS))
                fn = sp.lambdify(var, f.xreplace(values), "mpmath")
                if finite and not _grid_is_clean(fn, a, b):
                    continue
                got, err = mpmath.quad(fn, [a, b], error=True)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
                continue
            if err > QUAD_ERROR_TOL * max(1.0, abs(got)) or got != got:
                continue  # the quadrature itself is not trustworthy here (pole, divergence)
            checked += 1
            tol = float(QUAD_REL_TOL) * max(1.0, abs(want), abs(complex(got)))
            agreed += abs(want - complex(got)) <= tol
            if checked >= 8:
                break
    passed = checked >= MIN_VALID_POINTS if params else checked >= 1
    passed = passed and agreed == checked
    if checked == 0 or (params and checked < MIN_VALID_POINTS):
        note = "too few parameter points where the quadrature was reliable"
    elif agreed != checked:
        note = "the quadrature disagrees with the symbolic value"
    else:
        note = "the quadrature agreed with the symbolic value at every sampled point"
    return QuadratureVerification(
        points_checked=checked, points_agreed=agreed, passed=passed, note=note
    )


def _grid_is_clean(fn: object, a: mpmath.mpf, b: mpmath.mpf) -> bool:
    """No non-finite or huge integrand value on interior grid points of the (finite) interval.

    End points are skipped: quadrature never evaluates them, and an integrable singularity there
    (`1/sqrt(x)` at 0) is legitimate; a divergent one is caught by the quadrature error estimate.
    """
    lo, hi = (a, b) if a <= b else (b, a)
    for k in range(1, GRID_POINTS - 1):
        t = lo + (hi - lo) * k / (GRID_POINTS - 1)
        try:
            y = complex(fn(t))  # type: ignore[operator]
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return False
        if y != y or abs(y) > BLOW_UP:
            return False
    return True


def _definite(
    f: sp.Expr,
    symbol: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    *,
    notation: str,
    base: dict[str, object],
    deadline: float,
) -> IntegrateResult:
    common = {
        **base,
        "definite": True,
        "lower_latex": to_latex(lower),
        "upper_latex": to_latex(upper),
        "notation_latex": notation,
    }
    try:
        raw = sp.integrate(f, (symbol, lower, upper), conds="separate")
    except Exception:  # SymPy raises many things for what it cannot do
        return _fail(Status.UNSUPPORTED, "Minerva could not integrate this expression.", **common)
    value, cond = raw if isinstance(raw, tuple) else (raw, sp.true)
    if cond is not sp.true and cond is not True:
        parts = list(cond.args) if isinstance(cond, sp.And) else [cond]
        return _fail(
            Status.CONDITIONS_NOT_SATISFIED,
            "The value exists only under conditions that your assumptions do not imply. Declare "
            "them (for example that a parameter is positive) and try again.",
            conditions_latex=[to_latex(c) for c in parts],
            **common,
        )
    if value.has(sp.Integral):
        return _fail(
            Status.UNSUPPORTED,
            "No closed form was found. This does not prove that none exists.",
            **common,
        )
    if value.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        return _fail(
            Status.UNSUPPORTED,
            "SymPy reports that the integral diverges or is undefined. Minerva does not verify "
            "divergence, so it shows no value.",
            **common,
        )
    check = check_definite(f, symbol, lower, upper, value, deadline=deadline)
    if not check.passed:
        return _fail(
            Status.NOT_VERIFIED,
            "A value was produced but could not be verified by quadrature, so it is not shown. "
            f"{check.note}.",
            quadrature=check,
            **common,
        )
    numeric = None
    if not value.free_symbols:
        try:
            numeric = str(sp.N(value, 15))
        except (TypeError, ValueError):
            numeric = None
    return IntegrateResult(
        status=Status.OK,
        result_latex=to_latex(value),
        result_text=str(value),
        numeric=numeric,
        quadrature=check,
        notes=["Checked against numerical quadrature, not against an antiderivative."],
        message="Integrated with SymPy; the value was verified by independent quadrature.",
        **common,
    )


def integrate(
    data: ExpressionInput,
    variable: str,
    limits: Limits,
    lower: str | None = None,
    upper: str | None = None,
) -> IntegrateResult:
    from .ops import parse_expression  # local import: ops imports this module

    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    if (lower is None) != (upper is None):
        return _fail(Status.PARSE_ERROR, "Give both bounds, or neither.", variable=variable)
    try:
        f = parse_expression(data, limits)
        symbol = Builder(data.assumptions, limits).symbol(variable)
        low = _bound(lower, data, limits) if lower is not None else None
        high = _bound(upper, data, limits) if upper is not None else None
    except LimitError as exc:
        return _fail(Status.RESOURCE_LIMIT, str(exc), variable=variable)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc), variable=variable)

    base: dict[str, object] = {"input_latex": to_latex(f), "variable": variable}
    if symbol.is_integer or symbol.is_rational:
        return _fail(
            Status.UNSUPPORTED,
            f"{variable} is declared integer or rational, so it is not a continuous variable. "
            "Remove that assumption or declare it real.",
            **base,
        )
    if low is not None and high is not None:
        return _definite(
            f,
            symbol,
            low,
            high,
            notation=str(to_latex(sp.Integral(f, (symbol, low, high)))),
            base=base,
            deadline=deadline,
        )
    notation = str(to_latex(sp.Integral(f, symbol)))

    try:
        candidate = sp.integrate(f, symbol)
    except Exception:  # SymPy raises many things for what it cannot do
        return _fail(Status.UNSUPPORTED, "Minerva could not integrate this expression.", **base)
    if candidate.has(sp.Integral):
        return _fail(
            Status.UNSUPPORTED,
            "No closed form was found. This does not prove that none exists.",
            notation_latex=notation,
            **base,
        )
    if _nodes(candidate) > limits.max_tree_nodes * RESULT_NODE_FACTOR:
        return _fail(Status.RESOURCE_LIMIT, "The antiderivative became too large to check.", **base)

    verification = check_antiderivative(f, symbol, candidate)
    if not verification.passed:
        return _fail(
            Status.NOT_VERIFIED,
            "An antiderivative was produced but could not be verified, so it is not shown. "
            f"{verification.note}.",
            verification=verification,
            notation_latex=notation,
            **base,
        )

    final = candidate
    if (
        not isinstance(candidate, sp.Piecewise)
        and _nodes(candidate) > 1
        and sp.count_ops(candidate) <= TIDY_MAX_OPS
        and time.monotonic() < deadline
    ):
        tidy = sp.simplify(candidate)
        if tidy != candidate and check_antiderivative(f, symbol, tidy).passed:
            final = tidy

    if sp.count_ops(final) <= TIDY_MAX_OPS and time.monotonic() < deadline:
        try:
            verification.symbolic_zero = bool(sp.simplify(sp.diff(final, symbol) - f) == 0)
        except Exception:
            verification.symbolic_zero = False

    notes = [
        "This is one antiderivative: add an arbitrary constant, which may differ on each "
        "connected piece of the domain."
    ]
    if final.has(*SPECIAL_FUNCTIONS):
        notes.append("The result uses special functions that are not elementary.")
    if symbol not in f.free_symbols:
        notes.append(f"The integrand does not depend on {variable}.")
    return IntegrateResult(
        status=Status.OK,
        notation_latex=notation,
        result_latex=to_latex(final),
        result_text=str(final),
        verification=verification,
        notes=notes,
        message="Integrated with SymPy; the antiderivative was verified by a numerical derivative.",
        **base,
    )
