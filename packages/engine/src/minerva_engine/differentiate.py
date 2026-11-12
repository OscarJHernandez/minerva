"""Symbolic differentiation with an independent numerical check of every step.

`sympy.diff` cannot vouch for itself, so each differentiation step is compared with a central
finite difference of the *function it came from*, evaluated at 40 digits at random points that
respect the declared assumptions:

    (f(x0 + h) - f(x0 - h)) / 2h   versus   candidate(x0)          h = 1e-12

For a variable that may be complex the difference is taken along the real *and* the imaginary
direction. A function that is holomorphic gives the same value in both (Cauchy-Riemann); `|z|`,
`conjugate(z)`, `re(z)` and `im(z)` do not, so they are reported as unsupported instead of
producing a formula that only looks like a derivative.

Order n is checked as n first-order steps, each against the previous one, so a tenth derivative is
never an ill-conditioned tenth-order difference. There is no symbolic-zero leg: no independent
symbolic test of a derivative exists, and the report does not pretend otherwise.
"""

from __future__ import annotations

import random
import time
from typing import Literal

import sympy as sp
from pydantic import BaseModel, Field

from .build import Builder
from .errors import LimitError, MinervaMathError
from .models import ExpressionInput, Limits, Status
from .verify import MIN_VALID_POINTS, REL_TOL, SAMPLE_ATTEMPTS, _sample, verify_equal

MAX_ORDER = 10
MAX_VARIABLES = 4
STEP = sp.Float("1e-12", 50)
DIGITS = 40
MAX_POINTS = 12
RESULT_NODE_FACTOR = 5  # a derivative may be this many times larger than the input limit
TIDY_MAX_OPS = 150  # sp.simplify cannot be interrupted, so only small results are tidied


class DiffVariable(BaseModel):
    name: str = Field(min_length=1, max_length=32)
    order: int = Field(default=1, ge=1, le=MAX_ORDER)


class DerivativeStep(BaseModel):
    variable: str
    order_so_far: int  # derivatives taken in total when this expression was reached
    latex: str


class DerivativeVerification(BaseModel):
    method: str = "central finite difference of the previous expression, 40 digits"
    directions: list[Literal["real", "imaginary"]] = Field(default_factory=list)
    steps_checked: int = 0
    points_checked: int = 0
    points_agreed: int = 0
    passed: bool = False
    note: str = ""


class DifferentiateResult(BaseModel):
    status: Status
    input_latex: str | None = None
    variables: list[DiffVariable] = Field(default_factory=list)
    total_order: int = 0
    notation_latex: str | None = None
    result_latex: str | None = None
    result_text: str | None = None
    steps: list[DerivativeStep] = Field(default_factory=list)
    verification: DerivativeVerification | None = None
    notes: list[str] = Field(default_factory=list)
    message: str = ""
    duration_ms: int = 0


def _fail(status: Status, message: str, **extra: object) -> DifferentiateResult:
    return DifferentiateResult(status=status, message=message, **extra)


# --- numerical check ---------------------------------------------------------------------------
def _wide(value: sp.Expr) -> sp.Expr:
    """The same number at 50 digits (samples are drawn at double precision)."""
    re_part, im_part = value.as_real_imag()
    out = sp.Float(re_part, 50)
    if im_part != 0:
        out += sp.I * sp.Float(im_part, 50)
    return out


def _value(expr: sp.Expr, values: dict[sp.Symbol, sp.Expr]) -> sp.Expr | None:
    try:
        number = expr.xreplace(values).evalf(DIGITS)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
        return None
    if not number.is_number or number.has(sp.zoo, sp.oo, sp.nan):
        return None
    return number


def _close(a: sp.Expr, b: sp.Expr) -> bool | None:
    """True/False if the two numbers (nearly) agree; None if they cannot be compared."""
    try:
        x, y = complex(a), complex(b)
    except (TypeError, ValueError, OverflowError):
        return None
    if x != x or y != y or abs(x) == float("inf") or abs(y) == float("inf"):
        return None
    return abs(x - y) <= REL_TOL * max(1.0, abs(x), abs(y))


def check_derivative(
    f: sp.Expr, var: sp.Symbol, candidate: sp.Expr, *, seed: int = 20260101
) -> DerivativeVerification:
    """Compare `candidate` with a numerical derivative of `f` in `var`.

    `passed` needs at least `MIN_VALID_POINTS` valid points, and every valid point must agree
    in every direction that applies to `var`.
    """
    directions: list[Literal["real", "imaginary"]] = ["real"]
    if var.is_real is not True:
        directions.append("imaginary")
    symbols = sorted(f.free_symbols | candidate.free_symbols | {var}, key=lambda s: s.name)
    rng = random.Random(seed)  # noqa: S311 - deterministic sampling, not cryptographic
    checked = agreed = 0
    for _ in range(SAMPLE_ATTEMPTS):
        values = {s: _wide(_sample(s, rng)) for s in symbols}
        x0 = values[var]
        if (var.is_nonnegative or var.is_nonpositive) and abs(complex(x0)) < 0.05:
            continue  # a step of either sign would leave the declared domain
        want = _value(candidate, values)
        if want is None:
            continue
        verdicts: list[bool] = []
        for direction in directions:
            unit = sp.Integer(1) if direction == "real" else sp.I
            plus = _value(f, {**values, var: x0 + unit * STEP})
            minus = _value(f, {**values, var: x0 - unit * STEP})
            if plus is None or minus is None:
                break
            verdict = _close((plus - minus) / (2 * unit * STEP), want)
            if verdict is None:
                break
            verdicts.append(verdict)
        else:
            checked += 1
            agreed += all(verdicts)
        if checked >= MAX_POINTS:
            break
    passed = checked >= MIN_VALID_POINTS and agreed == checked
    if checked < MIN_VALID_POINTS:
        note = "too few valid sample points to verify numerically"
    elif agreed != checked:
        note = "the numerical derivative disagrees with the symbolic one"
    else:
        note = "the numerical derivative agreed at every sampled point"
    return DerivativeVerification(
        directions=directions,
        steps_checked=1,
        points_checked=checked,
        points_agreed=agreed,
        passed=passed,
        note=note,
    )


# --- the operation -----------------------------------------------------------------------------
def _nodes(expr: sp.Basic) -> int:
    return sum(1 for _ in sp.preorder_traversal(expr))


def _notation(expr: sp.Expr, variables: list[tuple[sp.Symbol, int]]) -> str | None:
    try:
        args: list[object] = []
        for symbol, order in variables:
            args.extend([symbol, order])
        return str(sp.latex(sp.Derivative(expr, *args, evaluate=False)))
    except Exception:  # printing only; the result does not depend on it
        return None


def differentiate(
    data: ExpressionInput, variables: list[DiffVariable], limits: Limits
) -> DifferentiateResult:
    from .ops import parse_expression  # local import: ops imports this module

    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    if not variables or len(variables) > MAX_VARIABLES:
        return _fail(Status.UNSUPPORTED, f"Give between 1 and {MAX_VARIABLES} variables.")
    total = sum(v.order for v in variables)
    if total > MAX_ORDER:
        return _fail(
            Status.UNSUPPORTED,
            f"The total order is {total}; Minerva differentiates up to order {MAX_ORDER}.",
            variables=variables,
            total_order=total,
        )
    try:
        expr = parse_expression(data, limits)
        builder = Builder(data.assumptions, limits)
        symbols = [(builder.symbol(v.name), v.order) for v in variables]
    except LimitError as exc:
        return _fail(Status.RESOURCE_LIMIT, str(exc), variables=variables, total_order=total)
    except MinervaMathError as exc:
        return _fail(Status.PARSE_ERROR, str(exc), variables=variables, total_order=total)

    input_latex = sp.latex(expr)
    base = {"input_latex": input_latex, "variables": variables, "total_order": total}
    for symbol, _ in symbols:
        if symbol.is_integer or symbol.is_rational:
            return _fail(
                Status.UNSUPPORTED,
                f"{symbol.name} is declared integer or rational, so there is no derivative "
                "with respect to it. Remove that assumption or declare it real.",
                **base,
            )

    notes: list[str] = []
    steps: list[DerivativeStep] = []
    current = expr
    done = points = agreed = checked_steps = 0
    directions: list[Literal["real", "imaginary"]] = []
    for symbol, order in symbols:
        for _ in range(order):
            if time.monotonic() > deadline:
                return _fail(
                    Status.RESOURCE_LIMIT,
                    "The time budget was used up before the last step.",
                    **base,
                )
            if symbol not in current.free_symbols:
                nxt: sp.Expr = sp.Integer(0)
                notes.append(
                    f"The expression no longer depends on {symbol.name}, so that step is 0."
                )
            else:
                nxt = sp.diff(current, symbol)
                if nxt.has(sp.Derivative, sp.Subs):
                    return _fail(
                        Status.UNSUPPORTED,
                        f"This expression has no complex derivative with respect to {symbol.name} "
                        "(for example |z|, re z, im z or conjugates). If the variable is real, "
                        "declare it real and try again.",
                        **base,
                    )
                if _nodes(nxt) > limits.max_tree_nodes * RESULT_NODE_FACTOR:
                    return _fail(
                        Status.RESOURCE_LIMIT, "The derivative became too large to check.", **base
                    )
                check = check_derivative(current, symbol, nxt)
                if not check.passed:
                    return _fail(
                        Status.NOT_VERIFIED,
                        "A derivative was produced but could not be verified, so it is not "
                        f"shown. {check.note}.",
                        verification=check,
                        **base,
                    )
                checked_steps += 1
                points += check.points_checked
                agreed += check.points_agreed
                seen = set(directions) | set(check.directions)
                directions = [name for name in ("real", "imaginary") if name in seen]
            current = nxt
            done += 1
            steps.append(
                DerivativeStep(variable=symbol.name, order_so_far=done, latex=sp.latex(nxt))
            )

    if current.has(sp.DiracDelta, sp.Heaviside):
        notes.append(
            "The result contains a Dirac delta or step function: it holds away from the "
            "points where the original function is not differentiable."
        )
    final = current
    if (
        _nodes(current) > 1
        and sp.count_ops(current) <= TIDY_MAX_OPS
        and time.monotonic() < deadline
    ):
        tidy = sp.simplify(current)
        if tidy != current and verify_equal(current, tidy, symbolic=False).passed:
            final = tidy
    if final != current:
        steps[-1] = DerivativeStep(
            variable=steps[-1].variable, order_so_far=done, latex=sp.latex(final)
        )

    verification = DerivativeVerification(
        directions=directions or ["real"],
        steps_checked=checked_steps,
        points_checked=points,
        points_agreed=agreed,
        passed=True,
        note=(
            "every step agreed with a numerical derivative at all sampled points"
            if checked_steps
            else "the expression does not depend on the variable, so the derivative is exactly 0"
        ),
    )
    return DifferentiateResult(
        status=Status.OK,
        notation_latex=_notation(expr, symbols),
        result_latex=sp.latex(final),
        result_text=str(final),
        steps=steps,
        verification=verification,
        notes=notes,
        message="Differentiated with SymPy; every step was verified numerically.",
        **base,
    )
