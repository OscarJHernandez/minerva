"""Leading-order behaviour and limits.

SymPy proposes (`as_leading_term`, `limit`); the verdict comes from evaluating the original
expression with mpmath along a geometric approach to the point, which uses neither.
"""

from __future__ import annotations

import itertools
from typing import Any

import mpmath
import sympy as sp

from ..latexprint import to_latex
from ..models import Status
from .checks import DPS, compile_mp, parameter_sets
from .common import Context, Refused, finish, parse_text, refuse_non_analytic
from .models import ApproximationReport, Check

APPROACH = (6, 8, 10, 12)  # |x - x0| (or 1/|x|) = 10^-k
RATIO_FINAL = mpmath.mpf("1e-3")
LIMIT_FINAL = mpmath.mpf("1e-5")


class Approach:
    """The variable's way to the point: x0 + sign*h with h = 10^-k, or sign/h at infinity."""

    def __init__(self, ctx: Context) -> None:
        text = ctx.data.point
        if text is None:
            raise Refused(
                Status.CONDITIONS_NOT_SATISFIED,
                "domain_unspecified",
                "The point (or oo / -oo) the variable approaches is yours to choose.",
                ["Where does the variable go?"],
            )
        self.point = parse_text(text, ctx.data, ctx.limits)
        self.infinite = self.point in (sp.oo, -sp.oo)
        if not self.infinite and (
            self.point.free_symbols
            or self.point.is_finite is not True
            or self.point.is_real is not True
        ):
            raise Refused(
                Status.UNSUPPORTED,
                "point_invalid",
                "The point must be a finite real number, oo or -oo.",
            )
        if self.infinite:
            self.sign = 1 if self.point == sp.oo else -1
        else:
            if ctx.data.direction is None:
                self.sign = 0
            else:
                self.sign = 1 if ctx.data.direction == "+" else -1

    def sides(self) -> list[int]:
        return [self.sign] if self.sign else [1, -1]

    def x(self, sign: int, k: int) -> Any:
        h = mpmath.mpf(10) ** -k
        if self.infinite:
            return sign / h
        return mpmath.mpf(str(sp.N(self.point, DPS))) + sign * h


def _substitute(ctx: Context, approach: Approach, sign: int) -> tuple[sp.Symbol, sp.Expr]:
    """The expression as a function of a positive h that goes to 0."""
    h = sp.Symbol("h_", positive=True)
    if approach.infinite:
        return h, ctx.expr.subs(ctx.var, sign / h)
    return h, ctx.expr.subs(ctx.var, approach.point + sign * h)


def _side_text(approach: Approach, sign: int) -> str:
    if approach.infinite:
        return r"\infty" if sign > 0 else r"-\infty"
    p = to_latex(approach.point)
    return rf"{p}^{{+}}" if sign > 0 else rf"{p}^{{-}}"


def _values(ctx: Context, approach: Approach, sign: int, expr: sp.Expr) -> list[list[Any]]:
    """The expression at each approach distance, for several parameter sets."""
    rows = []
    for values in parameter_sets(expr, ctx.var, 4, seed=20260902):
        fn = compile_mp(expr.subs(values), ctx.var)
        rows.append([fn(approach.x(sign, k)) for k in APPROACH])
    return rows


def leading(ctx: Context) -> ApproximationReport:
    approach = Approach(ctx)
    refuse_non_analytic(ctx.expr, ctx.var)
    parts: list[str] = []
    checks: list[Check] = []
    last_lead = sp.Integer(0)
    found: list[tuple[int, sp.Expr]] = []
    for sign in approach.sides():
        h, in_h = _substitute(ctx, approach, sign)
        try:
            lead_h = in_h.as_leading_term(h)
        except (NotImplementedError, ValueError, TypeError, RecursionError, AttributeError) as exc:
            raise Refused(
                Status.UNSUPPORTED,
                "unsupported_input",
                f"No leading term was found ({type(exc).__name__}).",
            ) from exc
        if lead_h in (0, sp.nan, sp.zoo) or lead_h.has(sp.Order, sp.nan, sp.zoo):
            raise Refused(Status.UNSUPPORTED, "unsupported_input", "No leading term was found.")
        # back to the variable: h = (x - x0) * sign, or 1/(sign * x)
        x_value = ctx.var
        h_of_x = (1 / (sign * x_value)) if approach.infinite else (x_value - approach.point) * sign
        lead_x = lead_h.subs(h, h_of_x)
        # numerical check of f / lead -> 1, at several parameter sets
        verdict = _ratio_check(ctx, approach, sign, ctx.expr, lead_x)
        side = _side_text(approach, sign)
        checks.append(
            Check(
                name=f"f / leading term -> 1 ({side})",
                passed=verdict[0],
                points=verdict[1],
                detail=verdict[2],
            )
        )
        if verdict[0] is not True:
            raise Refused(
                Status.NOT_VERIFIED if verdict[0] is False else Status.UNSUPPORTED,
                "verification_failed",
                "A leading term was proposed but the ratio to the function did not approach 1, so "
                "it is not shown.",
            )
        last_lead = lead_x
        found.append((sign, lead_x))
        parts.append(rf"f \sim {to_latex(lead_x)}\quad ({to_latex(ctx.var)} \to {side})")
    leads = {str(lead) for _, lead in found}
    if len(found) == 2 and len(leads) == 1:
        statement = rf"f \sim {to_latex(last_lead)}\quad ({to_latex(ctx.var)} \to {to_latex(approach.point)})"
    else:
        statement = r",\quad ".join(parts)
    report = ApproximationReport(
        status=Status.OK,
        method="leading",
        input_latex=to_latex(ctx.expr),
        variable=ctx.var.name,
        statement_latex=statement,
        result_latex=to_latex(last_lead) if len(leads) == 1 else None,
        checks=checks,
    )
    report.message = (
        "Leading behaviour: the ratio of the function to this term was evaluated at high precision "
        "along a geometric approach and tends to 1. This is a numerical check, not a proof."
    )
    if len(found) == 2 and len(leads) > 1:
        report.notes.append("The two sides differ; they are listed separately.")
    warnings: list[Any] = []
    if not approach.infinite and last_lead.is_number and last_lead != 0:
        value = ctx.expr.subs(ctx.var, approach.point)
        if value.has(sp.nan, sp.zoo):
            warnings.append("removable_singularity")
    return finish(report, evidence="numeric_estimate", warnings=warnings)


def _shrinks(errors: list[Any]) -> bool:
    """Each error is at most 1/1.5 of the one before it (or negligible): a steady approach, not a plateau."""
    floor = mpmath.mpf(10) ** -40
    return all(b <= a / 1.5 or b < floor for a, b in itertools.pairwise(errors))


def _ratio_check(
    ctx: Context, approach: Approach, sign: int, f: sp.Expr, lead: sp.Expr
) -> tuple[bool | None, int, str]:
    decided = 0
    with mpmath.workdps(DPS):
        for values in parameter_sets(f, ctx.var, 4, seed=20260903):
            fv = compile_mp(f.subs(values), ctx.var)
            lv = compile_mp(lead.subs(values), ctx.var)
            errors: list[Any] = []
            for k in APPROACH:
                x = approach.x(sign, k)
                a, b = fv(x), lv(x)
                if a is None or b is None or b == 0:
                    errors = []
                    break
                errors.append(abs(a / b - 1))
            if len(errors) < len(APPROACH):
                continue
            decided += 1
            if not _shrinks(errors) or errors[-1] > RATIO_FINAL:
                return False, decided, "the ratio did not approach 1"
    if decided == 0:
        return None, 0, "the function could not be evaluated along the approach"
    return (
        True,
        decided,
        f"|f/lead - 1| decreased along h = 1e-6 ... 1e-12 and ended below {float(RATIO_FINAL):g}",
    )


# --- limits -----------------------------------------------------------------------------------
def limit(ctx: Context) -> ApproximationReport:
    approach = Approach(ctx)
    outcomes: list[tuple[int, sp.Expr]] = []
    checks: list[Check] = []
    for sign in approach.sides():
        try:
            if approach.infinite:
                value = sp.limit(ctx.expr, ctx.var, approach.point)
            else:
                value = sp.limit(ctx.expr, ctx.var, approach.point, "+" if sign > 0 else "-")
        except (NotImplementedError, ValueError, TypeError, RecursionError, AttributeError) as exc:
            raise Refused(
                Status.UNSUPPORTED,
                "limit_undecided",
                f"SymPy could not decide this limit ({type(exc).__name__}).",
            ) from exc
        if value.has(sp.AccumBounds, sp.nan, sp.zoo, sp.Limit) or value.free_symbols - (
            ctx.expr.free_symbols - {ctx.var}
        ):
            raise Refused(
                Status.UNSUPPORTED,
                "limit_undecided",
                "The limit does not exist, oscillates, or could not be decided.",
            )
        verdict = _limit_check(ctx, approach, sign, value)
        side = _side_text(approach, sign)
        checks.append(
            Check(
                name=f"values approach the limit ({side})",
                passed=verdict[0],
                points=verdict[1],
                detail=verdict[2],
            )
        )
        if verdict[0] is not True:
            raise Refused(
                Status.NOT_VERIFIED if verdict[0] is False else Status.UNSUPPORTED,
                "verification_failed",
                "A limit was proposed but the values did not approach it, so it is not shown.",
            )
        outcomes.append((sign, value))
    parts = [
        rf"\lim_{{{to_latex(ctx.var)} \to {_side_text(approach, sign)}}} f = {to_latex(v)}"
        for sign, v in outcomes
    ]
    report = ApproximationReport(
        status=Status.OK,
        method="limit",
        input_latex=to_latex(ctx.expr),
        variable=ctx.var.name,
        statement_latex=r",\quad ".join(parts),
        checks=checks,
        message=(
            "The limit was found with SymPy and checked by evaluating the original expression at "
            "high precision along a geometric approach (a numerical check, not a proof)."
        ),
    )
    warnings: list[Any] = []
    if len(outcomes) == 2 and outcomes[0][1] != outcomes[1][1]:
        warnings.append("one_sided_limits_differ")
        report.notes.append("The one-sided limits differ, so there is no two-sided limit.")
    elif len(outcomes) == 2:
        report.result_latex = to_latex(outcomes[0][1])
        report.statement_latex = rf"\lim_{{{to_latex(ctx.var)} \to {to_latex(approach.point)}}} f = {to_latex(outcomes[0][1])}"
    else:
        report.result_latex = to_latex(outcomes[0][1])
    return finish(report, evidence="numeric_estimate", warnings=warnings)


def _limit_check(
    ctx: Context, approach: Approach, sign: int, value: sp.Expr
) -> tuple[bool | None, int, str]:
    decided = 0
    with mpmath.workdps(DPS):
        for values in parameter_sets(ctx.expr, ctx.var, 4, seed=20260904):
            fv = compile_mp(ctx.expr.subs(values), ctx.var)
            samples = [fv(approach.x(sign, k)) for k in APPROACH]
            if any(s is None for s in samples):
                continue
            decided += 1
            if value in (sp.oo, -sp.oo):
                want = 1 if value == sp.oo else -1
                real = [mpmath.re(s) * want for s in samples]
                growing = all(b >= a for a, b in itertools.pairwise(real))
                if not growing or real[-1] < 10 or real[-1] < 1.5 * real[0]:
                    return (
                        False,
                        decided,
                        "the values do not grow without bound in the stated direction",
                    )
                continue
            target = value.subs(values)
            lv = mpmath.mpmathify(sp.N(target, DPS))
            errors = [abs(s - lv) for s in samples]
            scale = max(mpmath.mpf(1), abs(lv))
            if not _shrinks([e / scale for e in errors]) or errors[-1] > LIMIT_FINAL * scale:
                return False, decided, "the values do not settle at the proposed limit"
    if decided == 0:
        return None, 0, "the function could not be evaluated along the approach"
    return True, decided, "the values approach the limit along h = 1e-6 ... 1e-12"
