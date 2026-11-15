"""Helpers shared by the approximation methods: parsing, refusals, report assembly.

Nothing here decides correctness. A method proposes an approximation (usually with SymPy) and a
check that does not use the route that built it decides whether it is shown.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import sympy as sp

from ..build import Builder
from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..latexprint import to_latex
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from ..series import NON_ANALYTIC, SeriesVerification
from .models import (
    ApproximationInput,
    ApproximationReport,
    Check,
    Diagnostics,
    Evidence,
    Reason,
    Warning,
)

LOCAL_NOTE = (
    "A local approximation: it describes the function near the stated point, and the measured errors "
    "are measurements at those distances, not a bound."
)


class Refused(Exception):
    """A request the operation cannot answer. Carries the status, the fixed reason and a safe message."""

    def __init__(
        self,
        status: Status,
        reason: Reason,
        message: str,
        questions: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message
        self.questions = questions or []


@dataclass
class Context:
    """A parsed request: the expression, the variable and the limits in force."""

    data: ApproximationInput
    limits: Limits
    started: float
    expr: sp.Expr
    var: sp.Symbol

    def expired(self, share: float = 0.8) -> bool:
        return time.monotonic() - self.started > share * self.limits.timeout_seconds


def parse_text(text: str, data: ExpressionInput, limits: Limits) -> sp.Expr:
    """Read a small piece of text (a point, a parameter) with the request's assumptions."""
    return parse_expression(
        ExpressionInput(input=text, format="text", assumptions=data.assumptions), limits
    )


def build_context(data: ApproximationInput, limits: Limits, started: float) -> Context:
    try:
        expr = parse_expression(data, limits)
        var = Builder(data.assumptions, limits).symbol(data.variable)
    except LimitError as exc:
        raise Refused(Status.RESOURCE_LIMIT, "resource_limit", str(exc)) from exc
    except UnsupportedInput as exc:
        raise Refused(Status.UNSUPPORTED, "unsupported_input", str(exc)) from exc
    except MinervaMathError as exc:
        raise Refused(Status.PARSE_ERROR, "unsupported_input", str(exc)) from exc
    return Context(data, limits, started, expr, var)


def refuse_non_analytic(expr: sp.Expr, var: sp.Symbol) -> None:
    for atom in expr.atoms(*NON_ANALYTIC):
        if var in atom.free_symbols:
            raise Refused(
                Status.UNSUPPORTED,
                "not_analytic",
                "This expression contains a function with no complex derivative (|x|, re, im, "
                "conjugate, sign, a step); a local series in it is not defined.",
            )
    if expr.atoms(sp.Derivative, sp.Integral, sp.Sum, sp.Product, sp.Indexed):
        raise Refused(
            Status.UNSUPPORTED,
            "unsupported_input",
            "Unevaluated derivatives, sums and integrals are not supported here.",
        )


def require_order(data: ApproximationInput, low: int = 1, high: int = 12) -> int:
    if data.order is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The order of the expansion is yours to choose.",
            ["Up to which order should the expansion go?"],
        )
    if not low <= data.order <= high:
        raise Refused(
            Status.UNSUPPORTED,
            "order_out_of_range",
            f"Give an order between {low} and {high}.",
        )
    return data.order


def finite_real_point(ctx: Context) -> sp.Expr:
    text = ctx.data.point
    if text is None:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "The expansion point is yours to choose.",
            ["About which point should the expansion be made?"],
        )
    x0 = parse_text(text, ctx.data, ctx.limits)
    if x0.free_symbols or x0.is_finite is not True or x0.is_real is not True:
        raise Refused(
            Status.UNSUPPORTED,
            "point_invalid",
            "The point must be a finite real number here (use asymptotic for infinity).",
        )
    return x0


def distance_symbol(var: sp.Symbol, x0: sp.Expr) -> sp.Symbol:
    """A symbol whose name is the LaTeX of the distance from the point, so it prints as itself."""
    latex = to_latex(var) if x0 == 0 else rf"\left({to_latex(var - x0)}\right)"
    return sp.Symbol(latex)


def powers_latex(coeffs: dict[int, sp.Expr], symbol: sp.Symbol, *, descending: bool = False) -> str:
    """A sum of c_k * symbol**k printed in ascending (or descending) powers of the symbol."""
    parts = sorted(coeffs.items(), reverse=descending)
    if not parts:
        return "0"
    text = ""
    for k, c in parts:
        term = to_latex(c * symbol**k)
        if not text:
            text = term
        elif term.startswith("-"):
            text += f" - {term[1:].lstrip()}"
        else:
            text += f" + {term}"
    return text


def join_terms(terms: list[sp.Expr]) -> str:
    """The LaTeX of a sum written term by term in the order given (SymPy would reorder it)."""
    text = ""
    for term in terms:
        piece = to_latex(term)
        if not text:
            text = piece
        elif piece.startswith("-"):
            text += f" - {piece[1:].lstrip()}"
        else:
            text += f" + {piece}"
    return text or "0"


def remainder(symbol_latex: str, power: int) -> str:
    return rf"O\left({symbol_latex}^{{{power}}}\right)"


def checks_from_series(v: SeriesVerification) -> list[Check]:
    """The two checks of `series.check_series` as report rows."""
    return [
        Check(
            name="coefficients",
            passed=v.coefficients_passed,
            points=v.parameter_sets_checked,
            detail=v.coefficient_method,
        ),
        Check(
            name="remainder",
            passed=v.remainder_passed,
            points=v.parameter_sets_checked,
            detail=v.remainder_method,
        ),
    ]


def failure(
    method: str,
    refused: Refused,
    *,
    data: ApproximationInput | None = None,
    started: float | None = None,
) -> ApproximationReport:
    report = ApproximationReport(
        status=refused.status,
        reason=refused.reason,
        method=method,
        message=refused.message,
        questions=refused.questions,
        variable=data.variable if data else None,
    )
    if started is not None:
        report.duration_ms = (time.monotonic() - started) * 1000
    return report


def finish(
    report: ApproximationReport,
    *,
    evidence: Evidence,
    warnings: list[Warning] | None = None,
) -> ApproximationReport:
    report.evidence = evidence
    for w in warnings or []:
        if w not in report.warnings:
            report.warnings.append(w)
    return report


def diagnostics(report: ApproximationReport, started: float) -> Diagnostics:
    decided = [c for c in report.checks if c.passed is not None]
    return Diagnostics(
        method=report.method,
        stages=len(report.checks),
        checks=len(decided),
        checks_passed=sum(1 for c in decided if c.passed),
        warnings=len(report.warnings),
        candidates=len(report.candidates),
        duration_ms=(time.monotonic() - started) * 1000,
    )


def sympy_failed(exc: BaseException) -> Refused:
    return Refused(
        Status.UNSUPPORTED,
        "unsupported_input",
        "SymPy could not expand this expression "
        f"({type(exc).__name__}); there may be a branch point or an essential singularity.",
    )


def first_arg(value: Any) -> Any:
    return value[0] if isinstance(value, (list, tuple)) else value
