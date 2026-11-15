"""The ``expectation`` operation: E, Var and Cov by linearity, with independence only where declared.

A result is shown only if it agrees with the original on explicit finite joint distributions (dependent
unless independence was declared). A result that used declared independence is `conditional`: it holds
given that premise, which Minerva never checks. Moments are assumed to exist (finite support in the
check; stated as a condition).
"""

from __future__ import annotations

import time

import sympy as sp

from ..distributions.models import Condition
from ..errors import LimitError, MinervaMathError, RangeNotProvable, UnsupportedInput
from ..indexed.instantiate import InstanceError
from ..indexed.models import SumStep
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from .expectation import Moments
from .models import (
    ExpectationInput,
    ExpectationReason,
    ExpectationResult,
)
from .pmf_check import PmfError, verify_moments

TITLES = {
    "expand": "Linearity of expectation and bilinearity of covariance",
    "to_moments": "Variance and covariance through raw moments",
    "check": "Check an answer",
}


def _fail(
    data: ExpectationInput,
    status: Status,
    reason: ExpectationReason | None,
    message: str,
    started: float,
) -> ExpectationResult:
    return ExpectationResult(
        status=status,
        action=data.action,
        reason=reason,
        message=message,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def _read(text: str, data: ExpectationInput, limits: Limits) -> sp.Expr:
    return parse_expression(
        ExpressionInput(
            input=text, format=data.expression.format, assumptions=data.expression.assumptions
        ),
        limits,
        allow_sums=True,
        random_variables=data.random,
    )


def _declared(data: ExpectationInput) -> list[str]:
    out = []
    for group in data.independent:
        if len(group) > 1:
            out.append(", ".join(group[:-1]) + f" and {group[-1]} are independent")
        else:
            out.append(f"the components of {group[0]} are mutually independent")
    return out


def expectation(data: ExpectationInput, limits: Limits) -> ExpectationResult:
    started = time.monotonic()
    names = set(data.random)
    for group in data.independent:
        if not group or not set(group) <= names:
            return _fail(
                data,
                Status.PARSE_ERROR,
                "parse_error",
                "an independence group names a variable that is not declared random",
                started,
            )
    try:
        expr = _read(data.expression.input, data, limits)
        candidate: sp.Expr | None = None
        if data.action == "check":
            text = (data.candidate or {}).get("result")
            if text is None:
                return _fail(
                    data, Status.PARSE_ERROR, "missing_candidate", "no answer to check", started
                )
            candidate = _read(text, data, limits)
    except RangeNotProvable as exc:
        return _fail(data, Status.CONDITIONS_NOT_SATISFIED, "parse_error", str(exc), started)
    except LimitError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)
    except UnsupportedInput as exc:
        return _fail(data, Status.UNSUPPORTED, "unsupported_form", str(exc), started)
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc), started)

    moments = Moments(frozenset(data.random), data.independent)
    if data.action == "check":
        assert candidate is not None  # noqa: S101 - read above
        result = candidate
        premises = _declared(data)
    else:
        result = moments.to_moments(expr) if data.action == "to_moments" else moments.expand(expr)
        premises = moments.premises
        if sp.srepr(result) == sp.srepr(sp.expand(expr)) or result == expr:
            return ExpectationResult(
                status=Status.UNCHANGED,
                action=data.action,
                input_latex=to_latex(expr),
                message="nothing more can be derived without further assumptions "
                "(independence is never assumed)",
                duration_ms=(time.monotonic() - started) * 1000,
            )

    try:
        verification = verify_moments(expr, result, frozenset(data.random), data.independent)
    except PmfError as exc:
        return _fail(data, Status.UNSUPPORTED, "unsupported_form", str(exc), started)
    except InstanceError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)
    elapsed = (time.monotonic() - started) * 1000
    if not verification.passed:
        return ExpectationResult(
            status=Status.NOT_VERIFIED,
            action=data.action,
            reason="verification_failed",
            input_latex=to_latex(expr),
            verification=verification,
            message="the result did not agree with the original on explicit joint distributions; "
            "not shown",
            duration_ms=elapsed,
        )
    evidence = "conditional" if premises else "identity"
    return ExpectationResult(
        status=Status.OK,
        action=data.action,
        input_latex=to_latex(expr),
        result_latex=to_latex(result),
        result_text=to_text(result),
        evidence=evidence,
        given=premises,
        steps=[
            SumStep(
                title=TITLES[data.action],
                before_latex=to_latex(expr),
                after_latex=to_latex(result),
                after_text=to_text(result),
                evidence=evidence,
            )
        ],
        verification=verification,
        message=("holds given: " + "; ".join(premises) + ". " if premises else "")
        + verification.describe(),
        duration_ms=elapsed,
    )


__all__ = ["Condition", "expectation"]
