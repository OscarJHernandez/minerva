"""The ``sums`` operation: linearity, constants, factoring, log of a product and derivatives of finite sums.

A rewrite is shown only if the original and the result agree on every explicit instance (several sizes,
independent data per element); the derivative also passes the independent numerical derivative on every
instance. Agreement is reported as "checked for n = ...", never as a proof for every n.
"""

from __future__ import annotations

import time

import sympy as sp

from ..distributions.models import Condition
from ..errors import LimitError, MinervaMathError, RangeNotProvable, UnsupportedInput
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import rewrite
from .instantiate import InstanceError, check_derivative_indexed, verify_equal_indexed
from .models import SumReason, SumsInput, SumsResult, SumStep

TITLES = {
    "expand": "Linearity of finite sums",
    "evaluate_constant": "Closed form of a sum without data",
    "factor": "Product of sums",
    "log_of_product": "Logarithm of a product",
    "differentiate": "Derivative of a finite sum",
    "check": "Check an answer",
}


def _fail(
    data: SumsInput, status: Status, reason: SumReason | None, message: str, started: float
) -> SumsResult:
    return SumsResult(
        status=status,
        action=data.action,
        reason=reason,
        message=message,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def _read(text: str, data: SumsInput, limits: Limits) -> sp.Expr:
    return parse_expression(
        ExpressionInput(
            input=text, format=data.expression.format, assumptions=data.expression.assumptions
        ),
        limits,
        allow_sums=True,
    )


def sums(data: SumsInput, limits: Limits) -> SumsResult:
    started = time.monotonic()
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
        return _fail(data, Status.CONDITIONS_NOT_SATISFIED, "range_not_provable", str(exc), started)
    except LimitError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)
    except UnsupportedInput as exc:
        return _fail(data, Status.UNSUPPORTED, "unsupported_form", str(exc), started)
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc), started)

    conditions: list[Condition] = []
    try:
        result = _apply(data, expr, candidate, conditions)
    except (InstanceError, LimitError) as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)
    except MinervaMathError as exc:
        return _fail(data, Status.PARSE_ERROR, "parse_error", str(exc), started)
    if isinstance(result, SumsResult):  # a refusal from the rewrite itself
        result.duration_ms = (time.monotonic() - started) * 1000
        return result
    if data.action != "check" and sp.srepr(result) == sp.srepr(expr):
        return SumsResult(
            status=Status.UNCHANGED,
            action=data.action,
            input_latex=to_latex(expr),
            message="nothing to rewrite in this form",
            duration_ms=(time.monotonic() - started) * 1000,
        )

    try:
        if data.action == "differentiate":
            variable = _variable(data, expr)
            derivative, sizes = check_derivative_indexed(expr, variable, result)
            from .models import InstanceVerification

            verification = InstanceVerification(
                sizes=sizes,
                instances_checked=len(sizes),
                instances_agreed=len(sizes) if derivative.passed else 0,
                passed=derivative.passed,
                note=derivative.note,
            )
            passed = derivative.passed
        else:
            derivative = None
            verification = verify_equal_indexed(expr, result)
            passed = verification.passed
    except InstanceError as exc:
        return _fail(data, Status.RESOURCE_LIMIT, "resource_limit", str(exc), started)

    elapsed = (time.monotonic() - started) * 1000
    if not passed:
        return SumsResult(
            status=Status.NOT_VERIFIED,
            action=data.action,
            reason="verification_failed",
            input_latex=to_latex(expr),
            verification=verification,
            derivative_verification=derivative,
            message="the result did not agree with the original on explicit instances; not shown",
            duration_ms=elapsed,
        )
    return SumsResult(
        status=Status.OK,
        action=data.action,
        input_latex=to_latex(expr),
        result_latex=to_latex(result),
        result_text=to_text(result),
        steps=[
            SumStep(
                title=TITLES[data.action],
                before_latex=to_latex(expr),
                after_latex=to_latex(result),
                after_text=to_text(result),
            )
        ],
        verification=verification,
        derivative_verification=derivative,
        conditions=conditions,
        message=verification.describe(),
        duration_ms=elapsed,
    )


def _variable(data: SumsInput, expr: sp.Expr) -> sp.Symbol:
    found = [s for s in expr.free_symbols if isinstance(s, sp.Symbol) and s.name == data.variable]
    if not found:
        raise MinervaMathError(f"the variable {data.variable!r} does not occur in the expression")
    return found[0]


def _apply(
    data: SumsInput, expr: sp.Expr, candidate: sp.Expr | None, conditions: list[Condition]
) -> sp.Expr | SumsResult:
    match data.action:
        case "check":
            assert candidate is not None  # noqa: S101 - read above
            return candidate
        case "expand":
            return rewrite.expand_sums(expr)
        case "evaluate_constant":
            return rewrite.evaluate_constant_sums(expr)
        case "factor":
            return rewrite.factor_sums(expr)
        case "log_of_product":
            result, unproved = rewrite.log_of_product(expr)
            if unproved:
                return SumsResult(
                    status=Status.CONDITIONS_NOT_SATISFIED,
                    action=data.action,
                    reason="conditions_unmet",
                    input_latex=to_latex(expr),
                    conditions=[
                        Condition(
                            latex=to_latex(f) + r" > 0",
                            established=False,
                            how="not implied by the declared assumptions",
                        )
                        for f in unproved
                    ],
                    message="log(a b) = log a + log b needs every factor positive",
                )
            for node in expr.atoms(sp.log):
                if isinstance(node.args[0], sp.Product):
                    conditions.append(
                        Condition(
                            latex=to_latex(node.args[0].function) + r" > 0",
                            established=True,
                            how="from the declared assumptions",
                        )
                    )
            return result
        case "differentiate":
            variable = _variable(data, expr)
            bounds = (
                set().union(
                    *[
                        n.limits[k][1].free_symbols | n.limits[k][2].free_symbols
                        for n in expr.atoms(sp.Sum, sp.Product)
                        for k in range(len(n.limits))
                    ]
                )
                if expr.has(sp.Sum, sp.Product)
                else set()
            )
            if variable in bounds:
                raise MinervaMathError("the bounds of a sum depend on the variable")
            return rewrite.differentiate_sums(expr, variable)
    raise MinervaMathError("unknown action")
