"""The ``distribution`` operation: simplify, integrate or differentiate with deltas and steps.

A result is shown only if both legs hold, and they are reported separately:

1. the exact conditions (variable real, scale nonzero, roots real and simple, positions decided),
   established symbolically in ``normal_form``;
2. an independent test-function pairing (``pairing``).

Every failure is a status plus one of the enumerated reasons; the events record the stages with
durations and never carry expression text.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import sympy as sp

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..latexprint import to_latex
from ..models import Limits, Status
from .guard import contains_distribution
from .models import (
    CITED_ENTRIES,
    Blocked,
    Diagnostics,
    DistributionEvent,
    DistributionInput,
    DistributionResult,
    DistributionStep,
    DistributionVerification,
    Reason,
    Stage,
)
from .normal_form import (
    Outcome,
    Refusal,
    differentiate_steps,
    integrate_deltas,
    order_bounds,
    simplify_deltas,
)
from .pairing import verify_pairing

DEADLINE_FRACTION = 0.8

_STATUS_FOR_REASON: dict[Reason, Status] = {
    "unsupported_distribution": Status.UNSUPPORTED,
    "singular_root": Status.UNSUPPORTED,
    "assumptions_insufficient": Status.CONDITIONS_NOT_SATISFIED,
    "integration_domain_ambiguous": Status.CONDITIONS_NOT_SATISFIED,
    "verification_failed": Status.NOT_VERIFIED,
    "resource_limit": Status.RESOURCE_LIMIT,
    "parse_error": Status.PARSE_ERROR,
}


class _Trace:
    """Collects the stage events with the time each stage took."""

    def __init__(self) -> None:
        self.events: list[DistributionEvent] = []
        self._last = time.monotonic()

    def mark(self, stage: Stage, *, failed: bool = False, reason: Reason | None = None) -> None:
        now = time.monotonic()
        self.events.append(
            DistributionEvent(
                stage=stage,
                status="failed" if failed else "ok",
                duration_ms=round((now - self._last) * 1000, 3),
                reason=reason,
            )
        )
        self._last = now


def _failure(
    data: DistributionInput,
    trace: _Trace,
    reason: Reason,
    message: str,
    *,
    entry_id: str | None = None,
    input_latex: str | None = None,
) -> DistributionResult:
    trace.mark("rejected", failed=True, reason=reason)
    blocked = (
        [Blocked(entry_id=entry_id, reason=reason, detail=message)]
        if entry_id in CITED_ENTRIES and entry_id is not None
        else []
    )
    return DistributionResult(
        status=_STATUS_FOR_REASON[reason],
        action=data.action,
        reason=reason,
        input_latex=input_latex,
        variable=data.variable,
        blocked=blocked,
        diagnostics=Diagnostics(reason=reason, events=trace.events),
        message=message,
    )


def _bound(text: str, data: DistributionInput, limits: Limits) -> sp.Expr:
    from ..integrate import _bound as parse_bound  # the integrate operation's own bound parser

    return parse_bound(text, data.expression, limits)


def distribution(data: DistributionInput, limits: Limits) -> DistributionResult:
    from ..build import Builder
    from ..ops import parse_expression  # local import: ops imports this package lazily

    started = time.monotonic()
    deadline = started + DEADLINE_FRACTION * limits.timeout_seconds
    trace = _Trace()
    input_latex: str | None = None
    try:
        expr = parse_expression(data.expression, limits, allow_distributions=True)
        x = Builder(data.expression.assumptions, limits).symbol(data.variable)
        lower = upper = None
        if data.action == "integrate":
            if data.lower is None or data.upper is None:
                return _failure(
                    data,
                    trace,
                    "unsupported_distribution",
                    "Integrating against a delta needs both limits (use -oo and oo for the whole line).",
                )
            lower = _bound(data.lower, data, limits)
            upper = _bound(data.upper, data, limits)
        input_latex = to_latex(expr)
        trace.mark("parsed")
        if not contains_distribution(expr):
            return _failure(
                data,
                trace,
                "unsupported_distribution",
                "The expression contains no Dirac delta or Heaviside function. Use the simplify, "
                "differentiate or integrate operation for ordinary expressions.",
                input_latex=input_latex,
            )
        if x.is_real is not True:
            return _failure(
                data,
                trace,
                "assumptions_insufficient",
                f"Declare {data.variable} real: delta and step functions are distributions on the "
                "real line.",
                input_latex=input_latex,
            )
        if not expr.has(x):
            return _failure(
                data,
                trace,
                "unsupported_distribution",
                f"The expression does not depend on {data.variable}.",
                input_latex=input_latex,
            )
        runner: Callable[[], Outcome]
        if data.action == "simplify":
            runner = lambda: simplify_deltas(expr, x)  # noqa: E731
        elif data.action == "integrate":
            assert lower is not None and upper is not None  # noqa: S101
            lo, hi = lower, upper
            runner = lambda: integrate_deltas(expr, x, lo, hi, data.endpoint)  # noqa: E731
        else:
            runner = lambda: differentiate_steps(expr, x)  # noqa: E731
        trace.mark("identities_identified")
        outcome = runner()
        trace.mark("transformation_selected")
        trace.mark("assumptions_evaluated")
    except Refusal as refusal:
        return _failure(
            data,
            trace,
            refusal.reason,
            refusal.message,
            entry_id=refusal.entry_id,
            input_latex=input_latex,
        )
    except LimitError as exc:
        return _failure(data, trace, "resource_limit", str(exc), input_latex=input_latex)
    except UnsupportedInput as exc:
        return _failure(data, trace, "unsupported_distribution", str(exc), input_latex=input_latex)
    except MinervaMathError as exc:
        return _failure(data, trace, "parse_error", str(exc), input_latex=input_latex)
    except RecursionError:
        return _failure(
            data,
            trace,
            "resource_limit",
            "The expression is nested too deeply.",
            input_latex=input_latex,
        )

    if contains_distribution(outcome.result) and data.action == "integrate":
        return _failure(
            data,
            trace,
            "unsupported_distribution",
            "The integral did not reduce to a value.",
            input_latex=input_latex,
        )

    trace.mark("verification_attempted")
    try:
        bounds = None
        target = outcome.result
        if lower is not None and upper is not None:
            low, high, sign = order_bounds(lower, upper)
            bounds, target = (
                (low, high),
                sign * outcome.result,
            )  # the check runs on the ordered interval
        verification = verify_pairing(
            expr,
            target,
            x,
            action=data.action,
            bounds=bounds,
            endpoint=data.endpoint,
            deadline=deadline,
        )
    except Refusal as refusal:
        return _failure(data, trace, refusal.reason, refusal.message, input_latex=input_latex)
    except RecursionError:
        return _failure(
            data,
            trace,
            "resource_limit",
            "The verification ran out of resources.",
            input_latex=input_latex,
        )
    verification.conditions_established = all(
        c.established for step in outcome.steps for c in step.conditions
    )
    if not (verification.passed and verification.conditions_established):
        failed = _failure(
            data,
            trace,
            "verification_failed",
            "A candidate was produced but the test-function check did not confirm it, so it is "
            "not shown. " + verification.note,
            input_latex=input_latex,
        )
        failed.verification = verification
        return failed

    trace.mark("accepted")
    trace.mark("completed")
    equality = "value" if data.action == "integrate" else "distributional"
    notes = list(outcome.notes)
    if data.action == "integrate":
        notes = [n for n in notes if not n.startswith("Equality is as distributions")]
    return DistributionResult(
        status=Status.OK,
        action=data.action,
        input_latex=input_latex,
        variable=data.variable,
        lower_latex=to_latex(lower) if lower is not None else None,
        upper_latex=to_latex(upper) if upper is not None else None,
        endpoint=data.endpoint,
        result_latex=to_latex(outcome.result),
        result_text=str(outcome.result),
        equality=equality,
        steps=_dedupe(outcome),
        verification=verification,
        diagnostics=Diagnostics(events=trace.events),
        notes=notes,
        message=_message(data, verification),
    )


def _dedupe(outcome: Outcome) -> list[DistributionStep]:
    for step in outcome.steps:
        seen: set[tuple[str, str]] = set()
        unique = []
        for condition in step.conditions:
            key = (condition.latex, condition.how)
            if key not in seen:
                seen.add(key)
                unique.append(condition)
        step.conditions = unique
    return outcome.steps


def _message(data: DistributionInput, verification: DistributionVerification) -> str:
    what = {
        "simplify": "Simplified",
        "integrate": "Integrated",
        "differentiate": "Differentiated",
    }[data.action]
    return (
        f"{what} with handbook identities under conditions that were established exactly; "
        f"{verification.pairings_agreed} of {verification.pairings_checked} test-function "
        "pairings agree (supporting evidence, not a proof)."
    )
