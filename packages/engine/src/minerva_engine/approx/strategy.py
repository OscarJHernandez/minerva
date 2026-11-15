"""Choosing a method, deterministically, from the shape of the request.

The rules read what the user has said (a point or infinity, a small parameter, an interval, a period,
a number) and the expression itself. They are heuristics: each one is recorded as applying or not,
with its reason, whether or not it was tried. At most `MAX_TRIED` methods are run; the first that
returns a verified result is the answer, and every method tried is listed. Nothing is chosen by a
language model, and a request that does not say where the approximation is wanted gets a question.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import Status
from .common import Context, Refused
from .models import ApproximationReport, Suggestion

MAX_TRIED = 3

Handler = Callable[[Context], ApproximationReport]


def _infinite(ctx: Context) -> bool:
    return (ctx.data.point or "").strip().lstrip("+-").lower() in {
        "oo",
        "inf",
        "infinity",
        "\\infty",
    }


def plan(ctx: Context) -> tuple[list[tuple[str, dict[str, Any], str]], list[Suggestion]]:
    """The methods to try, in order, with the changes to the request each needs; and every rule's verdict."""
    d = ctx.data
    tries: list[tuple[str, dict[str, Any], str]] = []
    notes: list[Suggestion] = []

    def rule(method: str, applies: bool, why: str, update: dict[str, Any] | None = None) -> None:
        notes.append(Suggestion(method=method, applies=applies, why=why))
        if applies:
            tries.append((method, update or {}, why))

    has_interval = d.lower is not None and d.upper is not None
    constant = not ctx.expr.free_symbols and ctx.expr.is_real is True
    multi = len(d.variables) >= 2
    rule(
        "continued_fraction",
        constant,
        "the expression is a real number: rational approximations by convergents"
        if constant
        else "the expression has symbols, so it is not a constant",
    )
    rule(
        "multivariable",
        multi,
        "several variables with expansion points were given" if multi else "a single variable",
    )
    rule(
        "asymptotic",
        _infinite(ctx) and not multi,
        "the variable goes to infinity: an expansion in inverse powers"
        if _infinite(ctx)
        else "no limit at infinity was asked for",
    )
    rule(
        "perturbation",
        bool(d.small) and not constant and not _infinite(ctx),
        "a small parameter was named: expand in it before simplifying"
        if d.small
        else "no small parameter was named",
    )
    rule(
        "fourier",
        d.period is not None and not multi,
        "a period was given: a trigonometric series" if d.period else "no period was given",
    )
    local = (
        d.point is not None
        and not _infinite(ctx)
        and d.order is not None
        and not has_interval
        and not multi
        and not constant
        and not d.small
    )
    rule(
        "taylor",
        local,
        "a regular point and an order were given: Taylor series"
        if local
        else "no single expansion point with an order",
    )
    rule(
        "laurent",
        local,
        "if the point is a pole, a Laurent series (tried when Taylor refuses)"
        if local
        else "no single expansion point with an order",
    )
    interval = has_interval and d.period is None and not multi and not constant and not d.small
    rule(
        "pade",
        interval and d.denominator_degree is not None and d.degree is not None,
        "numerator and denominator degrees were given"
        if d.denominator_degree is not None
        else "no [m/n] degrees were given",
    )
    rule(
        "compare",
        interval
        and d.denominator_degree is None
        and (d.degree is not None or d.tolerance is not None),
        "a finite interval with a degree cap or a tolerance: compare Taylor, Padé, Chebyshev (piecewise if the function jumps)"
        if interval
        else "no finite interval was given",
    )
    return tries, notes


def auto(ctx: Context, handlers: dict[str, Handler]) -> ApproximationReport:
    from .operation import approximation_with  # local import: operation imports this module

    tries, notes = plan(ctx)
    if not tries:
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "domain_unspecified",
            "Say where the approximation is wanted: a point (with an order), infinity, a small parameter, "
            "a finite interval (with a degree or an accuracy), or a period.",
            ["Near which point, or over which interval, should the approximation hold?"],
        )
    last: ApproximationReport | None = None
    for method, update, _why in tries[:MAX_TRIED]:
        if ctx.expired():
            break
        for note in notes:
            if note.method == method:
                note.tried = True
        candidate = approximation_with(ctx, method, update, handlers)
        last = candidate
        if candidate.status in (Status.OK, Status.UNCHANGED):
            candidate.suggestions = notes
            candidate.notes.insert(0, f"Chosen by rule: {_why}.")
            return candidate
        # a pole is the one case where the next method is a better reading of the same request
        if method == "taylor" and candidate.reason not in (
            "unsupported_input",
            "not_analytic",
            "not_laurent",
        ):
            break
    if last is None:
        raise Refused(
            Status.RESOURCE_LIMIT,
            "resource_limit",
            "The time budget ended before any method was tried.",
        )
    last.suggestions = notes
    return last
