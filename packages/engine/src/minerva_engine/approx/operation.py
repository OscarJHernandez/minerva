"""The approximation operation: parse, dispatch to a method, return one report.

Each method returns a report or raises `Refused`. Whatever happens, the caller gets a typed report;
nothing that was not checked is returned as an approximation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..models import Limits, Status
from .cfrac import continued_fraction
from .chebyshev import chebyshev
from .common import Context, Refused, build_context, diagnostics, failure
from .compare import compare
from .composed import integral_expansion, reversion, root_expansion
from .expand import asymptotic, laurent, perturbation, taylor
from .fourier import fourier
from .laplace import laplace
from .leading import leading, limit
from .legendre import legendre
from .models import ApproximationInput, ApproximationReport
from .multivar import multivariable
from .pade import pade
from .piecewise import piecewise
from .sums import accelerate, euler_maclaurin

Handler = Callable[[Context], ApproximationReport]

HANDLERS: dict[str, Handler] = {
    "taylor": taylor,
    "perturbation": perturbation,
    "laurent": laurent,
    "asymptotic": asymptotic,
    "multivariable": multivariable,
    "leading": leading,
    "limit": limit,
    "pade": pade,
    "chebyshev": chebyshev,
    "compare": compare,
    "fourier": fourier,
    "legendre": legendre,
    "piecewise": piecewise,
    "continued_fraction": continued_fraction,
    "euler_maclaurin": euler_maclaurin,
    "accelerate": accelerate,
    "integral_expansion": integral_expansion,
    "root_expansion": root_expansion,
    "reversion": reversion,
    "laplace": laplace,
}


# Methods that work in the one `variable`. The others name what they expand in themselves
# (`perturbation`: `small`; `multivariable`: `variables`; `continued_fraction`: a number).
NO_SINGLE_VARIABLE = frozenset({"auto", "perturbation", "multivariable", "continued_fraction"})
# Methods `variables` and `points` are meant for (`auto` routes them to `multivariable`).
TAKES_VARIABLES = frozenset({"auto", "multivariable"})


def refuse_mismatched_request(ctx: Context, method: str) -> None:
    """Refuse a request whose fields say something the method would silently not do.

    `variables` sent to a method of one variable used to be dropped, and the method then expanded
    in the default `variable`, which the expression did not contain: a verified series of the
    unchanged expression, presented as the answer. Nothing is rerouted here: `point` is not
    `points`, so which expansion was meant is the caller's to say.
    """
    data = ctx.data
    if method not in TAKES_VARIABLES and (data.variables or data.points):
        raise Refused(
            Status.CONDITIONS_NOT_SATISFIED,
            "fields_not_for_method",
            f"`variables` and `points` belong to the method `multivariable` (one expansion point for "
            f"each variable); `{method}` works in the one `variable`. Nothing was computed.",
        )
    if method in NO_SINGLE_VARIABLE:
        return
    free = ctx.expr.free_symbols
    if free and ctx.var not in free:
        raise Refused(
            Status.UNSUPPORTED,
            "variable_missing",
            f"The expression does not contain the variable `{data.variable}`, so there is nothing to "
            "approximate in it. Name a variable of the expression (several variables at once: the "
            "method `multivariable` with `variables` and `points`).",
        )


def approximation_with(
    ctx: Context, method: str, update: dict[str, Any], handlers: dict[str, Handler]
) -> ApproximationReport:
    """Run `method` on the same expression with some fields of the request changed (the strategy engine)."""
    handler = handlers[method]
    data = ctx.data.model_copy(update={"method": method, **update})
    sub = Context(data, ctx.limits, ctx.started, ctx.expr, ctx.var)
    try:
        refuse_mismatched_request(sub, method)
        return handler(sub)
    except Refused as refused:
        return failure(method, refused, data=data)


def _auto(ctx: Context) -> ApproximationReport:
    from .strategy import auto

    return auto(ctx, HANDLERS)


HANDLERS["auto"] = _auto


def approximation(data: ApproximationInput, limits: Limits) -> ApproximationReport:
    started = time.monotonic()
    handler = HANDLERS.get(data.method)
    if handler is None:
        report = failure(
            data.method,
            Refused(
                Status.UNSUPPORTED,
                "not_implemented",
                f"The method {data.method!r} is not available yet.",
            ),
            data=data,
        )
    else:
        try:
            ctx = build_context(data, limits, started)
            refuse_mismatched_request(ctx, data.method)
            report = handler(ctx)
        except Refused as refused:
            report = failure(data.method, refused, data=data)
        except LimitError as exc:
            report = failure(
                data.method, Refused(Status.RESOURCE_LIMIT, "resource_limit", str(exc)), data=data
            )
        except UnsupportedInput as exc:
            report = failure(
                data.method, Refused(Status.UNSUPPORTED, "unsupported_input", str(exc)), data=data
            )
        except MinervaMathError as exc:
            report = failure(
                data.method,
                Refused(Status.NOT_VERIFIED, "verification_failed", str(exc)),
                data=data,
            )
    report.method = report.method or data.method
    report.duration_ms = (time.monotonic() - started) * 1000
    report.diagnostics = diagnostics(report, started)
    return report
