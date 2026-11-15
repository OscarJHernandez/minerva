"""The approximation operation: parse, dispatch to a method, return one report.

Each method returns a report or raises `Refused`. Whatever happens, the caller gets a typed report;
nothing that was not checked is returned as an approximation.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..models import Limits, Status
from .chebyshev import chebyshev
from .common import Context, Refused, build_context, diagnostics, failure
from .compare import compare
from .expand import asymptotic, laurent, perturbation, taylor
from .leading import leading, limit
from .models import ApproximationInput, ApproximationReport
from .multivar import multivariable
from .pade import pade

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
}


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
            report = handler(build_context(data, limits, started))
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
    report.method = data.method
    report.duration_ms = (time.monotonic() - started) * 1000
    report.diagnostics = diagnostics(report, started)
    return report
