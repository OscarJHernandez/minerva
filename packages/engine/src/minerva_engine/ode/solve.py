"""One entry for the analytical approaches: the method names the function."""

from __future__ import annotations

from ..models import Limits
from .models import OdeSolution, OdeSolveInput
from .symbolic import solve_closed_form


def solve_symbolic(data: OdeSolveInput, limits: Limits, trace_id: str | None = None) -> OdeSolution:
    if data.method == "series":
        from .approximation import solve_series

        return solve_series(data, limits, trace_id)
    if data.method == "perturbation":
        from .approximation import solve_perturbation

        return solve_perturbation(data, limits, trace_id)
    return solve_closed_form(data, limits, trace_id)
