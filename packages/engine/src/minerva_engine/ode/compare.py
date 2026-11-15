"""A numerical solution against a closed form, on the numerical grid.

The closed form is the user's (a verified result of `ode_solve_symbolic`, passed as text); it is parsed
with the strict parser, may contain only the variable and the parameters, and is evaluated with mpmath.
The result is a measurement of the difference, never a verdict on which of the two is right.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np

from ..errors import MinervaMathError
from ..models import Limits, Status
from ..ops import _failure
from ..textparse import parse_text
from .models import OdeComparison, OdeNumericInput, OdeSeries
from .reading import OdeProblem
from .verification import DIGITS, Unevaluable, _lambdify


def compare_closed_form(
    problem: OdeProblem,
    data: OdeNumericInput,
    values: dict[str, float],
    times: np.ndarray,
    series: list[OdeSeries],
    limits: Limits,
) -> OdeComparison | None:
    if not data.compare_with:
        return None
    if len(data.compare_with) != len(problem.unknowns):
        return OdeComparison(
            status=Status.PARSE_ERROR,
            message="Give one closed form for each unknown, in the order of the unknowns.",
        )
    mp.mp.dps = DIGITS
    t = problem.variable
    allowed = {t.name, *(p.name for p in problem.parameters)}
    assumptions = {k: list(v) for k, v in data.assumptions.items()}
    rows: list[list[float]] = []
    labels: list[str] = []
    worst = 0.0
    worst_at = float(times[0])
    for unknown, text in zip(problem.unknowns, data.compare_with, strict=True):
        try:
            expression = parse_text(text, {**assumptions, t.name: ["real"]}, limits)
        except MinervaMathError as exc:
            status, message = _failure(exc)
            return OdeComparison(
                status=status, message=f"The closed form could not be read: {message}"
            )
        extra = sorted(s.name for s in expression.free_symbols if s.name not in allowed)
        if extra:
            return OdeComparison(
                status=Status.PARSE_ERROR,
                message="The closed form has symbols that are neither the variable nor a parameter: "
                + ", ".join(extra)
                + " (a closed form with constants must have them fixed by the conditions first).",
            )
        symbols = sorted(expression.free_symbols, key=lambda s: s.name)
        try:
            fn = _lambdify(symbols, expression)
        except Unevaluable:
            return OdeComparison(
                status=Status.UNSUPPORTED,
                message="The closed form cannot be evaluated numerically.",
            )
        row: list[float] = []
        try:
            for x in times:
                args = [
                    mp.mpf(float(x)) if s.name == t.name else mp.mpf(values[s.name])
                    for s in symbols
                ]
                value = fn(*args)
                if abs(mp.im(value)) > mp.mpf(10) ** -12 * max(1, abs(value)):
                    raise ArithmeticError("complex value")
                row.append(float(mp.re(value)))
        except (ArithmeticError, ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError):
            return OdeComparison(
                status=Status.NOT_VERIFIED,
                message="The closed form is not real and finite on the whole interval, so it is not compared.",
            )
        mine = next(s for s in series if s.unknown == unknown and s.order == 0)
        scale = max(max(abs(v) for v in row), 1e-9)
        differences = [abs(a - b) / scale for a, b in zip(mine.values, row, strict=True)]
        biggest = max(differences)
        if biggest > worst:
            worst, worst_at = biggest, float(times[differences.index(biggest)])
        rows.append([float(f"{v:.10g}") for v in row])
        labels.append(mine.label_latex)
    return OdeComparison(
        status=Status.OK,
        labels_latex=labels,
        closed_form=rows,
        max_difference=worst,
        worst_at=worst_at,
        message=(
            "Two separately verified results compared on the numerical grid: the difference is a "
            "measurement, not a bound on the error of either."
        ),
    )
