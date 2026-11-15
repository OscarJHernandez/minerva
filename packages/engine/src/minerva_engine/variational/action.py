"""Action values of user-supplied trial paths, with independent quadrature and honest error figures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp

from ..errors import LimitError, ParseError, UnsupportedInput
from .models import (
    Evaluation,
    PathCondition,
    TrialPath,
    TrialResult,
    VariationInput,
    VariationReport,
)
from .numeric_setup import path_field, prepare
from .numerics import JetField, Numerics, _to_mp
from .paths import SampledField, numeric_value
from .quadrature import Integral, integrate, integrate_breaks
from .setup import Problem

AGREE_REL = mp.mpf("1e-8")  # the quadrature rules must agree to this (relative to max(1, |value|))


def action_of(
    num: Numerics, field: JetField, *, breaks: list[mp.mpf] | None = None
) -> tuple[Any, Any, Integral]:
    """(total, endpoint terms, integral) for one path; the integral carries the rule disagreement."""
    problem = num.problem
    lagrangian = num.compile(problem.lagrangian)

    def f(s: mp.mpf) -> Any:
        value = lagrangian(num.jet_values(s, field))
        if not mp.isfinite(value):
            raise LimitError("the integrand is not finite on the interval")
        return value

    try:
        integral = (
            integrate_breaks(f, breaks)
            if breaks is not None
            else integrate(f, num.lower, num.upper)
        )
    except (ZeroDivisionError, ValueError, OverflowError) as exc:
        raise LimitError("the integrand could not be evaluated on the whole interval") from exc
    endpoint: Any = mp.mpf(0)
    for term, point in ((problem.lower_term, num.lower), (problem.upper_term, num.upper)):
        if term != 0:
            endpoint += num.compile(term)(num.jet_values(point, field))
    return integral.value + endpoint, endpoint, integral


def _show(value: Any) -> str:
    return str(mp.nstr(value, 15))


def conditions_of(
    problem: Problem, num: Numerics, field: JetField, fmt: str
) -> list[PathCondition]:
    out: list[PathCondition] = []
    for (end, i, j), cond in sorted(problem.conditions.items()):
        if cond.kind != "fixed" or cond.value is None:
            continue
        point = num.lower if end == "lower" else num.upper
        required = _to_mp(numeric_value(problem, cond.value, fmt))
        row = field.at(point)[i]
        if j >= len(row):
            continue
        found = row[j]
        met = abs(found - required) <= mp.mpf("1e-9") * (1 + abs(required))
        out.append(
            PathCondition(
                function=problem.jets.names[i],
                order=j,
                end=end,
                required=_show(required),
                found=_show(found),
                met=bool(met),
            )
        )
    return out


def evaluate(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from ..models import Status
    from .models import Diagnostics

    if not data.paths:
        raise ParseError("give at least one trial path")
    if data.sweep is not None and (len(data.paths) != 1 or data.paths[0].sampled):
        raise ParseError("a sweep needs exactly one analytic path")
    num = prepare(problem, data, seed=20260401)
    jets = problem.jets
    depth = max(jets.orders)
    trials: list[TrialResult] = []
    values: list[Any] = []
    runs: list[tuple[TrialPath, str, Any]] = []
    if data.sweep is not None:
        for source in data.sweep.values:
            value = _to_mp(numeric_value(problem, source, data.format))
            runs.append((data.paths[0], f"{data.sweep.name} = {source}", (source, value)))
    else:
        runs = [(p, p.label or f"path {n + 1}", None) for n, p in enumerate(data.paths)]

    problems: list[str] = []
    for path, label, sweep in runs:
        field, route = path_field(
            num, problem, path, data, depth=depth, sweep_value=None if sweep is None else sweep[1]
        )
        breaks = _breaks(field, num) if isinstance(field, SampledField) else None
        total, endpoint, integral = action_of(num, field, breaks=breaks)
        tolerance = AGREE_REL * max(1, abs(integral.value))
        if integral.error > tolerance:
            problems.append(label)
        values.append(total)
        half = None
        if isinstance(field, SampledField) and route == "sampled":
            half = _half_data(num, problem, path, data, depth, total)
        trials.append(
            TrialResult(
                label=label,
                parameter_value=None if sweep is None else sweep[0],
                action=_show(total),
                integral=_show(integral.value),
                endpoint_terms=_show(endpoint),
                quadrature_error=float(integral.error),
                route="sampled" if route == "sampled" else "analytic",
                conditions=conditions_of(problem, num, field, data.format),
                half_data_difference=half,
            )
        )
    if problems:
        return VariationReport(
            status=Status.NOT_VERIFIED,
            action="evaluate",
            message="The independent quadrature rules did not agree for: "
            + ", ".join(problems)
            + ". No action value is shown.",
        )
    notes: list[str] = []
    ordering = None
    if len(trials) >= 2 and all(abs(mp.im(v)) < mp.mpf("1e-20") for v in values):
        order = sorted(range(len(trials)), key=lambda n: mp.re(values[n]))
        ordering = [trials[n].label for n in order]
    if len(trials) >= 2 and any(not c.met for t in trials for c in t.conditions):
        notes.append(
            "Some paths do not satisfy the declared fixed conditions, so these values do not "
            "compare like with like."
        )
    if any(t.conditions == [] for t in trials) and not problem.conditions:
        notes.append("No endpoint conditions were declared, so none were checked.")
    return VariationReport(
        status=Status.OK,
        action="evaluate",
        evaluation=Evaluation(
            lower=_show(num.lower),
            upper=_show(num.upper),
            parameters={p.name: _show(v) for p, v in num.params.items()},
            trials=trials,
            ordering=ordering,
            notes=notes,
        ),
        message="Action evaluated by two independent quadrature rules.",
        diagnostics=Diagnostics(stages=len(trials)),
    )


def _breaks(field: SampledField, num: Numerics) -> list[mp.mpf]:
    points = {num.lower, num.upper}
    for piece in field.pieces:
        points.update(mp.mpf(t) for t in piece.t if num.lower < t < num.upper)
    return sorted(points)


def _half_data(
    num: Numerics, problem: Problem, path: TrialPath, data: VariationInput, depth: int, full: Any
) -> float | None:
    """S from every second sample: how much the answer moves when the data are halved (a measurement)."""
    from .models import SampledFunction

    halved: dict[str, SampledFunction] = {}
    for name, fn in path.sampled.items():
        if len(fn.t) < 8:
            return None
        keep = list(range(0, len(fn.t), 2))
        if keep[-1] != len(fn.t) - 1:
            keep.append(len(fn.t) - 1)
        halved[name] = SampledFunction(
            t=[fn.t[k] for k in keep],
            values=[fn.values[k] for k in keep],
            slopes=None if fn.slopes is None else [fn.slopes[k] for k in keep],
        )
    try:
        field, _ = path_field(num, problem, TrialPath(sampled=halved), data, depth=depth)
    except (ParseError, UnsupportedInput):
        return None
    assert isinstance(field, SampledField)  # noqa: S101 - constructed from sampled data above
    total, _, _ = action_of(num, field, breaks=_breaks(field, num))
    return float(abs(total - full))


def check_action(
    integrand: Callable[[mp.mpf], Any],
    lower: Any,
    upper: Any,
    claimed: Any,
    *,
    rel_tol: Any = AGREE_REL,
) -> bool:
    """True if `claimed` equals ∫ integrand to `rel_tol` by two independent rules that also agree."""
    integral = integrate(integrand, lower, upper)
    scale = max(1, abs(integral.value))
    return bool(
        integral.error <= rel_tol * scale and abs(claimed - integral.value) <= rel_tol * scale
    )
