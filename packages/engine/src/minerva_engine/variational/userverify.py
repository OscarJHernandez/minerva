"""Finite-difference verification of the first variation along the user's own path.

Perturbations are chosen so that the *declared* conditions are respected where that is established
(a fixed value or derivative at an end is kept at zero) and are left free where nothing is fixed, so
free-end boundary terms are exercised. Each comparison reports the interior and the boundary part
of the analytic variation separately. The user's perturbations are accepted as given and flagged
if they violate a declared fixed condition.

For sampled paths the integrated (Euler-Lagrange plus boundary) form does not apply, because a
spline has corners in its higher derivatives; the comparison is then with the direct form
`∫ Σ ∂L/∂u^(k) η^(k) dt`, which is still an independent check of the finite differences but no
longer a check of the Euler-Lagrange expressions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mpmath as mp

from ..errors import ParseError
from . import polypath
from .firstvar import FirstVariationData, first_variation
from .models import (
    Diagnostics,
    PerturbationResult,
    VariationInput,
    VariationReport,
    VerificationReport,
)
from .numeric_setup import path_field, prepare
from .numerics import JetField, Numerics, Poly
from .paths import AnalyticField, SampledField, parse_in_variable
from .quadrature import Integral, integrate, integrate_breaks
from .report import _fmt, comparison_out
from .setup import Problem
from .verify import Claim, Trial, VariationForm, claimed_variation, compare_first_variation


def _fixed_slots(problem: Problem, function: int) -> list[tuple[str, int]]:
    return sorted(
        (end, j)
        for (end, i, j), cond in problem.conditions.items()
        if i == function and cond.kind == "fixed"
    )


def _poly(num: Numerics, kind: str, only: int | None, degree: int) -> Poly:
    problem, jets = num.problem, num.jets
    coefficients: list[polypath.Coeffs] = []
    for i in range(jets.count):
        if only is not None and i != only:
            coefficients.append([mp.mpf(0)] * (degree + 1))
            continue
        vanishing = set(_fixed_slots(problem, i))
        if kind == "bubble":
            vanishing |= {(e, j) for e in ("lower", "upper") for j in range(jets.orders[i])}
        elif kind == "generic":
            vanishing = set()
        coefficients.append(
            polypath.admissible(num.rng, degree, num.lower, num.upper, sorted(vanishing))
        )
    return Poly(coefficients)


def admissible(problem: Problem, num: Numerics, eta: JetField) -> bool:
    for (end, i, j), cond in problem.conditions.items():
        if cond.kind != "fixed":
            continue
        point = num.lower if end == "lower" else num.upper
        row = eta.at(point)[i]
        if j >= len(row):
            continue
        size = max(abs(v) for v in eta.at((num.lower + num.upper) / 2)[i][:1]) + 1
        if abs(row[j]) > mp.mpf("1e-12") * size:
            return False
    return True


def _direct_claim(num: Numerics, derived: FirstVariationData, breaks: list[mp.mpf] | None) -> Claim:
    problem, jets = num.problem, num.jets
    delta = num.compile(derived.delta_integrand)
    ends = [(problem.lower_term, num.lower), (problem.upper_term, num.upper)]

    def claim(path: JetField, eta: JetField) -> tuple[Any, mp.mpf, mp.mpf]:
        def f(s: mp.mpf) -> Any:
            return delta(num.jet_values(s, path, eta))

        def size(s: mp.mpf) -> Any:
            return abs(f(s))

        def rule(g: Callable[[mp.mpf], Any]) -> Integral:
            if breaks is not None:
                return integrate_breaks(g, breaks)
            return integrate(g, num.lower, num.upper, fast=True)

        integral, scale = rule(f), rule(size).value
        value = integral.value
        for term, point in ends:
            if term != 0:
                piece = num.compile(jets.vary(term))(num.jet_values(point, path, eta))
                value += piece
                scale += abs(piece)
        return value, mp.mpf(scale), integral.error

    return claim


def verify(problem: Problem, data: VariationInput, deadline: float) -> VariationReport:
    from ..models import Status

    if len(data.paths) != 1:
        raise ParseError("give exactly one path to verify")
    num = prepare(problem, data, seed=20260402)
    jets = problem.jets
    derived = first_variation(problem)
    form = VariationForm.of(derived)
    sampled = bool(data.paths[0].sampled)
    depth = max(jets.orders) if sampled else num.depth
    field, _ = path_field(num, problem, data.paths[0], data, depth=depth)
    breaks = None
    if isinstance(field, SampledField):
        from .action import _breaks

        breaks = _breaks(field, num)

    degree = max(2 * max(jets.orders) + 3, len(problem.conditions) + 4)
    trials: list[Trial] = []
    for spec in data.perturbations:
        unknown = set(spec.expressions) - set(jets.names)
        if unknown:
            raise ParseError(f"{sorted(unknown)[0]!r} is not one of the unknown functions")
        exprs = [
            parse_in_variable(problem, spec.expressions.get(name, "0"), data.format)
            for name in jets.names
        ]
        named = {p.name: v for p, v in num.params.items()}
        mapping = {s: named[s.name] for e in exprs for s in e.free_symbols if s.name in named}
        eta = AnalyticField(problem, exprs, mapping, depth=depth, lower=num.lower, upper=num.upper)
        trials.append(
            Trial("given", None, eta, spec.label or "given", admissible(problem, num, eta))
        )
    if data.automatic_perturbations or not trials:
        plans: list[tuple[str, int | None]] = [
            ("bubble", None),
            ("free_end", None),
            ("free_end", None),
        ]
        if jets.count > 1:
            plans += [("free_end", i) for i in range(jets.count)]
        for n, (kind, only) in enumerate(plans):
            auto_eta = num.field(_poly(num, kind, only, degree))
            label = {"bubble": "vanishing at the ends", "free_end": "free at undeclared ends"}[kind]
            if only is not None:
                label += f" ({jets.names[only]} only)"
            trials.append(
                Trial(kind, only, auto_eta, f"{label} #{n + 1}", admissible(problem, num, auto_eta))
            )

    claim = _direct_claim(num, derived, breaks) if sampled else None
    check = compare_first_variation(
        num, form, field, trials, seed=20260402, deadline=deadline, claim=claim, breaks=breaks
    )
    results: list[PerturbationResult] = []
    interior_form = VariationForm(form.interior, {})
    for trial, comparison in zip(trials, check.comparisons, strict=True):
        if sampled:
            interior = mp.mpf(0)
            boundary = comparison.analytic
        else:
            interior = claimed_variation(num, interior_form, field, trial.eta)[0]
            boundary = comparison.analytic - interior
        results.append(
            PerturbationResult(
                label=trial.label,
                kind=trial.kind,
                admissible=trial.admissible,
                interior_part="not separated on a sampled path" if sampled else _fmt(interior),
                boundary_part=_fmt(boundary),
                analytic=_fmt(comparison.analytic),
                comparison=comparison_out(problem, comparison),
            )
        )
    notes: list[str] = []
    if sampled:
        notes.append(
            "Sampled path: compared with the direct form of the variation, not with the "
            "Euler-Lagrange expressions and boundary terms."
        )
    if any(t.admissible is False for t in trials):
        notes.append(
            "Some perturbations violate a declared fixed condition; the comparison with the full "
            "first variation (boundary terms included) still applies to them."
        )
    report = VerificationReport(
        route="direct" if sampled else "integrated",
        perturbations=results,
        verdict=check.verdict,
        notes=notes,
    )
    ok = check.verdict == "passed"
    return VariationReport(
        status=Status.OK if ok else Status.NOT_VERIFIED,
        action="verify",
        verification=report,
        message=(
            "Finite differences of the action agree with the analytic first variation."
            if ok
            else "The finite differences did not confirm the analytic first variation on this path."
        ),
        diagnostics=Diagnostics(stages=len(results)),
    )
