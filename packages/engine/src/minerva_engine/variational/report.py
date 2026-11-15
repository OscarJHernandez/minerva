"""Assembling the first-variation report: LaTeX for every step, evidence from the checks."""

from __future__ import annotations

from typing import Any

import mpmath as mp
import sympy as sp

from ..models import Verification
from . import boundary as bnd
from .firstvar import FirstVariationData
from .jets import latex_eta_name, latex_jet_name
from .models import (
    BoundaryEntry,
    Comparison,
    ComparisonRow,
    EulerLagrangeEquation,
    FirstVariation,
    FirstVariationChecks,
    IbpStage,
)
from .setup import Problem
from .verify import Comparison as RawComparison
from .verify import PointwiseCheck, StageCheck, VariationCheck


def collect_eta(problem: Problem, expr: sp.Expr) -> str:
    """LaTeX of an expression that is linear in the perturbation jets, grouped by perturbation."""
    jets = problem.jets
    expanded = sp.expand(expr)
    parts: list[str] = []
    for i in range(jets.count):
        for k in range(jets.depth + 1):
            symbol = jets.eta[i][k]
            coefficient = expanded.coeff(symbol)
            if coefficient == 0:
                continue
            tex = problem.latex(coefficient)
            if coefficient.is_Add:
                tex = rf"\left({tex}\right)"
            elif coefficient == 1:
                tex = ""
            elif coefficient == -1:
                tex = "-"
            parts.append(
                f"{tex}\\,{problem.latex(symbol)}"
                if tex not in ("", "-")
                else f"{tex}{problem.latex(symbol)}"
            )
    if not parts:
        return "0"
    return " + ".join(parts).replace("+ -", "- ")


def _fmt(value: Any) -> str:
    return str(mp.nstr(value, 12))


def comparison_out(problem: Problem, c: RawComparison) -> Comparison:
    table = c.table
    return Comparison(
        kind="bubble" if c.kind == "bubble" else "generic",
        function=None if c.function is None else problem.jets.names[c.function],
        analytic=_fmt(c.analytic),
        scale=_fmt(c.scale),
        rows=[
            ComparisonRow(
                step=_fmt(r.step),
                quotient=_fmt(r.fd),
                abs_error=float(r.abs_error),
                rel_error=float(r.rel_error),
            )
            for r in table.rows
        ],
        observed_order=table.observed_order,
        richardson_rel_error=float(table.richardson_rel_error),
        verdict=table.verdict,
    )


def build(
    problem: Problem,
    data: FirstVariationData,
    check: VariationCheck,
    pointwise: PointwiseCheck,
    stage_checks: list[StageCheck],
    sympy_agrees: bool | None,
) -> FirstVariation:
    jets = problem.jets
    lo, hi = problem.latex(problem.lower), problem.latex(problem.upper)
    t = problem.variable.name

    stages = [
        IbpStage(
            index=s.index,
            function=None if s.function < 0 else jets.names[s.function],
            moved_order=None if s.function < 0 else s.order,
            integrand_latex=collect_eta(problem, s.interior),
            boundary_latex=collect_eta(problem, s.boundary),
            needs_derivatives=s.needs_derivatives,
            checked=None
            if s.index == 0
            else next((c.passed for c in stage_checks if c.index == s.index), None),
        )
        for s in data.stages
    ]
    equations = [
        EulerLagrangeEquation(
            function=name,
            equation_latex=rf"{problem.latex(data.euler[i])} = 0",
            expression_latex=problem.latex(data.euler[i]),
            order=data.el_orders[i],
        )
        for i, name in enumerate(jets.names)
    ]
    entries = bnd.classify(problem, data)
    boundary = [
        BoundaryEntry(
            end=e.end,
            function=jets.names[e.function],
            order=e.order,
            perturbation_latex=latex_eta_name(jets.names[e.function], e.order),
            coefficient_latex=problem.latex(e.coefficient),
            state=e.state,
            effect=e.effect,
            natural_condition_latex=(
                rf"{problem.latex(e.coefficient)} = 0"
                if e.effect in ("natural_condition", "kept_unspecified")
                else None
            ),
        )
        for e in entries
    ]

    interior_sum = sum((data.euler[i] * jets.eta[i][0] for i in range(jets.count)), sp.Integer(0))
    endpoint = ""
    for term, point in ((problem.upper_term, hi), (problem.lower_term, lo)):
        if term != 0:
            variation = jets.vary(term)
            endpoint += (
                rf" + \left.\left[{collect_eta(problem, variation)}\right]\right|_{{{t}={point}}}"
            )
    boundary_form = rf"\left[{collect_eta(problem, data.boundary)}\right]_{{{lo}}}^{{{hi}}}"
    first_variation = (
        rf"\delta S = \int_{{{lo}}}^{{{hi}}} \left({collect_eta(problem, interior_sum)}\right)\mathrm{{d}}{t}"
        rf" + {boundary_form}{endpoint}"
    )

    stationarity: list[str] = []
    if not data.null:
        stationarity += [
            rf"{problem.latex(data.euler[i])} = 0 \quad ({lo} < {t} < {hi})"
            for i in range(jets.count)
            if data.euler[i] != 0
        ]
    for e in entries:
        if e.effect == "natural_condition":
            where = lo if e.end == "lower" else hi
            stationarity.append(rf"{problem.latex(e.coefficient)} = 0 \quad ({t} = {where})")

    checks = FirstVariationChecks(
        deciding=check.verdict,
        comparisons=[comparison_out(problem, c) for c in check.comparisons],
        pointwise_passed=pointwise.passed,
        stages_checked=len(stage_checks),
        stages_passed=all(c.passed for c in stage_checks) if stage_checks else None,
        sympy_agrees=sympy_agrees,
        seed=check.seed,
    )
    verification = Verification(
        # SymPy's own derivation agreeing is a second route, reported as checks.sympy_agrees; it is
        # not a symbolic zero test of the claim, so it must never read as one.
        symbolic_zero=False,
        numeric_points_checked=check.points_checked,
        numeric_points_agreed=check.points_agreed,
        passed=check.passed,
        note="finite differences of the action along polynomial perturbations (numerical evidence)",
    )
    return FirstVariation(
        varied_latex=[
            rf"{latex_jet_name(name, 0)} \to {latex_jet_name(name, 0)} + \varepsilon\,{latex_eta_name(name, 0)}"
            for name in jets.names
        ],
        delta_integrand_latex=collect_eta(problem, data.delta_integrand),
        stages=stages,
        equations=equations,
        null_lagrangian=data.null,
        boundary_form_latex=boundary_form,
        endpoint_terms_latex=endpoint.removeprefix(" + "),
        boundary=boundary,
        first_variation_latex=first_variation,
        stationarity_latex=stationarity,
        notes=bnd.notes(problem, data, entries),
        checks=checks,
        verification=verification,
    )


def checks_out(problem: Problem, check: VariationCheck, statement: str) -> FirstVariationChecks:
    """The finite-difference evidence of a single-leg check (second variation)."""
    return FirstVariationChecks(
        deciding=check.verdict,
        comparisons=[comparison_out(problem, c) for c in check.comparisons],
        seed=check.seed,
        statement=statement,
    )
