"""Generate the capability record from docs/capabilities.yaml by running every example for real.

Writes docs/capabilities.md (for the repository) and apps/site/src/data/capabilities.json (for the site's
landing page and /docs page). Nothing in the output is typed by hand except the YAML's prose: results,
statuses and verification summaries come from the engine and the handbook. Timings are left out so the output
is stable.

Usage:
  uv run python scripts/capabilities.py            # regenerate both files
  uv run python scripts/capabilities.py --check    # fail if the files on disk differ (CI runs this)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from minerva_engine import (
    ApproximationInput,
    DistributionInput,
    Engine,
    ExpectationInput,
    GeometryInput,
    Limits,
    StatMechInput,
    StatModelInput,
    SumsInput,
    TelescopingInput,
    VariationInput,
    VectorInput,
)
from minerva_engine.differentiate import DiffVariable
from minerva_engine.matrix import MatrixCalculusInput
from minerva_engine.models import ExpressionInput, TensorInput
from minerva_engine.ode import OdeAnalyzeInput, OdeNumericInput, OdeSolveInput
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.identities import (
    find_identities,
    simplify_expression,
    vector_identity_rules,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "capabilities.yaml"
MARKDOWN = ROOT / "docs" / "capabilities.md"
JSON_OUT = ROOT / "apps" / "site" / "src" / "data" / "capabilities.json"

WHERE = {
    "workbench": "workbench",
    "reference": "handbook pages",
    "chat": "chat",
    "api": "HTTP API",
}
NUMBER = re.compile(r"-?\d+\.\d+")


def _assumptions_text(assumptions: dict[str, list[str]]) -> str | None:
    if not assumptions:
        return None
    return "; ".join(f"{name} {', '.join(items)}" for name, items in assumptions.items())


def _short_numbers(text: str, digits: int = 6) -> str:
    return NUMBER.sub(lambda m: f"{float(m.group()):.{digits}f}", text)


def _sample_check(v: Any) -> str:
    zero = "symbolic zero test passed" if v.symbolic_zero else "symbolic zero test not established"
    return f"{zero}; numeric check agreed at {v.numeric_points_agreed} of {v.numeric_points_checked} points"


def _example(
    spec: dict[str, Any],
    status: str,
    inputs: list[dict[str, str]],
    outputs: list[dict[str, str]],
    checks: list[str],
    notes: list[str],
) -> dict[str, Any]:
    if spec.get("note"):
        notes = [*notes, " ".join(str(spec["note"]).split())]
    return {
        "id": spec["id"],
        "title": spec["title"],
        "op": spec["op"],
        "status": status,
        "input": inputs,
        "assumptions": _assumptions_text(spec.get("assumptions") or {}),
        "outputs": outputs,
        "checks": checks,
        "notes": notes,
    }


def _syntax(spec: dict[str, Any]) -> str:
    return "text" if spec.get("format") == "text" else "latex"


def _expr(spec: dict[str, Any], assumptions: dict[str, list[str]] | None = None) -> ExpressionInput:
    """The example's input in the syntax it names (`format: text`; LaTeX by default)."""
    return ExpressionInput(
        input=spec["input"],
        format=_syntax(spec),  # type: ignore[arg-type]
        assumptions=spec.get("assumptions") or {} if assumptions is None else assumptions,
    )


def _source_line(
    spec: dict[str, Any],
    label: str,
    engine: Engine | None = None,
    reading: str | None = None,
    source: str | None = None,
) -> dict[str, str]:
    """An input line: what was typed, its syntax, and (for plain text) how the engine read it.

    The reading comes from the engine, never from typesetting the typed string here.
    """
    line = {"label": label, "source": source if source is not None else spec["input"]}
    line["syntax"] = _syntax(spec)
    if _syntax(spec) == "text":
        if reading is None and engine is not None:
            status, _, _, latex = engine.features(_expr(spec))
            reading = latex if status.value == "ok" else None
        if reading:
            line["latex"] = reading
    return line


def run_simplify(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = simplify_expression(
        kb,
        engine,
        _expr(spec),
        spec.get("goal", "simplify"),
        spec.get("target") or [],
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        outputs.append({"label": "Result", "latex": r.result_latex})
        for i, step in enumerate(r.steps, start=1):
            entry = r.entries.get(step.name)
            how = f"{entry.title} ({step.name})" if entry else step.name
            if step.direction == "reverse":
                how += ", read right to left"
            outputs.append(
                {"label": f"Step {i}: {how}", "latex": f"{step.before_latex} = {step.after_latex}"}
            )
            checks.append(f"step {i}: " + _sample_check(step.verification))
        if r.verification:
            checks.append("end to end: " + _sample_check(r.verification))
    elif r.status.value in {"unchanged", "conditions_not_satisfied"} and r.input_latex:
        # Nothing better was found; under declared assumptions the input may still be read differently.
        outputs.append({"label": "Read as", "latex": r.input_latex})
    for blocked in r.blocked:
        entry = r.entries.get(blocked.rule_id)
        needs = ", ".join(f"{c.variable} {c.assumption}" for c in blocked.unmet_conditions)
        outputs.append(
            {
                "label": f"Not applied: {entry.title if entry else blocked.rule_id}",
                "text": f"{blocked.rule_id} matches but needs {needs}",
            }
        )
    return _example(
        spec,
        r.status.value,
        [_source_line(spec, "Input", engine, r.input_latex)],
        outputs,
        checks,
        [r.message] if r.message and r.status.value != "ok" else [],
    )


def run_differentiate(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.differentiate(
        _expr(spec),
        [DiffVariable(**v) for v in spec["variables"]],
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex and r.notation_latex:
        outputs.append({"label": "Derivative", "latex": f"{r.notation_latex} = {r.result_latex}"})
        if r.verification:
            v = r.verification
            checks.append(
                f"{v.steps_checked} step(s) checked against a numerical derivative; "
                f"agreed at {v.points_agreed} of {v.points_checked} points"
            )
    return _example(
        spec,
        r.status.value,
        [_source_line(spec, "Input", engine)],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_time_derivative(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.time_derivative(
        _expr(spec), spec["coordinates"], spec.get("time", "t"), spec.get("lagrangian")
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex and r.notation_latex:
        outputs.append(
            {"label": "Total time derivative", "latex": f"{r.notation_latex} = {r.result_latex}"}
        )
        if r.on_motion_latex is not None:
            zero = r.zero_on_motion is True
            outputs.append(
                {
                    "label": "On the equations of motion" + (" (zero)" if zero else ""),
                    "latex": r.on_motion_latex,
                }
            )
        if r.verification:
            v = r.verification
            checks.append(
                f"central difference of the input along random motions agreed at "
                f"{v.points_agreed} of {v.points_checked} points"
            )
    return _example(
        spec,
        r.status.value,
        [_source_line(spec, "Input", engine, r.input_latex)],
        outputs,
        checks,
        [r.message, *r.notes] if r.message else list(r.notes),
    )


def run_series(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.series(
        _expr(spec),
        spec.get("variable", "x"),
        str(spec.get("point", "0")),
        int(spec["order"]),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex and r.remainder_latex:
        outputs.append({"label": "Series", "latex": f"{r.result_latex} + {r.remainder_latex}"})
        if r.verification:
            v = r.verification
            checks.append(
                "coefficients agree with a Cauchy-integral estimate and the remainder shrinks at "
                f"order {r.order + 1} ({v.parameter_sets_checked} parameter set(s))"
            )
    return _example(
        spec,
        r.status.value,
        [_source_line(spec, "Input", engine)],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_approximation(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    fields = set(ApproximationInput.model_fields) - {"input", "format", "assumptions"}
    r = engine.approximation(
        ApproximationInput(
            input=spec["input"],
            format=_syntax(spec),  # type: ignore[arg-type]
            assumptions=spec.get("assumptions") or {},
            **{k: v for k, v in spec.items() if k in fields},
        )
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value in ("ok", "unchanged"):
        if r.result_latex:
            shown = (
                f"{r.result_latex} + {r.remainder_latex}" if r.remainder_latex else r.result_latex
            )
            outputs.append({"label": "Result", "latex": shown})
        if r.statement_latex:
            outputs.append({"label": "Statement", "latex": r.statement_latex})
        outputs.extend({"label": e.label, "latex": e.latex} for e in r.extras)
        for c in r.checks:
            word = "passed" if c.passed else "FAILED" if c.passed is False else "undecided"
            checks.append(f"{c.name}: {word}")
        if r.error and r.error.max_abs:
            notes.append(
                f"Largest error on {r.error.samples} sampled points: {r.error.max_abs} "
                "(a measurement, not a bound)."
            )
        if r.tolerance_met is not None:
            notes.append(
                "The tolerance is "
                + ("met" if r.tolerance_met else "NOT met")
                + " on the sampled points."
            )
        notes.extend(f"Warning: {w.replace('_', ' ')}." for w in r.warnings)
    notes.extend(f"Question: {q}" for q in r.questions)
    if r.message:
        notes.insert(0, r.message)
    return _example(
        spec, r.status.value, [_source_line(spec, "Input", engine)], outputs, checks, notes
    )


def run_identities(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = find_identities(
        kb,
        engine,
        _expr(spec),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    status = r.status.value
    wanted = list(spec["show"])
    by_rule = {m.application.rule_id: m for m in r.matches}
    for rule_id in wanted:
        m = by_rule.get(rule_id)
        if m is None:
            raise SystemExit(f"{spec['id']}: rule {rule_id} was not among the matches")
        app = m.application
        status = app.applicability.value
        if app.applicability.value == "applicable" and app.rewritten_latex:
            outputs.append({"label": m.entry.title, "latex": app.rewritten_latex})
            if app.verification:
                checks.append(_sample_check(app.verification))
        else:
            needs = ", ".join(f"{c.variable} {c.assumption}" for c in app.unmet_conditions)
            outputs.append(
                {
                    "label": m.entry.title,
                    "text": f"not applied; not provable from the declared assumptions: {needs} (the pattern's variables)"
                    if needs
                    else app.message,
                }
            )
    return _example(spec, status, [_source_line(spec, "Input", engine)], outputs, checks, notes)


def run_tensor(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.tensor_canonicalize(
        TensorInput(
            input=spec["input"], tensors=spec["tensors"], dimension=spec.get("dimension", 4)
        )
    )
    outputs: list[dict[str, str]] = []
    if r.status.value == "ok" and r.result_latex:
        outputs.append({"label": "Canonical form", "latex": r.result_latex})
    declared = "; ".join(
        f"{name}: rank {t['rank']}, {t.get('symmetry', 'no symmetry')}"
        for name, t in spec["tensors"].items()
    )
    checks: list[
        str
    ] = []  # the engine's message already says whether it was independently verified
    return _example(
        spec,
        r.status.value,
        [
            {"label": "Input", "source": spec["input"]},
            {"label": "Declared", "text": declared},
        ],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_solve(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.solve(spec["equation"], spec.get("variable"))
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok":
        factors = r" \cdot ".join(
            f"\\left({f.factor_latex}\\right)"
            + (f"^{{{f.multiplicity}}}" if f.multiplicity > 1 else "")
            for f in r.factors
        )
        outputs.append({"label": "Factorization", "latex": factors})
        exact = [x for f in r.factors if f.radicals == "yes" for x in f.exact_roots_latex]
        if exact:
            outputs.append({"label": "Roots in radicals", "latex": ",\\ ".join(exact)})
        for f in r.factors:
            if f.radicals == "no":
                outputs.append(
                    {"label": "Formula in radicals", "text": f"none: {f.radicals_reason}"}
                )
        if any(root.exact_latex is None for root in r.roots):
            outputs.append(
                {
                    "label": "Numerical roots",
                    "text": "; ".join(_short_numbers(root.numeric) for root in r.roots),
                }
            )
        if r.verification:
            v = r.verification
            parts = []
            if v.roots_substituted_back:
                parts.append(f"roots substituted back (largest residual {v.max_residual})")
            if v.count_matches_degree:
                parts.append("root count equals the degree")
            if v.vieta_holds:
                parts.append("Vieta's relations hold")
            checks.append("; ".join(parts))
    return _example(
        spec,
        r.status.value,
        [{"label": "Equation", "source": spec["equation"]}],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_lagrangian(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.analyze_lagrangian(spec["input"], spec["coordinates"], format=_syntax(spec))  # type: ignore[arg-type]
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        for eq in r.equations:
            outputs.append({"label": f"Equation of motion ({eq.coordinate})", "latex": eq.latex})
        for c in r.conserved:
            outputs.append(
                {"label": f"Conserved: {c.kind}", "latex": c.expression_latex, "text": c.reason}
            )
            if c.verified and c.verification:
                v = c.verification
                checks.append(
                    f"{c.kind}: time derivative is zero on the equations of motion "
                    f"({'symbolic zero' if v.symbolic_zero else 'numeric only'}; "
                    f"{v.numeric_points_agreed} of {v.numeric_points_checked} points)"
                )
        notes.extend(r.tested)
        notes.extend(r.notes)
    elif r.message:
        notes.append(r.message)
    return _example(
        spec,
        r.status.value,
        [
            _source_line(spec, "Lagrangian", reading=r.lagrangian_latex),
            {"label": "Coordinates", "text": ", ".join(spec["coordinates"])},
        ],
        outputs,
        checks,
        notes,
    )


def run_variation(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    request = VariationInput(
        action=spec.get("action", "first"),
        integrand=spec["input"],
        format=_syntax(spec),  # type: ignore[arg-type]
        variable=spec.get("variable", "t"),
        functions=spec["functions"],
        given_functions=spec.get("given_functions", []),
        lower=spec.get("lower"),
        upper=spec.get("upper"),
        lower_term=spec.get("lower_term"),
        upper_term=spec.get("upper_term"),
        conditions=spec.get("conditions", []),
        assumptions=spec.get("assumptions") or {},
        values=spec.get("values", {}),
        definitions=spec.get("definitions", {}),
        paths=spec.get("paths", []),
        with_second=bool(spec.get("with_second", False)),
    )
    r = engine.variation(request)
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    functional = r.analysis.functional_latex if r.analysis else None
    if r.status.value == "ok":
        if r.first is not None:
            f = r.first
            outputs.append({"label": "First variation", "latex": f.first_variation_latex})
            for eq in f.equations:
                outputs.append(
                    {"label": f"Euler-Lagrange ({eq.function})", "latex": eq.equation_latex}
                )
            outputs.append({"label": "Boundary form", "latex": f.boundary_form_latex})
            for b in f.boundary:
                if b.effect == "natural_condition" and b.natural_condition_latex:
                    outputs.append(
                        {
                            "label": f"Natural condition ({b.end} end, {b.function}, order {b.order})",
                            "latex": b.natural_condition_latex,
                        }
                    )
            checks.append(f.checks.statement)
            notes.extend(f.notes)
        if r.second is not None:
            outputs.append({"label": "Second variation", "latex": r.second.quadratic_form_latex})
            checks.append(r.second.checks.statement)
        if r.evaluation is not None:
            for t in r.evaluation.trials:
                outputs.append({"label": f"Action of {t.label}", "text": t.action})
            notes.append(r.evaluation.statement)
        if r.solution is not None:
            for p in r.solution.paths:
                outputs.append({"label": f"Stationary path action ({p.method})", "text": p.action})
            if r.solution.second is not None:
                outputs.append({"label": "Facts", "text": ", ".join(r.solution.second.facts)})
                notes.extend(r.solution.second.not_established)
            notes.append(r.solution.statement)
    elif r.message:
        notes.append(r.message)
    return _example(
        spec,
        r.status.value,
        [
            _source_line(spec, "Integrand", reading=functional),
            {"label": "Unknown functions", "text": ", ".join(spec["functions"])},
        ],
        outputs,
        checks,
        notes,
    )


def run_invariants(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.search_invariants(
        spec["input"],
        spec["coordinates"],
        spec.get("angles", []),
        spec.get("degree", 2),
        spec.get("power_bound", 3),
        spec.get("harmonics", 1),
        format=_syntax(spec),  # type: ignore[arg-type]
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        for c in r.found:
            outputs.append({"label": c.name, "latex": c.expression_latex})
            if c.verification:
                v = c.verification
                checks.append(
                    "time derivative is zero on the equations of motion "
                    f"({'symbolic zero' if v.symbolic_zero else 'numeric only'}; "
                    f"{v.numeric_points_agreed} of {v.numeric_points_checked} points)"
                )
        if not r.found:
            outputs.append({"label": "Result", "text": "no new constant in the searched family"})
        notes.append(r.ansatz)
        if r.known:
            notes.append("Already known and left out of the count: " + "; ".join(r.known) + ".")
        notes.extend(r.notes)
    elif r.message:
        notes.append(r.message)
    inputs = [
        _source_line(spec, "Lagrangian", reading=r.lagrangian_latex),
        {"label": "Coordinates", "text": ", ".join(spec["coordinates"])},
    ]
    if spec.get("angles"):
        inputs.append({"label": "Angles", "text": ", ".join(spec["angles"])})
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_discovery(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    system = spec.get("system", "lagrangian")
    r = engine.discover_invariants(
        spec["input"],
        spec["coordinates"],
        system,
        spec.get("momenta"),
        spec.get("angles", []),
        spec.get("budget", "standard"),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        for c in r.constants:
            outputs.append(
                {
                    "label": f"{c.id}, " + ("newly found" if c.kind == "new" else c.name.lower()),
                    "latex": c.expression_latex,
                }
            )
            if c.kind == "new" and c.verification:
                v = c.verification
                checks.append(
                    "time derivative is zero on the equations of motion "
                    f"({'symbolic zero' if v.symbolic_zero else 'numeric only'}; "
                    f"{v.numeric_points_agreed} of {v.numeric_points_checked} points)"
                )
        if not r.new_constants:
            outputs.append(
                {"label": "Result", "text": "no additional constant within the explored families"}
            )
        ind = r.independence
        if ind is not None and ind.constants:
            notes.append(
                f"Functionally independent: {ind.rank_lower}"
                + (
                    " (proved)."
                    if ind.rank_proved
                    else f" to {ind.rank_upper} (lower bound proved)."
                )
            )
            for rel in ind.relations:
                outputs.append({"label": "Exact relation", "latex": rel.latex})
        searched = [t for t in r.tiers if t.outcome == "searched"]
        if searched:
            notes.append(
                f"{len(searched)} families searched, from {searched[0].code} to {searched[-1].code} "
                "(d: degree in the velocities or momenta, p: power bound, h: harmonics, t: powers of t, "
                "c: powers of a cyclic coordinate)."
            )
        if r.unexplored:
            notes.append("Not searched: " + r.unexplored[0])
    elif r.message:
        notes.append(r.message)
    inputs = [
        {
            "label": "Hamiltonian" if system == "hamiltonian" else "Lagrangian",
            "source": spec["input"],
        },
        {"label": "Coordinates", "text": ", ".join(spec["coordinates"])},
    ]
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_entry(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    entry = kb.get(spec["entry"])
    if entry is None:
        raise SystemExit(f"{spec['id']}: no handbook entry {spec['entry']}")
    level = getattr(entry.verification, "value", entry.verification)
    notes = [f"Validity: {entry.validity}"] if entry.validity else []
    if entry.reference:
        ref = entry.reference
        notes.append(f"Source: {ref.source} {ref.locator or ''}".strip())
    return _example(
        spec,
        "stated" if level == "stated" else "verified",
        [{"label": "Entry", "text": f"{entry.id}: {entry.title}"}],
        [{"label": "Statement", "latex": kb.statement_latex(entry)}],
        [f"verification level: {level}"],
        notes,
    )


def run_read_formulas(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    formulas = [str(f) for f in spec["formulas"]]
    readings = engine.read_formulas(formulas)
    inputs = [{"label": "Transcribed", "source": f} for f in formulas]
    outputs: list[dict[str, str]] = []
    for reading in readings:
        if reading.status.value == "ok" and reading.read_as:
            outputs.append({"label": "Read as", "latex": reading.read_as})
        else:
            outputs.append({"label": "Not read", "text": reading.message or reading.status.value})
    status = next((r.status.value for r in readings if r.status.value != "ok"), "ok")
    return _example(
        spec,
        status,
        inputs,
        outputs,
        ["strict parser: the formula is parsed, never simplified or evaluated"],
        [],
    )


def _matrix_text(m: dict[str, Any]) -> str:
    if m.get("entries"):
        return f"{m['name']} = [" + "; ".join(", ".join(row) for row in m["entries"]) + "]"
    rows = m["rows"]
    return f"{m['name']}: {rows} by {m.get('cols') or rows}, entries not given"


def run_matrix(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    request = MatrixCalculusInput(
        expression=_expr(spec),
        matrices=spec["matrices"],
        commuting=[tuple(p) for p in spec.get("commuting", [])],
        invertible=spec.get("invertible", []),
        self_commuting=spec.get("self_commuting", []),
        action=spec["action"],
        variable=spec.get("variable"),
        lower=spec.get("lower"),
        upper=spec.get("upper"),
    )
    r = engine.matrix_calculus(request)
    inputs = [_source_line(spec, "Expression")]
    inputs += [{"label": "Matrix", "source": _matrix_text(m)} for m in spec["matrices"]]
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        heading = {
            "explicit": "Result",
            "expression": "Result",
            "equation": "Equation (not an evaluated integral)",
            "representation": "Exact representation (not evaluated)",
            "numeric": "Numerical approximation (not a closed form)",
            "gradient": "Gradient",
            "jacobian": "Jacobian",
            "hessian": "Hessian",
        }[r.result_kind or "expression"]
        if r.result_latex:
            outputs.append({"label": heading, "latex": r.result_latex})
        if r.equation_latex:
            outputs.append({"label": "Verified equation", "latex": r.equation_latex})
        if r.derivative is not None:
            d = r.derivative
            shown = (
                "gradient within the symmetric matrices (Frobenius inner product)"
                if d.convention == "symmetric_frobenius"
                else "unconstrained derivative (every entry free)"
            )
            notes.append(
                f"Convention: {shown}; real field, a transpose and not a conjugate transpose."
            )
            for alt in d.alternatives:
                outputs.append(
                    {"label": "Entrywise form (independent entries, i <= j)", "latex": alt.latex}
                )
        if r.method:
            notes.append(f"Method: {r.method_title} ({r.method}).")
        notes += [f"Assumes: {a}." for a in r.assumptions_used if not a.startswith("scalar")]
        if r.verification:
            v = r.verification
            checks.append(
                f"numeric check agreed at {v.points_agreed} of {v.points_checked} random samples"
            )
            if v.symbolic_zero is not None:
                checks.append(
                    "exact check through degree 4: " + ("passed" if v.symbolic_zero else "failed")
                )
    for b in r.blocked:
        notes.append(f"Not applied: {b.title} needs {b.needs}.")
    if r.reason:
        notes.append(f"Refused ({r.reason}).")
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


_GEOMETRY_ENGINE: Engine | None = None


def run_geometry(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    # Schwarzschild takes a few seconds: use a roomy budget so the generated page never depends on speed.
    global _GEOMETRY_ENGINE
    if _GEOMETRY_ENGINE is None:
        _GEOMETRY_ENGINE = Engine(Limits(timeout_seconds=120, max_concurrency=1))
    request = GeometryInput(
        coordinates=spec["coordinates"],
        metric=spec["metric"],
        format=spec.get("format", "latex"),
        assumptions=spec.get("assumptions") or {},
        functions=spec.get("functions") or [],
        quantities=spec["quantities"],
        tensor=spec.get("tensor"),
        tensor_index=spec.get("tensor_index", 0),
    )
    r = _GEOMETRY_ENGINE.geometry(request)
    inputs = [
        {
            "label": "Coordinates",
            "source": ", ".join(spec["coordinates"]),
            "syntax": _syntax(spec),
        },
        *(
            {"label": f"Metric row {i}", "source": ",  ".join(row), "syntax": _syntax(spec)}
            for i, row in enumerate(spec["metric"], start=1)
        ),
    ]
    if spec.get("functions"):
        decls = ", ".join(f"{f['name']}({', '.join(f['args'])})" for f in spec["functions"])
        inputs.append({"label": "Functions", "source": decls})
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    for q in r.quantities:
        if q.status.value != "ok":
            notes.append(f"{q.kind}: {q.status.value}. {q.message}")
            continue
        if q.scalar_latex is not None:
            outputs.append({"label": q.title, "latex": q.scalar_latex})
        elif q.all_zero and q.kind != "geodesics":
            outputs.append({"label": q.title, "latex": "0 \\text{ (every component)}"})
        for c in q.components:
            text = c.latex if q.kind == "geodesics" else f"{c.label_latex} = {c.latex}"
            outputs.append({"label": q.title.split(" (")[0], "latex": text})
        v = q.verification
        if v is not None:
            numeric = [c for c in v.checks if c.kind == "numeric"]
            proof = {True: "proved", False: "not proved", None: "not attempted"}[v.symbolic_zero]
            checks.append(
                f"{q.kind}: {len(numeric)} independent numeric checks passed; symbolic proof {proof}"
            )
    if r.signature:
        notes.append(f"Signature {r.signature}, decided from the declared assumptions.")
    elif r.dimension and r.status.value == "ok":
        notes.append("Signature not determined: " + r.signature_note)
    if r.restrictions_latex:
        notes.append(
            "Valid only where "
            + ", ".join(f"${x}$" for x in r.restrictions_latex)
            + " (the coordinates "
            "break down there; this is not a statement about curvature singularities)."
        )
    notes += [f"Assumes: {a}." for a in r.assumptions_used[:-1]]
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_vector(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    action = spec.get("action", "evaluate")
    rules = (
        vector_identity_rules(kb) if action in ("identities", "apply_identity", "derive") else []
    )
    r = engine.vector_calculus(
        VectorInput(
            expression=spec.get("input", ""),
            format=spec.get("format", "latex"),
            coordinates=spec.get("coordinates", "cartesian"),
            fields=spec.get("fields") or [],
            constants=spec.get("constants") or [],
            assumptions=spec.get("assumptions") or {},
            action=action,
            to=spec.get("to"),
            rules=rules,
            rule_id=spec.get("rule_id"),
            path=spec.get("path"),
            reverse=spec.get("reverse", False),
            premises=[{"equation": p} for p in spec.get("premises") or []],
            premise=spec.get("premise"),
            operator=spec.get("operator"),
            target=spec.get("target"),
            integral=spec.get("integral"),
            curve=spec.get("curve"),
            surface=spec.get("surface"),
            region=spec.get("region"),
        )
    )
    inputs = [{"label": "Coordinates", "source": spec.get("coordinates", "cartesian")}]
    if spec.get("integral"):
        inputs.append({"label": "Integral", "source": spec["integral"].replace("_", " ")})
    for key in ("curve", "surface", "region"):
        if spec.get(key):
            inputs.append({"label": key.capitalize(), "source": _geometry_text(spec[key])})
    if spec.get("input"):
        inputs.append({"label": "Expression", "source": spec["input"], "syntax": _syntax(spec)})
    for i, p in enumerate(spec.get("premises") or []):
        inputs.append(
            {"label": f"Premise {i} (given by the user)", "source": p, "syntax": _syntax(spec)}
        )
    if spec.get("target"):
        inputs.append({"label": "Target", "source": spec["target"], "syntax": _syntax(spec)})
    for f in spec.get("fields") or []:
        what = f["kind"] + (
            " field, components " + ", ".join(f["components"])
            if f.get("components")
            else " field" + (f", {f['smoothness']}" if f.get("smoothness") else "")
        )
        inputs.append({"label": f"Field {f['name']}", "source": what})
    return _vector_example(spec, r, inputs)


def _geometry_text(geometry: dict[str, Any]) -> str:
    parts = []
    if "q" in geometry:
        parts.append("coordinates (" + ", ".join(geometry["q"]) + ")")
    for key in ("parameter", "u", "v"):
        if key in geometry:
            p = geometry[key]
            parts.append(f"{p['name']} from {p['lower']} to {p['upper']}")
    if "bounds" in geometry:
        parts.append("bounds " + "; ".join(f"[{lo}, {hi}]" for lo, hi in geometry["bounds"]))
    return ", ".join(parts)


def _vector_example(spec: dict[str, Any], r: Any, inputs: list[dict[str, str]]) -> dict[str, Any]:
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        if r.result_kind == "vector" and r.basis_latex and r.components_latex:
            outputs += [
                {"label": f"Component along {b}", "latex": c}
                for b, c in zip(r.basis_latex, r.components_latex, strict=True)
            ]
        elif (
            r.result_kind in ("scalar", "vector")
            and r.result_latex
            and r.integral is None
            and r.singular is None
        ):
            outputs.append({"label": "Result", "latex": r.result_latex})
        if r.singular is not None:
            sg = r.singular
            outputs.append(
                {
                    "label": "Classical result (pointwise, away from the source)",
                    "latex": rf"{r.input_latex} = {sg.classical_latex}",
                }
            )
            outputs.append(
                {
                    "label": "Distributional result (as a distribution on R^3)",
                    "latex": sg.distributional_latex,
                }
            )
            agreed = sum(w.passed for w in sg.weak_checks)
            checks.append(
                f"{sg.proof_status}: the hypotheses of the shrinking-sphere argument are exact "
                f"({', '.join(d.title.lower() for d in sg.derivation[:3])}); {agreed} of "
                f"{len(sg.weak_checks)} independent pairings with test functions and "
                f"{sum(f.passed for f in sg.flux_checks)} of {len(sg.flux_checks)} sphere fluxes agree. "
                "Derived is not proved; the pairings are supporting evidence"
            )
            if sg.regularisation:
                checks.append(
                    "regularised field r/(r^2 + eps^2)^(3/2): the pairing error falls as eps shrinks "
                    "(reported, never deciding)"
                )
            notes.append(sg.scope)
            if sg.claim is not None:
                notes.append(
                    f"Statement checked: ${sg.claim.latex}$ "
                    + ("holds" if sg.claim.holds else f"does not hold ({sg.claim.reason})")
                    + "."
                )
        elif r.result_kind == "matrix" and r.result_latex and r.determinant_latex:
            outputs.append({"label": "Jacobian", "latex": r.result_latex})
            outputs.append({"label": "Determinant", "latex": r.determinant_latex})
        for i, step in enumerate(r.steps, start=1):
            kind = (
                "conditional on " + "; ".join(step.given)
                if step.evidence == "conditional"
                else "identity"
            )
            outputs.append({"label": f"Step {i} ({kind}): {step.title}", "latex": step.after_latex})
        if r.integral is not None:
            for iv in r.integral.values:
                shown = iv.latex if iv.latex is not None else (iv.numeric or "")
                kind = "exact" if iv.exact else "numerical approximation"
                outputs.append({"label": f"Integral ({kind})", "latex": f"{iv.label} = {shown}"})
            notes.append("Orientation: " + r.integral.orientation + ".")
            for h in r.integral.hypotheses:
                notes.append(f"Hypothesis ({h.status}): {h.text}. {h.note}.")
            if r.integral.theorem:
                checks.append(
                    f"{r.integral.theorem}: both sides computed separately, each an exact value checked "
                    "against an independent quadrature of the input read in Cartesian coordinates"
                    if r.integral.method == "symbolic"
                    else f"{r.integral.theorem}: both sides computed separately and agree numerically"
                )
            else:
                checks.append(
                    "exact value checked against an independent quadrature"
                    if r.integral.method == "symbolic"
                    else "two quadrature rules agree; a numerical approximation, not an exact value"
                )
        if r.em is not None:
            for item in r.em.items:
                shown = item.latex if item.latex is not None else (item.numeric or "")
                outputs.append({"label": item.label, "latex": shown})
                if item.note:
                    notes.append(f"{item.label}: {item.note}.")
            checks.extend(r.em.checks)
            if r.em.approximate:
                notes.append("A numerical approximation agreeing with an independent method.")
        for eq in r.equations_latex:
            outputs.append({"label": "Derived equation", "latex": eq})
        if r.model_check is not None and r.model_check.status == "passed":
            checks.append(r.model_check.note)
        for a in r.applicable[:6]:
            outputs.append(
                {
                    "label": f"Match: {a.rule_id}" + (" (blocked)" if a.blocked else ""),
                    "latex": rf"{a.part_latex} \to {a.result_latex}",
                }
            )
        v = r.verification
        if v is not None and not r.steps:
            checks.append(
                f"{v.points_agreed} of {v.points_checked} sample points agree with an independent "
                "Cartesian finite-difference evaluation; no symbolic proof is claimed"
            )
        if r.evidence == "identity":
            checks.append(
                "each identity step: the whole expressions before and after differ by an exactly zero "
                "component residual, and an independent Cartesian evaluation agrees"
            )
        if r.evidence == "conditional":
            notes.append(
                "Holds only given: "
                + "; ".join(r.given)
                + ". Supplied by the user and not checked: "
                "not an identity."
            )
        if r.restrictions_latex and r.singular is None:
            notes.append(
                "Valid only where "
                + ", ".join(f"${x}$" for x in r.restrictions_latex)
                + " (the coordinates break down there)."
            )
    for b in r.blocked:
        notes.append(f"Not shown: needs {b.needs} smoothness of {', '.join(b.fields)}. {b.note}.")
    if r.needs_premises:
        notes.append("Needs these premises (the user adopts them): " + "; ".join(r.needs_premises))
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_electromagnetism(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    from minerva_engine.vector import FieldDecl, Premise
    from minerva_engine.vector.em import TEMPLATES
    from minerva_knowledge.identities import resolve_premise_ids

    if "problem" in spec:
        from minerva_engine.vector.models import EMProblem
        from pydantic import TypeAdapter

        problem = TypeAdapter(EMProblem).validate_python(spec["problem"])
        r = engine.vector_calculus(
            VectorInput(
                expression="",
                format="text",
                action="em_compute",
                em=problem,
                constants=spec.get("constants") or [],
            )
        )
        return _vector_example(
            spec,
            r,
            [{"label": "Problem", "source": json.dumps(spec["problem"]), "syntax": "text"}],
        )
    template = TEMPLATES[spec["derivation"]]
    chosen = resolve_premise_ids(kb, spec.get("premise_ids") or [])
    premises = [Premise(equation=o.equation, label=o.title) for o in chosen]
    premises += [Premise(equation=p) for p in spec.get("premises") or []]
    smoothness = spec.get("smoothness")
    r = engine.vector_calculus(
        VectorInput(
            expression="",
            format="text",
            fields=[
                FieldDecl(
                    name=n,
                    kind=k,
                    depends_on=["x", "y", "z", "t"],
                    smoothness=smoothness if n in template.smooth else None,
                )
                for n, k in template.fields
            ],
            constants=list(template.constants),
            premises=premises,
            action="em_derive",
            template=template.id,
            rules=vector_identity_rules(kb),
        )
    )
    inputs = [{"label": "Derivation", "source": template.title}]
    inputs += [
        {
            "label": f"Premise (adopted by the user): {o.title}",
            "source": o.equation,
            "syntax": "text",
        }
        for o in chosen
    ]
    if smoothness:
        inputs.append({"label": "Smoothness (declared by the user)", "source": smoothness})
    return _vector_example(spec, r, inputs)


_READING_WORDS = {
    "dirac_delta": "a Dirac delta",
    "heaviside": "a step function",
    "symbol": "an ordinary symbol",
}


def run_statmech(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    del kb
    request = StatMechInput(**spec["request"])
    r = engine.statmech(request)
    system = request.system
    descriptions = {
        "levels": lambda s: f"{len(s.levels)} levels",
        "level_family": lambda s: (
            f"E({s.index}) = {s.energy}, {s.index} from {s.lower} to {s.upper}"
        ),
        "density_of_states": lambda s: (
            f"rho({s.variable}) = {s.density}, from {s.lower} to {s.upper}"
        ),
        "classical": lambda s: f"H = {s.hamiltonian}",
        "quantum": lambda s: (
            f"a {len(s.hamiltonian)}x{len(s.hamiltonian)} Hermitian matrix H"
            + (f" and operators {', '.join(s.observables)}" if s.observables else "")
        ),
        "occupation": lambda s: (
            f"{s.statistics} particles in {len(s.levels)} levels, chemical potential {s.mu}"
        ),
        "chain": lambda s: f"periodic chain of N = {s.length} sites, bond {s.bond}, site {s.site}",
    }
    inputs = [{"label": "System", "source": descriptions[system.kind](system), "syntax": "text"}]
    if request.ensemble != "canonical":
        inputs.append({"label": "Ensemble", "source": request.ensemble.replace("_", " ")})
    if request.total_energy is not None:
        inputs.append({"label": "Fixed total energy", "source": request.total_energy})
    if request.copies is not None:
        inputs.append(
            {
                "label": "Copies",
                "source": f"{request.copies.count}, "
                + (
                    "indistinguishable (you chose)"
                    if request.copies.identical
                    else "distinguishable"
                ),
            }
        )
    if request.method is not None:
        inputs.append({"label": "Method", "source": request.method.replace("_", " ")})
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        for q in r.quantities:
            outputs.append({"label": q.title, "latex": q.latex})
            names = sorted({c.name.replace("_", " ") for c in q.checks if c.passed})
            if names:
                checks.append(f"{q.name}: {', '.join(names)} agreed")
        if r.verification.passed:
            checks.append(
                f"{r.verification.points_agreed} of {r.verification.points_checked} sample points "
                "agree with an independent evaluation (evidence, not a proof)"
            )
        if r.approximation is not None:
            a = r.approximation
            notes.append(
                f"An approximation, not an identity. Expansion parameter: {a.parameter}. "
                f"Expected accuracy: {a.order}. Valid: {a.validity}."
            )
            notes.append("Assumes: " + "; ".join(a.assumptions) + ".")
            if a.error_table:
                notes.append(
                    f"Measured error against the independent evaluation ({len(a.error_table)} points; "
                    "a measurement, not a bound)."
                )
        notes += r.notes
    else:
        notes.append(f"Refused ({r.reason}).")
        notes += r.questions
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_distribution(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    del kb
    variable = spec["variable"]
    assumptions = dict(spec.get("assumptions") or {})
    assumptions[variable] = sorted({*assumptions.get(variable, []), "real"})
    r = engine.distribution(
        DistributionInput(
            expression=_expr(spec, assumptions),
            action=spec["action"],
            variable=variable,
            lower=spec.get("lower"),
            upper=spec.get("upper"),
            endpoint=spec.get("endpoint"),
            notation=spec.get("notation") or {},
        )
    )
    inputs = [_source_line(spec, "Expression", reading=r.input_latex)]
    if spec.get("notation"):
        inputs.append(
            {
                "label": "You confirmed",
                "text": "; ".join(
                    f"{name}(…) is {_READING_WORDS[reading]}"
                    for name, reading in spec["notation"].items()
                ),
            }
        )
    if spec["action"] == "integrate":
        inputs.append({"label": "Limits", "source": f"{spec.get('lower')} to {spec.get('upper')}"})
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        label = "Value" if r.equality == "value" else "Result (as distributions)"
        outputs.append({"label": label, "latex": r.result_latex})
        notes += [f"Step: {s.title} ({s.entry_id})." for s in r.steps]
        if r.verification:
            v = r.verification
            checks.append(
                "conditions established exactly: " + ("yes" if v.conditions_established else "no")
            )
            checks.append(
                f"{v.pairings_agreed} of {v.pairings_checked} test-function pairings agree "
                "(supporting evidence, not a proof)"
            )
    elif r.reason:
        notes.append(f"Refused ({r.reason}).")
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def _given(r: Any) -> list[dict[str, str]]:
    return [
        {"label": "Given (your relation, not checked by Minerva)", "latex": premise.latex}
        for premise in r.given
    ]


def run_substitute(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.substitute(
        ExpressionInput(input=spec["input"], assumptions=spec.get("assumptions") or {}),
        spec["relations"],
        order=spec.get("order"),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        for i, step in enumerate(r.steps, start=1):
            shown = f"{step.before_latex} \\to "
            shown += f"{step.display_latex} \\to " if step.display_latex else ""
            outputs.append({"label": f"Step {i}: substitute", "latex": shown + step.after_latex})
        outputs.append({"label": "Result", "latex": r.result_latex})
        if r.evidence == "conditional":
            outputs += _given(r)
            notes.append(
                "Conditional: the result holds given the relations, which Minerva did not check."
            )
        else:
            notes.append("The result also holds without the relation.")
        if r.restrictions_latex:
            outputs.append({"label": "Valid only where", "latex": ", ".join(r.restrictions_latex)})
        if r.verification:
            checks.append(_sample_check(r.verification))
    elif r.message:
        notes.append(r.message)
    inputs = [{"label": "Expression", "source": spec["input"]}] + [
        {"label": f"Relation {i}", "source": text}
        for i, text in enumerate(spec["relations"], start=1)
    ]
    if spec.get("order"):
        inputs.append({"label": "Order", "source": spec["order"]})
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_rewrite(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    rule = None
    if spec.get("rule"):
        entry = kb.get(spec["rule"])
        rule = entry.rule if entry is not None else None
    r = engine.rewrite_at(
        ExpressionInput(
            input=spec["input"],
            format=_syntax(spec),  # type: ignore[arg-type]
            assumptions=spec.get("assumptions") or {},
        ),
        spec["action"],
        part=spec.get("part"),
        occurrence=spec.get("occurrence"),
        rule=rule,
        by=spec.get("by") or [],
        target=spec.get("target"),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        for step in r.steps:
            outputs.append(
                {
                    "label": f"Applied {step.name} to the part",
                    "latex": f"{step.matched_latex}" if step.matched_latex else step.name,
                }
            )
        outputs.append({"label": "Result", "latex": r.result_latex})
        if r.restrictions_latex:
            outputs.append({"label": "Valid only where", "latex": ", ".join(r.restrictions_latex)})
        if r.verification:
            checks.append("whole expression: " + _sample_check(r.verification))
    for cond in r.unmet_conditions:
        outputs.append(
            {
                "label": "Not applied",
                "text": f"the identity needs its variable {cond.variable} to be {cond.assumption}",
            }
        )
    inputs = [{"label": "Expression", "source": spec["input"]}]
    if spec.get("part"):
        inputs.append({"label": "Part", "source": spec["part"]})
    inputs.append({"label": "Move", "source": spec.get("rule") or spec["action"]})
    notes = [r.message] if r.message and r.status.value != "ok" else []
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_isolate(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.isolate(
        ExpressionInput(input=spec["input"], assumptions=spec.get("assumptions") or {}),
        spec["variable"],
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        for i, sol in enumerate(r.solutions, start=1):
            outputs.append({"label": f"Solution {i}", "latex": sol.latex})
            if sol.conditions_latex:
                outputs.append(
                    {"label": f"Solution {i} needs", "latex": ", ".join(sol.conditions_latex)}
                )
            if sol.branch:
                notes.append(f"Solution {i} is a family: {sol.branch}.")
        if not r.solutions:
            outputs.append({"label": "Result", "text": "no solution"})
        if r.conditions_latex:
            outputs.append({"label": "All solutions need", "latex": ", ".join(r.conditions_latex)})
        if r.excluded_latex:
            outputs.append({"label": "The equation excludes", "latex": ", ".join(r.excluded_latex)})
        if r.exceptional:
            outputs.append({"label": "Not covered if", "latex": ", ".join(r.exceptional)})
        if r.verification and r.solutions:
            checks.append(
                f"each solution substituted back into the equation: {r.verification.numeric_points_agreed} "
                f"of {r.verification.numeric_points_checked} sampled points agree"
            )
        if r.completeness_note:
            checks.append(f"nothing dropped: {r.completeness_note}")
        if r.assumption_note:
            notes.append(r.assumption_note)
    elif r.message:
        notes.append(r.message)
    inputs = [
        {"label": "Equation", "source": spec["input"]},
        {"label": "Unknown", "source": spec["variable"]},
    ]
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def _ode_inputs(spec: dict[str, Any]) -> list[dict[str, str]]:
    syntax = _syntax(spec)
    lines = [{"label": "Equation", "source": e, "syntax": syntax} for e in spec["equations"]]
    lines += [
        {"label": "Condition", "source": c, "syntax": syntax} for c in spec.get("conditions", [])
    ]
    lines += [
        {"label": "Parameter", "source": f"{k} = {v:g}", "syntax": "text"}
        for k, v in (spec.get("parameters") or {}).items()
    ]
    for key, label in (("independent", "Independent variable"), ("t_end", "End of the interval")):
        if spec.get(key) is not None:
            lines.append({"label": label, "source": str(spec[key]), "syntax": "text"})
    return lines


def run_ode_analyze(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.ode_analyze(
        OdeAnalyzeInput(
            equations=spec["equations"],
            format=_syntax(spec),  # type: ignore[arg-type]
            independent=spec.get("independent"),
            conditions=spec.get("conditions", []),
            assumptions=spec.get("assumptions") or {},
            detail=spec.get("detail", "structure"),
            parameters=spec.get("parameters") or {},
            window=spec.get("window"),
        )
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.structure and r.interpretation:
        s = r.structure
        outputs += [
            {"label": "As Minerva read it", "latex": e} for e in r.interpretation.equations_latex
        ]
        notes.append(
            f"{'System' if s.kind == 'system' else 'Equation'} of order {s.order}, "
            f"{'linear' if s.linear else 'nonlinear'}, {'autonomous' if s.autonomous else 'not autonomous'}; "
            f"{s.state_dimension} first-order states."
        )
        if r.classes:
            notes.append(
                "Classes (SymPy's classification, each solver verifies its own result): "
                + ", ".join(r.classes)
                + "."
            )
        outputs += [
            {"label": "Solved for the highest derivative", "latex": e} for e in r.solved_latex
        ]
        ranked = sorted((m for m in r.methods if m.rank), key=lambda m: m.rank or 0)
        notes += [f"{m.rank}. {m.title}: {m.reason}." for m in ranked]
        notes += [f"Not applicable: {m.title}: {m.reason}." for m in r.methods if not m.applicable]
        q = r.qualitative
        if q is not None and q.status.value == "ok":
            for e in q.equilibria:
                outputs.append(
                    {
                        "label": f"Equilibrium: {e.classification.replace('_', ' ')}",
                        "latex": ", ".join(e.point_latex),
                    }
                )
                if e.eigenvalues_latex:
                    outputs.append(
                        {"label": "Eigenvalues", "latex": ", ".join(e.eigenvalues_latex)}
                    )
                elif e.eigenvalues_numeric:
                    outputs.append(
                        {"label": "Eigenvalues", "text": ", ".join(e.eigenvalues_numeric)}
                    )
            checks.append("Jacobian agrees with numerical differences of the field")
            checks.append(
                "every equilibrium substituted back into the field"
                + (
                    "; the list is complete (polynomial field)"
                    if q.complete
                    else "; the list may be incomplete"
                )
            )
            notes.append(q.local_note)
            notes += q.notes
            if q.direction_field is not None:
                notes.append("A direction field was computed on the window given.")
        elif q is not None:
            notes.append(q.message)
    else:
        notes.append(r.message)
    notes += [n for n in r.notes[1:3]]
    return _example(spec, r.status.value, _ode_inputs(spec), outputs, checks, notes)


def run_ode_solve(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.ode_solve_symbolic(
        OdeSolveInput(
            equations=spec["equations"],
            format=_syntax(spec),  # type: ignore[arg-type]
            independent=spec.get("independent"),
            conditions=spec.get("conditions", []),
            assumptions=spec.get("assumptions") or {},
            method=spec.get("method", "closed_form"),
            order=spec.get("order", 6),
            small_parameter=spec.get("small_parameter"),
            equilibrium=spec.get("equilibrium"),
        )
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    label = {"closed_form": "Solution", "series": "Series", "perturbation": "Expansion"}[
        r.method or "closed_form"
    ]
    for item in r.solutions:
        for equation in item.equations_latex:
            outputs.append({"label": label, "latex": equation})
        v = item.verification
        if item.expansion:
            checks.append(
                f"the truncated result put into the equation as written: its residual falls at the claimed order "
                f"({v.points_agreed} of {v.points_checked} confirmations)"
            )
            notes.append(item.expansion.validity)
            if item.expansion.secular_orders:
                notes.append(
                    "Terms that grow with the variable appear at order "
                    + ", ".join(map(str, item.expansion.secular_orders))
                    + ": valid only on a bounded interval."
                )
        else:
            checks.append(
                f"substituted into the equation as written, derivatives taken numerically: {v.points_agreed} of "
                f"{v.points_checked} points agree; "
                + (
                    "also proved by a symbolic zero test"
                    if v.symbolic_zero
                    else "no symbolic proof (numerically checked)"
                )
            )
            notes.append(
                {
                    "general": "The general solution (independent constants as many as the order).",
                    "particular": "A particular solution: every constant is fixed by the conditions.",
                    "partial": "A family with fewer independent constants than the order.",
                }[item.family]
            )
        for c in item.conditions:
            checks.append(f"condition {'holds' if c.satisfied else 'FAILS'}: {c.latex}")
        if item.restrictions_latex:
            outputs.append(
                {"label": "Valid only where", "latex": ", ".join(item.restrictions_latex)}
            )
    if r.status.value != "ok":
        notes.append(r.message)
    elif r.dropped:
        notes.append(f"{r.dropped} candidate(s) failed their checks and are not shown.")
    return _example(spec, r.status.value, _ode_inputs(spec), outputs, checks, notes)


def run_ode_numeric(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.ode_solve_numerical(
        OdeNumericInput(
            equations=spec["equations"],
            format=_syntax(spec),  # type: ignore[arg-type]
            independent=spec.get("independent"),
            conditions=spec.get("conditions", []),
            assumptions=spec.get("assumptions") or {},
            parameters=spec.get("parameters") or {},
            t_end=spec.get("t_end"),
            kind=spec.get("kind", "ivp"),
            method=spec.get("method", "auto"),
            rtol=spec.get("rtol", 1e-8),
            atol=spec.get("atol", 1e-10),
        )
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.t:
        end = r.t_end
        for series in r.series:
            outputs.append(
                {
                    "label": f"{series.label_latex} at the end of the interval",
                    "text": f"{series.label_latex.replace(chr(92), '')}({end:g}) = {series.values[-1]:.6g}",
                }
            )
        notes.append(f"{r.method_used}: {r.method_reason}; {r.steps} steps.")
        if r.verification:
            for c in r.verification.checks:
                value = (
                    f" ({c.value:.1e} against {c.tolerance:.0e})"
                    if c.value is not None and c.tolerance
                    else ""
                )
                checks.append(f"{c.name.replace('_', ' ')}: {c.status}{value}")
        notes += [
            "A numerical approximation on the interval given: not exact, not valid beyond it; the checks are measurements, not a bound on the error."
        ]
        notes += [f"Warning: {w.replace('_', ' ')}." for w in r.warnings]
    else:
        notes.append(r.message)
    return _example(spec, r.status.value, _ode_inputs(spec), outputs, checks, notes)


def run_sums(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    del kb
    r = engine.sums(
        SumsInput(
            expression=_expr(spec),
            action=spec.get("action", "expand"),
            variable=spec.get("variable"),
            candidate=spec.get("candidate"),
        )
    )
    inputs = [_source_line(spec, "Expression", reading=r.input_latex)]
    if spec.get("candidate"):
        inputs.append({"label": "Answer to check", "source": spec["candidate"]["result"]})
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        outputs.append({"label": "Result", "latex": r.result_latex})
        if r.verification:
            v = r.verification
            checks.append(
                f"{v.instances_agreed} of {v.instances_checked} explicit instances agree; "
                + v.describe()
            )
    elif r.reason:
        notes.append(f"Refused ({r.reason}).")
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_expectation(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    del kb
    r = engine.expectation(
        ExpectationInput(
            expression=_expr(spec),
            random=spec["random"],
            independent=spec.get("independent") or [],
            action=spec.get("action", "expand"),
            candidate=spec.get("candidate"),
        )
    )
    inputs = [
        _source_line(spec, "Expression", reading=r.input_latex),
        {"label": "Random variables", "source": ", ".join(spec["random"])},
    ]
    if spec.get("independent"):
        inputs.append(
            {
                "label": "Independence you stated",
                "source": "; ".join(" and ".join(g) for g in spec["independent"]),
            }
        )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        outputs.append({"label": "Result", "latex": r.result_latex})
        if r.given:
            notes.append(
                "Holds given: "
                + "; ".join(r.given)
                + " (Minerva does not check that this is true)."
            )
        else:
            notes.append("No independence was used.")
        if r.verification:
            checks.append(r.verification.describe())
    elif r.reason:
        notes.append(f"Refused ({r.reason}).")
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def _matrix_latex(rows: list[list[str]]) -> str:
    return "\\begin{pmatrix}" + " \\\\ ".join(" & ".join(row) for row in rows) + "\\end{pmatrix}"


_GREEK_TEX = {
    "alpha": "\\alpha",
    "beta": "\\beta",
    "mu": "\\mu",
    "sigma": "\\sigma",
    "theta": "\\theta",
    "lam": "\\lambda",
}


def run_statistics(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    del kb
    request = StatModelInput(
        family=spec["family"],
        parametrization=spec.get("parametrization"),
        parameters=spec.get("parameters") or {},
        observation=spec.get("observation"),
        sample_size=spec.get("sample_size"),
        iid=spec.get("iid"),
        estimate=spec.get("estimate") or [],
        action=spec["action"],
        prior=spec.get("prior"),
        data=spec.get("data"),
        candidate=spec.get("candidate"),
        assumptions=spec.get("assumptions") or {},
    )
    r = engine.statistical_model(request)
    inputs = [{"label": "Model", "source": f"{spec['family']}, action {spec['action']}"}]
    if spec.get("parameters"):
        inputs.append(
            {
                "label": "Known parameters",
                "source": ", ".join(f"{k} = {v}" for k, v in spec["parameters"].items()),
            }
        )
    if spec.get("sample_size"):
        said = (
            "stated: independent and identically distributed" if spec.get("iid") else "NOT stated"
        )
        inputs.append(
            {
                "label": "Observations",
                "source": f"{spec['sample_size']} observations of {spec.get('observation', 'x')} ({said})",
            }
        )
    if spec.get("data"):
        inputs.append({"label": "Data", "source": ", ".join(spec["data"])})
    if spec.get("prior"):
        pr = spec["prior"]
        inputs.append(
            {
                "label": "Prior (yours)",
                "source": f"{pr['family']}({', '.join(f'{k} = {v}' for k, v in pr['parameters'].items())})",
            }
        )
    if spec.get("estimate"):
        inputs.append({"label": "Parameters", "source": ", ".join(spec["estimate"])})
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.status.value == "ok":
        outputs += [
            {"label": name.capitalize(), "latex": latex} for name, latex in r.moments_latex.items()
        ]
        outputs += [{"label": name, "latex": latex} for name, latex in r.expressions_latex.items()]
        for e in r.estimates:
            outputs.append({"label": f"Estimator of {e.parameter}", "latex": e.latex})
            if e.optimum:
                notes.append(
                    f"{e.parameter}: {e.optimum} maximum ({e.optimum_evidence}); "
                    + (
                        " ".join(f"valid only if ${c.latex}$." for c in e.admissible_if)
                        or "no condition on the data."
                    )
                )
        if r.matrix_latex:
            outputs.append({"label": "Fisher information", "latex": _matrix_latex(r.matrix_latex)})
        if r.posterior:
            outputs.append(
                {
                    "label": f"Posterior ({r.posterior.family})",
                    "latex": ",\\; ".join(
                        f"{_GREEK_TEX.get(k, k)} = {v}"
                        for k, v in r.posterior.parameters_latex.items()
                    ),
                }
            )
        checks += [f"{c.method}: {c.points_agreed} of {c.points_checked}" for c in r.checks]
        if r.premises:
            notes.append("Holds given: " + "; ".join(r.premises) + ".")
    elif r.reason:
        notes.append(f"Refused ({r.reason}).")
    notes.append(r.message)
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_telescoping(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.creative_telescoping(
        TelescopingInput(
            input=spec["input"],
            format=_syntax(spec),  # type: ignore[arg-type]
            assumptions=spec.get("assumptions") or {},
            variable=spec["variable"],
            parameter=spec["parameter"],
            lower=spec.get("lower"),
            upper=spec.get("upper"),
        )
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.operator:
        outputs.append({"label": f"Operator (order {r.operator.order})", "latex": r.operator.latex})
    if r.certificate:
        outputs.append({"label": "Certificate G", "latex": r.certificate.latex})
    if r.certificate_check:
        c = r.certificate_check
        checks.append(
            f"{'proved exactly' if c.symbolic_zero else 'not proved'}: {c.note}; numerical legs "
            f"{c.numeric_points_agreed}/{c.numeric_points_checked} and "
            f"{c.derivative_points_agreed}/{c.derivative_points_checked}"
        )
    for cond in r.conditions:
        checks.append(f"{'established' if cond.established else 'NOT established'}: {cond.how}")
    if r.boundary:
        outputs.append({"label": "Boundary contribution", "latex": r.boundary.contribution_latex})
    if r.equation:
        outputs.append({"label": "Differential equation", "latex": r.equation.latex})
    if r.result_latex:
        outputs.append({"label": "Value", "latex": f"I = {r.result_latex}"})
    if r.value_verification:
        checks.append(r.value_verification.note)
    notes.extend(r.unresolved)
    notes.append(f"{r.outcome}: {r.message}")
    inputs = [_source_line(spec, "Integrand", engine)]
    return _example(spec, r.status.value, inputs, outputs, checks, notes)


def run_integrate(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.integrate(
        _expr(spec),
        spec.get("variable", "x"),
        spec.get("lower"),
        spec.get("upper"),
        spec.get("strategy"),
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    notes: list[str] = []
    if r.result_latex:
        outputs.append(
            {"label": "Value" if r.definite else "Antiderivative", "latex": r.result_latex}
        )
    if r.result_kind == "numeric" and r.numeric:
        notes.append(
            f"Numerical estimate {r.numeric} (estimated error {r.numeric_error}); not a formula."
        )
    s = r.strategy
    if s is not None:
        if s.technique and s.technique != "direct":
            notes.append(
                f"Technique: {s.technique.replace('_', ' ')} ({s.why}); verification: {s.verification}."
            )
            outputs += [
                {"label": step.what.capitalize(), "latex": step.latex} for step in s.steps[:-1]
            ]
        for a in s.attempts:
            if a.outcome in ("blocked", "failed_verification"):
                notes.append(
                    f"{a.technique.replace('_', ' ')}: {a.outcome}"
                    + (f" ({a.detail})" if a.detail else "")
                )
    if r.quadrature and r.quadrature.passed:
        checks.append(
            f"quadrature at {r.quadrature.points_agreed}/{r.quadrature.points_checked} parameter points"
        )
    if r.verification and r.verification.passed:
        checks.append(
            f"numerical derivative at {r.verification.points_agreed}/{r.verification.points_checked} points"
            + ("; symbolic derivative equals the integrand" if r.verification.symbolic_zero else "")
        )
    notes.append(r.message)
    return _example(
        spec, r.status.value, [_source_line(spec, "Integrand", engine)], outputs, checks, notes
    )


RUNNERS = {
    "ode_analyze": run_ode_analyze,
    "ode_solve": run_ode_solve,
    "ode_numeric": run_ode_numeric,
    "simplify": run_simplify,
    "differentiate": run_differentiate,
    "time_derivative": run_time_derivative,
    "series": run_series,
    "approximation": run_approximation,
    "matrix": run_matrix,
    "sums": run_sums,
    "expectation": run_expectation,
    "statistics": run_statistics,
    "geometry": run_geometry,
    "vector": run_vector,
    "electromagnetism": run_electromagnetism,
    "distribution": run_distribution,
    "statmech": run_statmech,
    "identities": run_identities,
    "tensor": run_tensor,
    "solve": run_solve,
    "lagrangian": run_lagrangian,
    "variation": run_variation,
    "invariants": run_invariants,
    "discovery": run_discovery,
    "telescoping": run_telescoping,
    "integrate": run_integrate,
    "entry": run_entry,
    "read_formulas": run_read_formulas,
    "substitute": run_substitute,
    "rewrite": run_rewrite,
    "isolate": run_isolate,
}


EXAMPLE_TIMEOUT_SECONDS = 120.0


def build() -> dict[str, Any]:
    spec = yaml.safe_load(SOURCE.read_text())
    # Generous on purpose: a result must not depend on how fast the machine is. With the default 10 s, five
    # examples near their limit turned into `resource_limit` on a slow or busy runner and the generated files
    # no longer matched. A call that really cannot finish is still stopped by its own size and step caps.
    engine = Engine(Limits(timeout_seconds=EXAMPLE_TIMEOUT_SECONDS))
    kb = KnowledgeBase.load()
    seen: set[str] = set()
    capabilities: list[dict[str, Any]] = []
    groups = [
        {"id": g["id"], "title": g["title"], "summary": " ".join(g["summary"].split())}
        for g in spec["groups"]
    ]
    group_ids = [g["id"] for g in groups]
    if len(set(group_ids)) != len(group_ids):
        raise SystemExit("duplicate group id")
    for cap in spec["capabilities"]:
        if cap.get("group") not in group_ids:
            raise SystemExit(
                f"{cap['id']}: group must be one of {', '.join(group_ids)} (got {cap.get('group')!r})"
            )
    for gid in group_ids:
        if gid in {c["id"] for c in spec["capabilities"]}:
            raise SystemExit(f"group id {gid} clashes with a capability id (page anchors)")
        if not any(c["group"] == gid for c in spec["capabilities"]):
            raise SystemExit(f"group {gid} has no capability")
    for cap in spec["capabilities"]:
        if cap["id"] in seen:
            raise SystemExit(f"duplicate capability id {cap['id']}")
        seen.add(cap["id"])
        if cap["status"] == "planned" and cap.get("examples"):
            raise SystemExit(f"{cap['id']}: planned capabilities carry no examples")
        examples = []
        for ex in cap.get("examples", []):
            if ex["id"] in seen:
                raise SystemExit(f"duplicate example id {ex['id']}")
            seen.add(ex["id"])
            examples.append(RUNNERS[ex["op"]](ex, engine, kb))
        capabilities.append(
            {
                "id": cap["id"],
                "title": cap["title"],
                "status": cap["status"],
                "group": cap["group"],
                "where": [WHERE[w] for w in cap.get("where", [])],
                "needs_model": bool(cap.get("needs_model", False)),
                "summary": " ".join(cap["summary"].split()),
                "limits": [" ".join(item.split()) for item in cap.get("limits", [])],
                "examples": examples,
            }
        )
    return {
        "handbook_entries": len(kb.entries),
        "groups": groups,
        "capabilities": capabilities,
    }


def _tex(latex: str) -> str:
    return f"${latex}$"


def _render_capability(cap: dict[str, Any]) -> list[str]:
    lines = [f"### {cap['title']}", ""]
    where = ", ".join(cap["where"])
    extra = " Needs a model you connect." if cap["needs_model"] else ""
    lines += [f"*Available in:* {where}.{extra}", "", cap["summary"], ""]
    if cap["limits"]:
        lines += ["**Limits**", ""] + [f"- {item}" for item in cap["limits"]] + [""]
    for ex in cap["examples"]:
        lines += [f"#### {ex['title']}", ""]
        for item in ex["input"]:
            value = f"`{item['source']}`" if "source" in item else item["text"]
            if item.get("syntax"):
                value += " (plain text)" if item["syntax"] == "text" else " (LaTeX)"
            if item.get("latex"):
                value += f", read as {_tex(item['latex'])}"
            lines.append(f"- **{item['label']}:** {value}")
        if ex["assumptions"]:
            lines.append(f"- **Assumptions:** {ex['assumptions']}")
        lines.append(f"- **Status:** `{ex['status']}`")
        for out in ex["outputs"]:
            parts = []
            if "latex" in out:
                parts.append(_tex(out["latex"]))
            if "text" in out:
                parts.append(out["text"])
            lines.append(f"- **{out['label']}:** {' — '.join(parts)}")
        for check in ex["checks"]:
            lines.append(f"- **Check:** {check}")
        for note in ex["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    return lines


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Capabilities",
        "",
        "What Minerva can do today, with examples.",
        "",
        "<!-- Generated by scripts/capabilities.py from capabilities.yaml. Do not edit by hand. -->",
        "",
        "Every example below was run through the real engine and handbook by `scripts/capabilities.py`;",
        "the prose comes from [capabilities.yaml](capabilities.yaml). CI fails if this file is out of date,",
        "so a change in behaviour appears as a diff here. Regenerate with",
        "`uv run python scripts/capabilities.py`.",
        "",
        "**Available in** says which interface exposes a capability: the *workbench* forms, the *handbook pages*,",
        "the *chat* (needs a model you connect, see [self-hosting.md](self-hosting.md)) or the *HTTP API*.",
        "",
    ]
    available = [c for c in data["capabilities"] if c["status"] == "available"]
    planned = [c for c in data["capabilities"] if c["status"] == "planned"]
    for group in data["groups"]:
        members = [c for c in available if c["group"] == group["id"]]
        if not members:
            continue
        lines += [f"## {group['title']}", "", group["summary"], ""]
        for cap in members:
            lines += _render_capability(cap)
    lines += [
        "## Planned",
        "",
        "Not implemented. Listed so that nobody mistakes them for features.",
        "",
    ]
    titles = {g["id"]: g["title"] for g in data["groups"]}
    for cap in planned:
        lines.append(f"- **{cap['title']}** ({titles[cap['group']]}): {cap['summary']}")
    lines.append("")
    return "\n".join(lines)


def render_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def stale_outputs(data: dict[str, Any]) -> list[str]:
    """The generated files that do not match `data` (repository-relative paths)."""
    outputs = {MARKDOWN: render_markdown(data), JSON_OUT: render_json(data)}
    return [
        str(path.relative_to(ROOT))
        for path, text in outputs.items()
        if not path.exists() or path.read_text() != text
    ]


def main(argv: list[str]) -> int:
    data = build()
    if "--check" in argv:
        stale = stale_outputs(data)
        if stale:
            print(
                "out of date: " + ", ".join(stale) + "\nrun: uv run python scripts/capabilities.py",
                file=sys.stderr,
            )
            return 1
        print("capabilities are up to date")
        return 0
    for path, text in {MARKDOWN: render_markdown(data), JSON_OUT: render_json(data)}.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
