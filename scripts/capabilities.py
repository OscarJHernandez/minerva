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
from minerva_engine import Engine
from minerva_engine.differentiate import DiffVariable
from minerva_engine.models import ExpressionInput, TensorInput
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.identities import find_identities

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


def run_simplify(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.simplify(
        ExpressionInput(input=spec["input"], assumptions=spec.get("assumptions") or {})
    )
    outputs: list[dict[str, str]] = []
    checks: list[str] = []
    if r.status.value == "ok" and r.result_latex:
        outputs.append({"label": "Result", "latex": r.result_latex})
        if r.verification:
            checks.append(_sample_check(r.verification))
    elif r.status.value == "unchanged" and r.input_latex:
        # Nothing simpler exists; under declared assumptions the input may still be read differently.
        outputs.append({"label": "Read as", "latex": r.input_latex})
    return _example(
        spec,
        r.status.value,
        [{"label": "Input", "source": spec["input"]}],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_differentiate(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = engine.differentiate(
        ExpressionInput(input=spec["input"], assumptions=spec.get("assumptions") or {}),
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
        [{"label": "Input", "source": spec["input"]}],
        outputs,
        checks,
        [r.message] if r.message else [],
    )


def run_identities(spec: dict[str, Any], engine: Engine, kb: KnowledgeBase) -> dict[str, Any]:
    r = find_identities(
        kb,
        engine,
        ExpressionInput(input=spec["input"], assumptions=spec.get("assumptions") or {}),
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
    return _example(
        spec, status, [{"label": "Input", "source": spec["input"]}], outputs, checks, notes
    )


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
    r = engine.analyze_lagrangian(spec["input"], spec["coordinates"])
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
            {"label": "Lagrangian", "source": spec["input"]},
            {"label": "Coordinates", "text": ", ".join(spec["coordinates"])},
        ],
        outputs,
        checks,
        notes,
    )


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


RUNNERS = {
    "simplify": run_simplify,
    "differentiate": run_differentiate,
    "identities": run_identities,
    "tensor": run_tensor,
    "solve": run_solve,
    "lagrangian": run_lagrangian,
    "entry": run_entry,
    "read_formulas": run_read_formulas,
}


def build() -> dict[str, Any]:
    spec = yaml.safe_load(SOURCE.read_text())
    engine = Engine()
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


def main(argv: list[str]) -> int:
    data = build()
    outputs = {MARKDOWN: render_markdown(data), JSON_OUT: render_json(data)}
    if "--check" in argv:
        stale = [
            str(path.relative_to(ROOT))
            for path, text in outputs.items()
            if not path.exists() or path.read_text() != text
        ]
        if stale:
            print(
                "out of date: " + ", ".join(stale) + "\nrun: uv run python scripts/capabilities.py",
                file=sys.stderr,
            )
            return 1
        print("capabilities are up to date")
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
