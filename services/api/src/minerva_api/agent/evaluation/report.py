"""Aggregate case results into metrics, a JSON document, a Markdown report and a comparison.

The JSON is the record (schema-versioned, one results list per model); the Markdown is a view of it.
Reports hold the fixed benchmark prompts' results only: no host name, no user data.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .cases import CATEGORIES
from .score import CaseResult

SCHEMA = 1

# The targets of the benchmark. They are goals, not results: a report shows both.
TARGETS: dict[str, tuple[str, float]] = {
    "tool_selection_accuracy": (">=", 0.95),
    "valid_arguments_rate": (">=", 0.99),
    "completion_rate": (">=", 0.90),
    "false_verified": ("==", 0),
}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _dist(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Metrics over the scored (non-informational) runs; informational ones are listed apart."""
    scored = [r for r in results if not r.informational]
    calls = sum(r.calls for r in scored)
    per_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        rows = [r for r in scored if r.category == category]
        if rows:
            passed = sum(r.passed for r in rows)
            per_category[category] = {
                "runs": len(rows),
                "passed": passed,
                "rate": passed / len(rows),
                "tool_selection": sum(r.tool_selected for r in rows) / len(rows),
            }
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for r in scored:
        confusion[r.expected_tool][r.first_tool or "none"] += 1

    by_case: dict[str, list[bool]] = defaultdict(list)
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    for r in scored:
        by_case[r.case_id].append(r.passed)
        by_repeat[r.repeat].append(r.passed)
    flaky = sorted(c for c, v in by_case.items() if 0 < sum(v) < len(v))
    repeat_counts = [sum(v) for _, v in sorted(by_repeat.items())]

    tokens_in = [r.tokens_in for r in scored if r.tokens_in is not None]
    tokens_out = [r.tokens_out for r in scored if r.tokens_out is not None]
    costs = [r.cost_usd for r in scored if r.cost_usd is not None]
    return {
        "runs": len(scored),
        "cases": len(by_case),
        "informational_runs": len(results) - len(scored),
        "passed": sum(r.passed for r in scored),
        "pass_rate": _rate(sum(r.passed for r in scored), len(scored)),
        "tool_selection_accuracy": _rate(sum(r.tool_selected for r in scored), len(scored)),
        "tool_reached_rate": _rate(sum(r.tool_reached for r in scored), len(scored)),
        "valid_arguments_rate": (1 - sum(r.invalid_calls for r in scored) / calls)
        if calls
        else None,
        "completion_rate": _rate(sum(r.completed for r in scored), len(scored)),
        "false_verified": sum(r.false_verified for r in scored),
        "checker_false_positives": sum(r.checker_false_positive for r in scored),
        "unchecked_formulas": sum(r.unchecked for r in scored),
        "contradicted_formulas": sum(r.contradicted for r in scored),
        "tool_calls": calls,
        "unnecessary_calls": sum(r.unnecessary_calls for r in scored),
        "repeated_calls": sum(r.repeated_calls for r in scored),
        "latency_seconds": _dist([r.seconds for r in scored]),
        "llm_ms": _dist([r.llm_ms for r in scored if r.llm_calls]),
        "tool_ms": _dist([r.tool_ms for r in scored if r.calls]),
        "tool_failures": sum(r.tool_failures for r in scored),
        "unverified_results": sum(r.unverified_results for r in scored),
        "cost_usd": sum(costs) if costs else None,
        "first_token_ms": _dist([r.first_token_ms for r in scored if r.first_token_ms is not None]),
        "rate_limited": sum(r.error_kind == "rate_limit" for r in scored),
        "timeouts": sum(r.error_kind == "timeout" for r in scored),
        "tokens_in": sum(tokens_in) if tokens_in else None,
        "tokens_out": sum(tokens_out) if tokens_out else None,
        "per_category": per_category,
        "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
        "failure_classes": dict(
            Counter(str(r.failure_class) for r in scored if not r.passed and r.failure_class)
        ),
        "unclassified_failures": sum(1 for r in scored if not r.passed and not r.failure_class),
        "flaky_cases": flaky,
        "pass_counts_per_repeat": repeat_counts,
        "pass_count_stdev": statistics.pstdev(repeat_counts) if len(repeat_counts) > 1 else 0.0,
    }


def build_document(
    meta: dict[str, Any], results_by_model: dict[str, list[CaseResult] | None]
) -> dict[str, Any]:
    """The JSON record. A model that could not be used maps to None and is listed as unusable."""
    models: dict[str, Any] = {}
    for model, results in results_by_model.items():
        if results is None:
            models[model] = {"usable": False, "summary": None, "results": []}
        else:
            models[model] = {
                "usable": True,
                "summary": summarise(results),
                "results": [r.model_dump(mode="json") for r in results],
            }
    return {"schema": SCHEMA, "meta": meta, "models": models}


# --------------------------------------------------------------------------------------- markdown
def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _meets(metric: str, value: float | None) -> str:
    if value is None:
        return "not measured"
    op, target = TARGETS[metric]
    ok = value >= target if op == ">=" else value == target
    return "met" if ok else "NOT met"


def render_markdown(doc: dict[str, Any]) -> str:
    meta = doc["meta"]
    lines = [
        f"# Agent benchmark: {meta.get('label', 'run')}",
        "",
        f"Generated {meta.get('date', '?')} by `scripts/eval_models.py`. Case set `{meta.get('cases_digest', '?')}` "
        f"({meta.get('case_count', '?')} cases, {meta.get('repeats', 1)} repeat(s), seed {meta.get('seed', 'none')}, "
        f"temperature {meta.get('temperature', '?')}). Code `{meta.get('git_sha', '?')}`, "
        f"prompt `{meta.get('prompt_sha', '?')}`. The targets are goals, not results.",
        "",
        "## Headline",
        "",
        "| Model | Passed | Tool selection | Valid arguments | Completion | Unsupported shown as verified | Unnecessary calls | Median s | p95 s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model, entry in doc["models"].items():
        s = entry["summary"]
        if not s:
            lines.append(f"| {model} | not usable | | | | | | | |")
            continue
        lat = s["latency_seconds"]
        lines.append(
            f"| {model} | {s['passed']}/{s['runs']} ({_pct(s['pass_rate'])}) "
            f"| {_pct(s['tool_selection_accuracy'])} | {_pct(s['valid_arguments_rate'])} "
            f"| {_pct(s['completion_rate'])} | {s['false_verified']} "
            f"| {s['unnecessary_calls']} of {s['tool_calls']} | {_num(lat['median'])} | {_num(lat['p95'])} |"
        )
    lines += [
        "",
        "Targets: tool selection >= 95%, valid arguments >= 99%, completion >= 90%, "
        "unsupported formulas shown as verified = 0.",
        "",
    ]
    for model, entry in doc["models"].items():
        s = entry["summary"]
        if not s:
            continue
        lines += [f"## {model}", ""]
        lines += [
            "| Target | Value | Status |",
            "| --- | --- | --- |",
        ]
        for metric in TARGETS:
            value = s[metric]
            shown = str(value) if metric == "false_verified" else _pct(value)
            lines.append(f"| {metric.replace('_', ' ')} | {shown} | {_meets(metric, value)} |")
        lines += [
            "",
            "### By category",
            "",
            "| Category | Passed | Tool selection |",
            "| --- | --- | --- |",
        ]
        for category, row in s["per_category"].items():
            lines.append(
                f"| {category} | {row['passed']}/{row['runs']} ({_pct(row['rate'])}) | {_pct(row['tool_selection'])} |"
            )
        lines += ["", "### Tool selection (expected rows, first tool called columns)", ""]
        columns = sorted({c for row in s["confusion"].values() for c in row})
        lines += [
            "| expected \\ called | " + " | ".join(columns) + " |",
            "| --- |" + " --- |" * len(columns),
        ]
        for expected, row in s["confusion"].items():
            lines.append(
                f"| {expected} | " + " | ".join(str(row.get(c, "")) for c in columns) + " |"
            )
        lat, llm, tool = s["latency_seconds"], s["llm_ms"], s["tool_ms"]
        lines += [
            "",
            "### Latency",
            "",
            "| | median | p95 | max |",
            "| --- | --- | --- | --- |",
            f"| whole turn (s) | {_num(lat['median'])} | {_num(lat['p95'])} | {_num(lat['max'])} |",
            f"| model calls (ms) | {_num(llm['median'], 0)} | {_num(llm['p95'], 0)} | {_num(llm['max'], 0)} |",
            f"| tool calls (ms) | {_num(tool['median'], 0)} | {_num(tool['p95'], 0)} | {_num(tool['max'], 0)} |",
            "",
            f"Tokens in/out: {s['tokens_in'] if s['tokens_in'] is not None else 'not reported by the host'} / "
            f"{s['tokens_out'] if s['tokens_out'] is not None else 'not reported by the host'}.",
            "",
            _extras_line(s),
            "",
            "### Failures by class",
            "",
        ]
        if s["failure_classes"]:
            lines += ["| Class | Runs |", "| --- | --- |"]
            lines += [
                f"| {c} | {n} |"
                for c, n in sorted(s["failure_classes"].items(), key=lambda kv: -kv[1])
            ]
        else:
            lines.append("No failures.")
        lines += [
            "",
            f"Formulas the model wrote: {s['unchecked_formulas']} unchecked, {s['contradicted_formulas']} "
            f"contradicted, {s['checker_false_positives']} correct ones marked contradicted. "
            f"Repeated identical calls: {s['repeated_calls']}.",
        ]
        if s["flaky_cases"]:
            lines += [
                "",
                f"Unstable across repeats ({_num(s['pass_count_stdev'], 2)} std of passes per repeat): "
                + ", ".join(f"`{c}`" for c in s["flaky_cases"])
                + ".",
            ]
        failed = [r for r in entry["results"] if not r["passed"] and not r["informational"]]
        if failed:
            lines += ["", "### Failed runs", ""]
            seen: set[str] = set()
            for r in failed:
                if r["case_id"] in seen:
                    continue
                seen.add(r["case_id"])
                lines.append(f"- `{r['case_id']}` ({r['failure_class']}): {'; '.join(r['notes'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- comparison
_COMPARED = (
    ("pass_rate", "pass rate", True),
    ("tool_selection_accuracy", "tool selection", True),
    ("valid_arguments_rate", "valid arguments", True),
    ("completion_rate", "completion", True),
    ("false_verified", "unsupported shown as verified", False),
    ("unnecessary_calls", "unnecessary calls", False),
    ("repeated_calls", "repeated calls", False),
    ("checker_false_positives", "checker false positives", False),
)


def _pass_rates(entry: dict[str, Any]) -> dict[str, float]:
    by_case: dict[str, list[bool]] = defaultdict(list)
    for r in entry["results"]:
        if not r["informational"]:
            by_case[r["case_id"]].append(bool(r["passed"]))
    return {c: sum(v) / len(v) for c, v in by_case.items()}


def compare(base: dict[str, Any], new: dict[str, Any]) -> str:
    """Baseline against new, per model present in both (a single model on each side is paired)."""
    names = [m for m in base["models"] if m in new["models"]]
    if not names and len(base["models"]) == 1 and len(new["models"]) == 1:
        names = [next(iter(base["models"]))]
        pairs = [(names[0], next(iter(new["models"])))]
    else:
        pairs = [(m, m) for m in names]
    lines = [
        "# Baseline against new",
        "",
        f"Baseline `{base['meta'].get('label', '?')}` (code `{base['meta'].get('git_sha', '?')}`, "
        f"cases `{base['meta'].get('cases_digest', '?')}`) against `{new['meta'].get('label', '?')}` "
        f"(code `{new['meta'].get('git_sha', '?')}`, cases `{new['meta'].get('cases_digest', '?')}`).",
    ]
    if base["meta"].get("cases_digest") != new["meta"].get("cases_digest"):
        lines += ["", "**The case sets differ: only cases present in both are compared per case.**"]
    if not pairs:
        return "\n".join([*lines, "", "No model appears in both reports."]) + "\n"
    for old_name, new_name in pairs:
        old, cur = base["models"][old_name], new["models"][new_name]
        title = old_name if old_name == new_name else f"{old_name} against {new_name}"
        lines += ["", f"## {title}", ""]
        if not old["summary"] or not cur["summary"]:
            lines.append("One side was not usable.")
            continue
        a, b = old["summary"], cur["summary"]
        lines += ["| Metric | Baseline | New | Change |", "| --- | --- | --- | --- |"]
        for key, label, is_rate in _COMPARED:
            va, vb = a[key], b[key]
            if va is None or vb is None:
                lines.append(f"| {label} | {va} | {vb} | |")
            elif is_rate:
                lines.append(
                    f"| {label} | {_pct(va)} | {_pct(vb)} | {100 * (vb - va):+.1f} points |"
                )
            else:
                lines.append(f"| {label} | {va} | {vb} | {vb - va:+d} |")
        for key, label in (
            ("latency_seconds", "turn latency (s)"),
            ("llm_ms", "model call (ms)"),
            ("tool_ms", "tool call (ms)"),
        ):
            for stat in ("median", "p95"):
                va, vb = a[key][stat], b[key][stat]
                digits = 1 if key == "latency_seconds" else 0
                lines.append(f"| {label} {stat} | {_num(va, digits)} | {_num(vb, digits)} | |")
        lines += ["", "### By category", "", "| Category | Baseline | New |", "| --- | --- | --- |"]
        for category in CATEGORIES:
            ca, cb = a["per_category"].get(category), b["per_category"].get(category)
            if ca or cb:
                lines.append(
                    f"| {category} | {_pct(ca['rate']) if ca else '-'} | {_pct(cb['rate']) if cb else '-'} |"
                )
        classes = sorted(set(a["failure_classes"]) | set(b["failure_classes"]))
        if classes:
            lines += [
                "",
                "### Failures by class",
                "",
                "| Class | Baseline | New |",
                "| --- | --- | --- |",
            ]
            lines += [
                f"| {c} | {a['failure_classes'].get(c, 0)} | {b['failure_classes'].get(c, 0)} |"
                for c in classes
            ]
        ra, rb = _pass_rates(old), _pass_rates(cur)
        better = sorted(c for c in ra.keys() & rb.keys() if rb[c] > ra[c])
        worse = sorted(c for c in ra.keys() & rb.keys() if rb[c] < ra[c])
        lines += [
            "",
            f"Cases that improved ({len(better)}): "
            + (", ".join(f"`{c}`" for c in better) or "none"),
        ]
        lines += [
            "",
            f"Cases that got worse ({len(worse)}): "
            + (", ".join(f"`{c}`" for c in worse) or "none"),
        ]
        lines += [
            "",
            "Differences of one case between single runs are within the run-to-run noise of a sampling "
            "model; use repeats before reading a change into them.",
        ]
    return "\n".join(lines) + "\n"


def _extras_line(s: dict[str, Any]) -> str:
    """Provider-facing numbers; read defensively so a summary saved by an older version still renders."""
    first = (s.get("first_token_ms") or {}).get("median")
    cost = s.get("cost_usd")
    return (
        f"First output after {_num(first, 0)} ms (median) · tool calls that errored: "
        f"{s.get('tool_failures', 'n/a')} · tool results not verified: {s.get('unverified_results', 'n/a')} · "
        f"provider rate limits / timeouts: {s.get('rate_limited', 'n/a')} / {s.get('timeouts', 'n/a')} · "
        f"cost: {'n/a' if cost is None else f'{cost:.4f} USD'}."
    )
