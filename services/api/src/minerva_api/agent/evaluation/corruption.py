"""Corruption test: every known model mistake, applied to every reference call, must be harmless.

The benchmark has about 130 reference calls, each checked against the engine (`reference.check_reference`).
This module takes each one, applies one *corruption* (a mistake models are known to make: a velocity written
`xdot`, a formula wrapped in `$`, markdown-escaped operators, `L =` in front...), runs it through the same
pipeline a model's call takes (`prepare_arguments`, then `execute`), and compares it with the clean call.
Each corrupted call must end in exactly one of:

``same``                      the result is the clean call's result (after cleaning, or after the engine read
                              the spelling);
``diagnosed``                 a failure that carries a typed reason (an `InputCode` or schema field), which is
                              what lets code, not another model call, decide what happens next;

and anything else is a defect:

``different_verified_result`` a verified result that differs from the clean one: a normalisation changed the
                              mathematics (must be zero, always, and is reported first);
``untyped_failure``           a failure with no typed reason: the model gets only prose.

The operators are the inverses of the normalisations (`notation.transport.DAMAGE` and the derivative
spellings), so the two lists cannot drift apart. A corruption whose arguments are *identical* to the clean
call's after cleaning needs no engine run (the same function of the same arguments): it is counted as
``same_after_cleaning``; `full=True` runs them all anyway. A recovery rate is a measurement, never a
verification: the clean reference calls are the benchmark's ground truth, and nothing here relaxes a check.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minerva_engine.notation.models import Normalization
from minerva_engine.notation.transport import DAMAGE

from ..arguments import repair_arguments
from ..mathinput import MATH_FIELDS, MATH_LIST_FIELDS, prepare_arguments
from ..tools import ToolContext, ToolOutput, execute
from .cases import load_suite

Arguments = dict[str, Any]
Operator = Callable[[str, Arguments], Arguments | None]

# Tools that declare dependents (coordinates): only there is `xdot` the velocity of `x`.
DEPENDENT_TOOLS = frozenset(
    {
        "analyze_lagrangian",
        "search_invariants",
        "time_derivative",
        "expand_motion",
        "perturbation_solve",
        "integrate_motion",
        "find_conserved_quantities",
    }
)
# Tools with no reference call, and why. A registered tool that is neither covered nor listed fails a test.
EXEMPT: dict[str, str] = {
    "geometry": "its inputs are nested grids, not a formula string; corrupted by tests of its own parser",
    "load_tools": "takes group names, no mathematics",
    "search_invariants": "no benchmark case yet; shares the Lagrangian reader with analyze_lagrangian",
    "sums": "no benchmark case yet; shares the parser with expectation",
}

_VELOCITY = re.compile(r"\b([A-Za-z][A-Za-z0-9]*)_dot\b")
_LATEX_DOT = re.compile(r"\\dot\{([A-Za-z]+)\}")


def _map_fields(arguments: Arguments, change: Callable[[str, str], str | None]) -> Arguments | None:
    """A copy of `arguments` with `change(field, text)` applied to each mathematical string; None if no field
    changed (the corruption does not apply to this call)."""
    out = copy.deepcopy(arguments)
    changed = False
    for key in MATH_FIELDS:
        if isinstance(out.get(key), str):
            new = change(key, out[key])
            if new is not None and new != out[key]:
                out[key], changed = new, True
    for key in MATH_LIST_FIELDS:
        if isinstance(out.get(key), list) and all(isinstance(v, str) for v in out[key]):
            items = [change(key, v) or v for v in out[key]]
            if items != out[key]:
                out[key], changed = items, True
    return out if changed else None


def _damage(name: str) -> Operator:
    def apply(tool: str, arguments: Arguments) -> Arguments | None:
        function = DAMAGE[name]
        return _map_fields(arguments, lambda _key, text: function(text))

    return apply


def _assignment(tool: str, arguments: Arguments) -> Arguments | None:
    letters = {"lagrangian": "L", "hamiltonian": "H"}
    return _map_fields(
        arguments, lambda key, text: f"{letters[key]} = {text}" if key in letters else None
    )


def _velocity(rewrite: Callable[[re.Match[str]], str]) -> Operator:
    def apply(tool: str, arguments: Arguments) -> Arguments | None:
        if tool not in DEPENDENT_TOOLS or arguments.get("format") == "latex":
            return None
        return _map_fields(arguments, lambda _key, text: _VELOCITY.sub(rewrite, text))

    return apply


def _latex_dot_as_letters(tool: str, arguments: Arguments) -> Arguments | None:
    """`\\dot{x}` -> `xdot` in a LaTeX field: a product of letters there, which must be refused, not read."""
    if tool not in DEPENDENT_TOOLS:
        return None

    def spell(text: str) -> str | None:
        if "\\" not in text:
            return None

        def replace(match: re.Match[str]) -> str:
            glued = match.start() > 0 and text[match.start() - 1].isalpha()
            return (" " if glued else "") + f"{match.group(1)}dot"

        return _LATEX_DOT.sub(replace, text)

    return _map_fields(arguments, lambda _key, text: spell(text))


def _shaped(
    tool: str, arguments: Arguments, change: Callable[[str, Any, Any], Any]
) -> Arguments | None:
    """Apply `change(field, annotation, value)` to each argument of a known tool; None if nothing changed."""
    from ..tools import TOOLS

    known = TOOLS.get(tool)
    if known is None:
        return None
    out = copy.deepcopy(arguments)
    changed = False
    for key, value in arguments.items():
        info = known.args_model.model_fields.get(key)
        if info is None:
            continue
        new = change(key, info.annotation, value)
        if new is not None:
            out[key], changed = new, True
    return out if changed else None


def _list_as_string(tool: str, arguments: Arguments) -> Arguments | None:
    def change(key: str, annotation: Any, value: Any) -> Any:
        names = isinstance(value, list) and value and all(isinstance(v, str) for v in value)
        if (
            names
            and key not in MATH_LIST_FIELDS
            and all(re.fullmatch(r"[A-Za-z]\w*", v) for v in value)
        ):
            return ", ".join(value)
        return None

    return _shaped(tool, arguments, change)


def _json_as_string(tool: str, arguments: Arguments) -> Arguments | None:
    def change(key: str, annotation: Any, value: Any) -> Any:
        if isinstance(value, dict | list) and value and key not in MATH_LIST_FIELDS:
            return json.dumps(value)
        return None

    return _shaped(tool, arguments, change)


def _number_as_string(tool: str, arguments: Arguments) -> Arguments | None:
    def change(key: str, annotation: Any, value: Any) -> Any:
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and annotation in (int, float)
        ):
            return str(value)
        return None

    return _shaped(tool, arguments, change)


def _formula_alias(tool: str, arguments: Arguments) -> Arguments | None:
    from ..tools import TOOLS

    known = TOOLS.get(tool)
    if known is None:
        return None
    formula = [f for f in known.args_model.model_fields if f in MATH_FIELDS]
    if (
        len(formula) != 1
        or formula[0] not in arguments
        or "expression" in known.args_model.model_fields
    ):
        return None
    out = dict(arguments)
    out["expression"] = out.pop(formula[0])
    return out


OPERATORS: dict[str, Operator] = {
    **{f"transport:{name}": _damage(name) for name in sorted(DAMAGE)},
    "shape:assignment_prefix": _assignment,
    "shape:list_as_string": _list_as_string,
    "shape:json_as_string": _json_as_string,
    "shape:number_as_string": _number_as_string,
    "shape:formula_alias": _formula_alias,
    "derivative:joined": _velocity(lambda m: f"{m.group(1)}dot"),
    "derivative:leibniz": _velocity(lambda m: f"d{m.group(1)}_dt"),
    "derivative:spaced": _velocity(lambda m: f"{m.group(1)} dot"),
    "derivative:upper": _velocity(lambda m: f"{m.group(1)}_DOT"),
    "derivative:latex_letter_run": _latex_dot_as_letters,
}

SAME = "same"
SAME_AFTER_CLEANING = "same_after_cleaning"
DIAGNOSED = "diagnosed"
DIFFERENT = "different_verified_result"
UNTYPED = "untyped_failure"
OUTCOMES = (DIFFERENT, UNTYPED, SAME, SAME_AFTER_CLEANING, DIAGNOSED)
DEFECTS = (DIFFERENT, UNTYPED)


@dataclass(frozen=True)
class Reference:
    case: str
    tool: str
    arguments: Arguments


def references() -> list[Reference]:
    """Every reference call of the benchmark that needs no earlier result."""
    out: list[Reference] = []
    for case in load_suite("all"):
        for turn in case.turns():
            ref = turn.reference
            if ref is not None and "@" not in str(ref.arguments):
                out.append(Reference(turn.id, ref.tool, ref.arguments))
    return sorted(out, key=lambda r: (r.tool, r.case))


def pipeline_arguments(
    tool: str, arguments: Arguments
) -> tuple[Arguments, tuple[Normalization, ...]]:
    """The arguments after what the loop does to them before a call is keyed: shape repair, then cleaning."""
    from ..tools import TOOLS

    known = TOOLS.get(tool)
    shaped, shape_notes = (
        repair_arguments(known.args_model, arguments) if known is not None else (arguments, ())
    )
    cleaned, notes = prepare_arguments(tool, shaped)
    return cleaned, (*shape_notes, *notes)


def run_call(
    tool: str, arguments: Arguments, ctx: ToolContext, *, clean: bool = True
) -> ToolOutput:
    """What the loop does with a model's call, without the model: clean the arguments, then execute.

    `clean=False` skips `prepare_arguments` (the engine still reads what it reads): the measurement of what
    the cleaning layer adds."""
    cleaned, notes = pipeline_arguments(tool, arguments) if clean else (arguments, ())
    ctx.input_notes = notes
    try:
        return execute(tool, cleaned, ctx)
    finally:
        ctx.input_notes = ()


def _signature(output: ToolOutput) -> tuple[str, tuple[str, ...]]:
    return output.status, tuple(sorted("".join(f.split()) for f in output.known_latex))


def _typed(output: ToolOutput) -> bool:
    return bool(output.reason_code) or bool(output.invalid_fields)


def classify(clean: ToolOutput, corrupted: ToolOutput) -> str:
    if _signature(clean) == _signature(corrupted):
        return SAME
    if corrupted.status != "ok" and _typed(corrupted):
        return DIAGNOSED
    if corrupted.status == "ok" or (clean.status != "ok" and corrupted.known_latex):
        return DIFFERENT
    return UNTYPED


def _tally() -> dict[str, int]:
    return dict.fromkeys(OUTCOMES, 0)


def run_corruptions(
    ctx: ToolContext,
    *,
    full: bool = False,
    per_pair: int | None = 1,
    operators: dict[str, Operator] | None = None,
    refs: list[Reference] | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    """Apply each operator to the reference calls and tally the outcomes.

    `per_pair` limits how many reference calls per (operator, tool) are *run through the engine* (a
    corruption identical to the clean call after cleaning is counted without a run); None runs all.
    """
    chosen = operators or OPERATORS
    pool = refs if refs is not None else references()
    clean_cache: dict[str, ToolOutput] = {}
    by_operator: dict[str, dict[str, int]] = defaultdict(_tally)
    by_tool: dict[str, dict[str, int]] = defaultdict(_tally)
    skipped: Counter[str] = Counter()
    defects: list[dict[str, str]] = []
    ran: Counter[tuple[str, str]] = Counter()

    for name, operator in chosen.items():
        for ref in pool:
            damaged = operator(ref.tool, ref.arguments)
            if damaged is None:
                skipped[name] += 1
                continue
            cleaned_clean, _ = pipeline_arguments(ref.tool, ref.arguments)
            cleaned_damaged, _ = pipeline_arguments(ref.tool, damaged)
            outcome: str
            if clean and cleaned_damaged == cleaned_clean and not full:
                outcome = SAME_AFTER_CLEANING
            else:
                if per_pair is not None and not full and ran[(name, ref.tool)] >= per_pair:
                    continue
                ran[(name, ref.tool)] += 1
                if ref.case not in clean_cache:
                    clean_cache[ref.case] = run_call(ref.tool, ref.arguments, ctx, clean=clean)
                outcome = classify(
                    clean_cache[ref.case], run_call(ref.tool, damaged, ctx, clean=clean)
                )
            by_operator[name][outcome] += 1
            by_tool[ref.tool][outcome] += 1
            if outcome in DEFECTS:
                defects.append(
                    {"operator": name, "tool": ref.tool, "case": ref.case, "outcome": outcome}
                )

    totals = _tally()
    for tally in by_operator.values():
        for outcome, count in tally.items():
            totals[outcome] += count
    applied = sum(totals.values())
    recovered = totals[SAME] + totals[SAME_AFTER_CLEANING]
    return {
        "format": 1,
        "mode": ("full" if full else "pr") + ("" if clean else " (without the cleaning layer)"),
        "references": len(pool),
        "tools_covered": len({r.tool for r in pool}),
        "applied": applied,
        "recovered_without_a_model_call": recovered,
        "recovery_rate": round(recovered / applied, 4) if applied else None,
        "totals": totals,
        "by_operator": {k: dict(v) for k, v in sorted(by_operator.items())},
        "by_tool": {k: dict(v) for k, v in sorted(by_tool.items())},
        "not_applicable": dict(sorted(skipped.items())),
        "defects": defects,
    }


def deterministic_view(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k != "format"}


def markdown(report: dict[str, Any]) -> str:
    t = report["totals"]
    out = [
        "# Corruption test",
        "",
        "Every known model mistake applied to every reference call of the benchmark, through the same pipeline a "
        "model's call takes. Each corrupted call must give the clean call's result or fail with a typed "
        "reason; anything else is a defect.",
        "",
        f"{report['applied']} corrupted calls over {report['references']} reference calls "
        f"({report['tools_covered']} tools), mode `{report['mode']}`: "
        f"**{t[DIFFERENT]} different verified results**, {t[UNTYPED]} untyped failures; "
        f"{report['recovered_without_a_model_call']} recovered without a model call "
        f"({100 * (report['recovery_rate'] or 0):.1f}%), {t[DIAGNOSED]} diagnosed.",
        "",
        "| Operator | Applied | Same | Same after cleaning | Diagnosed | Untyped | Different |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, tally in report["by_operator"].items():
        out.append(
            f"| {name} | {sum(tally.values())} | {tally[SAME]} | {tally[SAME_AFTER_CLEANING]} | "
            f"{tally[DIAGNOSED]} | {tally[UNTYPED]} | {tally[DIFFERENT]} |"
        )
    if report["defects"]:
        out += ["", "## Defects", ""]
        out += [
            f"* `{d['outcome']}`: {d['operator']} on {d['tool']} ({d['case']})"
            for d in report["defects"]
        ]
    return "\n".join(out) + "\n"
