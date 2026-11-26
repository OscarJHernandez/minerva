"""What the loop does about a tool call that did not deliver, decided by code and not by another model call.

A failure is one of a few kinds, and each kind has one policy. Most failures need the model to change something
(it must re-spell a name, ask the user, or say what is not supported), so the policy is about what the loop does
*before* that: try the one deterministic correction that exists, never run an equivalent call twice, and stop
offering a tool that keeps failing. Nothing here touches verification: a candidate that was not verified stays
withheld, and a correction is run through exactly the same checks as any call.

======================  =================================  ============================================
failure                 deterministic action first         then
======================  =================================  ============================================
invalid arguments       shape repair (`arguments.py`)      per-field message
parse error, text/LaTeX one re-run with the other parser   typed diagnostic to the model
parse error, other      none (cleaning already ran)        typed diagnostic with the spellings that work
semantic (ambiguous)    none                               the model is told to ask or declare
unsupported             none                               an answer: never retried with equivalent input
not verified            none, never reworded               withheld
timeout / resource      none                               never re-run with the same input
internal                none                               fixed text
======================  =================================  ============================================

An equivalent call (same tool, same arguments after cleaning, defaults filled) that already failed is not run
again; a tool that fails `max_failures_per_tool` times in a turn is not run again in that turn.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from .mathinput import GRID_FIELDS, MATH_FIELDS, input_record, resolve_syntax
from .tools import TOOLS, ToolOutput, _coerce_numbers

# Reason codes (`minerva_engine.notation.models.InputCode`) that mean the input is understood but needs a
# decision: the model must ask the user or declare something, so repeating or re-spelling cannot help.
SEMANTIC_CODES = frozenset(
    {
        "ambiguous_derivative",
        "mixed_derivative_spellings",
        "undeclared_dependent",
        "ambiguous_notation",
        "acceleration_in_input",
        "derivative_of_expression",
        "total_derivative_requested",
    }
)


class FailureClass(StrEnum):
    INVALID_ARGUMENTS = "invalid_arguments"
    PARSE = "parse"
    SEMANTIC = "semantic"
    UNSUPPORTED = "unsupported"
    NOT_VERIFIED = "not_verified"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    INTERNAL = "internal"


# Classes that count against a tool's failure budget. `unsupported` is an answer from the engine, not a
# failure to fix; a repeat of it with equivalent input is stopped by the call key, not by the budget.
BUDGETED = frozenset(
    {
        FailureClass.INVALID_ARGUMENTS,
        FailureClass.PARSE,
        FailureClass.SEMANTIC,
        FailureClass.NOT_VERIFIED,
        FailureClass.TIMEOUT,
        FailureClass.RESOURCE,
        FailureClass.INTERNAL,
    }
)
# Classes whose equivalent call must not be run again: the result would be the same.
NEVER_RERUN = frozenset(FailureClass)

EQUIVALENT_FAILED = (
    "These are the same arguments as the call above, which did not succeed: change the input as the "
    "diagnostic or the hint above describes, or tell the user what could not be established."
)
BUDGET_EXHAUSTED = (
    "error: not run. This tool has failed several times in a row in this answer. Do not call it again: say what could "
    "not be established, mark anything you write about it as not verified, and suggest what the user "
    "could do next."
)

_BY_CATEGORY = {
    "invalid_arguments": FailureClass.INVALID_ARGUMENTS,
    "unknown_tool": FailureClass.INVALID_ARGUMENTS,
    "malformed_model_output": FailureClass.INVALID_ARGUMENTS,
    "parse_error": FailureClass.PARSE,
    "unsupported": FailureClass.UNSUPPORTED,
    "not_verified": FailureClass.NOT_VERIFIED,
    "timeout": FailureClass.TIMEOUT,
    "resource_limit": FailureClass.RESOURCE,
    "sandbox_killed": FailureClass.RESOURCE,
    "busy": FailureClass.RESOURCE,
    "tool_failure": FailureClass.INTERNAL,
}


def classify(output: ToolOutput) -> FailureClass | None:
    """The kind of failure a tool result is, or None for a result that is not a failure (ok, an answer)."""
    if output.status == "ok":
        return None
    if output.reason_code in SEMANTIC_CODES:
        return FailureClass.SEMANTIC
    if not output.arguments_valid:
        return FailureClass.INVALID_ARGUMENTS
    return _BY_CATEGORY.get(output.error_category or "")


def canonical_key(name: str, arguments: dict[str, Any]) -> str:
    """The identity of a call: two calls with the same key would do the same thing.

    The tool, and its arguments as the schema reads them (defaults filled in, numbers and text unified), with
    `format: auto` replaced by the syntax it resolves to. Arguments that do not validate keep their own text.
    """
    tool = TOOLS.get(name)
    shaped: dict[str, Any] = arguments
    if tool is not None:
        try:
            shaped = tool.args_model(**_coerce_numbers(tool.args_model, arguments)).model_dump(
                mode="json"
            )
        except (ValueError, TypeError):
            shaped = arguments
    if shaped.get("format") == "auto" and not any(f in shaped for f in GRID_FIELDS):
        # A grid under `auto` is read by the engine, entry by entry when it must: it is its own call.
        text = next((shaped[f] for f in MATH_FIELDS if isinstance(shaped.get(f), str)), "")
        shaped = {**shaped, "format": resolve_syntax(text, "auto")}
    return name + "\0" + json.dumps(shaped, sort_keys=True, default=str)


# reason code -> (the syntax the call was read in, the other one). A letter run such as `Xdot` in a LaTeX
# field is only the same mistake as `sin(x)` there when the text has no backslash at all (it is text).
_OTHER_SYNTAX = {
    "text_syntax_in_latex": ("latex", "text"),
    "unrecognized_derivative": ("latex", "text"),
    "latex_in_text": ("text", "latex"),
}


def rerun_arguments(
    name: str, arguments: dict[str, Any], output: ToolOutput
) -> tuple[dict[str, Any], str] | None:
    """The one deterministic correction of a failed call, if there is one: (new arguments, policy name).

    A parse error that says the text is of the other syntax, in a tool that takes a `format`, is run once
    with the other strict parser. The input is not rewritten, and the re-run is checked exactly like any call.
    """
    tool = TOOLS.get(name)
    switch = _OTHER_SYNTAX.get(output.reason_code or "")
    if tool is None or switch is None or "format" not in tool.args_model.model_fields:
        return None
    current = arguments.get("format", tool.args_model.model_fields["format"].default)
    if any(f in tool.args_model.model_fields for f in GRID_FIELDS):
        # A grid (a metric, tensor components) is re-read under `auto`: the whole grid in the other syntax when
        # that reads every entry, otherwise entry by entry where that is unambiguous. Never flipped wholesale,
        # since one entry in each syntax (`2M` next to `sin(theta)`) fails either way.
        return None if current == "auto" else ({**arguments, "format": "auto"}, "format_switch")
    if current == "auto":
        text = next((arguments[f] for f in MATH_FIELDS if isinstance(arguments.get(f), str)), "")
        current = resolve_syntax(text, "auto")
    if current != switch[0]:
        return None
    if switch[1] == "text" and any(
        isinstance(arguments.get(f), str) and "\\" in arguments[f] for f in MATH_FIELDS
    ):
        return None  # a backslash is LaTeX: the text reader would refuse it too
    return {**arguments, "format": switch[1]}, "format_switch"


def rerun_summary(syntax: str, retried: ToolOutput) -> str:
    """One line for the model: what the corrective re-run in `syntax` reported (its diagnostic when it has one)."""
    lines = retried.for_model.splitlines()
    detail = next(
        (line for line in lines if line.startswith("diagnostic:")), lines[0] if lines else ""
    )
    return f"re-read with format {syntax}, it failed too: {detail}"


def failure_signature(output: ToolOutput) -> tuple[str, tuple[str, ...]] | None:
    """What a failure was about, for the failure budget: its diagnostic code and symbols, or None.

    Only a recoverable diagnostic has a signature: it names what to change, so a failure with a signature not seen
    before in the turn is the model correcting its input (`sin(` fixed, now `theta`), not a stuck loop. A failure
    without one (an uncoded error, a timeout, invalid arguments) has none and always counts. The symbols are the
    user's names: kept in memory for the turn, never logged.
    """
    _, diagnostic = input_record(output.card.data)
    if diagnostic is None or not diagnostic.recoverable:
        return None
    return diagnostic.code.value, tuple(diagnostic.symbols)
