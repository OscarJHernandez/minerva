"""What a tool tells the model about how its mathematical input was read, or why it could not be.

The engine reports two typed records (`minerva_engine.notation.models`): an `InputReading` with the changes made
to read an accepted input (`Xdot` read as the velocity of `X`), and an `InputDiagnostic` for one that could not
be read (a closed `code`, the field, the user's symbols and the spellings that would work). Both are shown to the
model in the same fixed shape by every tool, so a small model reads one convention rather than one prose format
per tool, and the loop can act on the `code` without parsing text.

Nothing here is logged except the `code` (a closed vocabulary, see `tracing.REASON_CODES`): the symbols and
suggestions are the user's names and live only in the stored message.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from minerva_engine.notation.derivatives import DERIVATIVE_NAME
from minerva_engine.notation.models import (
    InputDiagnostic,
    InputReading,
    Normalization,
    NormalizationKind,
)
from minerva_engine.notation.transport import clean_transport, localize_symbols

# Every argument that carries mathematics as one string, in every tool (the single list: references
# `@rN`/`@eN`, the scalar-tool guard, hints, dropped-assumption notes and transport cleaning all read it).
MATH_FIELDS = ("latex", "expression", "equation", "lagrangian", "hamiltonian", "integrand")
# Arguments that are lists of such strings.
MATH_LIST_FIELDS = ("equations",)

SyntaxChoice = Literal["auto", "latex", "text"]
FORMAT_DESCRIPTION = (
    "`auto` (default) reads text syntax (`theta_dot`, `*`, `sin(x)`) as text and anything with a "
    "backslash as LaTeX; set `latex` or `text` to say which."
)
_TEXT_MARKERS = re.compile(rf"\*|{DERIVATIVE_NAME}|\b[A-Za-z]\w*\s*\(")

# Tools that read `format: auto` by default, and tools whose expression field is text syntax by nature.
AUTO_TOOLS = frozenset({"analyze_lagrangian", "search_invariants"})
TEXT_TOOLS = frozenset({"tensor_canonicalize"})
_ASSIGNED = {"lagrangian": "L", "hamiltonian": "H"}


def resolve_syntax(source: str, requested: SyntaxChoice) -> Literal["latex", "text"]:
    """Choose between the two strict parsers; the input itself is never rewritten."""
    if requested != "auto":
        return requested
    return "text" if "\\" not in source and _TEXT_MARKERS.search(source) else "latex"


def strip_assignment(source: str, name: str) -> str:
    """`L = expr` -> `expr` for a single `=` whose left side is exactly `name`; anything else unchanged."""
    left, sign, right = source.partition("=")
    if sign and left.strip() == name and "=" not in right:
        return right.strip()
    return source


def _syntax(tool: str, arguments: dict[str, Any], text: str) -> Literal["latex", "text"]:
    requested = arguments.get("format")
    if requested in ("latex", "text"):
        return requested  # type: ignore[no-any-return]
    if tool in TEXT_TOOLS:
        return "text"
    if requested == "auto" or (requested is None and tool in AUTO_TOOLS):
        return resolve_syntax(text, "auto")
    return "latex"


def _prepare_text(
    tool: str, arguments: dict[str, Any], key: str, value: str
) -> tuple[str, list[Normalization]]:
    text, notes = clean_transport(value)
    out = list(notes)
    letter = _ASSIGNED.get(key)
    if letter is not None:
        bare = strip_assignment(text, letter)
        if bare != text:
            out.append(
                Normalization(
                    kind=NormalizationKind.ASSIGNMENT,
                    source=f"{letter} =",
                    target="",
                    note="the expression after the equals sign is the one analysed",
                )
            )
            text = bare
    text, local = localize_symbols(text, _syntax(tool, arguments, text))
    return text, [*out, *local]


def prepare_arguments(
    tool: str, arguments: dict[str, Any]
) -> tuple[dict[str, Any], tuple[Normalization, ...]]:
    """The arguments of a call with transport damage removed from every mathematical string, and what changed.

    Runs on the arguments after `@rN` references are filled in and before the call is keyed, validated or
    run, so that a call differing from an earlier one only by `\\*`, a `$` wrapper or `L =` is the same call.
    The list of changes is closed (`notation.transport`); a field that is not a string is left alone.
    """
    out = dict(arguments)
    notes: list[Normalization] = []
    for key in MATH_FIELDS:
        value = arguments.get(key)
        if isinstance(value, str):
            out[key], found = _prepare_text(tool, arguments, key, value)
            notes.extend(found)
    for key in MATH_LIST_FIELDS:
        values = arguments.get(key)
        if isinstance(values, list) and all(isinstance(v, str) for v in values):
            cleaned: list[str] = []
            for v in values:
                text, found = _prepare_text(tool, arguments, key, v)
                cleaned.append(text)
                notes.extend(found)
            out[key] = cleaned
    return (out if notes else arguments), tuple(dict.fromkeys(notes))


def transport_line(notes: tuple[Normalization, ...]) -> str | None:
    """One line telling the model (and the stored message) what was removed to read a call's input."""
    if not notes:
        return None
    return "input cleaned: " + "; ".join(n.describe() for n in notes)


def input_record(data: dict[str, Any]) -> tuple[InputReading | None, InputDiagnostic | None]:
    """The `reading` and `diagnostic` an engine report carried into a card (under `report` or `result`)."""
    for key in ("report", "result"):
        inner = data.get(key)
        if not isinstance(inner, dict):
            continue
        reading, diagnostic = inner.get("reading"), inner.get("diagnostic")
        if reading is None and diagnostic is None:
            continue
        try:
            return (
                InputReading.model_validate(reading) if reading else None,
                InputDiagnostic.model_validate(diagnostic) if diagnostic else None,
            )
        except ValueError:
            return None, None
    return None, None


def reading_lines(reading: InputReading | None) -> list[str]:
    """One line saying what was changed to read the input; empty when the input was read as written."""
    if reading is None or not reading.normalizations:
        return []
    return ["input: " + "; ".join(n.describe() for n in reading.normalizations)]


def diagnostic_line(status: str, diagnostic: InputDiagnostic | None) -> str | None:
    """The machine-readable form of why an input was refused: one compact JSON object on its own line."""
    if diagnostic is None:
        return None
    payload = {
        "status": "invalid_input" if status == "parse_error" else status,
        "code": diagnostic.code.value,
        "field": diagnostic.field,
        "symbols": diagnostic.symbols,
        "suggestions": diagnostic.suggestions,
        "recoverable": diagnostic.recoverable,
    }
    return "diagnostic: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def input_lines(
    status: str, reading: InputReading | None, diagnostic: InputDiagnostic | None
) -> tuple[list[str], str | None]:
    """The lines to add to a tool result and the `reason_code` to record (None when there is no diagnostic)."""
    lines = reading_lines(reading)
    line = diagnostic_line(status, diagnostic)
    if line is not None:
        lines.append(line)
    return lines, diagnostic.code.value if diagnostic is not None else None
