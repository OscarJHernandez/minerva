"""Argument shapes a model gets wrong in predictable ways, repaired from a closed list.

A small model sends `"X, Y, s"` where the schema wants `["X", "Y", "s"]`, a JSON object as a string, an
`expression` where the tool's one formula field is `lagrangian`, or one equation where a list is expected.
Each of these has exactly one sensible reading, and the schema says what the tool wants; guessing is never
needed. This module changes the *shape* of an argument (never a formula: that is `mathinput`'s transport
cleaning) and reports each change, so the trace and the model see what was done.

Repairs, in this order, each only when the schema decides the target:

* a field the tool does not have, named like a formula (`expression`, `expr`, `formula`, `input`), becomes the
  tool's only formula field when that field is absent;
* a string given where a list of names is wanted is split on commas (`"X, Y, s"`, `"[X, Y]"`) when every piece
  is a valid name;
* a string given where a list of formulas is wanted becomes a one-item list;
* a JSON string given where an object or list is wanted is parsed;
* a number given where text is wanted becomes its text (`_coerce_numbers`, the same rule `execute` applies, so
  that validity, the trace and the call key agree).

Anything else stays an `invalid_arguments` refusal with the per-field message.
"""

from __future__ import annotations

import json
import re
import types
from typing import Any, Union, get_args, get_origin

from minerva_engine.notation.models import Normalization, NormalizationKind
from pydantic import BaseModel

from .mathinput import MATH_FIELDS, MATH_LIST_FIELDS

# Names a model uses for "the formula" in a tool whose formula field is called something else.
FORMULA_ALIASES = ("expression", "expr", "formula", "input", "math")
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _unwrap(annotation: Any) -> Any:
    """`X | None` -> `X`."""
    if get_origin(annotation) in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            return options[0]
    return annotation


def _is_list_of_str(annotation: Any) -> bool:
    annotation = _unwrap(annotation)
    return get_origin(annotation) is list and get_args(annotation) == (str,)


def _is_container(annotation: Any) -> bool:
    annotation = _unwrap(annotation)
    return get_origin(annotation) in (list, dict) or (
        isinstance(annotation, type) and issubclass(annotation, BaseModel)
    )


def _note(field: str, before: str, after: str, why: str) -> Normalization:
    return Normalization(
        kind=NormalizationKind.SHAPE, source=f"{field}: {before}", target=after, note=why
    )


def _split_names(text: str) -> list[str] | None:
    pieces = [p.strip().strip("'\"") for p in text.strip().strip("[]").split(",")]
    if pieces and all(_NAME.fullmatch(p) for p in pieces):
        return pieces
    return None


def repair_arguments(
    model: type[BaseModel], arguments: dict[str, Any]
) -> tuple[dict[str, Any], tuple[Normalization, ...]]:
    """`arguments` with the repairable shape errors fixed (the same dict when there are none), and the changes."""
    fields = model.model_fields
    out = dict(arguments)
    notes: list[Normalization] = []

    formula_fields = [f for f in fields if f in MATH_FIELDS]
    if len(formula_fields) == 1 and formula_fields[0] not in out:
        for alias in FORMULA_ALIASES:
            if alias in out and alias not in fields and isinstance(out[alias], str):
                out[formula_fields[0]] = out.pop(alias)
                notes.append(
                    _note(
                        alias,
                        "the formula",
                        formula_fields[0],
                        f"this tool's formula is `{formula_fields[0]}`",
                    )
                )
                break

    for name, value in list(out.items()):
        info = fields.get(name)
        if info is None or not isinstance(value, str):
            continue
        annotation = info.annotation
        if _is_list_of_str(annotation):
            if name in MATH_LIST_FIELDS:
                out[name] = [value]
                notes.append(_note(name, "one formula", "a list of one", "this field takes a list"))
                continue
            names = _split_names(value)
            if names is not None:
                out[name] = names
                notes.append(
                    _note(name, repr(value), repr(names), "this field takes a list of names")
                )
                continue
        if _is_container(annotation) and value.lstrip()[:1] in "[{":
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, (list, dict)):
                out[name] = parsed
                notes.append(
                    _note(
                        name,
                        "a JSON string",
                        type(parsed).__name__,
                        "this field takes JSON, not text",
                    )
                )

    for name, value in list(out.items()):
        info = fields.get(name)
        if (
            info is not None
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and _unwrap(info.annotation) is str
        ):
            out[name] = str(value)
            notes.append(_note(name, str(value), repr(str(value)), "this field takes text"))

    return (out if notes else arguments), tuple(notes)
