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
* a theorem (`gradient_theorem`, `divergence_theorem`, `stokes_theorem`) sent with no `expression` and exactly one
  declared field of the kind the theorem takes (a scalar for the gradient theorem, a vector for the others)
  gets that field's name as its expression: the theorem is about that field and there is no other reading.

Anything else stays an `invalid_arguments` refusal with the per-field message.
"""

from __future__ import annotations

import json
import re
import types
from collections.abc import Mapping
from typing import Any, Union, get_args, get_origin

from minerva_engine.notation.models import Normalization, NormalizationKind
from pydantic import BaseModel

from .mathinput import MATH_FIELDS, MATH_LIST_FIELDS

# Names a model uses for "the formula" in a tool whose formula field is called something else.
FORMULA_ALIASES = ("expression", "expr", "formula", "input", "math", "integrand")
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

    _kind_wrapper(fields, out, notes)
    _theorem_expression(fields, out, notes)
    _item_keys(fields, out, notes)
    _default_assumption(fields, out, notes)
    _nested_numbers(fields, out, notes)

    return (out if notes else arguments), tuple(notes)


def _tags(info: Any) -> dict[str, Any]:
    """tag -> model class, for a field whose type is a union chosen by a tag (`Field(discriminator="kind")`)."""
    tag = getattr(info, "discriminator", None)
    if not isinstance(tag, str):
        return {}
    found: dict[str, Any] = {}
    for member in get_args(info.annotation):
        member_field = getattr(member, "model_fields", {}).get(tag)
        if member_field is not None and isinstance(member_field.default, str):
            found[member_field.default] = member
    return found


def _kind_wrapper(
    fields: Mapping[str, Any], out: dict[str, Any], notes: list[Normalization]
) -> None:
    """`{"level_family": {...}}` for `{"kind": "level_family", ...}`: the member nested under its own tag.

    Closed rule: the value is an object with exactly one key, that key is one of the union's tags and its value
    is an object that is not tagged differently. Anything else (two keys, an unknown key) is left to the
    validation message, never guessed."""
    for name, value in list(out.items()):
        info = fields.get(name)
        tags = _tags(info) if info is not None else {}
        if not tags or not isinstance(value, dict) or len(value) != 1:
            continue
        ((key, inner),) = value.items()
        if key not in tags or not isinstance(inner, dict) or inner.get("kind", key) != key:
            continue
        out[name] = {**inner, "kind": key}
        notes.append(
            _note(
                name,
                f"{{{key!r}: {{...}}}}",
                f"{{'kind': {key!r}, ...}}",
                "the system's type is its `kind` key, not an outer key",
            )
        )


def _stringify(annotation: Any, value: Any) -> tuple[Any, bool]:
    """`value` with every number put where the schema wants text (a list of text, a model's text field),
    and whether anything changed. Numbers stay numbers wherever the schema takes a number."""
    annotation = _unwrap(annotation)
    if annotation is str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), True
        return value, False
    if get_origin(annotation) is list and isinstance(value, list):
        (inner,) = get_args(annotation)
        items = [_stringify(inner, item) for item in value]
        if any(changed for _, changed in items):
            return [item for item, _ in items], True
        return value, False
    if (
        isinstance(annotation, type)
        and issubclass(annotation, BaseModel)
        and isinstance(value, dict)
    ):
        out = dict(value)
        changed = False
        for key, item in value.items():
            info = annotation.model_fields.get(key)
            if info is not None:
                new, did = _stringify(info.annotation, item)
                if did:
                    out[key] = new
                    changed = True
        return (out, True) if changed else (value, False)
    return value, False


def _nested_numbers(
    fields: Mapping[str, Any], out: dict[str, Any], notes: list[Normalization]
) -> None:
    """`components: [1, 0, 0]` inside a list of objects, where the tool takes text: 1 is "1"."""
    for name, value in list(out.items()):
        info = fields.get(name)
        if info is None or not isinstance(value, (list, dict)):
            continue
        new, changed = _stringify(info.annotation, value)
        if changed:
            out[name] = new
            notes.append(
                _note(
                    name,
                    "numbers",
                    "text",
                    "this field takes text, so each number is written as text",
                )
            )


# Keys a model was seen to use for a list item's field (2026-11-18 live traces), with the field the tool means.
# Closed: a key is renamed only when the item has no field of the right name already.
_ITEM_KEY_ALIASES: dict[str, dict[str, str]] = {"matrices": {"symbol": "name"}}


def _item_keys(fields: Mapping[str, Any], out: dict[str, Any], notes: list[Normalization]) -> None:
    """`matrices: [{"symbol": "A", ...}]` -> `name`; and a condition without `unknown` when exactly one
    unknown is listed (`unknowns: ["x"]`), which can only mean that one."""
    for name, aliases in _ITEM_KEY_ALIASES.items():
        items = out.get(name)
        if name not in fields or not isinstance(items, list):
            continue
        fixed: list[Any] = []
        changed = False
        for item in items:
            if isinstance(item, dict):
                for alias, right in aliases.items():
                    if alias in item and right not in item:
                        item = {right if k == alias else k: v for k, v in item.items()}
                        changed = True
            fixed.append(item)
        if changed:
            out[name] = fixed
            notes.append(
                _note(
                    name,
                    "items keyed by a symbol",
                    "items keyed by name",
                    "each item is named by `name`",
                )
            )
    conditions, unknowns = out.get("conditions"), out.get("unknowns")
    if (
        "conditions" in fields
        and "unknowns" in fields
        and isinstance(conditions, list)
        and isinstance(unknowns, list)
        and len(unknowns) == 1
        and isinstance(unknowns[0], str)
        and any(isinstance(c, dict) and "unknown" not in c for c in conditions)
    ):
        out["conditions"] = [
            {"unknown": unknowns[0], **c} if isinstance(c, dict) and "unknown" not in c else c
            for c in conditions
        ]
        notes.append(
            _note(
                "conditions",
                "conditions without an unknown",
                f"unknown {unknowns[0]!r}",
                "the one listed unknown is the one every condition is about",
            )
        )


def _default_assumption(
    fields: Mapping[str, Any], out: dict[str, Any], notes: list[Normalization]
) -> None:
    """`assumptions: {"x": ["complex"]}`: complex is what every symbol is unless declared otherwise, so stating
    it changes nothing. It is not a declarable assumption (the tool refuses it); drop it, never the others."""
    assumptions = out.get("assumptions")
    if "assumptions" not in fields or not isinstance(assumptions, dict):
        return
    cleaned: dict[str, Any] = {}
    changed = False
    for symbol, items in assumptions.items():
        if isinstance(items, list) and "complex" in items:
            changed = True
            kept = [i for i in items if i != "complex"]
            if kept:  # a symbol that only said "complex" has nothing left to declare
                cleaned[symbol] = kept
        else:
            cleaned[symbol] = items
    if changed:
        out["assumptions"] = cleaned
        notes.append(
            _note(
                "assumptions",
                "complex",
                "nothing",
                "symbols are complex unless declared, so it is already the case",
            )
        )


_THEOREM_FIELD_KIND = {
    "gradient_theorem": "scalar",
    "divergence_theorem": "vector",
    "stokes_theorem": "vector",
}


def _theorem_expression(
    fields: dict[str, Any], out: dict[str, Any], notes: list[Normalization]
) -> None:
    """A theorem about the one declared field of its kind, sent without an expression, is about that field."""
    wanted = _THEOREM_FIELD_KIND.get(str(out.get("integral")))
    declared = out.get("fields")
    if (
        wanted is None
        or not {"expression", "fields", "integral"} <= set(fields)
        or str(out.get("expression") or "").strip()
        or not isinstance(declared, list)
    ):
        return
    candidates = [
        f["name"]
        for f in declared
        if isinstance(f, dict) and f.get("kind") == wanted and isinstance(f.get("name"), str)
    ]
    if len(declared) == 1 and len(candidates) == 1:
        out["expression"] = candidates[0]
        notes.append(
            _note(
                "expression",
                "(missing)",
                candidates[0],
                f"the theorem is about the one declared {wanted} field {candidates[0]}",
            )
        )
