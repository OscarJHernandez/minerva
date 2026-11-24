"""Did the user say what a quantity is, in the unit the assistant used? (Else the unit is the assistant's.)

`check_units` finds the dimension of an unknown from the units of the other symbols. Those units come from the
model's call, and the answer holds "given these inputs". This module decides which inputs the USER stated, by
one closed rule that errs towards "assumed": a wrong "assumed" costs a sentence on the card, a wrong "stated"
would present the assistant's choice as the user's.

A quantity counts as stated only if one sentence of one user message ties its name directly to a word of the
kind table or to a unit the unit reader accepts, in one of these shapes, and that word reads to the same
dimension the call sent:

    KIND NAME          mass m
    NAME is (a) KIND   m is a mass, r: radius
    NAME in UNIT       m in kg, m = 2 kg, m [kg], m (kg), m has units of kg

Merely holding the name and a unit somewhere in a sentence does not count (a Lagrangian with a mass `m` and a
radius `r` holds the name `r` and the unit letter `m` and says nothing about either). Messages are the user's
only; the model's own text never counts. Nothing here is logged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from minerva_engine.dimensions import (
    QUANTITY_KINDS,
    Dim,
    dim_text,
    kind_key,
    read_kind,
    read_unit,
)
from minerva_engine.errors import MinervaMathError

_SENTENCES = re.compile(r"(?<=[.!?;])\s+|\n+")
_KINDS = "|".join(
    re.escape(k)
    for k in sorted(
        [*QUANTITY_KINDS, "velocities", "gravitational acceleration"], key=len, reverse=True
    )
)
_UNIT = r"[A-Za-z][A-Za-z0-9*/^()-]{0,40}"
_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _name_forms(name: str) -> str:
    """The ways a name is written in a message: `theta`, `\\theta`, `r_0`, `r_{0}`, `$r$`, `C_3`."""
    base, _, sub = name.partition("_")
    forms = {name, f"\\{name}"}
    if sub:
        forms |= {f"{base}_{{{sub}}}", f"\\{base}_{sub}", f"\\{base}_{{{sub}}}"}
    return "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"


def _patterns(name: str) -> list[re.Pattern[str]]:
    n = rf"(?<![\w\\])\$?{_name_forms(name)}\$?(?![\w])"
    kind_first = rf"\b(?P<kind>(?i:{_KINDS}))\s+(?:of\s+)?{n}"
    kind_after = (
        rf"{n}\s*(?:is|are|be|denotes|means|:|=)\s*(?:the\s+|an?\s+)?(?P<kind>(?i:{_KINDS}))\b"
    )
    unit_after = (
        rf"{n}\s*(?:(?:\(|\[|:|(?:\b(?:is|are|be)\s+)?(?:\b(?:measured|given)\s+)?\bin\b|\bhas units? (?:of )?)\s*"
        rf"(?:{_NUMBER}\s*)?|=\s*{_NUMBER}\s*)(?P<unit>{_UNIT})"
    )
    return [
        re.compile(kind_first),
        re.compile(kind_after),
        re.compile(unit_after),
    ]


def _dimension(match: re.Match[str]) -> Dim | None:
    try:
        kind = match.groupdict().get("kind")
        if kind:
            return read_kind(kind_key(kind))
        unit = match["unit"]
        while unit.endswith(")") and unit.count("(") < unit.count(")"):
            unit = unit[:-1]
        return read_unit(unit)
    except MinervaMathError:
        return None


def stated(name: str, dimension: str, messages: Sequence[str]) -> bool:
    """Whether a user sentence ties `name` to a kind or unit whose dimension reads as `dimension` (the text
    the engine prints, `kg m^-1 s^-2`; `1` when dimensionless)."""
    patterns = _patterns(name)
    for message in messages:
        for sentence in _SENTENCES.split(message):
            if name not in re.sub(r"[\\${}]", "", sentence):
                continue
            for pattern in patterns:
                for match in pattern.finditer(sentence):
                    found = _dimension(match)
                    if found is not None and dim_text(found) == dimension:
                        return True
    return False


def assumed_quantities(
    quantities: Iterable[tuple[str, str]], user_messages: Sequence[str]
) -> list[str]:
    """The names of the quantities the user did not state in the dimension the call used; each pair is a name
    and the engine's text of the dimension the call gave it."""
    return [name for name, dim in quantities if not stated(name, dim, user_messages)]
