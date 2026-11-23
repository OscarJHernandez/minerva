"""The time symbol must not be the name of a coordinate (or of a momentum).

A Lagrangian with a coordinate `t` (the Schwarzschild geodesic Lagrangian in `(t, r, theta, phi)`) and the
default time symbol `t` would give the one symbol two roles, and the derivation cannot be confirmed. This is a
fact about the call, not about the mathematics: the call is refused with a typed, recoverable diagnostic that
names the fix. Nothing is renamed and no time symbol is chosen for the caller; `suggestions` only lists
names that are free, so the caller can repeat the call with one of them (`time: s`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import InputCode, InputDiagnostic

# Tried in this order; the first that is not in use is suggested first. Names that both parsers read as
# one symbol.
CANDIDATES: tuple[str, ...] = ("s", "tau", "u", "w")
_WORDS = re.compile(r"[A-Za-z]+")


def time_clash(
    coordinates: Iterable[str],
    time: str,
    *,
    also_used: Iterable[str] = (),
    source: str = "",
) -> InputDiagnostic | None:
    """The diagnostic when `time` is the name of a coordinate (or of a name in `also_used`), else None.

    `also_used` are the other declared names (momenta); `source` is the formula as sent, whose letter runs
    are never suggested (a parameter called `s` stays free of the time symbol).
    """
    coords = list(coordinates)
    declared = {*coords, *also_used}
    if time not in declared:
        return None
    taken = {
        *declared,
        *(f"{c}_dot" for c in coords),
        *_WORDS.findall(source),
    }
    free = [name for name in CANDIDATES if name not in taken]
    return InputDiagnostic(
        code=InputCode.TIME_SYMBOL_IS_COORDINATE,
        field="time",
        symbols=[time],
        suggestions=free[:3],
        recoverable=True,
    )


def time_clash_message(diagnostic: InputDiagnostic) -> str:
    """The sentence that goes with the diagnostic: the cause and the fix, in the user's own names."""
    name = diagnostic.symbols[0] if diagnostic.symbols else "t"
    fix = (
        f"for example `time: {diagnostic.suggestions[0]}`"
        if diagnostic.suggestions
        else "any symbol that is not a coordinate"
    )
    return (
        f"The time symbol `{name}` is also the name of a coordinate, so one symbol would play both "
        f"roles and nothing was computed. Repeat the call with a time symbol that is not a coordinate "
        f"({fix}); the Lagrangian is not rewritten."
    )
