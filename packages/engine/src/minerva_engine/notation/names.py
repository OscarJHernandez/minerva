"""Spellings of a declared name, and the symbols read that a misspelt declared name most likely means.

A declared coordinate is matched by exact name. These helpers only *describe* a mismatch (the suggestion of a
`declared_name_absent` diagnostic); they never decide which symbol a name refers to.
"""

from __future__ import annotations

from collections.abc import Iterable

_DERIVATIVE_SUFFIXES = ("_ddot", "_dot")


def spelling_key(name: str) -> str:
    """The name with backslashes, braces and underscores removed: `\\theta_{1}`, `theta_1`, `theta1` agree."""
    return "".join(ch for ch in name.strip() if ch not in "\\{}_ ")


def position_name(symbol: str) -> str:
    """`x_1_dot` and `x_1_ddot` are the velocity and acceleration of `x_1`."""
    for suffix in _DERIVATIVE_SUFFIXES:
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def near_misses(declared: str, read: Iterable[str]) -> list[str]:
    """The position names among `read` that are spelt like `declared` up to backslashes, braces and underscores.

    Sorted; empty when nothing is close. Several entries mean the spelling does not single one out.
    """
    key = spelling_key(declared)
    if not key:
        return []
    return sorted(
        {position_name(s) for s in read if spelling_key(position_name(s)) == key} - {declared}
    )
