"""Spellings of a declared name, and the symbols read that a misspelt declared name most likely means.

`clean_name` removes LaTeX spelling from a declared name (`\\theta_{1}` is `theta_1`): the parser reads
`\\theta_{1}` in an expression as the symbol `theta_1`, so these are one name. `near_misses` only *describes* a
mismatch (the suggestion of a `declared_name_absent` diagnostic); `alias_in_latex` is the one place a declared
name is read as a different symbol, under the narrow conditions it states.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

from .models import Normalization, NormalizationKind

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


def clean_name(name: str) -> str:
    """A declared name without its LaTeX spelling: backslashes, braces and spaces removed (`\\theta_{1}`)."""
    return "".join(ch for ch in name.strip() if ch not in "\\{} ")


def alias_in_latex(
    declared: list[str], read: Collection[str]
) -> tuple[dict[str, str], list[Normalization]]:
    """Declared names that a LaTeX input can only mean as one symbol read, and the notes that say so.

    In LaTeX `x1` is never one symbol (it reads as `x*1`), so a declared `x1` absent from the input can only mean
    a symbol written `x_1`. It is read as that symbol only if: the name has no underscore, exactly one position
    name read is spelt alike (`near_misses`), that symbol's velocity is read too (a coordinate has one; a mass
    `m_1` spelt like a declared `m1` does not), it is not itself declared, and no other declared name maps onto
    it. Anything else is left as declared, so the caller refuses it with the near misses as suggestions.
    """
    names = set(read)
    taken = set(declared)
    aliases: dict[str, str] = {}
    for name in declared:
        if "_" in name or name in names or f"{name}_dot" in names:
            continue
        close = near_misses(name, names)
        if len(close) != 1:
            continue
        target = close[0]
        if f"{target}_dot" not in names or target in taken or target in aliases.values():
            continue
        aliases[name] = target
    notes = [
        Normalization(
            kind=NormalizationKind.DECLARED_NAME,
            source=name,
            target=target,
            note=f"the declared coordinate {name} is written {target} in the input, with its velocity",
        )
        for name, target in aliases.items()
    ]
    return aliases, notes
