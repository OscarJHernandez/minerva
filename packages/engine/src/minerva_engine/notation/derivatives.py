"""One reader for derivative spellings: `Xdot`, `X_dot`, `dX_dt`, `X'`-as-a-name, `dot X`, ...

Operations that declare *dependents* (coordinates of a mechanical system, unknowns of an ODE) all need the
same answer to "is this name the velocity of a declared one?". Before, each asked its own closed question
(`f"{q}_dot"`); a model that wrote `Xdot` got "no velocity" and had to guess the spelling. Here the families
of spellings are generated from parts, never listed per tool, and a name is read as a derivative only when
the reading is unique:

* the keyword part is case-insensitive (`XDot`, `x_DOT`); **the stem is matched exactly** (`xdot` is the
  velocity of `x`, never of `X`);
* a name that is itself declared (a coordinate, the independent variable, a symbol the caller gave
  assumptions for) is not rewritten;
* a name with two readings (`ddot` with dependents `d` and `dd`), two spellings of one derivative in the
  same expression (`xdot` and `x_dot`), and a derivative of an undeclared stem are refused with a typed
  diagnostic naming the candidates. Nothing is guessed.

The reader works on the *symbols of an already parsed tree* (every name is one symbol in text syntax) and
renames them with `xreplace`, so it cannot change anything but a name. Every rename is returned as a
`Normalization`, which the card and the model's tool result show. Two things need the source text, and only
inspect or splice it where text syntax has no other reading: `x dot` (two adjacent names, which text syntax
cannot mean anything else by) and, in LaTeX, a letter run such as `Xdot`, which is a product of letters
there and is refused with the fix, never reinterpreted.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cache

import sympy as sp

from ..errors import ParseError
from ..models import ALLOWED_ASSUMPTIONS
from .models import InputCode, Normalization, NormalizationKind

MAX_ORDER = 4

# `dot`, `ddot`, `dddot`, `ddddot`, or the repeated words `dot_dot`, `dotdot`: the order is the count.
_KEYWORD = r"(?i:(?P<kw>d{1,4}ot|dot(?:_?dot){1,3}))"
_PRIME = r"(?i:(?P<pr>prime(?:_?prime){0,3}|p{2,4}rime))"


def canonical(stem: str, order: int) -> str:
    """The name the parsers give the derivative of `stem`: `q_dot`, `q_ddot`, `q_d3`, `q_d4`."""
    if order == 1:
        return f"{stem}_dot"
    if order == 2:
        return f"{stem}_ddot"
    return f"{stem}_d{order}"


def _keyword_order(text: str) -> int:
    lowered = text.lower()
    if lowered.startswith("dot"):
        return lowered.count("dot")
    return lowered.count("d")


def _prime_order(text: str) -> int:
    lowered = text.lower()
    if lowered.startswith("pp"):
        return len(lowered) - len("rime")
    return lowered.count("prime")


@cache
def _patterns(stem: str, variable: str) -> tuple[tuple[re.Pattern[str], str], ...]:
    q, v = re.escape(stem), re.escape(variable)
    return (
        (re.compile(rf"^{q}_?{_KEYWORD}$"), "kw"),
        (re.compile(rf"^{_KEYWORD}_?{q}$"), "kw"),
        (re.compile(rf"^{q}_d(?P<n>[1-9])$"), "n"),
        (re.compile(rf"^d(?P<a>[2-4]?){q}_?d{v}(?P<b>[2-4]?)$"), "leibniz"),
        (re.compile(rf"^{q}_?{_PRIME}$"), "pr"),
    )


def orders_of(name: str, stem: str, variable: str) -> set[int]:
    """The derivative orders `name` can stand for as a spelling of a derivative of `stem` (empty: none)."""
    found: set[int] = set()
    for pattern, kind in _patterns(stem, variable):
        match = pattern.match(name)
        if match is None:
            continue
        if kind == "kw":
            order = _keyword_order(match.group("kw"))
        elif kind == "pr":
            order = _prime_order(match.group("pr"))
        elif kind == "n":
            order = int(match.group("n"))
        else:
            a, b = match.group("a"), match.group("b")
            if a != b:
                continue
            order = int(a) if a else 1
        if 1 <= order <= MAX_ORDER:
            found.add(order)
    return found


def spellings(stem: str, order: int, variable: str = "t") -> list[str]:
    """Every spelling this module reads as the derivative of `order` of `stem` (the canonical one excluded).

    The inverse of `orders_of`, used by tests and by the corruption harness so that a spelling that is read
    and a spelling that is generated can never drift apart.
    """
    words = {
        1: ["dot"],
        2: ["ddot", "dot_dot", "dotdot"],
        3: ["dddot", "dot_dot_dot"],
        4: ["ddddot", "dot_dot_dot_dot"],
    }[order]
    primes = {1: ["prime"], 2: ["pprime", "prime_prime", "primeprime"]}.get(order, [])
    out: list[str] = []
    for word in words:
        cased = [word, word.upper(), word.capitalize()]
        for w in cased:
            out += [f"{stem}{w}", f"{stem}_{w}", f"{w}{stem}", f"{w}_{stem}"]
    out += [f"{stem}_d{order}"]
    count = "" if order == 1 else str(order)
    out += [f"d{count}{stem}_d{variable}{count}", f"d{count}{stem}d{variable}{count}"]
    for p in primes:
        out += [f"{stem}{p}", f"{stem}_{p}"]
    target = canonical(stem, order)
    return [name for name in dict.fromkeys(out) if name != target]


@dataclass(frozen=True)
class Resolved:
    expr: sp.Expr
    normalizations: tuple[Normalization, ...] = ()


def _describe(stem: str, order: int) -> str:
    return f"the velocity of {stem}" if order == 1 else f"the order-{order} derivative of {stem}"


def resolve_derivatives(
    expr: sp.Expr,
    dependents: Sequence[str],
    variable: str,
    assumptions: dict[str, list[str]] | None = None,
    *,
    declared: Collection[str] = (),
) -> Resolved:
    """Read derivative spellings among the free symbols of `expr` as the canonical `q_dot`, `q_ddot`, ...

    `dependents` are the declared stems and `variable` the independent variable; `declared` are other names
    that must stay what they are. Raises `ParseError` (with a typed code) for an ambiguous, mixed or
    undeclared-stem spelling. A name that matches nothing is left alone: it is an ordinary symbol.
    """
    assumptions = assumptions or {}
    stems = list(dict.fromkeys(dependents))
    fixed = set(stems) | {variable} | set(declared)
    symbols = {s.name: s for s in expr.free_symbols if isinstance(s, sp.Symbol)}

    canonical_used: dict[tuple[str, int], str] = {}
    for stem in stems:
        for order in range(1, MAX_ORDER + 1):
            name = canonical(stem, order)
            if name in symbols:
                canonical_used[(stem, order)] = name

    aliases: dict[str, tuple[str, int]] = {}
    for name in sorted(symbols):
        if name in fixed or name in canonical_used.values():
            continue
        readings = {(stem, order) for stem in stems for order in orders_of(name, stem, variable)}
        if not readings:
            continue
        if len(readings) > 1:
            options = sorted(canonical(stem, order) for stem, order in readings)
            raise ParseError(
                f"`{name}` could be read as more than one derivative ({', '.join(options)}); "
                "write the one you mean.",
                code=InputCode.AMBIGUOUS_DERIVATIVE,
                symbols=[name],
                suggestions=options,
            )
        stem, order = next(iter(readings))
        if assumptions.get(name):
            raise ParseError(
                f"`{name}` has declared assumptions, so it is not read as {_describe(stem, order)}: "
                f"write {canonical(stem, order)} (or remove the assumptions on {name}).",
                code=InputCode.AMBIGUOUS_DERIVATIVE,
                symbols=[name],
                suggestions=[canonical(stem, order)],
            )
        aliases[name] = (stem, order)

    by_reading: dict[tuple[str, int], list[str]] = {}
    for name, reading in aliases.items():
        by_reading.setdefault(reading, []).append(name)
    for reading, names in by_reading.items():
        used = canonical_used.get(reading)
        spellings = sorted([*names, *([used] if used else [])])
        if len(spellings) > 1:
            raise ParseError(
                f"{' and '.join(f'`{n}`' for n in spellings)} are two spellings of "
                f"{_describe(*reading)}; use one of them throughout, so they are not mistaken for "
                "different quantities.",
                code=InputCode.MIXED_DERIVATIVE_SPELLINGS,
                symbols=spellings,
                suggestions=[canonical(*reading)],
            )

    _refuse_undeclared_stems(symbols, stems, variable, fixed, set(aliases))

    if not aliases:
        return Resolved(expr)
    mapping: dict[sp.Basic, sp.Basic] = {}
    notes: list[Normalization] = []
    for name, (stem, order) in sorted(aliases.items()):
        target = canonical(stem, order)
        flags = {a: True for a in assumptions.get(target, ()) if a in ALLOWED_ASSUMPTIONS}
        mapping[symbols[name]] = sp.Symbol(target, **flags)
        notes.append(
            Normalization(
                kind=NormalizationKind.DERIVATIVE_SPELLING,
                source=name,
                target=target,
                note=_describe(stem, order),
            )
        )
    return Resolved(expr.xreplace(mapping), tuple(notes))


_UNDECLARED_SUFFIX = re.compile(rf"^(?P<stem>[A-Za-z][A-Za-z0-9]*?)_?{_KEYWORD}$")


def _refuse_undeclared_stems(
    symbols: dict[str, sp.Symbol],
    stems: list[str],
    variable: str,
    fixed: set[str],
    resolved: set[str],
) -> None:
    """`ydot` or `y_dot` with `y` not declared: a velocity of something nobody listed, never a parameter.

    The canonical `y_dot` is always refused (as a parameter it would silently be a constant); an alias such
    as `ydot` is refused only when `y` itself also appears in the expression, which is what a coordinate
    that was forgotten looks like. Both name the stem to add.
    """
    for name in sorted(symbols):
        if name in fixed or name in resolved:
            continue
        match = _UNDECLARED_SUFFIX.match(name)
        if match is None:
            continue
        stem = match.group("stem")
        if stem in stems or stem == variable:
            continue
        spelled_canonically = name in (f"{stem}_dot", f"{stem}_ddot")
        if spelled_canonically or stem in symbols:
            listed = ", ".join(stems) or "none"
            raise ParseError(
                f"`{name}` looks like the velocity of `{stem}`, which is not among the declared "
                f"coordinates ({listed}). List `{stem}` as a coordinate, or rename the symbol if it is "
                "not a velocity.",
                code=InputCode.UNDECLARED_DEPENDENT,
                symbols=[name],
                suggestions=[stem],
                recoverable=True,
            )


_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_SPLIT_KEYWORD = re.compile(rf"^_?{_KEYWORD}$")
# A text token that is a derivative spelling of *some* stem (no stem is declared yet): `Xdot`, `theta_ddot`.
DERIVATIVE_NAME = r"\b[A-Za-z][A-Za-z0-9]*_?(?i:d{1,4}ot)\b"


def generic_readings(name: str) -> set[tuple[str, int]]:
    """Every (stem, order) the suffix spelling `name` can be, when no stem is declared (`xddot`: two).

    Used to find the coordinates a user's formula talks about before any are declared (routing). Suffix
    spellings only; the declared-stem reader above knows the rest of the families.
    """
    found: set[tuple[str, int]] = set()
    for i in range(1, len(name)):
        stem, rest = name[:i].rstrip("_"), name[i:]
        if not stem or not stem[0].isalpha():
            continue
        match = _SPLIT_KEYWORD.match(rest)
        if match is not None:
            order = _keyword_order(match.group("kw"))
            if order <= MAX_ORDER:
                found.add((stem, order))
        number = re.fullmatch(r"_d([1-9])", rest)
        if number is not None:
            found.add((stem, int(number.group(1))))
    return found


def spelled_derivatives(source: str) -> list[tuple[str, int]] | None:
    """The (stem, order) of every derivative-spelled name in text `source`; None if any name is ambiguous."""
    out: list[tuple[str, int]] = []
    for name in dict.fromkeys(_TOKEN.findall(source)):
        readings = generic_readings(name)
        if len(readings) > 1:
            return None
        out.extend(readings)
    return out


def join_spaced_keywords(
    source: str, dependents: Collection[str]
) -> tuple[str, tuple[Normalization, ...]]:
    """Text syntax only: `x dot` / `dot x` (two adjacent names) becomes the canonical `x_dot`.

    Text syntax has no implicit multiplication, so two names with nothing between them cannot mean anything
    else. Only a declared dependent followed (or preceded) by a derivative keyword is joined; the source is
    otherwise untouched.
    """
    if "\n" in source or not dependents:
        return source, ()
    try:
        tokens = [
            t
            for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source, ()
    stems = set(dependents)
    edits: list[tuple[int, int, str, str, str]] = []
    skip = -1
    for i in range(len(tokens) - 1):
        if i <= skip:
            continue
        a, b = tokens[i], tokens[i + 1]
        if a.type != tokenize.NAME or b.type != tokenize.NAME or a.end[0] != b.start[0]:
            continue
        if a.string in stems and re.fullmatch(_KEYWORD, b.string):
            stem, order = a.string, _keyword_order(b.string)
        elif b.string in stems and re.fullmatch(_KEYWORD, a.string):
            stem, order = b.string, _keyword_order(a.string)
        else:
            continue
        if order > MAX_ORDER:
            continue
        edits.append(
            (a.start[1], b.end[1], canonical(stem, order), source[a.start[1] : b.end[1]], stem)
        )
        skip = i + 1
    if not edits:
        return source, ()
    out = source
    notes: list[Normalization] = []
    for start, end, target, original, stem in reversed(edits):
        out = out[:start] + target + out[end:]
        notes.append(
            Normalization(
                kind=NormalizationKind.DERIVATIVE_SPELLING,
                source=original,
                target=target,
                note=_describe(stem, 1 if target.endswith("_dot") else 2),
            )
        )
    return out, tuple(reversed(notes))


def refuse_latex_letter_runs(source: str, dependents: Collection[str]) -> None:
    """LaTeX reads `Xdot` as the product X·d·o·t. For a declared dependent, say so and give the fix.

    Only inspects: the run is never reinterpreted, because in LaTeX a run of letters is a product.
    """
    for stem in dict.fromkeys(dependents):
        pattern = re.compile(rf"(?<![\\A-Za-z]){re.escape(stem)}_?{_KEYWORD}(?![A-Za-z])")
        match = pattern.search(source)
        if match is None:
            continue
        order = _keyword_order(match.group("kw"))
        command = {1: r"\dot", 2: r"\ddot"}.get(order, r"\dot")
        base = rf"\{stem}" if re.fullmatch(_GREEK, stem) else stem
        fix = f"{command}{{{base}}}" if order <= 2 else f"{canonical(stem, order)} (text syntax)"
        raise ParseError(
            f"`{match.group(0)}` is read as the product of the letters in LaTeX, not as the velocity of "
            f"`{stem}`: write `{fix}`, or use format text (`{canonical(stem, order)}`).",
            code=InputCode.UNRECOGNIZED_DERIVATIVE,
            symbols=[match.group(0)],
            suggestions=[fix, canonical(stem, order)],
            recoverable=True,
        )


_GREEK = (
    r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|rho|sigma|tau|"
    r"upsilon|phi|chi|psi|omega"
)
