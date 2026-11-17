"""Damage done to a formula on its way here, undone when it can only mean one thing.

A language model's tool call passes through JSON, markdown and a chat template before it reaches a parser. The
text that arrives is then not what the model meant: `*` became `\\*`, a formula grew `$...$` or a code fence,
`\\sin` became `\\\\sin`, a minus sign became U+2212. None of these is mathematics, none changes a meaning, and
the strict parsers refuse all of them. This module removes exactly those, from a closed list, and says what it
did (`Normalization`, kind `transport`). It never changes grouping, operators between operands, numbers or
names, and it never decides between two readings: anything else stays for the parser to refuse.

Two steps, because the second needs the syntax the first helps to choose:

* `clean_transport(source)`: format-independent (wrappers, markdown escapes, doubled backslashes, Unicode
  look-alikes, entities, trailing punctuation);
* `localize_symbols(source, format)`: Unicode Greek letters and combining dots (`θ̇`), which are spelled
  differently in LaTeX (`\\dot{\\theta}`) and in text (`theta_dot`).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

from .models import Normalization, NormalizationKind

_FENCE = re.compile(r"^```[A-Za-z]*\n?(?P<body>.*?)\n?```$", re.DOTALL)
_WRAPPERS: tuple[tuple[str, str], ...] = (
    ("$$", "$$"),
    ("$", "$"),
    (r"\(", r"\)"),
    (r"\[", r"\]"),
    ("`", "`"),
)
# A backslash in front of one of these is markdown escaping: no LaTeX command is spelled this way and
# the parser would refuse it ("unsupported LaTeX command \*").
_MARKDOWN_ESCAPES = re.compile(r"\\([*_^+\-=])")
# `\\sin`, `\\frac`: a backslash doubled by a second round of escaping, in front of a command name.
_DOUBLED = re.compile(r"\\{2,}(?=[A-Za-z])")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u2002\u2003\u2009\u200a\u202f\u2060\u200b"), " ")
_SYMBOLS = {
    "\u2212": "-",  # minus sign
    "\u2011": "-",  # non-breaking hyphen
    "\u00b7": "*",  # middle dot
    "\u22c5": "*",  # dot operator
    "\u2217": "*",  # asterisk operator
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u00b9": "^1",
}
_ENTITIES = {"&lt;": "<", "&gt;": ">", "&amp;": "&"}

_GREEK_NAMES = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]
_GREEK_CAPITALS = ["gamma", "delta", "theta", "lambda", "xi", "pi", "sigma", "phi", "psi", "omega"]
_UNICODE_NAME = {"lambda": "lamda"}  # Unicode spells it without the b
# Built from the Unicode names, not typed: a Greek letter in source code is easily mistaken for a Latin one.
GREEK: dict[str, str] = {
    **{
        unicodedata.lookup(f"GREEK SMALL LETTER {_UNICODE_NAME.get(n, n).upper()}"): n
        for n in _GREEK_NAMES
    },
    **{
        unicodedata.lookup(
            f"GREEK CAPITAL LETTER {_UNICODE_NAME.get(n, n).upper()}"
        ): n.capitalize()
        for n in _GREEK_CAPITALS
    },
}
_COMBINING = {"\u0307": 1, "\u0308": 2}  # dot above, diaeresis (one and two time derivatives)
_LETTER = r"[A-Za-z" + "".join(GREEK) + r"]"
_DOTTED = re.compile(rf"(?P<base>{_LETTER})(?P<mark>[\u0307\u0308])")


def _unwrap(text: str, notes: list[Normalization]) -> str:
    """Remove a delimiter pair that encloses the whole input (never one around a part of it)."""
    changed = True
    while changed:
        changed = False
        text = text.strip()
        fence = _FENCE.match(text)
        if fence is not None:
            notes.append(_note("code fence", "", "a code fence around the formula"))
            text, changed = fence.group("body"), True
            continue
        for opener, closer in _WRAPPERS:
            if (
                len(text) > len(opener) + len(closer)
                and text.startswith(opener)
                and text.endswith(closer)
            ):
                inner = text[len(opener) : -len(closer)]
                if opener in ("$", "$$") and "$" in inner:
                    continue  # `$a$ + $b$`: two formulas, not one wrapped
                if opener == "`" and "`" in inner:
                    continue
                notes.append(_note(opener + closer, "", f"{opener}...{closer} around the formula"))
                text, changed = inner, True
                break
    return text


def _note(source: str, target: str, why: str) -> Normalization:
    return Normalization(kind=NormalizationKind.TRANSPORT, source=source, target=target, note=why)


def clean_transport(source: str) -> tuple[str, tuple[Normalization, ...]]:
    """`source` without transport damage, and what was removed (nothing: the same string, no notes)."""
    notes: list[Normalization] = []
    text = _unwrap(source, notes)

    for entity, char in _ENTITIES.items():
        if entity in text:
            text = text.replace(entity, char)
            notes.append(_note(entity, char, "an HTML entity"))

    escaped = sorted(set(_MARKDOWN_ESCAPES.findall(text)))
    if escaped:
        text = _MARKDOWN_ESCAPES.sub(r"\1", text)
        notes.append(
            _note(
                " ".join("\\" + c for c in escaped),
                " ".join(escaped),
                "markdown escaping",
            )
        )
    if _DOUBLED.search(text):
        text = _DOUBLED.sub(r"\\", text)
        notes.append(_note("\\\\", "\\", "a backslash written twice in front of a command"))

    spaced = text.translate(_SPACES)
    if spaced != text:
        text = spaced
        notes.append(_note("non-breaking space", " ", "a space character that is not ASCII"))
    for char, ascii_char in _SYMBOLS.items():
        if char in text:
            text = text.replace(char, ascii_char)
            notes.append(_note(char, ascii_char, "a look-alike of an ASCII symbol"))

    stripped = text.rstrip()
    if stripped and stripped[-1] in ".,;" and len(stripped) > 1 and not stripped[-2].isdigit():
        text = stripped[:-1].rstrip()
        notes.append(_note(stripped[-1], "", "sentence punctuation after the formula"))
        # A period outside a wrapper (`$x^2$.`, a closing code fence followed by `.`) hid the wrapper from
        # the first pass; with the period gone it is the outermost layer.
        text = _unwrap(text, notes)
    return text.strip(), tuple(notes)


def localize_symbols(source: str, format: str) -> tuple[str, tuple[Normalization, ...]]:
    """Unicode Greek letters and combining dots in the spelling of `format` ("latex" or "text")."""
    if not any(c in GREEK or c in _COMBINING for c in unicodedata.normalize("NFD", source)):
        return source, ()
    notes: list[Normalization] = []
    text = unicodedata.normalize("NFD", source)

    def dotted(match: re.Match[str]) -> str:
        base, order = match.group("base"), _COMBINING[match.group("mark")]
        name = GREEK.get(base, base)
        if format == "latex":
            command = r"\dot" if order == 1 else r"\ddot"
            inner = rf"\{name}" if base in GREEK else name
            out = f"{command}{{{inner}}}"
        else:
            out = f"{name}_{'dot' if order == 1 else 'ddot'}"
        notes.append(
            _note(match.group(0), out, "a letter with a dot above it is a time derivative")
        )
        return out

    text = _DOTTED.sub(dotted, text)
    if any(c in _COMBINING for c in text):
        return (
            source,
            (),
        )  # a combining mark on something else: not read, left for the parser to refuse
    for char, name in GREEK.items():
        if char in text:
            replacement = rf"\{name} " if format == "latex" else name
            text = text.replace(char, replacement)
            notes.append(_note(char, replacement.strip(), "a Greek letter"))
    return re.sub(r"[ ]{2,}", " ", text).strip() if format == "latex" else text, tuple(notes)


# -- the inverse operators ---------------------------------------------------------------------------
# Each damages a string the way a transport does, and is the exact inverse of one row of `clean_transport`.
# They live here so that a row cannot be added without its inverse (a test pairs them); the corruption
# harness (`agent/evaluation/corruption.py`) applies them to every reference call of the benchmark.
Damage = Callable[[str], str]
DAMAGE: dict[str, Damage] = {}


def _damage(name: str) -> Callable[[Damage], Damage]:
    def register(function: Damage) -> Damage:
        DAMAGE[name] = function
        return function

    return register


@_damage("markdown_escape")
def escape_markdown(text: str) -> str:
    """`*` -> `\\*`, `_` -> `\\_`, `^` -> `\\^` (what chat markdown does to a formula)."""
    return re.sub(r"([*_^])", r"\\\1", text)


@_damage("dollar_wrapper")
def wrap_dollars(text: str) -> str:
    return f"${text}$"


@_damage("code_fence")
def wrap_fence(text: str) -> str:
    return f"```latex\n{text}\n```"


@_damage("doubled_backslash")
def double_backslashes(text: str) -> str:
    """`\\sin` -> `\\\\sin` (a second round of escaping)."""
    return re.sub(r"\\(?=[A-Za-z])", r"\\\\", text)


@_damage("unicode_look_alikes")
def unicode_look_alikes(text: str) -> str:
    """ASCII minus -> minus sign, `*` -> dot operator, `^2` -> superscript two, space -> no-break space."""
    minus, dot, square, nbsp = chr(0x2212), chr(0x22C5), chr(0x00B2), chr(0x00A0)
    return text.replace("^2", square).replace("*", dot).replace("-", minus).replace(" ", nbsp)


@_damage("trailing_period")
def add_trailing_period(text: str) -> str:
    """A sentence period after the formula (not after a digit: `2.` is a number, so that is not damage)."""
    return text if text[-1:].isdigit() else text + "."
