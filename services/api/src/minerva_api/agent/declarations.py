"""Matrix declarations written in words.

The claim checker never tests a statement about a matrix by sampling numbers (that would treat the
matrices as commuting scalars), but it can only protect names it knows are matrices. They are known
from `matrix_calculus` calls and `noncommutative` declarations in tool arguments; this module adds
the ones a person (or the model) wrote in prose: "Let A and B be 3x3 matrices", "X, Y are
noncommuting operators".

Deterministic and fail-safe: a name picked up by mistake only turns a "checked" claim about it into
an "unchecked" one, so the reading is deliberately generous. Nothing here computes anything.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MAX_NAMES = 64

_KEYWORD = re.compile(
    r"\b(?:matri(?:x|ces)|operators?|noncommut\w*|non-commut\w*|(?:do|does|don't|not)\s+commute)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
# A size such as 3x3 or n \times n must not make `n` or `N` a matrix name.
_SIZE = re.compile(r"[A-Za-z0-9]+\s*(?:\\times|\u00d7|x)\s*[A-Za-z0-9]+")
# A capital letter with an optional subscript, or a capital Greek letter written as a command.
_NAME = re.compile(
    r"(?<![A-Za-z\\])(?:\\([A-Z][a-z]+)|([A-Z])(?:_\{?([A-Za-z0-9]+)\}?)?)(?![A-Za-z])"
)
_ARTICLE = re.compile(r"^\s*(?:A|I)\s+[a-z]")


def _names_in(fragment: str) -> set[str]:
    found: set[str] = set()
    for command, letter, subscript in _NAME.findall(_SIZE.sub(" ", fragment)):
        if command:
            found.add(command)
        elif letter:
            found.add(f"{letter}_{subscript}" if subscript else letter)
    return found


def declared_matrices(texts: Iterable[str]) -> set[str]:
    """Names that `texts` describe as matrices or operators, or as not commuting."""
    names: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE.split(text):
            if not _KEYWORD.search(sentence):
                continue
            if _ARTICLE.match(sentence):  # "A matrix is ...": the article is not a name
                sentence = sentence.lstrip()[1:]
            math_parts = [
                next(g for g in m.groups() if g is not None) for m in _MATH.finditer(sentence)
            ]
            prose = _MATH.sub(" ", sentence)
            for part in (*math_parts, prose):
                names |= _names_in(part)
            if len(names) >= MAX_NAMES:
                return set(sorted(names)[:MAX_NAMES])
    return names
