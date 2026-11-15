"""Matrix and vector declarations written in words.

The claim checker never tests a statement about a matrix by sampling numbers (that would treat the
matrices as commuting scalars), but it can only protect names it knows are matrices. They are known
from `matrix_calculus` calls and `noncommutative` declarations in tool arguments; this module adds
the ones a person (or the model) wrote in prose: "Let A and B be 3x3 matrices", "X, Y are
noncommuting operators".

Vectors are read the same way ("let E and B be vector fields", "a and b are vectors"): the claim
checker never samples a statement about a vector either, since a dot or cross product tested as a
commuting scalar product would be "checked" falsely.

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
_VECTOR_KEYWORD = re.compile(r"\b(?:vectors?|vector[- ]fields?)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?:;])\s+|\n+")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
# A size such as 3x3 or n \times n must not make `n` or `N` a matrix name.
_SIZE = re.compile(r"[A-Za-z0-9]+\s*(?:\\times|\u00d7|x)\s*[A-Za-z0-9]+")
# A letter (either case) with an optional subscript, or a command such as \\omega: vectors are
# often lower case. Bold and arrow wrappers are looked through (`\\mathbf{E}` names E).
_VECTOR_NAME = re.compile(
    r"(?<![A-Za-z\\])(?:\\(?:mathbf|vec|boldsymbol|bm|hat)\s*\{?\s*)?"
    r"(?:\\([A-Za-z]+)|([A-Za-z])(?:_\{?([A-Za-z0-9]+)\}?)?)(?![A-Za-z])"
)
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


def declared_vectors(texts: Iterable[str]) -> set[str]:
    """Names that `texts` describe as vectors or vector fields. Only the formulas of such a sentence
    count (`$\\mathbf{E}$`, `$a$`): a bare word is not a name, so "vectors" cannot make `s` or `a`
    of the surrounding prose a vector. A wrong name only leaves claims about it unchecked."""
    names: set[str] = set()
    for text in texts:
        for sentence in _SENTENCE.split(text):
            if not _VECTOR_KEYWORD.search(sentence):
                continue
            for match in _MATH.finditer(sentence):
                fragment = next(g for g in match.groups() if g is not None)
                for command, letter, subscript in _VECTOR_NAME.findall(_SIZE.sub(" ", fragment)):
                    if command:
                        if command not in {"times", "cdot", "nabla", "partial"}:
                            names.add(command)
                    elif letter:
                        names.add(f"{letter}_{subscript}" if subscript else letter)
            if len(names) >= MAX_NAMES:
                return set(sorted(names)[:MAX_NAMES])
    return names
