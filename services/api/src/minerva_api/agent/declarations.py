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
from dataclasses import dataclass

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
# "A matrix is ..." starts with the article, but "A and B are matrices" starts with a name: a conjunction after the
# letter means a list of names, and reading it as an article dropped the first name (`A`), so a formula that
# mentioned only `A` was checked by sampling numbers.
_ARTICLE = re.compile(r"^\s*(?:A|I)\s+(?!(?:and|or|nor)\b)[a-z]")


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


# -- the strict reading, with sources, for the tools that refuse a declared name --------------------
#
# The generous reading above is right for the claim checker (a wrong name only leaves a claim
# "unchecked"). It is wrong for a refusal: "...the Laplace-Runge-Lenz vector, unique to $1/r$
# potentials" made `r` a vector and every later `differentiate` of `r` was refused. The strict reading
# takes a name only where a sentence *declares* it, by one of three shapes:
#   names + "is/are/be" + (adjectives) + keyword   "Let A and B be 3x3 matrices", "$a$ is a vector"
#   keyword + names                                "the vector $\mathbf{r}$", "matrices A and B"
#   names + "do not commute"                       "A and B do not commute"
# and only a bare name or list of names counts: `$1/r$` or `$H = T + V$` is an expression.
# A sentence of the USER that says a name is a scalar clears it from then on; the model's own text
# never clears anything, with one exception that is a definition and not an opinion: `r = |\mathbf{r}|`
# (the bare letter is the norm of the bold one) makes the bare letter a scalar whoever wrote it, and
# from then on the bold spelling names the vector and does not declare the bare letter. A scalar tool
# refuses `\mathbf{r}` itself (the parser has no such command), so the vector cannot get in that way.
# A name the norm rule or a user sentence about that very name ("r is a scalar") made a scalar is
# *held*: only a later USER sentence that declares the bare name brings it back, never the model's.

_BARE_NAME = (
    r"(?:\\(?:mathbf|vec|boldsymbol|bm|hat)\s*\{\s*)?"
    r"(?:\\[A-Za-z]+|[A-Za-z])(?:_\{?[A-Za-z0-9]+\}?)?\s*\}?"
)
_BARE_LIST = re.compile(rf"^\s*{_BARE_NAME}(?:\s*(?:,|;|\\,|\band\b)\s*{_BARE_NAME})*\s*$")
_ONE_NAME = re.compile(
    r"(?:\\(?:mathbf|vec|boldsymbol|bm|hat)\s*\{\s*)?(\\[A-Za-z]+|[A-Za-z])(?:_\{?([A-Za-z0-9]+)\}?)?"
)
# A size such as 3x3, n x n or n\\times n, with whole tokens around the x ("matrix again" is no size).
_STRICT_SIZE = re.compile(r"\b(?:\d+|[A-Za-z])\s*(?:\\times|\u00d7|x)\s*(?:\d+|[A-Za-z])\b")
_MARK = re.compile(r"⟦([^⟧]+)⟧")
# A prose name: one letter with an optional subscript. Math names arrive as ⟦name⟧.
_PROSE_NAME = r"(?<![\w\\])[A-Za-z](?:_(?:\{\w+\}|\w+))?(?![\w])"
_S_NAME = rf"(?:⟦[^⟧]+⟧|{_PROSE_NAME})"
_S_NAMES = rf"{_S_NAME}(?:\s*(?:,|;|\band\b|,\s*and)\s*{_S_NAME})*"
_STOP = r"(?:and|or|but|the|of|to|with|where|which|that)"
_FILLER = rf"(?:(?!{_STOP}\b)[\w\u00d7\-]+\s+){{0,3}}"
_MATRIX_KW = r"(?:matri(?:x|ces)|operators?|noncommut\w*|non-commut\w*)"
_VECTOR_KW = r"(?:vector[- ]fields?|vectors?)"
_SCALAR = re.compile(
    r"\b(?:scalars?|real numbers?|just numbers?|plain numbers?|not (?:a |an )?(?:matri(?:x|ces)|"
    r"vectors?|operators?))\b",
    re.IGNORECASE,
)

_WRAPPER = r"\\(?:mathbf|vec|boldsymbol|bm|hat)"
_LETTER = r"(\\[A-Za-z]+|[A-Za-z])"
_BAR_OPEN = r"(?:\\left\s*)?(?:\\lVert|\\\||\\lvert|\|)"
_BAR_CLOSE = r"(?:\\right\s*)?(?:\\rVert|\\\||\\rvert|\|)"
_NORM_OF = rf"{_BAR_OPEN}\s*{_WRAPPER}\s*\{{?\s*(?P=n)\s*\}}?\s*{_BAR_CLOSE}"
# `r = |\mathbf{r}|` or `|\vec{r}| = r`: the same letter on both sides, the wrapped one inside bars.
_NORM_DEFINITION = re.compile(
    rf"(?<![A-Za-z\\_^{{])(?P<n>\\[A-Za-z]+|[A-Za-z])\s*(?:=|\\equiv)\s*{_NORM_OF}(?![_^])"
    rf"|{_BAR_OPEN}\s*{_WRAPPER}\s*\{{?\s*(?P<m>\\[A-Za-z]+|[A-Za-z])\s*\}}?\s*{_BAR_CLOSE}"
    rf"\s*(?:=|\\equiv)\s*(?P=m)(?![A-Za-z_^])"
)


def norm_defined(sentence: str) -> set[str]:
    """Names a sentence defines as the norm of their own bold or arrowed spelling (`r = |\\mathbf{r}|`)."""
    names: set[str] = set()
    for match in _MATH.finditer(sentence):
        fragment = next(g for g in match.groups() if g is not None)
        for found in _NORM_DEFINITION.finditer(fragment):
            names.add((found.group("n") or found.group("m")).lstrip("\\"))
    return names


def _without_wrapped(sentence: str, names: set[str]) -> str:
    """`sentence` with the bold or arrowed spelling of each held name taken out of its formulas: that
    spelling is the vector, and it must not declare the bare letter, which is a scalar."""
    if not names:
        return sentence
    alternatives = "|".join(
        sorted((re.escape(n) if len(n) == 1 else rf"\\{re.escape(n)}" for n in names), key=len)[
            ::-1
        ]
    )
    wrapped = re.compile(
        rf"{_WRAPPER}\s*(?:\{{\s*(?:{alternatives})\s*\}}|(?:{alternatives})(?![A-Za-z]))"
    )

    def strip(match: re.Match[str]) -> str:
        return wrapped.sub("0", match.group(0))

    return _MATH.sub(strip, sentence)


@dataclass(frozen=True)
class Declared:
    kind: str  # "matrix" | "vector"
    role: str  # "user" | "assistant"
    quote: str  # the sentence, as written (shown to the user, never logged)


def _plain(sentence: str) -> str:
    """The sentence with each formula that is a bare name (list) written as ⟦name⟧ tokens and every
    other formula replaced by one opaque word, so that an expression never yields a name."""

    def swap(match: re.Match[str]) -> str:
        fragment = next(g for g in match.groups() if g is not None)
        if not _BARE_LIST.match(fragment):
            return " EXPR "
        names = [
            (c.lstrip("\\") + (f"_{sub}" if sub else "")) if c else ""
            for c, sub in _ONE_NAME.findall(fragment)
        ]
        return " " + ", ".join(f"⟦{n}⟧" for n in names if n) + " "

    return _MATH.sub(swap, sentence)


_NAME_FOLLOWS = re.compile(
    r"\s*(?:$|[,;.:)]|(?:and|or|is|are|be|do|does|denotes?)\b)", re.IGNORECASE
)


def _names_of(text: str, span: tuple[int, int], *, capitals_only: bool) -> set[str]:
    """The names inside `text[span]`. A bare `a`, `A` or `I` in prose is a name only when what follows
    is a copula, a list separator or the end ("Let A and B be ..."), not another word ("A matrix")."""
    found: set[str] = set()
    for token in re.finditer(_S_NAME, text[span[0] : span[1]]):
        raw = token.group(0)
        marked = _MARK.fullmatch(raw)
        name = marked.group(1) if marked else raw.replace("{", "").replace("}", "")
        article = not marked and name in {"a", "A", "I"}
        if article and not _NAME_FOLLOWS.match(text, span[0] + token.end()):
            continue  # the article or the pronoun, not a name
        if capitals_only and not name[0].isupper():
            continue
        found.add(name)
    return found


def _strict_names(sentence: str, kind: str) -> set[str]:
    text = _STRICT_SIZE.sub(" SIZE ", _plain(sentence))
    keyword = _MATRIX_KW if kind == "matrix" else _VECTOR_KW
    capitals_only = kind == "matrix"
    shapes = [
        rf"(?P<n>{_S_NAMES})\s+(?:is|are|be|denotes?)\s+(?:an?\s+|the\s+|some\s+)?{_FILLER}"
        rf"(?:{keyword}\s*)+",
        rf"\b(?:{keyword})\s+(?P<n>{_S_NAMES})(?![\w])",
    ]
    if kind == "matrix":
        shapes.append(rf"(?P<n>{_S_NAMES})\s+(?:do|does|don't|do not|does not)\s+commute")
    names: set[str] = set()
    for pattern in shapes:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            names |= _names_of(text, match.span("n"), capitals_only=capitals_only)
    return names


def _cleared_names(sentence: str) -> set[str]:
    """Names a USER sentence calls scalars ("r is a scalar", "$x$ and $y$ are numbers")."""
    if not _SCALAR.search(sentence):
        return set()
    text = _plain(sentence)
    return _names_of(text, (0, len(text)), capitals_only=False)


def _held_by_user(sentence: str) -> set[str]:
    """Names a USER sentence ties to the scalar keyword itself ("r is a scalar", "the scalar $r$",
    "r is not a vector"): narrower than `_cleared_names`, which takes every name of the sentence."""
    if not _SCALAR.search(sentence):
        return set()
    text = _plain(sentence)
    keyword = _SCALAR.pattern.replace(r"\b", "")
    names: set[str] = set()
    for pattern in (
        rf"(?P<n>{_S_NAMES})\s+(?:is|are|be)\s+(?:an?\s+|the\s+)?{_FILLER}{keyword}(?![\w-])",
        rf"\b{keyword}\s+(?P<n>{_S_NAMES})(?![\w])",
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            names |= _names_of(text, match.span("n"), capitals_only=False)
    return names


@dataclass(frozen=True)
class Scan:
    found: dict[str, Declared]
    cleared: set[str]  # names a user sentence called scalars and nothing declared again
    held: set[
        str
    ]  # scalars by a norm definition or by the user's word: the model cannot re-declare them


def scan(
    entries: Iterable[tuple[str, str]], *, strict: bool = True, held: Iterable[str] = ()
) -> Scan:
    """Read `(role, text)` messages in order. A user sentence that calls a name a scalar removes it and
    records it as cleared; a later declaration adds it back (and un-clears), except that a *held* name
    (see the comment above) comes back only through a user sentence that declares the bare name.
    `held` starts the scan with names held by earlier text (the claim check of a reply reads only the
    reply, but what the conversation settled before it still applies)."""
    found: dict[str, Declared] = {}
    cleared: set[str] = set()
    holding: set[str] = set(held)
    for role, text in entries:
        for sentence in _SENTENCE.split(text):
            quote = " ".join(sentence.split())[:240]
            if role == "user" and quote:
                for name in _cleared_names(sentence):
                    found.pop(name, None)
                    cleared.add(name)
                holding |= _held_by_user(sentence)
            defined_here = norm_defined(sentence)
            for name in defined_here:
                found.pop(name, None)
                holding.add(name)
            readable = _without_wrapped(sentence, holding)
            for kind, keyword, reader in (
                ("matrix", _KEYWORD, declared_matrices),
                ("vector", _VECTOR_KEYWORD, declared_vectors),
            ):
                if not keyword.search(readable):
                    continue
                names = _strict_names(readable, kind) if strict else reader([readable])
                for name in names:
                    if name in defined_here:
                        continue  # the sentence that defines `r` as a norm cannot also declare it
                    if name in holding:
                        if role != "user":
                            continue
                        holding.discard(name)
                    found[name] = Declared(kind, role, quote)
                    cleared.discard(name)
    return Scan(found, cleared, holding)


def scan_declarations(
    entries: Iterable[tuple[str, str]], *, strict: bool = True
) -> tuple[dict[str, Declared], set[str]]:
    """(declarations, cleared) for `(role, text)` messages in order (see `scan`)."""
    result = scan(entries, strict=strict)
    return result.found, result.cleared


def declared_in_reply(texts: Iterable[str], held: Iterable[str] = ()) -> tuple[set[str], set[str]]:
    """(matrices, vectors) the model's own prose declares, read generously, for the claim check of that
    reply. Names the conversation holds as scalars are not declared by it."""
    found = scan([("assistant", text) for text in texts], strict=False, held=held).found
    return (
        {n for n, d in found.items() if d.kind == "matrix"},
        {n for n, d in found.items() if d.kind == "vector"},
    )


def read_declarations(
    entries: Iterable[tuple[str, str]], *, strict: bool = True
) -> dict[str, Declared]:
    """Names declared as matrices/operators or vectors by `(role, text)` messages in order."""
    return scan_declarations(entries, strict=strict)[0]
