"""Split model prose into text and formulas, and mark which formulas Minerva can vouch for.

The model is told to quote formulas from tool results, but a prompt is not a guarantee. After the
fact, every formula in its prose is compared with the formulas that tool results and handbook
entries of *this conversation* contain. A formula that is not found is shown as model-written and
unchecked. This is a conservative string comparison on normalised LaTeX, not a proof of anything.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from .events import Segment

# `$` follows the pandoc rule: the opener is not followed by whitespace, the closer is not preceded by
# whitespace or followed by a digit, so prices such as "$5 to $10" stay text. A bare environment
# (`\\begin{aligned}...\\end{aligned}`) is a display formula. `apps/web/src/lib/inline-math.ts` mirrors this.
_MATH = re.compile(
    r"\$\$(?P<d1>.+?)\$\$"
    r"|\\\[(?P<d2>.+?)\\\]"
    r"|\\\((?P<i2>.+?)\\\)"
    r"|(?P<d3>\\begin\{(?P<env>align\*?|aligned|equation\*?|gather\*?|eqnarray\*?)\}.+?\\end\{(?P=env)\})"
    r"|(?<![\\$])\$(?!\s|\$)(?P<i1>[^$\n]*?[^$\s\\])\$(?![\d$])",
    re.DOTALL,
)
# Spacing commands carry no meaning; `\\` (line break) is not one and is kept. Anchored so that
# `\leftarrow` is not read as `\left` + `arrow`.
_SPACING = re.compile(
    r"\\(?:left|right|quad|qquad|thinspace|medspace|thickspace)(?![A-Za-z])|\\[,;:>! ]"
)
_ALIASES = (
    (re.compile(r"\\exponentialE(?![A-Za-z])"), r"\\mathrm{e}"),
    (re.compile(r"\\imaginaryI(?![A-Za-z])"), r"\\mathrm{i}"),
)
_COMMAND_GAP = re.compile(r"(\\[A-Za-z]+)\s+(?=[A-Za-z])")
_BRACED = re.compile(r"(?<!\\mathrm)\{(\w)\}")


def normalize(latex: str) -> str:
    """Whitespace-, spacing-command- and alias-insensitive form used for comparison.

    `\\exponentialE` and `\\imaginaryI` equal `\\mathrm{e}` and `\\mathrm{i}`; a bare `e` or `i` is a
    variable and is left alone. A space that ends a command name is kept (`\\in t` is not `\\int`).
    """
    text = latex
    for pattern, replacement in _ALIASES:
        text = pattern.sub(replacement, text)
    text = _SPACING.sub(" ", text.replace("\\\\", "\x00")).replace("\x00", "\\\\")
    text = text.replace("{}", "")
    text = _COMMAND_GAP.sub("\\1\x01", text)
    text = re.sub(r"\s+", "", text).replace("\x01", " ")
    # Redundant braces around single tokens, e.g. x^{2} versus x^2 (repeated: {{x}} is x)
    while (reduced := _BRACED.sub(r"\1", text)) != text:
        text = reduced
    return text


_TOKEN = re.compile(r"[\w^+\-*/().{}\\|=<>!,]+")
_FUNCTIONS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "exp", "log", "ln", "sqrt",
    "det", "pi", "int", "sum", "arcsin", "arccos", "arctan", "frac", "cdot", "dot", "ddot", "left",
    "right", "alpha", "beta", "gamma", "delta", "theta", "omega", "lambda", "sigma", "infty", "mathbf",
}  # fmt: skip
_WORD = re.compile(r"[A-Za-z]{3,}")


def _mathy(token: str) -> bool:
    """A whitespace-separated token that can be part of a bare formula (no ordinary words)."""
    if not _TOKEN.fullmatch(token):
        return False
    return all(w.lower() in _FUNCTIONS for w in _WORD.findall(token))


def _prose(token: str) -> bool:
    """A bare two-or-more-letter word like "So", "is", "of" (not a function name)."""
    word = token.strip("*_.,;:")
    return bool(re.fullmatch(r"[A-Za-z]{2,}", word)) and word.lower() not in _FUNCTIONS


def _bare_equations(text: str) -> list[tuple[int, int]]:
    """Spans of equations written without $...$, e.g. `x^3 + x^2 + 1 = 0`.

    Maximal runs of whitespace-separated tokens that contain no ordinary words and have a token
    holding `=`, with sentence punctuation and markdown emphasis left outside the span.
    """
    spans: list[tuple[int, int]] = []
    tokens = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
    i = 0
    while i < len(tokens):
        if not _mathy(tokens[i][2].strip("*_")):
            i += 1
            continue
        j = i
        while j < len(tokens) and _mathy(tokens[j][2].strip("*_")):
            j += 1
        # Two-letter words such as "So" or "is" pass the word test; trim them from the ends.
        while j - 1 > i and _prose(tokens[j - 1][2]):
            j -= 1
        while i < j - 1 and _prose(tokens[i][2]):
            i += 1
        start, end = tokens[i][0], tokens[j - 1][1]
        run = text[start:end]
        lead = len(run) - len(run.lstrip("*_"))
        trail = len(run) - len(run.rstrip("*_.,;:"))
        start, end = start + lead, end - trail
        body = text[start:end]
        parts = body.split("=")
        if (
            len(parts) >= 2
            and all(re.search(r"[A-Za-z0-9)]", q) for q in parts)
            and not re.search(r"[<>]=?=|==", body)
        ):
            spans.append((start, end))
        while j < len(tokens) and _mathy(tokens[j][2].strip("*_")):
            j += 1
        i = j
    return spans


def _mark(body: str, known: set[str], quoted: Mapping[str, str]) -> tuple[str, str | None]:
    """The mark of a formula the model wrote: produced by a tool, merely quoted, or unchecked."""
    key = normalize(body)
    if key in known:
        return "from_tool", None
    if key in quoted:
        return "quoted", quoted[key]
    return "unchecked", None


def _text_segments(text: str, known: set[str], quoted: Mapping[str, str]) -> list[Segment]:
    out: list[Segment] = []
    position = 0
    for start, end in _bare_equations(text):
        if start > position:
            out.append(Segment(type="text", value=text[position:start]))
        body = text[start:end].strip()
        status, note = _mark(body, known, quoted)
        out.append(Segment(type="math", value=body, status=status, note=note))
        position = end
    if position < len(text):
        out.append(Segment(type="text", value=text[position:]))
    return out


def split_segments(
    text: str, known: set[str], quoted: Mapping[str, str] | None = None
) -> list[Segment]:
    """`known` holds normalised formulas a tool produced as results, or machine-checked handbook
    statements. `quoted` maps normalised formulas that appeared in tool output without being a verified
    result (the user's own input as read, a handbook statement that is only stated) to the reason."""
    quoted = quoted or {}
    segments: list[Segment] = []
    position = 0
    for match in _MATH.finditer(text):
        if match.start() > position:
            segments.extend(_text_segments(text[position : match.start()], known, quoted))
        display = any(match.group(n) is not None for n in ("d1", "d2", "d3"))
        body = next(
            g
            for g in (
                match.group("d1"),
                match.group("d2"),
                match.group("i2"),
                match.group("d3"),
                match.group("i1"),
            )
            if g is not None
        )
        body = body.strip()
        status, note = _mark(body, known, quoted)
        segments.append(
            Segment(
                type="display_math" if display else "math",
                value=body,
                status=status,
                note=note,
            )
        )
        position = match.end()
    if position < len(text):
        segments.extend(_text_segments(text[position:], known, quoted))
    return segments
