"""Split model prose into text and formulas, and mark which formulas Minerva can vouch for.

The model is told to quote formulas from tool results, but a prompt is not a guarantee. After the
fact, every formula in its prose is compared with the formulas that tool results and handbook
entries of *this conversation* contain. A formula that is not found is shown as model-written and
unchecked. This is a conservative string comparison on normalised LaTeX, not a proof of anything.
"""

from __future__ import annotations

import re

from .events import Segment

_MATH = re.compile(
    r"\$\$(?P<d1>.+?)\$\$|\\\[(?P<d2>.+?)\\\]|\\\((?P<i2>.+?)\\\)|(?<!\\)\$(?P<i1>[^$\n]+?)(?<!\\)\$",
    re.DOTALL,
)
_STRIP = re.compile(r"\\left|\\right|\\,|\;|\\!|\\ |\s+|\{\}")


def normalize(latex: str) -> str:
    """Whitespace- and size-command-insensitive form used for comparison."""
    text = _STRIP.sub("", latex)
    # Redundant braces around single tokens, e.g. x^{2} versus x^2
    return re.sub(r"\{(\w)\}", r"\1", text)


def split_segments(text: str, known: set[str]) -> list[Segment]:
    """`known` holds normalised formulas that came from tools or handbook entries."""
    segments: list[Segment] = []
    position = 0
    for match in _MATH.finditer(text):
        if match.start() > position:
            segments.append(Segment(type="text", value=text[position : match.start()]))
        display = match.group("d1") is not None or match.group("d2") is not None
        body = next(
            g
            for g in (match.group("d1"), match.group("d2"), match.group("i2"), match.group("i1"))
            if g is not None
        )
        body = body.strip()
        segments.append(
            Segment(
                type="display_math" if display else "math",
                value=body,
                status="from_tool" if normalize(body) in known else "unchecked",
            )
        )
        position = match.end()
    if position < len(text):
        segments.append(Segment(type="text", value=text[position:]))
    return segments
