"""Check equations the language model wrote in its prose.

A claim like `\\frac{x^2-1}{x-1} = x+1 \\quad \\text{for } x \\neq 1` is parsed with the same strict parser
as user input and tested numerically at assumption-respecting sample points (complex by default).
`holds` means every valid sample agreed, `fails` that the equation is false, anything else that Minerva
could not decide. Conditions written next to the equation (`for x \\neq 1`) are ignored: the check is
"equal as functions at generic points", which is what simplification claims mean.
"""

from __future__ import annotations

import re
from typing import Literal

import sympy as sp
from pydantic import BaseModel

from .errors import MinervaMathError
from .latexparse import parse_latex
from .models import Limits
from .verify import verify_equal

ClaimStatus = Literal["holds", "conditional", "fails", "unverified", "not_an_equation", "unparsed"]

_TRAILER = re.compile(r"\\q?quad\b.*$|\\text\{[^}]*\}.*$|,\s*(?:for|if|where|when)\b.*$", re.DOTALL)


class ClaimCheck(BaseModel):
    status: ClaimStatus
    detail: str = ""


def split_equation(latex: str) -> list[str] | None:
    """Split on top-level `=`; None if there is no equation."""
    text = _TRAILER.sub("", latex).strip().rstrip(" ,;.")
    parts: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    parts = [p.strip() for p in parts]
    return parts if len(parts) >= 2 and all(parts) else None


def check_claim(latex: str, assumptions: dict[str, list[str]], limits: Limits) -> ClaimCheck:
    sides = split_equation(latex)
    if sides is None:
        return ClaimCheck(status="not_an_equation")
    try:
        parsed = [parse_latex(side, assumptions, limits) for side in sides]
    except MinervaMathError as exc:
        return ClaimCheck(status="unparsed", detail=str(exc))
    outcome, detail = _compare(parsed)
    if outcome == "holds":
        return ClaimCheck(status="holds", detail=detail)
    if outcome == "fails":
        # A claim that is false for complex symbols may be the usual real-variable statement whose
        # condition is written in the prose, not in the formula. Try the weakest natural assumptions.
        symbols = sorted({s.name for p in parsed for s in p.free_symbols} - set(assumptions))
        for label in ("real", "positive"):
            if not symbols:
                break
            relaxed = {**{name: [label] for name in symbols}, **assumptions}
            try:
                retry = [parse_latex(side, relaxed, limits) for side in sides]
            except MinervaMathError:
                break
            if _compare(retry)[0] == "holds":
                names = ", ".join(symbols)
                verb = "is" if len(symbols) == 1 else "are"
                return ClaimCheck(
                    status="conditional",
                    detail=f"false in general, but holds if {names} {verb} {label}",
                )
        return ClaimCheck(status="fails", detail=detail)
    return ClaimCheck(status="unverified", detail=detail)


def _compare(parsed: list[sp.Expr]) -> tuple[str, str]:
    checked = agreed = 0
    for other in parsed[1:]:
        verification = verify_equal(parsed[0], other, symbolic=False)
        checked += verification.numeric_points_checked
        agreed += verification.numeric_points_agreed
        if verification.numeric_points_checked < 4:
            return "unverified", "too few valid sample points"
    if agreed == checked:
        return "holds", f"{agreed}/{checked} sample points agree"
    return "fails", f"{checked - agreed} of {checked} sample points disagree"
