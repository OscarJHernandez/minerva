"""Check equations the language model wrote in its prose.

A claim like `\\frac{x^2-1}{x-1} = x+1 \\quad \\text{for } x \\neq 1` is parsed with the same strict parser
as user input and tested numerically at assumption-respecting sample points (complex by default).
`holds` means every valid sample agreed, `fails` that the equation is false, anything else that Minerva
could not decide. Conditions written next to the equation (`for x \\neq 1`) are ignored: the check is
"equal as functions at generic points", which is what simplification claims mean.

Not every equation is an identity. If a symbol occurs on one side only (`\\theta = \\phi + \\omega t`,
`F = ma`, `x = -1`) the equation relates quantities: it is a definition, a coordinate change, a constraint or
an equation to solve, and disagreement at independently sampled values says nothing about it. Such a
*relation* is never `fails` on its own. It is checked only against definitions stated earlier in the
conversation, by substitution, and the result names those definitions; Minerva has not verified them.
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
ClaimKind = Literal["identity", "relation"]

MAX_DEFINITIONS = 100
_SUBSTITUTION_ROUNDS = 20

_TRAILER = re.compile(r"\\q?quad\b.*$|\\text\{[^}]*\}.*$|,\s*(?:for|if|where|when)\b.*$", re.DOTALL)


class ClaimCheck(BaseModel):
    """`kind`: "identity" was tested for all values of its symbols; "relation" has a symbol on one side
    only. `given`: definitions stated earlier that the result depends on (not verified by Minerva)."""

    status: ClaimStatus
    detail: str = ""
    kind: ClaimKind | None = None
    given: list[str] = []


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
    """Check one equation on its own (no definitions from earlier in the conversation)."""
    return ClaimChecker(assumptions, limits).check(latex)


def check_claims(
    formulas: list[str],
    assumptions: dict[str, list[str]],
    relations: list[str],
    limits: Limits,
) -> list[ClaimCheck]:
    """Check equations in order. `relations` are formulas stated earlier in the conversation: the
    ones that are definitions (`\\theta = \\phi + \\omega t`) are used by substitution. Each checked
    formula that is itself a definition is remembered for the ones after it."""
    checker = ClaimChecker(assumptions, limits)
    for relation in relations:
        checker.learn(relation)
    results: list[ClaimCheck] = []
    for latex in formulas:
        try:
            result = checker.check(latex)
        except Exception:
            result = ClaimCheck(status="unparsed", detail="could not be checked")
        results.append(result)
        checker.learn(latex)
    return results


def _text(latex: str) -> str:
    return _TRAILER.sub("", latex).strip().rstrip(" ,;.")


def _definition(parsed: list[sp.Expr]) -> tuple[sp.Symbol, sp.Expr] | None:
    """`symbol = expression` where the expression does not contain the symbol."""
    if len(parsed) != 2:
        return None
    a, b = parsed
    if isinstance(a, sp.Symbol) and a not in b.free_symbols:
        return a, b
    if isinstance(b, sp.Symbol) and b not in a.free_symbols:
        return b, a
    return None


class ClaimChecker:
    """Checks equations one after another against the definitions stated so far."""

    def __init__(self, assumptions: dict[str, list[str]], limits: Limits) -> None:
        self.assumptions = assumptions
        self.limits = limits
        # symbol -> (right side with earlier definitions substituted, its text, texts it was built on)
        self.definitions: dict[sp.Symbol, tuple[sp.Expr, str, list[str]]] = {}

    def _parse(self, latex: str) -> list[sp.Expr] | None:
        sides = split_equation(latex)
        if sides is None:
            return None
        try:
            return [parse_latex(side, self.assumptions, self.limits) for side in sides]
        except MinervaMathError:
            return None

    def learn(self, latex: str) -> None:
        """Remember `latex` if it is a definition. A later definition of a symbol replaces the earlier
        one; one that would define a symbol in terms of itself is ignored."""
        parsed = self._parse(latex)
        found = _definition(parsed) if parsed else None
        if found is None:
            return
        symbol, rhs = found
        (rhs,), used = self._substitute([rhs])
        if symbol in rhs.free_symbols:
            return
        if symbol not in self.definitions and len(self.definitions) >= MAX_DEFINITIONS:
            return
        self.definitions[symbol] = (rhs, _text(latex), self._given(used))

    def _given(self, used: list[sp.Symbol]) -> list[str]:
        """Texts of the definitions in `used` and of those they were built on, each once."""
        given: list[str] = []
        for s in used:
            for text in [*self.definitions[s][2], self.definitions[s][1]]:
                if text not in given:
                    given.append(text)
        return given

    def _substitute(self, exprs: list[sp.Expr]) -> tuple[list[sp.Expr], list[sp.Symbol]]:
        used: list[sp.Symbol] = []
        for _ in range(_SUBSTITUTION_ROUNDS):
            present = {s for e in exprs for s in e.free_symbols if s in self.definitions}
            if not present:
                break
            used.extend(s for s in sorted(present, key=lambda s: s.name) if s not in used)
            mapping = {s: self.definitions[s][0] for s in present}
            exprs = [e.xreplace(mapping) for e in exprs]
        return exprs, used

    def check(self, latex: str) -> ClaimCheck:
        sides = split_equation(latex)
        if sides is None:
            return ClaimCheck(status="not_an_equation")
        try:
            parsed = [parse_latex(side, self.assumptions, self.limits) for side in sides]
        except MinervaMathError as exc:
            return ClaimCheck(status="unparsed", detail=str(exc))
        outcome, detail = _compare(parsed)
        if outcome == "holds":
            return ClaimCheck(status="holds", detail=detail, kind="identity")
        if outcome != "fails":
            return ClaimCheck(status="unverified", detail=detail)
        # Prose uses e and i for Euler's number and the imaginary unit; the parser keeps them as plain
        # symbols (see AGENTS.md). Give the claim the benefit of that reading, and say so.
        used = {s.name for p in parsed for s in p.free_symbols}
        if used & {"e", "i"}:
            swap = {
                s: (sp.E if s.name == "e" else sp.I)
                for p in parsed
                for s in p.free_symbols
                if s.name in ("e", "i")
            }
            if _compare([p.subs(swap) for p in parsed])[0] == "holds":
                return ClaimCheck(
                    status="holds",
                    detail="holds reading e as Euler's number and i as the imaginary unit",
                    kind="identity",
                )
        symbol_sets = [frozenset(p.free_symbols) for p in parsed]
        if any(s != symbol_sets[0] for s in symbol_sets[1:]):
            return self._check_relation(parsed)
        return self._check_identity(sides, parsed, detail)

    def _check_identity(self, sides: list[str], parsed: list[sp.Expr], detail: str) -> ClaimCheck:
        # A claim that is false for complex symbols may be the usual real-variable statement whose
        # condition is written in the prose, not in the formula. Try the weakest natural assumptions.
        symbols = sorted({s.name for p in parsed for s in p.free_symbols} - set(self.assumptions))
        for label in ("real", "positive"):
            if not symbols:
                break
            relaxed = {**{name: [label] for name in symbols}, **self.assumptions}
            try:
                retry = [parse_latex(side, relaxed, self.limits) for side in sides]
            except MinervaMathError:
                break
            if _compare(retry)[0] == "holds":
                names = ", ".join(symbols)
                verb = "is" if len(symbols) == 1 else "are"
                return ClaimCheck(
                    status="conditional",
                    detail=f"false in general, but holds if {names} {verb} {label}",
                    kind="identity",
                )
        return ClaimCheck(status="fails", detail=detail, kind="identity")

    def _check_relation(self, parsed: list[sp.Expr]) -> ClaimCheck:
        """A symbol is on one side only: not an identity, so sampling the symbols independently proves
        nothing. Use the definitions stated earlier; without them there is nothing to test."""
        unchecked = ClaimCheck(
            status="unverified",
            detail="a definition or relation between quantities, not an identity; not checked",
            kind="relation",
        )
        substituted, used = self._substitute(parsed)
        if not used:
            return unchecked
        given = self._given(used)
        sets = [frozenset(p.free_symbols) for p in substituted]
        if any(s != sets[0] for s in sets[1:]):
            return unchecked
        outcome, detail = _compare(substituted)
        if outcome == "holds":
            return ClaimCheck(
                status="conditional",
                detail="holds given " + "; ".join(given),
                kind="relation",
                given=given,
            )
        if outcome == "fails" and _definition(parsed) is None:
            return ClaimCheck(
                status="fails",
                detail="does not agree with " + "; ".join(given) + f" ({detail})",
                kind="relation",
                given=given,
            )
        return unchecked


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
