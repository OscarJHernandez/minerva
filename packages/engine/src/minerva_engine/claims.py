"""Check equations the language model wrote in its prose.

A claim like `\\frac{x^2-1}{x-1} = x+1 \\quad \\text{for } x \\neq 1` is parsed with the same strict parser
as user input and tested numerically at assumption-respecting sample points (complex by default).
`holds` means every valid sample agreed. `fails` is reserved for an equation that is false for every
value (the two sides differ by a nonzero constant). An equation that is merely false at generic points
is `not_identity`: Minerva cannot tell a false identity from an equation to solve, a definition or an
equation of motion (`x^2 = 4x`, `F = ma`, `m\\ddot x = -kx`), so it does not call it contradicted.
Anything else could not be decided. Conditions written next to the equation (`for x \\neq 1`) are ignored: the check is
"equal as functions at generic points", which is what simplification claims mean.

A *relation* (a symbol on one side only, or a time derivative: `\\theta = \\phi + \\omega t`, `F = ma`,
`m\\ddot x = -kx`) is not tested as an identity at all. It is checked only against definitions stated earlier in the
conversation, by substitution, and the result names them (`given`); Minerva has not verified those definitions.
A relation that disagrees with them after substitution `fails`; one they cannot decide stays `unverified`.
"""

from __future__ import annotations

import re
from typing import Literal

import sympy as sp
from pydantic import BaseModel

from .errors import MinervaMathError
from .latexparse import parse_latex
from .models import Limits
from .restrictions import domain_restrictions
from .verify import verify_equal

ClaimStatus = Literal[
    "holds", "conditional", "fails", "not_identity", "unverified", "not_an_equation", "unparsed"
]

_TRAILER = re.compile(r"\\q?quad\b.*$|\\text\{[^}]*\}.*$|,\s*(?:for|if|where|when)\b.*$", re.DOTALL)


ClaimKind = Literal["identity", "relation"]

MAX_DEFINITIONS = 100
_SUBSTITUTION_ROUNDS = 20


class ClaimCheck(BaseModel):
    """`kind`: "identity" was tested for all values of its symbols; "relation" was not (see the module
    docstring). `given`: definitions stated earlier that the result depends on (not verified by Minerva)."""

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
        denominators: list[sp.Expr] = []
        try:
            parsed = [
                parse_latex(side, self.assumptions, self.limits, denominators) for side in sides
            ]
        except MinervaMathError as exc:
            return ClaimCheck(status="unparsed", detail=str(exc))
        outcome, detail = _compare(parsed)
        if outcome == "holds":
            return _holds_with_domain(parsed, denominators, detail).model_copy(
                update={"kind": "identity"}
            )
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
        # The sides differ by the same nonzero number for every value: false whatever it means.
        if any(_nonzero_constant(parsed[0] - other) for other in parsed[1:]):
            return ClaimCheck(status="fails", detail=detail, kind="identity")
        # Definitions, coordinate changes and equations of motion are not identities: do not test them
        # as one. Use what the conversation stated about the symbols instead.
        kind = _not_an_identity_kind(parsed)
        if kind is None and len({frozenset(p.free_symbols) for p in parsed}) > 1:
            kind = "a relation between quantities"
        if kind is not None:
            return self._check_relation(parsed, kind)
        # A claim that is false for complex symbols may be the usual real-variable statement whose
        # condition is written in the prose, not in the formula. Try the weakest natural assumptions.
        conditional = _relaxed(sides, self.assumptions, self.limits)
        if conditional is not None:
            return conditional.model_copy(update={"kind": "identity"})
        return ClaimCheck(
            status="not_identity",
            detail=f"not an identity ({detail}); it may still be an equation to solve or a definition",
            kind="identity",
        )

    def _check_relation(self, parsed: list[sp.Expr], kind: str) -> ClaimCheck:
        """Not an identity, so sampling the symbols independently proves nothing. Use the definitions
        stated earlier; without them there is nothing to test."""
        unchecked = ClaimCheck(
            status="unverified", detail=f"{kind}; not tested as an identity", kind="relation"
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


def _nonzero_constant(difference: sp.Expr) -> bool:
    """True if the two sides differ by the same nonzero number everywhere (false for every value)."""
    try:
        value = sp.simplify(difference)
    except Exception:  # undecided is not "constant"
        return False
    return bool(value.is_number and value.is_finite and value != 0)


def _not_an_identity_kind(parsed: list[sp.Expr]) -> str | None:
    """Name the kind of equation that is not meant as an identity, or None.

    Time derivatives (`\\dot x` parses as the symbol `x_dot`) mean an equation of motion; a bare symbol
    on one side that does not occur on the other (`F = m a`, `L = T - V`, `x = 2`) is a definition or
    a value being stated.
    """
    names = {s.name for p in parsed for s in p.free_symbols}
    if any(name.endswith(("_dot", "_ddot")) for name in names):
        return "an equation of motion"
    if len(parsed) == 2:
        for one, other in (parsed, parsed[::-1]):
            if one.is_Symbol and one not in other.free_symbols:
                return "a definition or a value for one symbol"
        if bool(parsed[0].free_symbols) != bool(parsed[1].free_symbols):
            return "an equation to solve"
    return None


def _relaxed(
    sides: list[str], assumptions: dict[str, list[str]], limits: Limits
) -> ClaimCheck | None:
    """`conditional` if the claim holds once symbols are assumed real, then positive.

    A symbol the user declared only `real` is relaxed to `positive` too; other declared
    assumptions are never overridden.
    """
    names = sorted(
        {s.name for side in sides for s in parse_latex(side, assumptions, limits).free_symbols}
    )
    for label in ("real", "positive"):
        relaxed: dict[str, list[str]] = {}
        changed: list[str] = []
        for name in names:
            declared = assumptions.get(name)
            if declared is None:
                relaxed[name] = [label]
                changed.append(name)
            elif label == "positive" and set(declared) == {"real"}:
                relaxed[name] = ["positive"]
                changed.append(name)
            else:
                relaxed[name] = declared
        if not changed:
            continue
        try:
            retry = [parse_latex(side, relaxed, limits) for side in sides]
        except MinervaMathError:
            return None
        if _compare(retry)[0] == "holds":
            verb = "is" if len(changed) == 1 else "are"
            return ClaimCheck(
                status="conditional",
                detail=f"false in general, but holds if {', '.join(changed)} {verb} {label}",
            )
    return None


def _holds_with_domain(
    parsed: list[sp.Expr], denominators: list[sp.Expr], detail: str
) -> ClaimCheck:
    """`holds`, unless the two sides are defined at different points (a cancelled singularity)."""
    notes: list[str] = []
    for other in parsed[1:]:
        found, _ = domain_restrictions(parsed[0], other, denominators)
        notes.extend(n for n in found if n not in notes)
    if notes:
        return ClaimCheck(
            status="conditional",
            detail="equal as functions, but not defined at the same points: valid only where "
            + ", ".join(notes),
        )
    return ClaimCheck(status="holds", detail=detail)
