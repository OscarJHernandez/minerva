"""Constants of motion an earlier result of the conversation found and verified.

`find_conserved_quantities` names what it finds `C_{1}`, `C_{2}`, ... and the user then asks about a
relation among them ("the derivative of Equation 14 with respect to time"). Typed as bare names they
are symbols with no dependence on the motion, so nothing can be differentiated. The expressions they
stand for are in the stored discovery cards, verified there, and this module reads them: the SERVER
replaces a name by its expression, never the model (`time_derivative` has no field for a definition,
a test pins that), and only when the name has exactly one verified definition in one Lagrangian system.
Anything else is a typed refusal that says why, never a guess.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from minerva_engine.notation.models import InputCode


@dataclass(frozen=True)
class KnownConstant:
    name: str  # `C_1`
    expression: str  # the right side as LaTeX
    verified: bool
    system: str  # "lagrangian" | "hamiltonian"
    function: str  # the Lagrangian or Hamiltonian the search was run on
    coordinates: tuple[str, ...]
    card: str  # the id of the card it came from (for the note shown to the user)


@dataclass(frozen=True)
class Expansion:
    definitions: dict[str, str]  # name -> expression, for the names the expression uses
    function: str  # the one Lagrangian they were found for
    cards: tuple[str, ...]


@dataclass(frozen=True)
class Refusal:
    code: InputCode
    message: str
    symbols: tuple[str, ...]


def _plain_name(latex_name: str) -> str:
    return latex_name.replace("{", "").replace("}", "").replace(" ", "")


def constants_of_card(card: Mapping[str, Any]) -> list[KnownConstant]:
    """The constants a stored `discovery` card lists, in its order. Anything malformed is skipped."""
    report = (card.get("data") or {}).get("report")
    if card.get("type") != "discovery" or not isinstance(report, Mapping):
        return []
    function = report.get("function_latex")
    coordinates = report.get("coordinates")
    if not isinstance(function, str) or not isinstance(coordinates, list):
        return []
    found: list[KnownConstant] = []
    for item in report.get("constants") or []:
        text = item.get("expression_latex") if isinstance(item, Mapping) else None
        if not isinstance(text, str) or "=" not in text:
            continue
        left, right = text.split("=", 1)
        name = _plain_name(left)
        if not re.fullmatch(r"[A-Za-z]+_[A-Za-z0-9]+", name) or not right.strip():
            continue
        found.append(
            KnownConstant(
                name=name,
                expression=right.strip(),
                verified=bool(item.get("verified")),
                system=str(report.get("system", "")),
                function=function,
                coordinates=tuple(str(c) for c in coordinates),
                card=str(card.get("id", "")),
            )
        )
    return found


def constants_of_blocks(
    blocks_per_message: Iterable[Iterable[Mapping[str, Any]]],
) -> list[KnownConstant]:
    """Every constant in the cards of the stored messages (each message's `blocks`), oldest first."""
    found: list[KnownConstant] = []
    for blocks in blocks_per_message:
        for block in blocks:
            card = block.get("card") if block.get("kind") == "card" else None
            if isinstance(card, Mapping):
                found.extend(constants_of_card(card))
    return found


def _mentions(latex: str, name: str) -> bool:
    base, _, subscript = name.partition("_")
    pattern = (
        rf"(?<![A-Za-z\\]){re.escape(base)}_(?:\{{\s*{re.escape(subscript)}\s*\}}|{re.escape(subscript)})"
        r"(?![A-Za-z0-9])"
    )
    return re.search(pattern, latex) is not None


def expand_constants(latex: str, known: Iterable[KnownConstant]) -> Expansion | Refusal | None:
    """The definitions to insert for the constants `latex` names, `None` when it names none of them.

    Refused (`Refusal`, a typed reason) when a name has two different definitions, an unverified one, or
    belongs to a Hamiltonian search, or when the names come from different systems: a derivative on the
    equations of motion of one Lagrangian is not a statement about another."""
    candidates: dict[str, list[KnownConstant]] = {}
    for constant in known:
        if _mentions(latex, constant.name):
            candidates.setdefault(constant.name, []).append(constant)
    if not candidates:
        return None
    chosen: dict[str, KnownConstant] = {}
    for name in sorted(candidates):
        distinct = {(c.function, c.expression): c for c in candidates[name]}
        if len(distinct) > 1:
            return Refusal(
                InputCode.AMBIGUOUS_CONSTANT,
                f"{name} has more than one definition in this conversation (different results used "
                "the same name). Say which result is meant, or ask for the constants again.",
                (name,),
            )
        (only,) = distinct.values()
        if not only.verified:
            return Refusal(
                InputCode.CONSTANT_NOT_VERIFIED,
                f"{name} was listed by an earlier result but was not verified as a constant of motion, "
                "so it is not used.",
                (name,),
            )
        if only.system != "lagrangian":
            return Refusal(
                InputCode.UNSUPPORTED_DEFINITION,
                f"{name} was found for a Hamiltonian system; only constants of a Lagrangian system "
                "can be inserted here.",
                (name,),
            )
        chosen[name] = only
    functions = {c.function for c in chosen.values()}
    if len(functions) > 1:
        return Refusal(
            InputCode.AMBIGUOUS_CONSTANT,
            "These constants were found for different Lagrangians; a derivative on the equations of "
            "motion needs one system.",
            tuple(sorted(chosen)),
        )
    return Expansion(
        {name: c.expression for name, c in chosen.items()},
        functions.pop(),
        tuple(dict.fromkeys(c.card for c in chosen.values())),
    )
