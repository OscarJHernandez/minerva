"""Deterministic first steps: a request whose tool is not in doubt does not need the model to pick it.

A user who writes a Lagrangian and asks for its equations of motion wants `analyze_lagrangian`. A small
model costs several calls (and failed ones) to get there; this module reads the same message with fixed
rules and proposes the call. It only *proposes*: the call goes through `execute` like any other, its
result is checked and shown like any other, and if it does not come back verified the turn continues as if
no route existed. The model still writes every word the user reads. Nothing here computes or decides
correctness, and it never runs unless both signals are present: a formula that reads as a Lagrangian with
velocities, and a request for equations of motion in the words around it (a keyword alone never routes).

Adding a rule means adding a function to `RULES`; each is a pure function of the message.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .segments import split_segments
from .tools import strip_assignment

MAX_COORDINATES = 8

_ASKS_FOR_EQUATIONS = re.compile(
    r"equations?\s+of\s+motion|euler[\s\-]*lagrange\s+equations?|\be\.?o\.?m\.?s?\b|"
    r"\bequations?\s+(?:that\s+)?(?:govern|describe)s?\b",
    re.IGNORECASE,
)
# Requests the first call cannot answer alone: the model plans those from the start.
_NEEDS_PLANNING = re.compile(
    r"\bhamilton(?:ian)?\b|\blegendre\b|\bnoether\b.*\bwithout\b|\bcompare\b|\bversus\b|\bvs\.?\b",
    re.IGNORECASE,
)
_VELOCITY_LATEX = re.compile(r"\\dot\s*(\{\s*(\\?[A-Za-z]+)\s*\}|\\?([A-Za-z]+))(\s*[_^])?")
_VELOCITY_TEXT = re.compile(r"\b([A-Za-z]\w*?)_dot\b")
_ACCELERATION = re.compile(r"\\ddot|_ddot\b")
_LAGRANGIAN_LINE = re.compile(r"(?:^|\n|\s)L\s*(?:\(.*?\))?\s*=\s*([^\n]+)")


@dataclass(frozen=True)
class Route:
    """The call to run, and a fixed label for why (never the user's text)."""

    tool: str
    arguments: dict[str, object] = field(default_factory=dict)
    reason: str = "lagrangian_equations"


Rule = Callable[[str], "Route | None"]


_DELIMITED = re.compile(r"\$|\\\(|\\\[")


def _split(text: str) -> tuple[list[str], str]:
    """The formulas of the message and the words around them. Delimited math (`$...$`) when there is any;
    otherwise a Lagrangian typed as `L = ...` on its own, whose right side runs to the end of the line."""
    if _DELIMITED.search(text):
        segments = split_segments(text, set())
        return (
            [s.value for s in segments if s.type != "text"],
            " ".join(s.value for s in segments if s.type == "text"),
        )
    match = _LAGRANGIAN_LINE.search(text)
    if match is None:
        return [], text
    return [match.group(1).strip().rstrip(".")], text[: match.start(1)] + " " + text[match.end(1) :]


def _coordinates(formula: str) -> list[str] | None:
    """Coordinates from the velocities written in `formula`, or None when that is not unambiguous."""
    names: list[str] = []
    for match in _VELOCITY_LATEX.finditer(formula):
        if match.group(4) and match.group(4).strip() == "_":
            return None  # `\dot{q}_1`: an indexed coordinate, left to the model
        name = (match.group(2) or match.group(3) or "").lstrip("\\")
        if "_" in (match.group(2) or ""):
            return None
        names.append(name)
    names += [m.group(1) for m in _VELOCITY_TEXT.finditer(formula)]
    unique = list(dict.fromkeys(names))
    if not unique or len(unique) > MAX_COORDINATES or any(not n for n in unique):
        return None
    return unique


def lagrangian_equations(text: str) -> Route | None:
    """One Lagrangian in the message and a request for its equations of motion."""
    formulas, words = _split(text)
    if not _ASKS_FOR_EQUATIONS.search(words) or _NEEDS_PLANNING.search(words):
        return None
    candidates = [f for f in formulas if _coordinates(f) is not None]
    if len(candidates) != 1 or _ACCELERATION.search(candidates[0]):
        return None
    source = strip_assignment(candidates[0].strip(), "L")
    coordinates = _coordinates(candidates[0])
    if coordinates is None or "=" in source:
        return None
    return Route("analyze_lagrangian", {"lagrangian": source, "coordinates": coordinates})


RULES: tuple[Rule, ...] = (lagrangian_equations,)


def plan_route(user_text: str) -> Route | None:
    """The first rule that applies to this message, or None (the model plans)."""
    for rule in RULES:
        found = rule(user_text)
        if found is not None:
            return found
    return None
