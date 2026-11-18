"""Deterministic first steps: a request whose tool is not in doubt does not need the model to pick it.

A user who writes a Lagrangian and asks for its equations of motion wants `analyze_lagrangian`. A small
model costs several calls (and failed ones) to get there; this module reads the same message with fixed
rules and proposes the call. It only *proposes*: the call goes through the same resolve/clean/`execute` path
as any other, its result is checked and shown like any other, and if it does not come back as one of the
outcomes the rule accepts the turn continues as if no route existed. The model still writes every word the
user reads. Nothing here computes or decides correctness, and a rule never runs on one signal: it needs
structure it can read in the message or the conversation AND words that ask for the operation (a keyword
alone never routes). A rule that cannot read every argument from what the user wrote returns None and the
model plans; no rule fills in a variable, bound, size or assumption the user did not give.

Rules (`RULES`): `lagrangian_equations`; `followup_reuse` ("differentiate that result": `@rN` of the one
result the previous turn produced); `ode_analyze` (one equation with a derivative of an applied function:
classification only, solving stays the user's choice); `lagrangian_followup` ("what is conserved?" after one
analysed Lagrangian); `matrix_simplify` (a formula over matrices the user declared as n x n). Each is a pure function of the message and a `RouteContext`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from minerva_engine.notation.derivatives import spelled_derivatives

from .mathinput import strip_assignment
from .segments import split_segments

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
_ACCELERATION = re.compile(r"\\ddot")
_LAGRANGIAN_LINE = re.compile(r"(?:^|\n|\s)L\s*(?:\(.*?\))?\s*=\s*([^\n]+)")


@dataclass(frozen=True)
class Route:
    """The call to run, and a fixed label for why (never the user's text).

    `answers` are the tool statuses the rule counts as an answer worth keeping: a call that comes back with
    any other status is dropped and the model plans (a rule whose honest outcome is a refusal, like a
    non-commuting product, lists it here).
    """

    tool: str
    arguments: dict[str, object] = field(default_factory=dict)
    reason: str = "lagrangian_equations"
    answers: frozenset[str] = frozenset({"ok"})


@dataclass(frozen=True)
class RouteContext:
    """What a rule may read besides the message: the conversation, never the model's words."""

    user_messages: Sequence[str] = ()  # every user message of the conversation, oldest first
    results: Mapping[str, str] = field(default_factory=dict)  # verified result id -> engine LaTeX
    latest: tuple[str, ...] = ()  # result ids the previous assistant turn produced
    # The tool calls the previous assistant turn made, as (tool, arguments): what the user can point at.
    previous_calls: tuple[tuple[str, Mapping[str, object]], ...] = ()


NO_CONTEXT = RouteContext()
Rule = Callable[[str, RouteContext], "Route | None"]


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
    if "\\" not in formula:
        spelled = spelled_derivatives(formula)
        if spelled is None:
            return None  # a name with two readings (`xddot`): left to the model
        if any(order > 1 for _, order in spelled):
            return None  # an acceleration: not a first-order Lagrangian, left to the model
        names += [stem for stem, _ in spelled]
    unique = list(dict.fromkeys(names))
    if not unique or len(unique) > MAX_COORDINATES or any(not n for n in unique):
        return None
    return unique


def lagrangian_equations(text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
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


# --- reading a symbol out of engine-printed LaTeX ------------------------------------------------------
_FUNCTION_COMMANDS = frozenset(
    [
        "frac",
        "left",
        "right",
        "cdot",
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "ln",
        "sqrt",
        "arcsin",
        "arccos",
        "arctan",
        "pi",
    ]
)


def single_symbol(latex: str) -> str | None:
    """The one variable of an engine-printed expression, or None when there is not exactly one plain letter.

    Conservative on purpose: any command that is not an elementary function (a Greek letter, an operator),
    a subscript or a multi-letter name makes the answer None, because the letters would hide a symbol.
    """
    text = latex.replace(r"\mathrm{e}", " ").replace(r"\mathrm{i}", " ")
    if set(re.findall(r"\\([A-Za-z]+)", text)) - _FUNCTION_COMMANDS:
        return None
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    if "_" in text or "\\" in text:
        return None
    words = re.findall(r"[A-Za-z]+", text)
    if any(len(w) > 1 for w in words):
        return None
    names = set(words)
    return next(iter(names)) if len(names) == 1 else None


# --- followup_reuse: "differentiate that result", "integrate it from 0 to 1" -----------------------------
_REUSE = re.compile(
    r"\b(differentiate|integrate)\s+(?:(?:that|this|the)\s+(?:(?:last|previous|latest)\s+)?"
    r"(?:result|expression|answer)|it|that|this)\b",
    re.IGNORECASE,
)
_OPERATION_WORDS = re.compile(
    r"\b(differentiate|integrate|derivative|integral|simplify|expand|factor|series|taylor|solve|substitute)\b",
    re.IGNORECASE,
)
# Wording that changes the request beyond the plain operation: the model plans those.
_REUSE_COMPLICATIONS = re.compile(
    r"\b(second|third|fourth|twice|partial|mixed|numeric\w*|compare|verify|prove|check|then|also|both)\b",
    re.IGNORECASE,
)
_VARIABLE = re.compile(
    r"\b(?:with respect to|w\.r\.t\.?|wrt|over|in|by)\s+\$?([A-Za-z])\$?(?![A-Za-z])", re.IGNORECASE
)
_LONE_SYMBOL = re.compile(r"\s*-?[A-Za-z0-9.]+\s*")  # `$t$`, `$0$`: naming a variable or a bound
_NUMBER = r"-?\d+(?:\.\d+)?"
_BOUNDS = re.compile(
    rf"\b(?:from\s+\$?({_NUMBER})\$?\s+to\s+\$?({_NUMBER})\$?|between\s+\$?({_NUMBER})\$?\s+and\s+\$?({_NUMBER})\$?)",
    re.IGNORECASE,
)
_BOUND_WORDS = re.compile(r"\b(from|between|limits?|bounds?|interval|range)\b", re.IGNORECASE)


def followup_reuse(text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
    """`differentiate` or `integrate` of the one result the previous turn produced, written as `@rN`.

    Signals: a request that names the operation and points at "it" / "that result" (words), and exactly one
    verified result from the previous turn (state). The variable is the one the user named, or the one plain
    letter of the result; the bounds are numbers the user wrote. Anything else, the model plans.
    """
    if _DELIMITED.search(text) and any(
        not _LONE_SYMBOL.fullmatch(formula) for formula in _split(text)[0]
    ):
        return None  # a formula in the message: the user is not pointing at the earlier result
    matches = list(_REUSE.finditer(text))
    if len(matches) != 1 or len(_OPERATION_WORDS.findall(text)) != 1:
        return None
    if _REUSE_COMPLICATIONS.search(text):
        return None
    if len(context.latest) != 1 or context.latest[0] not in context.results:
        return None
    result_id = context.latest[0]
    rest = text[matches[0].end() :]
    named = _VARIABLE.findall(rest)
    if len(set(v.lower() for v in named)) > 1:
        return None
    variable = named[0] if named else single_symbol(context.results[result_id])
    if variable is None:
        return None
    reference = f"@{result_id}"
    if matches[0].group(1).lower() == "differentiate":
        if _BOUND_WORDS.search(rest):
            return None
        return Route(
            "differentiate",
            {"latex": reference, "variables": [{"name": variable}]},
            reason="followup_reuse",
        )
    arguments: dict[str, object] = {"latex": reference, "variable": variable}
    bounds = _BOUNDS.search(rest)
    if bounds is not None:
        low, high = (
            (bounds.group(1), bounds.group(2))
            if bounds.group(1)
            else (bounds.group(3), bounds.group(4))
        )
        arguments |= {"lower": low, "upper": high}
    elif _BOUND_WORDS.search(rest):
        return None  # limits in words the rule cannot read
    return Route("integrate", arguments, reason="followup_reuse")


# --- ode_analyze: an equation with a derivative of an applied function -----------------------------------
_ODE_CHARS = re.compile(r"^[A-Za-z0-9'()+\-*/^=.,\s]+$")
_ODE_DERIVATIVE = re.compile(r"(?<![A-Za-z])([A-Za-z])('{1,4})\(([A-Za-z])\)")
_ODE_APPLIED = re.compile(r"(?<![A-Za-z])([A-Za-z])'*\(([A-Za-z])\)")
_ODE_FUNCTIONS = frozenset(
    ["sin", "cos", "tan", "exp", "log", "ln", "sqrt", "sinh", "cosh", "tanh"]
)
_ODE_ASKS = re.compile(
    r"\b(solve|classify|analy[sz]e|what kind|what type|which method|how (?:can|could|do|would|to))\b",
    re.IGNORECASE,
)
# Words that pick a method, state conditions or ask for something else: the model plans those.
_ODE_OTHER = re.compile(
    r"\b(numeric\w*|series|closed|initial|boundary|conditions?|laplace|perturb\w*|phase|equilibri\w*|"
    r"stab\w*|plot|verify|check|prove|compare|simplif\w*|differentiate|integrate)\b",
    re.IGNORECASE,
)


def ode_analyze(text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
    """`ode_analyze` of one equation that contains a derivative of an applied function (`y''(x) = y(x)`).

    Signals: the formula reads as an ordinary differential equation (structure) and the words around it ask
    to classify or solve it, or are absent. The tool only classifies and lists the applicable methods: it
    solves nothing and the server still makes the user choose a method.
    """
    if not _DELIMITED.search(text):
        return None
    formulas, words = _split(text)
    if len(formulas) != 1 or "\\" in words or _ODE_OTHER.search(words):
        return None
    if _ASKS_FOR_EQUATIONS.search(words):
        return None  # Lagrangian wording
    residue = re.sub(r"[\s.,;:!?]+", "", words)
    if residue and not _ODE_ASKS.search(words):
        return None
    equation = formulas[0].strip().rstrip(".").strip()
    if not _ODE_CHARS.fullmatch(equation) or equation.count("=") != 1 or "dot" in equation:
        return None
    if any(
        len(name) > 1 and name not in _ODE_FUNCTIONS for name in re.findall(r"[A-Za-z]+", equation)
    ):
        return None
    derivatives = _ODE_DERIVATIVE.findall(equation)
    if not derivatives:
        return None
    arguments = {arg for _, arg in re.findall(r"(?<![A-Za-z])([A-Za-z])'*\(([A-Za-z])\)", equation)}
    if len(arguments) != 1:
        return None  # several independent variables: a partial equation, or an unclear one
    return Route(
        "ode_analyze",
        {"equations": [equation], "independent": arguments.pop()},
        reason="ode_analyze",
    )


# --- matrix_simplify: a formula over matrices the user declared as n x n ---------------------------------
_MATRIX_DECLARATION = re.compile(
    r"\b((?:[A-Z])(?:\s*(?:,|and|&)\s*[A-Z])*)\s+(?:are|is|be)\s+(?:both\s+)?(\d+)\s*(?:x|\u00d7|by)\s*(\d+)\s+matri(?:x|ces)\b"
)
_MATRIX_ASKS = re.compile(r"\b(simplify|expand|multiply out|reduce)\b", re.IGNORECASE)
_MATRIX_OTHER = re.compile(
    r"\b(commut\w*|invertib\w*|assum\w*|if|when|unless|evaluate|numeric\w*|gradient|derivative|"
    r"differentiate|integra\w*|trace|determinant|eigen\w*|compare|verify|prove|check)\b",
    re.IGNORECASE,
)
_MATRIX_COMMANDS = frozenset({"exp", "left", "right", "cdot"})


def declared_matrix_sizes(messages: Sequence[str]) -> dict[str, int]:
    """Square sizes the user gave in words ("A and B are 2x2 matrices"). A name given two sizes is dropped."""
    sizes: dict[str, int] = {}
    clashes: set[str] = set()
    for message in messages:
        for found in _MATRIX_DECLARATION.finditer(message):
            rows, cols = int(found.group(2)), int(found.group(3))
            if rows != cols or not 1 <= rows <= 8:
                continue
            for name in re.findall(r"[A-Z]", found.group(1)):
                if sizes.get(name, rows) != rows:
                    clashes.add(name)
                sizes[name] = rows
    return {k: v for k, v in sizes.items() if k not in clashes}


def matrix_simplify(text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
    """`matrix_calculus` (simplify) of a formula whose every letter is a matrix the user declared n x n.

    Signals: the user's words declare the matrices and sizes (conversation), and ask to simplify or expand
    one formula over exactly those names (message). Commutation, invertibility and every other property
    stay undeclared: a message that mentions them is left to the model, and the tool answers
    `conditions_not_satisfied` where a result needs one.
    """
    if not _MATRIX_ASKS.search(text):
        return None
    formulas, words = _split(text) if _DELIMITED.search(text) else ([], text)
    if len(formulas) != 1 or _MATRIX_OTHER.search(words) or "=" in formulas[0]:
        return None
    formula = formulas[0].strip().rstrip(".").strip()
    if set(re.findall(r"\\([A-Za-z]+)", formula)) - _MATRIX_COMMANDS:
        return None
    if re.search(r"[_']|\^\s*\{?\s*T\b", formula):
        return None  # indexed names, primes, transposes: left to the model
    bare = re.sub(r"\\[A-Za-z]+", " ", formula)
    letters = re.findall(r"[A-Za-z]+", bare)
    sizes = declared_matrix_sizes([*context.user_messages, text])
    if not letters or any(len(w) > 1 or w not in sizes for w in letters):
        return None
    names = list(dict.fromkeys(letters))
    return Route(
        "matrix_calculus",
        {
            "latex": formula,
            "matrices": [{"name": n, "rows": sizes[n]} for n in names],
            "action": "simplify",
        },
        reason="matrix_simplify",
        answers=frozenset({"ok", "conditions_not_satisfied"}),
    )


# --- lagrangian_followup: "what is conserved?" after the equations of motion of one Lagrangian -----------
_CONSERVED_ASKS = re.compile(
    r"\b(?:what|which|any|anything)\b[^.?!]*\bconserved\b|\bconserved quantit(?:y|ies)\b|"
    r"\bconstants? of (?:the )?motion\b|\bconservation laws?\b",
    re.IGNORECASE,
)
# Wording that asks for more than the default search or something else: the model plans those.
_CONSERVED_OTHER = re.compile(
    r"\b(hidden|thorough\w*|deep\w*|beyond|exhaustive|noether|verify|prove|check|numeric\w*|"
    r"hamilton\w*|compare|solve|integrate|derivative|differentiate)\b",
    re.IGNORECASE,
)


def lagrangian_followup(text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
    """`find_conserved_quantities` for the Lagrangian the previous turn analysed.

    Signals: the user asks what is conserved, in words, with no formula of their own (words), and the previous
    assistant turn analysed exactly one Lagrangian (conversation). The Lagrangian and the coordinates are the
    ones in that earlier call, unchanged; a Lagrangian that is itself a reference, or a message that names a
    deeper search, is left to the model.
    """
    if _DELIMITED.search(text) or not _CONSERVED_ASKS.search(text) or _CONSERVED_OTHER.search(text):
        return None
    analysed = [args for name, args in context.previous_calls if name == "analyze_lagrangian"]
    if len(analysed) != 1 or len(context.previous_calls) != 1:
        return None
    lagrangian, coordinates = analysed[0].get("lagrangian"), analysed[0].get("coordinates")
    if (
        not isinstance(lagrangian, str)
        or "@" in lagrangian
        or not isinstance(coordinates, list)
        or not coordinates
        or not all(isinstance(c, str) for c in coordinates)
    ):
        return None
    return Route(
        "find_conserved_quantities",
        {"lagrangian": lagrangian, "coordinates": list(coordinates)},
        reason="lagrangian_followup",
    )


RULES: tuple[Rule, ...] = (
    lagrangian_equations,
    followup_reuse,
    lagrangian_followup,
    ode_analyze,
    matrix_simplify,
)


def plan_route(user_text: str, context: RouteContext = NO_CONTEXT) -> Route | None:
    """The first rule that applies to this message, or None (the model plans)."""
    for rule in RULES:
        found = rule(user_text, context)
        if found is not None:
            return found
    return None
