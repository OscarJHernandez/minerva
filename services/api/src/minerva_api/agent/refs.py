"""References to earlier verified results: `@r1` in a tool argument means "the result of result r1".

Without this the model retypes LaTeX from one tool result into the next call, and every retyping can
change the mathematics. A reference is replaced by the engine's own printed LaTeX before the call is
validated, so what reaches the tool is exactly what an earlier tool returned. The substituted text goes
through the same strict parser as anything else: a reference is a shortcut for copying, not a new way to
get unchecked input into the engine.

Only a result that was accepted (status `ok`, with engine evidence) and is a single expression can be
referenced; a failed, unchanged or unsupported result has no id.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from minerva_engine.notation.models import InputDiagnostic

from .card_equations import (
    PREMISE_SUFFIX,
    premise_labels,
    result_formula,
    result_restrictions,
)
from .equations import Equation
from .mathinput import MATH_FIELDS
from .reference_types import refusal_diagnostic
from .tools import ToolOutput

REFERENCE = re.compile(r"@r(\d+)\b")
# `@e3`: the equation numbered (3) in the conversation (see equations.py).
EQ_REFERENCE = re.compile(r"@e(\d+)\b")
# Tools whose card holds one result expression in `result_latex`.
REFERENCEABLE_TOOLS = frozenset(
    {
        "simplify",
        "differentiate",
        "integrate",
        "series",
        "creative_telescoping",
        "time_derivative",
        "substitute",
        "rewrite_part",
        "routh_reduce",
        "effective_potential",
    }
)
# `substitute` and `rewrite_part` keep their result under `data["result"]` (`card_equations.result_formula`); the id
# names that result with its restrictions. A `substitute` that holds only given the user's relations (evidence
# `conditional`) keeps its premises: the id carries them (`premise_labels`), so a result built on `@rN` is quoted,
# worded "given", and never called proved.
# `time_derivative`: the id names `result_latex`, the total derivative before any equations of motion are used.
# Its conclusion on the equations of motion (`zero_on_motion`, `on_motion_latex`) is a statement about a system
# and never gets an id: a plain `0` would drop the system it was decided on.
# `routh_reduce`: the id names the Routhian R (`result_latex`) with the block-determinant restrictions. The rows of
# what was held fixed and of the rebuilt velocities are numbered list rows (names, no id).
# `effective_potential`: the id names the radial equation `r_dot^2 + V_eff(r) = alpha p^2` (`result_latex`) with its
# restrictions; `V_eff(r) = ...` is a numbered list row (a name, no id).
# `multivariate_normal` is NOT referenceable: it has no single result expression (a mean vector, a covariance matrix
# and a density) and a bare `@rN` would drop the model it was computed under; a follow-up re-sends the model the
# card returned (`mvn_tools.MODEL_LINE`) instead.
# Arguments that may contain references (the ones that hold a mathematical expression).
EXPRESSION_ARGUMENTS = MATH_FIELDS


@dataclass
class Resolution:
    arguments: dict[str, Any]
    count: int = 0  # how many references were replaced
    error: str | None = None  # safe to show the model
    # Set with `error` when the reference is of a kind the field cannot take (`reference_types.py`): the typed
    # reason the model sees as a `diagnostic:` line. Holds `@eN`, never a formula.
    diagnostic: InputDiagnostic | None = None
    # Restrictions that the referenced results carry ("x \\neq 2"), and which results they come from. A result
    # computed from `@r1` is valid only where `@r1` was, so these travel with it (see `inherited_note`).
    inherited: list[str] = field(default_factory=list)
    inherited_from: list[str] = field(default_factory=list)
    # Equations the arguments rest on that Minerva never verified ("equation (3)"): they were written by
    # the model. A result computed from them is a verified computation on an unestablished input, and
    # says so (see `unverified_note`). Also those carried by the earlier results that were referenced.
    unverified: list[str] = field(default_factory=list)


@dataclass
class ResultStore:
    """Verified results of this conversation, by id. Ids continue across turns."""

    results: dict[str, str] = field(default_factory=dict)  # "r1" -> engine-printed LaTeX
    # "r1" -> the restrictions that result is valid under (its own, plus those inherited from what it was
    # computed from). Without this, `simplify` cancelling (x-2) reports x != 2 once and every later step that
    # reuses @r1 is silent about x = 2, which the original expression excluded.
    restrictions: dict[str, list[str]] = field(default_factory=dict)
    # The numbered equations of the conversation, and for a result id the unverified equations it was
    # computed from: a later `@r1` reuse inherits them, exactly like the restrictions.
    equations: dict[int, Equation] = field(default_factory=dict)
    unverified: dict[str, list[str]] = field(default_factory=dict)
    _next: int = 1

    @classmethod
    def from_prior(
        cls,
        prior: Mapping[str, str] | None,
        restrictions: Mapping[str, Sequence[str]] | None = None,
        equations: Mapping[int, Equation] | None = None,
        unverified: Mapping[str, Sequence[str]] | None = None,
    ) -> ResultStore:
        store = cls(
            results=dict(prior or {}),
            restrictions={k: list(v) for k, v in (restrictions or {}).items()},
            equations=dict(equations or {}),
            unverified={k: list(v) for k, v in (unverified or {}).items()},
        )
        numbers = [int(k[1:]) for k in store.results if re.fullmatch(r"r\d+", k)]
        store._next = max(numbers, default=0) + 1
        return store

    def register(
        self,
        tool: str,
        output: ToolOutput,
        inherited: Sequence[str] = (),
        unverified: Sequence[str] = (),
    ) -> str | None:
        """Give an accepted single-expression result an id; None when it cannot be referenced.

        `inherited` are the restrictions of the results it was computed from; they are stored with its own.
        """
        evidence = output.evidence
        if tool not in REFERENCEABLE_TOOLS or evidence is None or not evidence.accepted:
            return None
        data = output.card.data
        latex = result_formula(output.card.type, data)
        if output.status != "ok" or latex is None:
            return None
        result_id = f"r{self._next}"
        self._next += 1
        self.results[result_id] = latex
        merged = result_restrictions(output.card.type, data)
        merged += [r for r in inherited if r not in merged]
        if merged:
            self.restrictions[result_id] = merged
        carried = [*unverified, *premise_labels(output.card.type, data)]
        if carried:
            self.unverified[result_id] = list(dict.fromkeys(carried))
        return result_id

    def resolve(self, arguments: dict[str, Any], tool: str | None = None) -> Resolution:
        """Replace `@eN` and `@rN` in the expression arguments. A reference to nothing is an error, not a guess.

        With `tool`, a reference whose kind the field cannot take (a relation for an expression field, an
        expression for an equation field; `reference_types.FIELD_POLICY`) is refused with a typed diagnostic
        and the call is not run. A kind that is unknown (an old stored equation, or text that cannot be
        decided) is never refused.
        """
        staged = dict(arguments)
        count = 0
        unverified: list[str] = []
        for key in EXPRESSION_ARGUMENTS:
            value = staged.get(key)
            if not isinstance(value, str) or "@e" not in value:
                continue
            numbers = [int(n) for n in EQ_REFERENCE.findall(value)]
            missing = [n for n in numbers if n not in self.equations]
            if missing:
                known = ", ".join(f"({n})" for n in sorted(self.equations)) or "none yet"
                return Resolution(
                    arguments,
                    error=f"unknown equation reference @e{missing[0]}; equations: {known}",
                )
            whole = EQ_REFERENCE.fullmatch(value.strip()) is not None
            if tool is not None:
                for n in dict.fromkeys(numbers):
                    kind = self.equations[n].kind
                    # An expression is refused only as the whole argument (`@e3 = 4` is a relation built
                    # from it); a relation is refused wherever it sits (it would be wrapped in brackets).
                    if kind is None or (kind == "expression" and not whole):
                        continue
                    refused = refusal_diagnostic(tool, key, n, kind)
                    if refused is not None:
                        return Resolution(arguments, error=refused[1], diagnostic=refused[0])

            def sub(m: re.Match[str], whole: bool = whole) -> str:
                equation = self.equations[int(m.group(1))]
                if equation.result_id and equation.result_id in self.results:
                    return f"@{equation.result_id}"  # a verified result: resolved as one below
                if equation.source == "written_by_model":
                    label = f"equation ({equation.number})"
                    if label not in unverified:
                        unverified.append(label)
                elif equation.part_of:
                    # One row of a tool's list: exact LaTeX, but not linked to the verification or the
                    # restrictions of the card it came from, so the result must say so.
                    label = f"equation ({equation.number}){COMPONENT_SUFFIX}"
                    if label not in unverified:
                        unverified.append(label)
                elif equation.source == "from_tool":
                    # A tool's result with no result id (it was not accepted as a whole, or its tool cannot
                    # be referenced): exact LaTeX, but its restrictions and verification are not carried.
                    label = f"equation ({equation.number}){NO_ID_SUFFIX}"
                    if label not in unverified:
                        unverified.append(label)
                return equation.latex if whole else rf"\left({equation.latex}\right)"

            staged[key] = EQ_REFERENCE.sub(sub, value)
            count += len(numbers)
        out = dict(staged)
        for key in EXPRESSION_ARGUMENTS:
            value = out.get(key)
            if not isinstance(value, str) or "@r" not in value:
                continue
            ids = REFERENCE.findall(value)
            missing_ids = [f"@r{i}" for i in ids if f"r{i}" not in self.results]
            if missing_ids:
                known = ", ".join(f"@{k}" for k in self.results) or "none yet"
                return Resolution(
                    arguments,
                    error=f"unknown result reference {missing_ids[0]}; available: {known}",
                )
            if REFERENCE.fullmatch(value.strip()):
                out[key] = self.results[f"r{ids[0]}"]  # the whole argument: exactly the result
            else:  # embedded in a larger expression: parenthesised so precedence is kept
                out[key] = REFERENCE.sub(
                    lambda m: rf"\left({self.results['r' + m.group(1)]}\right)", value
                )
            count += len(ids)
        inherited: list[str] = []
        sources: list[str] = []
        for key in EXPRESSION_ARGUMENTS:
            value = staged.get(key)
            for found in REFERENCE.findall(value) if isinstance(value, str) else []:
                carried = self.restrictions.get(f"r{found}", [])
                if carried and f"@r{found}" not in sources:
                    sources.append(f"@r{found}")
                inherited += [r for r in carried if r not in inherited]
                unverified += [
                    u for u in self.unverified.get(f"r{found}", []) if u not in unverified
                ]
        return Resolution(
            out, count, inherited=inherited, inherited_from=sources, unverified=unverified
        )

    def note_for(self, result_id: str) -> str:
        return reference_note(result_id)


def inherited_note(restrictions: Sequence[str], sources: Sequence[str]) -> str:
    """What a tool message says about restrictions the new result inherits (fixed text plus the formulas)."""
    listed = ", ".join(f"${r}$" for r in restrictions)
    return (
        f"Inherited from {', '.join(sources)}: this result is valid only where {listed}. "
        "Those restrictions came from an earlier step and are not repeated by this tool; state them with the result."
    )


def reference_note(result_id: str) -> str:
    """The line a tool message carries so the model knows how to reuse a verified result."""
    return (
        f"result_id={result_id} (write @{result_id} in a later call to reuse this result exactly)"
    )


UNVERIFIED_INPUT = "computed from an equation the model wrote and Minerva never verified"
# A row of a card's list (a Christoffel symbol): the tool computed it, but reusing it by number does not carry
# the card's verification or restrictions with it.
COMPONENT_SUFFIX = ", a component of an earlier result"
# A tool result that has an equation number but no result id.
NO_ID_SUFFIX = ", a tool result with no result id"
_CLASS_SUFFIXES = (COMPONENT_SUFFIX, NO_ID_SUFFIX, PREMISE_SUFFIX)
UNLINKED_INPUT = "computed from one component of an earlier result; its verification and restrictions were not carried over"
PREMISE_INPUT = "computed from a result that holds only given a relation you supplied as a premise, which Minerva never checked"
UNNUMBERED_INPUT = "computed from a tool result that has no result id; its verification and restrictions were not carried over"


def quote_reason(sources: Sequence[str]) -> str:
    """Why the formulas of a result computed from `sources` are quoted. Written > component > no result id."""
    if any(not s.endswith(_CLASS_SUFFIXES) for s in sources):
        return UNVERIFIED_INPUT
    if any(s.endswith(COMPONENT_SUFFIX) for s in sources):
        return UNLINKED_INPUT
    if any(s.endswith(PREMISE_SUFFIX) for s in sources):
        return PREMISE_INPUT
    return UNNUMBERED_INPUT


def unverified_note(sources: Sequence[str]) -> str:
    """What a tool message says when its input was not verified as a whole (fixed text)."""
    written = [s for s in sources if not s.endswith(_CLASS_SUFFIXES)]
    premises = [s for s in sources if s.endswith(PREMISE_SUFFIX)]
    parts = [s for s in sources if s.endswith(COMPONENT_SUFFIX)]
    unnumbered = [s for s in sources if s.endswith(NO_ID_SUFFIX)]
    lines: list[str] = []
    if written:
        lines.append(
            f"Input not verified: this result was computed from {', '.join(written)}, which the model wrote and "
            "Minerva never verified. The computation is checked, but whether that input is right is not established: "
            "say so with the result, and do not call anything built on it verified."
        )
    if premises:
        lines.append(
            f"Input given, not proved: this result was computed from {', '.join(premises)}. It holds only "
            "given that premise, which Minerva never checked: say 'given', never 'proved', and name the premise "
            "with the result. The restrictions of the earlier result are carried over."
        )
    if parts:
        lines.append(
            f"Input taken from a list: this result was computed from {', '.join(parts)}. Its LaTeX is exactly what "
            "the tool returned, but the tool's verification and its restrictions are not carried over by reusing "
            "one component by number: say so with the result, state the restrictions of the earlier result "
            "yourself, and do not call anything built on it verified."
        )
    if unnumbered:
        lines.append(
            f"Input without a result id: this result was computed from {', '.join(unnumbered)}. A tool computed "
            "it and its LaTeX is exact, but reusing it by number does not carry the tool's verification or its "
            "restrictions: say so with the result, state the restrictions of the earlier result yourself, and "
            "do not call anything built on it verified."
        )
    return " ".join(lines)
