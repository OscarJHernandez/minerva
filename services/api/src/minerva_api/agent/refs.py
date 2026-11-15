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

from .tools import ToolOutput

REFERENCE = re.compile(r"@r(\d+)\b")
# Tools whose card holds one result expression in `result_latex`.
REFERENCEABLE_TOOLS = frozenset({"simplify", "differentiate", "integrate", "series"})
# Arguments that may contain references (the ones that hold a mathematical expression).
EXPRESSION_ARGUMENTS = ("latex", "expression", "equation", "lagrangian", "integrand")


@dataclass
class Resolution:
    arguments: dict[str, Any]
    count: int = 0  # how many references were replaced
    error: str | None = None  # safe to show the model
    # Restrictions that the referenced results carry ("x \\neq 2"), and which results they come from. A result
    # computed from `@r1` is valid only where `@r1` was, so these travel with it (see `inherited_note`).
    inherited: list[str] = field(default_factory=list)
    inherited_from: list[str] = field(default_factory=list)


@dataclass
class ResultStore:
    """Verified results of this conversation, by id. Ids continue across turns."""

    results: dict[str, str] = field(default_factory=dict)  # "r1" -> engine-printed LaTeX
    # "r1" -> the restrictions that result is valid under (its own, plus those inherited from what it was
    # computed from). Without this, `simplify` cancelling (x-2) reports x != 2 once and every later step that
    # reuses @r1 is silent about x = 2, which the original expression excluded.
    restrictions: dict[str, list[str]] = field(default_factory=dict)
    _next: int = 1

    @classmethod
    def from_prior(
        cls,
        prior: Mapping[str, str] | None,
        restrictions: Mapping[str, Sequence[str]] | None = None,
    ) -> ResultStore:
        store = cls(
            results=dict(prior or {}),
            restrictions={k: list(v) for k, v in (restrictions or {}).items()},
        )
        numbers = [int(k[1:]) for k in store.results if re.fullmatch(r"r\d+", k)]
        store._next = max(numbers, default=0) + 1
        return store

    def register(self, tool: str, output: ToolOutput, inherited: Sequence[str] = ()) -> str | None:
        """Give an accepted single-expression result an id; None when it cannot be referenced.

        `inherited` are the restrictions of the results it was computed from; they are stored with its own.
        """
        evidence = output.evidence
        if tool not in REFERENCEABLE_TOOLS or evidence is None or not evidence.accepted:
            return None
        data = output.card.data
        latex = data.get("result_latex")
        if output.status != "ok" or not isinstance(latex, str) or not latex.strip():
            return None
        result_id = f"r{self._next}"
        self._next += 1
        self.results[result_id] = latex
        own = data.get("restrictions_latex")
        merged = [r for r in (own if isinstance(own, list) else []) if isinstance(r, str)]
        merged += [r for r in inherited if r not in merged]
        if merged:
            self.restrictions[result_id] = merged
        return result_id

    def resolve(self, arguments: dict[str, Any]) -> Resolution:
        """Replace `@rN` in the expression arguments. A reference to nothing is an error, not a guess."""
        out = dict(arguments)
        count = 0
        for key in EXPRESSION_ARGUMENTS:
            value = out.get(key)
            if not isinstance(value, str) or "@r" not in value:
                continue
            ids = REFERENCE.findall(value)
            missing = [f"@r{i}" for i in ids if f"r{i}" not in self.results]
            if missing:
                known = ", ".join(f"@{k}" for k in self.results) or "none yet"
                return Resolution(
                    arguments, error=f"unknown result reference {missing[0]}; available: {known}"
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
            value = arguments.get(key)
            for found in REFERENCE.findall(value) if isinstance(value, str) else []:
                carried = self.restrictions.get(f"r{found}", [])
                if carried and f"@r{found}" not in sources:
                    sources.append(f"@r{found}")
                inherited += [r for r in carried if r not in inherited]
        return Resolution(out, count, inherited=inherited, inherited_from=sources)

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
