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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .tools import ToolOutput

REFERENCE = re.compile(r"@r(\d+)\b")
# Tools whose card holds one result expression in `result_latex`.
REFERENCEABLE_TOOLS = frozenset({"simplify", "differentiate", "integrate", "series"})
# Arguments that may contain references (the ones that hold a mathematical expression).
EXPRESSION_ARGUMENTS = ("latex", "expression", "equation", "lagrangian")


@dataclass
class Resolution:
    arguments: dict[str, Any]
    count: int = 0  # how many references were replaced
    error: str | None = None  # safe to show the model


@dataclass
class ResultStore:
    """Verified results of this conversation, by id. Ids continue across turns."""

    results: dict[str, str] = field(default_factory=dict)  # "r1" -> engine-printed LaTeX
    _next: int = 1

    @classmethod
    def from_prior(cls, prior: Mapping[str, str] | None) -> ResultStore:
        store = cls(results=dict(prior or {}))
        numbers = [int(k[1:]) for k in store.results if re.fullmatch(r"r\d+", k)]
        store._next = max(numbers, default=0) + 1
        return store

    def register(self, tool: str, output: ToolOutput) -> str | None:
        """Give an accepted single-expression result an id; None when it cannot be referenced."""
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
        return Resolution(out, count)

    def note_for(self, result_id: str) -> str:
        return reference_note(result_id)


def reference_note(result_id: str) -> str:
    """The line a tool message carries so the model knows how to reuse a verified result."""
    return (
        f"result_id={result_id} (write @{result_id} in a later call to reuse this result exactly)"
    )
