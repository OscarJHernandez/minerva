"""Equation numbers: one counter per conversation, so an equation can be referred to later.

Every displayed formula (in a reply or in the user's own message) and every tool result with a primary
formula gets a number, (1), (2), ... in reading order, over the whole conversation and never reused. A
formula that is already numbered keeps its number when it is shown again. The number is a *name*: it says
nothing about whether Minerva verified the formula (the mark does), and the number a model might write
itself is never trusted: numbers are assigned here, after the turn.

A reference in the user's message ("equation 3", "eq. (3)", or a lone "(3)") is resolved from the stored
conversation and told to the model; a number that does not exist is said so, never guessed. In a tool
argument `@eN` stands for the equation's LaTeX (see refs.py). Using an equation the model wrote and Minerva
never verified is allowed, but the result then carries that fact.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .events import Block
from .reasoning import clean_reply
from .segments import normalize, split_segments

Source = Literal["from_tool", "written_by_model", "yours"]
MAX_EQUATION_CHARS = (
    500  # a longer formula is shown but not numbered, like the claim checker's limit
)


@dataclass
class Equation:
    number: int
    latex: str
    source: Source
    result_id: str | None = None  # "r2": the verified result this equation is, when it is one


@dataclass
class EquationTable:
    """The numbered equations of one conversation."""

    items: dict[int, Equation] = field(default_factory=dict)
    index: dict[str, int] = field(default_factory=dict)  # normalised LaTeX -> number
    counter: int = 0  # the highest number ever given (kept even when messages are removed)

    def number_for(self, latex: str, source: Source, result_id: str | None = None) -> int | None:
        """The number of this formula: the one it already has, or the next. None if it cannot be numbered."""
        if not latex.strip() or len(latex) > MAX_EQUATION_CHARS:
            return None
        key = normalize(latex)
        if not key:
            return None
        existing = self.index.get(key)
        if existing is not None:
            item = self.items[existing]
            if result_id and not item.result_id:  # the same formula is also a verified result
                item.result_id, item.source = result_id, "from_tool"
            return existing
        self.counter += 1
        self.items[self.counter] = Equation(self.counter, latex, source, result_id)
        self.index[key] = self.counter
        return self.counter

    def restore(
        self, number: int, latex: str, source: Source, result_id: str | None = None
    ) -> None:
        """Re-enter an equation that was numbered earlier (from a stored message)."""
        self.counter = max(self.counter, number)
        key = normalize(latex)
        if number not in self.items:
            self.items[number] = Equation(number, latex, source, result_id)
        elif result_id and not self.items[number].result_id:
            self.items[number].result_id = result_id
            self.items[number].source = "from_tool"
        if key:
            self.index.setdefault(key, number)


def _source_of(status: str | None) -> Source:
    return "from_tool" if status == "from_tool" else "written_by_model"


def build_table(messages: Iterable[Any], counter: int = 0) -> EquationTable:
    """The table of a conversation from its stored messages (`counter`: the stored high-water mark)."""
    table = EquationTable(counter=counter)
    for m in messages:
        content = m.content
        if m.role == "user":
            for e in content.get("equations", []) or []:
                if isinstance(e, dict) and isinstance(e.get("number"), int):
                    table.restore(e["number"], str(e.get("latex", "")), "yours")
            continue
        steps = {s.get("id"): s for s in content.get("trace", []) or [] if isinstance(s, dict)}
        for block in content.get("blocks", []) or []:
            if block.get("kind") == "card" and isinstance(block.get("card"), dict):
                card = block["card"]
                number = card.get("number")
                latex = (card.get("data") or {}).get("result_latex")
                if isinstance(number, int) and isinstance(latex, str):
                    result_id = (steps.get(card.get("id")) or {}).get("result_id")
                    table.restore(number, latex, "from_tool", result_id)
            for seg in block.get("segments", []) or []:
                number = seg.get("number") if isinstance(seg, dict) else None
                if isinstance(number, int):
                    table.restore(number, str(seg.get("value", "")), _source_of(seg.get("status")))
    return table


def number_blocks(
    blocks: Sequence[Block], table: EquationTable, result_ids: Mapping[str, str] | None = None
) -> None:
    """Give the displayed formulas and primary results of a reply their numbers, in reading order.

    `result_ids` maps a card's id to the id of the verified result it is ("r2"), when it is one.
    """
    for block in blocks:
        if block.kind == "card" and block.card is not None:
            latex = block.card.data.get("result_latex")
            if isinstance(latex, str):
                block.card.number = table.number_for(
                    latex, "from_tool", (result_ids or {}).get(block.card.id)
                )
            continue
        for seg in block.segments:
            if seg.type == "display_math":
                seg.number = table.number_for(seg.value, _source_of(seg.status))


def number_user_text(text: str, table: EquationTable) -> list[dict[str, Any]]:
    """Numbers for the displayed formulas of the user's message, in order (None-numbered ones are skipped)."""
    out: list[dict[str, Any]] = []
    for seg in split_segments(text, set()):
        if seg.type == "display_math":
            number = table.number_for(seg.value, "yours")
            if number is not None:
                out.append({"number": number, "latex": seg.value})
    return out


# --- references in the user's message --------------------------------------------------------------
_EXPLICIT = re.compile(r"\b(?:eq|eqn|equation)\b\.?\s*\(?\s*(\d{1,4})\s*\)?", re.IGNORECASE)
# A lone "(3)": not right after a letter, digit or closing bracket (f(3), x_(3)), not before one.
_PAREN = re.compile(r"(?<![\w)\]}])\(\s*(\d{1,4})\s*\)(?![\w(])")


@dataclass(frozen=True)
class Reference:
    number: int
    explicit: bool  # "equation 3" rather than a lone "(3)": only these are reported when unknown


def find_references(text: str) -> list[Reference]:
    """Explicit references in the text of a message (formulas inside $...$ are not searched).

    `f(3)`, a list marker like "(1) first" and a number inside a formula are not references. A lone "(3)"
    is one, because that is how the equation is displayed; whether it really is gets decided by whether
    the conversation has such an equation (the interface shows what it was taken to be).
    """
    plain = " ".join(s.value for s in split_segments(text, set()) if s.type == "text")
    found: list[Reference] = []
    seen: set[int] = set()

    def add(number: int, explicit: bool) -> None:
        if number not in seen and number > 0:
            seen.add(number)
            found.append(Reference(number, explicit))

    for line in plain.splitlines():
        hits: list[tuple[int, int, bool]] = [
            (m.start(), int(m.group(1)), True) for m in _EXPLICIT.finditer(line)
        ]
        # Blank out the explicit forms (same length, so positions line up) before looking for lone "(3)".
        blank = _EXPLICIT.sub(lambda m: " " * len(m.group(0)), line)
        for m in _PAREN.finditer(blank):
            marker = not blank[: m.start()].strip() and bool(blank[m.end() :].strip())
            if not marker:  # "(1) first thing" at the start of a line is a list item
                hits.append((m.start(), int(m.group(1)), False))
        for _, number, explicit in sorted(hits):  # in the order the user wrote them
            add(number, explicit)
    return found


def resolve_references(
    references: Sequence[Reference], table: EquationTable
) -> tuple[list[dict[str, Any]], str]:
    """What each reference was taken to be (for the user's bubble) and the note for the model."""
    shown: list[dict[str, Any]] = []
    lines: list[str] = []
    for ref in references:
        item = table.items.get(ref.number)
        if item is None:
            if ref.explicit:
                shown.append({"number": ref.number, "latex": None, "source": "unknown"})
                if ref.number <= table.counter:  # it was numbered once: its message was replaced
                    lines.append(
                        f"({ref.number}) no longer exists: the message that had it was replaced."
                    )
                else:
                    known = f"(1) to ({table.counter})" if table.counter else "none yet"
                    lines.append(
                        f"({ref.number}) does not exist in this conversation (equations: {known})."
                    )
            continue
        shown.append({"number": ref.number, "latex": item.latex, "source": item.source})
        if item.result_id:
            how = f"a verified result (write @{item.result_id} or @e{item.number} in a tool call to reuse it exactly)"
        elif item.source == "yours":
            how = f"written by the user (write @e{item.number} in a tool call to reuse it exactly)"
        else:
            how = (
                "written by the model and NOT verified by Minerva (@e"
                f"{item.number} reuses it exactly; say that it was not verified)"
            )
        lines.append(f"({ref.number}) = ${item.latex}$, {how}.")
    if not lines:
        return shown, ""
    note = (
        "[notice from Minerva, not from the user] The user's message refers to numbered equations of "
        "this conversation: " + " ".join(lines)
    )
    return shown, note


def replay_prose(content: dict[str, Any]) -> str:
    """The model's earlier prose with the number after each numbered displayed formula.

    The numbers are what the user sees. They are put into the replayed text so that "(3)" in the user's next
    message means something to the model; it is told not to write numbers itself.
    """
    # Reasoning that leaked into a reply stored earlier is not replayed to the model.
    raw = clean_reply(str(content.get("text", "")))
    numbered = [
        (str(s.get("value", "")), s["number"])
        for b in content.get("blocks", []) or []
        for s in b.get("segments", []) or []
        if isinstance(s, dict)
        and s.get("type") == "display_math"
        and isinstance(s.get("number"), int)
    ]
    cursor = 0
    for value, number in numbered:
        pos = raw.find(value, cursor)
        if pos < 0:
            continue
        end = pos + len(value)
        for closer in ("$$", "\\]"):
            tail = raw[end:].lstrip(" \t")
            if tail.startswith(closer):
                end = len(raw) - len(tail) + len(closer)
                break
        tag = f" ({number})"
        raw = raw[:end] + tag + raw[end:]
        cursor = end + len(tag)
    return raw


def number_stored(contents: Iterable[tuple[str, dict[str, Any]]], table: EquationTable) -> bool:
    """Number the stored messages of a conversation that predates equation numbers, in place.

    `contents` are (role, content) pairs in order; each content dict is edited. The same reading-order
    rule as for a live reply, so the numbers are the ones the conversation would have had. Returns
    whether any number was given.
    """
    gave = False
    for role, content in contents:
        if role == "user":
            found = number_user_text(str(content.get("text", "")), table)
            if found:
                content["equations"] = found
                gave = True
            continue
        steps = {s.get("id"): s for s in content.get("trace", []) or [] if isinstance(s, dict)}
        for block in content.get("blocks", []) or []:
            card = block.get("card") if block.get("kind") == "card" else None
            if isinstance(card, dict):
                latex = (card.get("data") or {}).get("result_latex")
                if isinstance(latex, str):
                    rid = (steps.get(card.get("id")) or {}).get("result_id")
                    card["number"] = table.number_for(latex, "from_tool", rid)
                    gave = gave or card["number"] is not None
            for seg in block.get("segments", []) or []:
                if isinstance(seg, dict) and seg.get("type") == "display_math":
                    seg["number"] = table.number_for(
                        str(seg.get("value", "")), _source_of(seg.get("status"))
                    )
                    gave = gave or seg["number"] is not None
    return gave


def unverified_inputs(messages: Iterable[Any]) -> dict[str, list[str]]:
    """For each verified result id, the unverified equations it was computed from (from stored cards)."""
    out: dict[str, list[str]] = {}
    for m in messages:
        if m.role != "assistant":
            continue
        cards = {
            b["card"].get("id"): b["card"]
            for b in m.content.get("blocks", []) or []
            if b.get("kind") == "card" and isinstance(b.get("card"), dict)
        }
        for step in m.content.get("trace", []) or []:
            rid = step.get("result_id")
            carried = (cards.get(step.get("id")) or {}).get("data", {}).get("unverified_inputs")
            if rid and isinstance(carried, list) and carried:
                out[rid] = [str(c) for c in carried]
    return out


_MODEL_TAG = re.compile(r"(\$\$.+?\$\$|\\\[.+?\\\])[ \t]*\(\s*\d{1,3}\s*\)", re.DOTALL)


def strip_equation_tags(text: str) -> str:
    """Remove a number the model wrote itself right after a displayed formula ("$$...$$ (3)").

    The server numbers equations; a number the model writes would be its guess (it copies the pattern
    from the replayed history) and could differ from the real one, so it is dropped before the reply is
    split, stored and numbered. A "(3)" anywhere else in the prose, in a sentence or a list, stays.
    """
    return _MODEL_TAG.sub(r"\1", text)
