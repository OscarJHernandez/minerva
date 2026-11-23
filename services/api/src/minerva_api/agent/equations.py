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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .card_equations import LIST_READERS, listed_equations
from .events import Block
from .reasoning import clean_reply
from .segments import MATH, normalize, split_segments

Source = Literal["from_tool", "written_by_model", "yours"]
# Whose text a formula is, which decides whether a trailing equation label may be read off it (see `strip_label`).
LabelMode = Literal["model", "user", "none"]
MAX_EQUATION_CHARS = (
    500  # a longer formula is shown but not numbered, like the claim checker's limit
)


@dataclass
class Equation:
    number: int
    latex: str
    source: Source
    result_id: str | None = None  # "r2": the verified result this equation is, when it is one
    # The card type whose list this equation is one row of ("geometry"). Such a row is exact LaTeX from a tool,
    # but it is not linked to that card's verification or restrictions (see `refs.UNLINKED_INPUT`).
    part_of: str | None = None
    # A stored equation that is the same formula as a lower number once its own label is read off (a
    # conversation numbered before labels inside display math were removed): the lower number.
    alias_of: int | None = None


# --- labels the model writes inside a displayed formula ----------------------------------------------
_SPACING_CMD = r"(?:\\(?:quad|qquad|hfill)(?![A-Za-z])|\\hspace\*?\{[^{}]*\})"
_PAREN_NUMBER = r"\(\s*\d{1,3}\s*\)"
_TEXT_NUMBER = rf"\\(?:text|textrm|mathrm)\{{\s*{_PAREN_NUMBER}\s*\}}"
_TAG = r"\\tag\*?\{\s*\d{1,3}\s*\}"
_TRAILER = r"(?P<punct>[.,]?)\s*$"
# Forms that can only be a label: `\tag{3}`, a spacing command then `(3)`, `\text{(3)}`.
_UNAMBIGUOUS = re.compile(
    rf"\s*(?:{_TAG}|(?:{_SPACING_CMD}\s*)+(?:{_PAREN_NUMBER}|{_TEXT_NUMBER})|{_TEXT_NUMBER}){_TRAILER}"
)
# Whitespace then `(3)` at the very end: a label or a factor; `_bare_label_ok` decides.
_BARE = re.compile(rf"(?<=\S)\s+{_PAREN_NUMBER}{_TRAILER}")
_SINGLE_ENV = re.compile(
    r"(?P<open>\\begin\{(?P<env>equation\*?)\})(?P<inner>.*)(?P<close>\\end\{(?P=env)\})\Z",
    re.DOTALL,
)
_OPERATOR_END = set("+-*/=<>,;:([{&^_")


def _balanced(text: str) -> bool:
    plain = re.sub(r"\\[{}]|\\left|\\right", "", text)
    return all(plain.count(a) == plain.count(b) for a, b in ("()", "[]", "{}")) and len(
        re.findall(r"\\begin\{", text)
    ) == len(re.findall(r"\\end\{", text))


def _bare_label_ok(prefix: str) -> bool:
    """Whether a final ` (N)` after `prefix` reads as a label rather than a factor or an argument.

    Kept (not a label): after a command name (`\\sin (2)`, `\\Gamma (3)`), after an operator, relation,
    comma or opening bracket, after a letter the formula applies as a function elsewhere (`f(x) + f (1)`),
    and whenever removing it would leave unbalanced brackets or environments.
    """
    p = prefix.rstrip()
    if not p or p[-1] in _OPERATOR_END or p.endswith("\\\\"):
        return False
    if re.search(r"\\[A-Za-z]+$", p) or not _balanced(p):
        return False
    applied = p[-1].isalpha() and re.search(
        rf"(?<![A-Za-z\\]){re.escape(p[-1])}\s*(?:\\left)?\(", p
    )
    return not applied


def strip_label(body: str, mode: LabelMode) -> tuple[str, str]:
    """The formula without an equation label the model (or user) wrote at its end, and the label removed.

    The server numbers equations; a number written inside the formula is presentation, not mathematics.
    Only the very end of the body is read, one label, once:

    - `model` (text the model wrote): `\\tag{3}`, a spacing command or `\\text{...}` around `(3)`, and a
      bare ` (3)` after whitespace unless it can be a factor or an argument (`_bare_label_ok`).
      Accepted risk: a formula new to the conversation such as `y = g (1)` loses its `(1)` when `g` is not
      applied elsewhere in it; every removal is recorded (`TurnDebug.removed_labels`).
    - `user` (the user's own message): the unambiguous forms only; a bare ` (3)` there is a factor.
    - `none` (tool output): nothing. A verified formula is keyed and stored exactly as the tool gave it.

    A single-formula environment (`equation`, `equation*`) is read before its `\\end`; multi-row ones
    (`align`, `aligned`, `gather`, `eqnarray`) are left alone, as are row labels inside them. A final `.` or
    `,` after the label stays: `a = b \\quad (1).` gives `a = b.`.
    """
    if mode == "none":
        return body, ""
    env = _SINGLE_ENV.match(body.strip())
    if env is not None:
        inner, removed = strip_label(env.group("inner"), mode)
        if not removed:
            return body, ""
        return f"{env.group('open')}{inner}{env.group('close')}", removed
    core = body.rstrip()
    tail = body[len(core) :]
    hit = _UNAMBIGUOUS.search(core)
    if hit is None and mode == "model":
        hit = _BARE.search(core)
        if hit is not None and not _bare_label_ok(core[: hit.start()]):
            hit = None
    if hit is None or not core[: hit.start()].strip():
        return body, ""
    return core[: hit.start()] + hit.group("punct") + tail, core[hit.start() : hit.end()].strip()


def equation_key(latex: str, mode: LabelMode) -> str:
    """The comparison key of a formula once the label `mode` allows is read off."""
    return normalize(strip_label(latex, mode)[0])


@dataclass
class EquationTable:
    """The numbered equations of one conversation."""

    items: dict[int, Equation] = field(default_factory=dict)
    index: dict[str, int] = field(default_factory=dict)  # normalised LaTeX -> number
    counter: int = 0  # the highest number ever given (kept even when messages are removed)
    reused: int = (
        0  # formulas that were given the number they already had (counted per table, per turn)
    )
    aliases_resolved: int = 0  # references to a number that is an alias of a lower one

    def number_for(
        self,
        latex: str,
        source: Source,
        result_id: str | None = None,
        part_of: str | None = None,
        *,
        mode: LabelMode,
        any_length: bool = False,
    ) -> int | None:
        """The number of this formula: the one it already has, or the next. None if it cannot be numbered.

        A formula already numbered *as written* is found first and never stripped (a model restating a tool
        result or the user's formula cannot change it); only otherwise is the label `mode` allows read off.
        `any_length` is for the one result of a tool card: its number is a name that resolves through the
        result id (or, without one, to the exact stored text), so it is given whatever the length. The limit
        stays for the model's prose and the user's formulas.
        Known gap: a tool result that ends in whitespace + `(N)` and is quoted by the model before its card
        in reading order is keyed stripped, before the card's exact entry exists.
        """
        if not latex.strip() or (len(latex) > MAX_EQUATION_CHARS and not any_length):
            return None
        exact = normalize(latex)
        if not exact:
            return None
        existing = self.index.get(exact)
        stored = latex
        key = exact
        if existing is None:
            stored = strip_label(latex, mode)[0]
            key = normalize(stored)
            existing = self.index.get(key) if key else None
        if not key:
            return None
        if existing is not None:
            item = self.items[existing]
            if result_id and not item.result_id:  # the same formula is also a verified result
                item.result_id, item.source = result_id, "from_tool"
            self.reused += 1
            return existing
        self.counter += 1
        self.items[self.counter] = Equation(self.counter, stored, source, result_id, part_of)
        self.index[key] = self.counter
        return self.counter

    def restore(
        self,
        number: int,
        latex: str,
        source: Source,
        result_id: str | None = None,
        part_of: str | None = None,
        *,
        mode: LabelMode,
    ) -> None:
        """Re-enter an equation that was numbered earlier (from a stored message).

        Same order as `number_for`: the exact key, then the key with the label `mode` allows read off. A
        number whose formula already belongs to a lower number is kept and recorded as its alias.
        """
        self.counter = max(self.counter, number)
        stored = latex
        key = normalize(latex)
        earlier = self.index.get(key)
        if earlier is None:
            stored = strip_label(latex, mode)[0]
            key = normalize(stored)
            earlier = self.index.get(key) if key else None
        if number not in self.items:
            alias = earlier if earlier is not None and earlier < number else None
            self.items[number] = Equation(number, stored, source, result_id, part_of, alias)
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
                    table.restore(e["number"], str(e.get("latex", "")), "yours", mode="user")
            continue
        steps = {s.get("id"): s for s in content.get("trace", []) or [] if isinstance(s, dict)}
        for block in content.get("blocks", []) or []:
            if block.get("kind") == "card" and isinstance(block.get("card"), dict):
                card = block["card"]
                number = card.get("number")
                latex = (card.get("data") or {}).get("result_latex")
                if isinstance(number, int) and isinstance(latex, str):
                    result_id = (steps.get(card.get("id")) or {}).get("result_id")
                    table.restore(number, latex, "from_tool", result_id, mode="none")
                _restore_listed(card, table)
            for seg in block.get("segments", []) or []:
                number = seg.get("number") if isinstance(seg, dict) else None
                if isinstance(number, int):
                    table.restore(
                        number,
                        str(seg.get("value", "")),
                        _source_of(seg.get("status")),
                        mode="model",
                    )
    return table


def _restore_listed(card: dict[str, Any], table: EquationTable) -> None:
    """Re-enter the numbered rows of a stored list card (its `numbers` by key, its rows from its own data)."""
    numbers = card.get("numbers")
    kind = card.get("type")
    if not isinstance(numbers, dict) or not isinstance(kind, str) or kind not in LIST_READERS:
        return
    data = card.get("data")
    for key, latex in listed_equations(kind, data if isinstance(data, dict) else {}):
        number = numbers.get(key)
        if isinstance(number, int):
            table.restore(number, latex, "from_tool", None, kind, mode="none")


def number_blocks(
    blocks: Sequence[Block], table: EquationTable, result_ids: Mapping[str, str] | None = None
) -> None:
    """Give the displayed formulas and primary results of a reply their numbers, in reading order.

    `result_ids` maps a card's id to the id of the verified result it is ("r2"), when it is one.
    """
    for block in blocks:
        if block.kind == "card" and block.card is not None:
            if block.card.type in LIST_READERS:
                block.card.numbers = _number_listed(block.card.type, block.card.data, table)
                continue
            latex = block.card.data.get("result_latex")
            if isinstance(latex, str):
                block.card.number = table.number_for(
                    latex,
                    "from_tool",
                    (result_ids or {}).get(block.card.id),
                    mode="none",
                    any_length=True,
                )
            continue
        for seg in block.segments:
            if seg.type == "display_math":
                seg.number = table.number_for(seg.value, _source_of(seg.status), mode="model")


def _number_listed(card_type: str, data: dict[str, Any], table: EquationTable) -> dict[str, int]:
    """One whole number per listed equation, in display order, from the conversation's one counter."""
    out: dict[str, int] = {}
    for key, latex in listed_equations(card_type, data):
        number = table.number_for(latex, "from_tool", None, card_type, mode="none")
        if number is not None:
            out[key] = number
    return out


def number_user_text(text: str, table: EquationTable) -> list[dict[str, Any]]:
    """Numbers for the displayed formulas of the user's message, in order (None-numbered ones are skipped)."""
    out: list[dict[str, Any]] = []
    for seg in split_segments(text, set()):
        if seg.type == "display_math":
            number = table.number_for(seg.value, "yours", mode="user")
            if number is not None:
                out.append({"number": number, "latex": table.items[number].latex})
    return out


MAX_HANDLES = 8


def handles_note(user_equations: Sequence[dict[str, Any]]) -> str:
    """The note that tells the model the handles of the displayed formulas in the user's message.

    The model is never told these numbers otherwise, so it copies a formula by hand and mistypes it (a
    Lagrangian's velocities, a sign). A handle in a tool argument is replaced by the user's exact text, so
    copying becomes unnecessary. Only the handles are listed, in the order written: the formulas are in the
    message above, and the note stays a few tokens long. Nothing about a handle says it is verified.
    """
    numbers = [int(item["number"]) for item in user_equations[:MAX_HANDLES]]
    if not numbers:
        return ""
    names = ", ".join(f"@e{n}" for n in numbers)
    more = len(user_equations) - len(numbers)
    return (
        "[notice from Minerva, not from the user] The displayed formulas of the user's message are, in the "
        f"order written: {names}"
        + (f" (and {more} more)" if more > 0 else "")
        + ". In a tool call, write the handle (for example `@e"
        + f"{numbers[0]}"
        + "`) as the whole formula argument instead of copying the formula: Minerva substitutes the "
        "user's exact text."
    )


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
        if item.alias_of is not None:
            shown[-1]["alias_of"] = item.alias_of
            table.aliases_resolved += 1
        if item.result_id:
            how = f"a verified result (write @{item.result_id} or @e{item.number} in a tool call to reuse it exactly)"
        elif item.source == "yours":
            how = f"written by the user (write @e{item.number} in a tool call to reuse it exactly)"
        elif item.part_of:
            how = (
                f"one row of a {item.part_of} tool result (@e{item.number} reuses its LaTeX exactly; the tool's "
                "verification and restrictions are not carried over, so say that)"
            )
        else:
            how = (
                "written by the model and NOT verified by Minerva (@e"
                f"{item.number} reuses it exactly; say that it was not verified)"
            )
        same = f"the same formula as ({item.alias_of}); " if item.alias_of is not None else ""
        lines.append(f"({ref.number}) = ${item.latex}$, {same}{how}.")
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
            if isinstance(card, dict) and card.get("type") in LIST_READERS:
                data = card.get("data")
                listed = _number_listed(
                    str(card["type"]), data if isinstance(data, dict) else {}, table
                )
                if listed and not card.get("numbers"):
                    card["numbers"] = listed
                    gave = True
            elif isinstance(card, dict):
                latex = (card.get("data") or {}).get("result_latex")
                if isinstance(latex, str):
                    rid = (steps.get(card.get("id")) or {}).get("result_id")
                    card["number"] = table.number_for(
                        latex, "from_tool", rid, mode="none", any_length=True
                    )
                    gave = gave or card["number"] is not None
            for seg in block.get("segments", []) or []:
                if isinstance(seg, dict) and seg.get("type") == "display_math":
                    seg["number"] = table.number_for(
                        str(seg.get("value", "")), _source_of(seg.get("status")), mode="model"
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


_AFTER_LABEL = re.compile(r"[ \t]*\(\s*\d{1,3}\s*\)")


def strip_equation_tags(
    text: str, keep: Callable[[str], bool] | None = None
) -> tuple[str, list[str]]:
    """Remove the numbers the model wrote itself on displayed formulas, and say what was removed.

    The server numbers equations; a number the model writes would be its guess (it copies the pattern
    from the replayed history) and could differ from the real one, so it is dropped before the reply is
    split, stored and numbered: after the closing delimiter (`$$...$$ (3)`) and inside the formula
    (`$$... (3)$$`, `\\tag{3}`, see `strip_label`). The display formulas are found with the splitter's own
    pattern (`segments.MATH`), so the two never disagree about where one ends. `keep(body)` is true for a
    formula that is already numbered as written (a tool result, the user's formula): its body is left alone.
    A "(3)" anywhere else in the prose, in a sentence or a list, stays. Returns the text and the removed
    labels in reading order.
    """
    removed: list[str] = []
    out: list[str] = []
    position = 0
    for match in MATH.finditer(text):
        group = next((g for g in ("d1", "d2", "d3") if match.group(g) is not None), None)
        if group is None:
            continue
        body = match.group(group)
        new_body = body
        if keep is None or not keep(body.strip()):
            new_body, label = strip_label(body, "model")
            if label:
                removed.append(label)
        end = match.end()
        after = _AFTER_LABEL.match(text, end)
        if after is not None:
            removed.append(after.group().strip())
            end = after.end()
        if new_body == body and end == match.end():
            continue
        start = match.start(group)
        out.append(text[position:start] + new_body)
        out.append(text[match.end(group) : match.end()])
        position = end
    out.append(text[position:])
    return "".join(out), removed
