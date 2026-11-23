"""What a turn is given from the conversation that came before it: one pure function for the chat route, the
evaluation harness and the replay script.

The live chat rebuilds a turn's context from the stored messages of the conversation: the history the model is
shown, the formulas earlier results verified or only quoted, the result handles (`@r1`) and the restrictions
they carry, the equation numbers and what "equation 3" means in the new message, the equations a result was
computed from that Minerva never verified, the claim checker's relations, assumptions and verified evidence, the
declarations in words, the tool groups the conversation needed and the constants of motion earlier discovery
cards verified. Each of these is read from the stored rows (`role` and `content`), never from a model.

`turn_context` does all of it and touches no database and no model. The route calls it with the rows it loaded,
`agent/evaluation/runner.py` calls it with the answers of the case so far, and `scripts/replay_trace.py` calls it
with the messages of a debug report (format 4 stores the cards it needs). Before this existed each of the three
assembled the context on its own and the harness and the replay had drifted: no equation table, no unverified
inputs, no reference or handle notes, and the replay restored nothing at all.

The functions below were moved here from `routers_chat.py` unchanged; `test_turn_context_pin.py` runs the route
over four turns and compares what it hands the loop with a file written before the move.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

from .budget import fit_history
from .card_equations import result_formula, result_restrictions
from .constants import KnownConstant, constants_of_blocks
from .declarations import scan
from .equations import (
    EquationTable,
    build_table,
    find_references,
    handles_note,
    number_user_text,
    replay_prose,
    resolve_references,
    unverified_inputs,
)
from .loop import matrix_names, vector_names
from .refs import reference_note
from .segments import normalize, split_segments
from .tools import ToolContext
from .toolsets import conversation_groups

HISTORY_MESSAGES = 20


class StoredMessage(Protocol):
    """A stored chat message as far as the context is concerned: the database row, or a stand-in made from a
    debug report or from an evaluation case's earlier answer."""

    @property
    def role(self) -> str: ...

    @property
    def content(self) -> dict[str, Any]: ...


LEGACY_QUOTED = "from an earlier tool result; how it was verified was not recorded"


OLD_RESULT_CHARS = 600  # what is kept of a tool result from an earlier exchange
RECENT_MESSAGES = 4  # the last two exchanges keep their tool results in full


def _shorten(text: str, limit: int = OLD_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n" in cut:
        cut = cut[: cut.rindex("\n")]
    return cut + "\n[shortened: an earlier result]"


def _history(
    messages: Sequence[StoredMessage],
    *,
    max_messages: int = HISTORY_MESSAGES,
    max_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], set[str], dict[str, str], dict[str, str], dict[str, list[str]]]:
    """Rebuild the model's view of earlier turns.

    Also returns what earlier turns established, so formulas are marked the same way across turns:
    the formulas verified results produced, the formulas that were only quoted (with why), and the
    verified results that a later call may refer to (`@r1`) with the restrictions each is valid under.
    What was established is read from the WHOLE conversation (so a result id is never reused and its
    restrictions are never lost when the conversation outgrows the window); what the model is shown
    is the newest `max_messages` stored messages, at most `max_tokens` of them, starting at a user
    message, with the tool results of older exchanges shortened. Messages stored before verification
    was recorded per step have only text: their formulas count as quoted, never as verified.
    """
    units: list[list[dict[str, Any]]] = []  # what the model sees of each stored message
    known: set[str] = set()
    quoted: dict[str, str] = {}
    results: dict[str, str] = {}
    restrictions: dict[str, list[str]] = {}
    recent_from = max(0, len(messages) - RECENT_MESSAGES)
    for index, m in enumerate(messages):
        if m.role == "user":
            text = str(m.content.get("text", ""))
            if units and len(units[-1]) == 1 and units[-1][0]["role"] == "user":
                # The reply to the previous message was never stored (the connection dropped): the
                # model sees one message, never two user messages in a row.
                units[-1][0]["content"] += "\n\n" + text
            else:
                units.append([{"role": "user", "content": text}])
            continue
        unit: list[dict[str, Any]] = []
        cards = {
            b["card"]["id"]: b["card"]
            for b in m.content.get("blocks", [])
            if b.get("kind") == "card" and isinstance(b.get("card"), dict)
        }
        for step in m.content.get("trace", []):
            call_id = str(step.get("call_id") or f"call_{step.get('id', uuid.uuid4().hex[:8])}")
            unit.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": call_id, "name": step["name"], "arguments": step["arguments"]}
                    ],
                }
            )
            content = step["for_model"] if index >= recent_from else _shorten(step["for_model"])
            number = cards.get(step["id"], {}).get("number")
            if isinstance(number, int):
                content += (
                    f"\nequation_number=({number}) (the user sees this result as equation ({number}); "
                    f"@e{number} reuses it exactly)"
                )
            result_id = step.get("result_id")
            if result_id:
                content += "\n" + reference_note(result_id)
                card = cards.get(step["id"], {})
                data = card.get("data", {})
                latex = result_formula(card.get("type"), data)
                if latex:
                    results[result_id] = latex
                carried = [
                    *result_restrictions(card.get("type"), data),
                    *(
                        r
                        for r in (data.get("inherited_restrictions_latex") or [])
                        if isinstance(r, str)
                    ),
                ]
                if carried:
                    restrictions[result_id] = list(dict.fromkeys(carried))
            unit.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": step["name"],
                    "content": content,
                }
            )
            if "verified_latex" in step or "quoted" in step:
                known.update(normalize(x) for x in step.get("verified_latex", []))
                quoted.update(step.get("quoted", {}))
            else:  # stored before verification was recorded: quoted, never verified
                for seg in split_segments(step["for_model"], set()):
                    if seg.type != "text":
                        quoted.setdefault(normalize(seg.value), LEGACY_QUOTED)
        unit.append({"role": "assistant", "content": replay_prose(m.content)})
        units.append(unit)
    tail = units[-max_messages:]
    if len(tail) < len(units):  # cut by the window: begin at a user message
        while tail and tail[0][0]["role"] != "user":
            tail = tail[1:]
    out = [message for unit in tail for message in unit]
    if max_tokens:
        out, _ = fit_history(out, fixed_chars=0, limit_tokens=max_tokens)
    return out, known, {k: v for k, v in quoted.items() if k not in known}, results, restrictions


def _conversation_events(previous: Sequence[StoredMessage], new_text: str) -> list[tuple[str, Any]]:
    """The user messages and tool calls of the whole conversation, for the tool groups it needs."""
    events: list[tuple[str, Any]] = []
    for m in previous:
        if m.role == "user":
            events.append(("user", str(m.content.get("text", ""))))
        else:
            for step in m.content.get("trace", []):
                events.append(("tool", (str(step.get("name", "")), step.get("arguments", {}))))
    events.append(("user", new_text))
    return events


MAX_RELATIONS = 60


def _claim_context(
    messages: Sequence[StoredMessage], current: str
) -> tuple[list[str], dict[str, list[str]]]:
    """What earlier turns established that the claim checker may use: formulas written in the
    conversation, in order (the checker keeps only the ones that are definitions, and names them
    whenever a result depends on one), and the assumptions declared in earlier tool calls."""
    texts: list[str] = []
    assumptions: dict[str, list[str]] = {}
    for m in messages[-HISTORY_MESSAGES:]:
        texts.append(str(m.content.get("text", "")))
        if m.role != "user":
            for step in m.content.get("trace", []):
                declared = step.get("arguments", {}).get("assumptions")
                if isinstance(declared, dict):
                    for name, items in declared.items():
                        if isinstance(items, list):
                            assumptions[str(name)] = [str(i) for i in items]
    texts.append(current)
    relations = [
        seg.value
        for text in texts
        for seg in split_segments(text, set())
        if seg.type != "text" and len(seg.value) <= 500
    ]
    return relations[-MAX_RELATIONS:], assumptions


def _declaration_inputs(
    messages: Sequence[StoredMessage], current: str
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """The tool calls of the whole conversation and every message in order, the current one last."""
    steps: list[dict[str, Any]] = []
    entries: list[tuple[str, str]] = []
    for m in messages:
        entries.append((m.role, str(m.content.get("text", ""))))
        if m.role != "user":
            steps.extend(s for s in m.content.get("trace", []) if isinstance(s, dict))
    entries.append(("user", current))
    return steps, entries


def _prior_evidence(messages: Sequence[StoredMessage]) -> list[dict[str, Any]]:
    """The verified results of earlier turns, as the claim check needs them (`claim_evidence.py`).

    Read from the stored trace of the whole conversation, never from the model's text: a message stored
    before verification was recorded has no `verified_latex` and so contributes nothing.
    """
    rows: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant":
            rows.extend(
                {
                    k: step.get(k)
                    for k in (
                        "status",
                        "repeated",
                        "evidence",
                        "formula_roles",
                        "verified_latex",
                        "variables",
                        "restrictions",
                        "result_id",
                    )
                }
                for step in m.content.get("trace", [])
                if step.get("verified_latex")
            )
    return rows


class Declarations(NamedTuple):
    """What a conversation declared as matrices or vectors, read three ways for three purposes."""

    # generous: leaves a claim unchecked. A name is cheap to remember and forgetting one makes a matrix claim
    # checkable as if it were about numbers; a name the user later called a scalar is dropped. Matrices only:
    # a vector name goes to `prior_vectors`, so the claim is told it is about a vector, not a matrix.
    prior_matrices: list[str]
    # strict: name -> where it came from, for the names that make a scalar tool refuse (tool-call declarations
    # and sentences that *declare* a name), minus names the user later called scalars.
    declared_for_tools: dict[str, str]
    # of the strict names, those declared as VECTORS (not matrices): a vector's components and coordinates are
    # scalars, so the tools that only a declared matrix makes wrong (`tools.MATRIX_ONLY_TOOLS`) keep answering.
    vector_kind_names: frozenset[str]
    # generous, like `prior_matrices`, for the names declared as vectors.
    prior_vectors: list[str]
    # names the conversation settled as scalars (`r = |\\mathbf{r}|`, or the user's "r is a scalar"): the
    # model's reply cannot declare them again for its own claim check (`declarations.declared_in_reply`).
    scalar_names: frozenset[str]


def declarations_of(steps: list[dict[str, Any]], entries: list[tuple[str, str]]) -> Declarations:
    """The one place a conversation's declarations are read. The chat route and the evaluation harness both call
    it (`routers_chat.send_message`, `evaluation.runner.observe`): they once computed it separately, and the
    harness gave the claim checker less than the chat did (A-008)."""
    generous = scan(entries, strict=False)
    tool_matrices, tool_vectors = matrix_names(steps), vector_names(steps)
    said_vector = {n for n, d in generous.found.items() if d.kind == "vector"}
    prior_vectors = ((tool_vectors - generous.cleared) | said_vector) - tool_matrices
    prior = ((tool_matrices - generous.cleared) | set(generous.found)) - prior_vectors
    strict = scan(entries, strict=True)
    found, cleared = strict.found, strict.cleared
    sources = {
        name: ("a matrix or vector declared in a tool call earlier in this conversation")
        for name in (tool_matrices | tool_vectors) - cleared
    }
    for name, d in found.items():
        who = "your message" if d.role == "user" else "my earlier reply"
        sources[name] = f"{who}: \u00ab{d.quote}\u00bb"
    by_sentence = {n for n, d in found.items() if d.kind == "vector"}
    vectors = frozenset((by_sentence | tool_vectors) - cleared - tool_matrices)
    return Declarations(
        sorted(prior), sources, vectors, sorted(prior_vectors), frozenset(generous.held)
    )


@dataclass
class TurnContext:
    """Everything a turn is given from the conversation before it (see the module docstring)."""

    history: list[dict[str, Any]]  # what the model is shown, the new message last, notes appended
    equations: (
        EquationTable  # the conversation's numbered equations, the user's new formulas included
    )
    user_equations: list[dict[str, Any]]  # the displayed formulas of the new message, numbered
    shown_references: list[dict[str, Any]]  # what "equation 3" in the new message was taken to mean
    prior_known: set[str]
    prior_quoted: dict[str, str]
    prior_results: dict[str, str]
    prior_restrictions: dict[str, list[str]]
    prior_groups: list[str]
    prior_unverified: dict[str, list[str]]
    prior_relations: list[str]
    prior_assumptions: dict[str, list[str]]
    prior_evidence: list[dict[str, Any]]
    declarations: Declarations
    constants: tuple[KnownConstant, ...]  # verified constants of earlier discovery cards
    user_messages: list[str]  # the user's messages of the whole conversation, the new one last

    def agent_kwargs(self) -> dict[str, Any]:
        """The keyword arguments `run_agent` takes from the conversation (not the model parameters or info)."""
        return {
            "prior_known": self.prior_known,
            "prior_quoted": self.prior_quoted,
            "prior_evidence": self.prior_evidence,
            "prior_relations": self.prior_relations,
            "prior_assumptions": self.prior_assumptions,
            "prior_matrices": self.declarations.prior_matrices,
            "declared_for_tools": self.declarations.declared_for_tools,
            "vector_kind_names": self.declarations.vector_kind_names,
            "prior_vectors": self.declarations.prior_vectors,
            "scalar_names": self.declarations.scalar_names,
            "prior_results": self.prior_results,
            "prior_restrictions": self.prior_restrictions,
            "prior_groups": self.prior_groups,
            "equations": self.equations,
            "prior_unverified": self.prior_unverified,
        }

    def apply_to(self, ctx: ToolContext) -> None:
        """Set on the tool context what tools read from the conversation: the user's messages and the verified
        constants of earlier discovery results (from the stored cards, never from the model's text)."""
        ctx.user_messages = self.user_messages
        ctx.constants = self.constants


def turn_context(
    previous: Sequence[StoredMessage],
    text: str,
    *,
    counter: int = 0,
    max_messages: int = HISTORY_MESSAGES,
    max_tokens: int | None = None,
) -> TurnContext:
    """The context of the turn that answers `text`, given the stored messages that came before it.

    `previous` are the surviving messages in order, the message being answered excluded (for a retry or an edit
    the reply being replaced and the message itself are already out); `counter` is the conversation's equation
    number high-water mark (a number given out once is never reused, even when its message is gone). The order
    of the steps is the chat route's: references in `text` are resolved against the numbers that existed before
    it, then its own displayed formulas are numbered.
    """
    equations = build_table(previous, counter)
    # What "equation 3" in this message means (resolved before this message's own formulas are numbered).
    shown_references, reference_note_text = resolve_references(find_references(text), equations)
    user_equations = number_user_text(text, equations)
    history, prior_known, prior_quoted, prior_results, prior_restrictions = _history(
        previous, max_messages=max_messages, max_tokens=max_tokens
    )
    prior_groups = conversation_groups(_conversation_events(previous, text))
    prior_unverified = unverified_inputs(previous)
    prior_relations, prior_assumptions = _claim_context(previous, text)
    steps, entries = _declaration_inputs(previous, text)
    declarations = declarations_of(steps, entries)
    prior_evidence = _prior_evidence(previous)
    if history and history[-1]["role"] == "user":  # the last reply was never stored: one message
        history[-1] = {"role": "user", "content": history[-1]["content"] + "\n\n" + text}
    else:
        history.append({"role": "user", "content": text})
    notes = [n for n in (reference_note_text, handles_note(user_equations)) if n]
    if notes:  # for the model only: the stored message stays as the user typed it
        history[-1] = {
            "role": "user",
            "content": history[-1]["content"] + "\n\n" + "\n".join(notes),
        }
    return TurnContext(
        history=history,
        equations=equations,
        user_equations=user_equations,
        shown_references=shown_references,
        prior_known=prior_known,
        prior_quoted=prior_quoted,
        prior_results=prior_results,
        prior_restrictions=prior_restrictions,
        prior_groups=prior_groups,
        prior_unverified=prior_unverified,
        prior_relations=prior_relations,
        prior_assumptions=prior_assumptions,
        prior_evidence=prior_evidence,
        declarations=declarations,
        constants=tuple(
            constants_of_blocks(
                m.content.get("blocks", []) for m in previous if m.role == "assistant"
            )
        ),
        user_messages=[t for role, t in entries if role == "user"],
    )
