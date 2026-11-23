#!/usr/bin/env python3
"""Re-run what a chat debug report recorded, and show what changed.

In the chat, switch on "Debug mode", press "Copy debug report" (or "Copy this turn"), and save the
text to a file. Then:

    uv run python scripts/replay_trace.py report.md              # re-run every recorded tool call
    uv run python scripts/replay_trace.py report.md --llm        # replay a whole turn, without a model
    uv run python scripts/replay_trace.py report.md --llm --turn 4

Tool replay executes each recorded call (the arguments after `@rN` references were replaced) through the
same sandboxed `tools.execute` the API uses, and compares what the model read back, the tool's status and
the evidence record with what was recorded. No model and no network are involved; tools are pure.

`--llm` rebuilds the model's replies of one turn from the recorded events (which tools it asked for at
each step, the prose it wrote) and runs the real agent loop on them with a `ReplayLLM`. Everything but the
model is real: tools, result references, formula marks, the claim check, repeated-call detection, stop
reasons. It reproduces what the agent did with those replies; it cannot reproduce the model. Nothing here
ever calls a model host.

A follow-up turn is given the context the live chat gave it: the earlier messages of the report are handed to
`minerva_api.agent.turn_context.turn_context`, the function the chat route calls, so the history, the
verified constants of earlier discovery cards, `@rN` results with their restrictions, the equation table,
unverified inputs, evidence and declarations are rebuilt from the stored cards and trace. A format 4 report
holds the cards. Formats 2 and 3 do not: for them the script says which categories it cannot restore and that
the follow-up cannot be faithfully replayed, and a difference that missing context may explain is listed
apart from a difference in what the engine established.

Exit status 1 means at least one call or turn now differs in what the engine did, which is what you want to
see after a fix and what a regression test pins. 2 means the report could not be read. 3 means a whole-turn
replay found no such difference but could not be faithful (context missing from the report): it is not a pass.

Reports contain the user's mathematics: keep them out of the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from minerva_api.agent.equations import (
    build_table,
    find_references,
    number_user_text,
    resolve_references,
)
from minerva_api.agent.llm import Chunk, LLMError, ToolCall
from minerva_api.agent.loop import STOP_SENTENCES, AgentLimits, run_agent
from minerva_api.agent.replay import ReplayLLM
from minerva_api.agent.tools import ToolContext, execute
from minerva_api.agent.tracing import StopReason
from minerva_api.agent.turn_context import HISTORY_MESSAGES, turn_context
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

JSON_START = "<!-- minerva-debug-json -->"
JSON_END = "<!-- /minerva-debug-json -->"
SUPPORTED_FORMATS = (1, 2, 3, 4)
SUPPORTED_FORMAT = SUPPORTED_FORMATS[-1]


class ReportError(ValueError):
    """The text is not a debug report this script understands."""


def _context_for(step: dict[str, Any], ctx: ToolContext) -> ToolContext:
    """The context a recorded call ran in: a distribution card run with the user's answer carries it."""
    notation = step["arguments"].get("notation") if step["name"] == "distribution" else None
    if isinstance(notation, dict):
        return ToolContext(engine=ctx.engine, kb=ctx.kb, notation=dict(notation))
    numeric = step["arguments"].get("numeric") if step["name"] == "variation" else None
    if isinstance(numeric, dict):
        return ToolContext(engine=ctx.engine, kb=ctx.kb, numeric=dict(numeric))
    return ctx


@dataclass
class Replayed:
    turn: int  # 1-based index of the message in the report
    name: str
    arguments: dict[str, Any]
    recorded_status: str | None
    recorded: str
    status: str
    now: str
    recorded_evidence: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    compare_status: bool = False  # format 2 reports record the real status; format 1 did not always
    skipped: str | None = None  # why the call was not run again (it was never run the first time)

    @property
    def same_text(self) -> bool:
        return self.recorded == self.now

    @property
    def same_status(self) -> bool:
        return not self.compare_status or self.recorded_status == self.status

    @property
    def same_evidence(self) -> bool:
        return self.recorded_evidence is None or self.recorded_evidence == self.evidence

    @property
    def same(self) -> bool:
        return self.skipped is not None or (
            self.same_text and self.same_status and self.same_evidence
        )


def extract_data(text: str) -> dict[str, Any]:
    """The JSON block between the markers (a code fence of any length wraps it)."""
    start, end = text.find(JSON_START), text.rfind(JSON_END)
    if start < 0 or end < start:
        raise ReportError("no minerva-debug-json block found; paste the whole report")
    lines = text[start + len(JSON_START) : end].strip().splitlines()
    if len(lines) < 3 or not lines[0].startswith("```") or not lines[-1].startswith("```"):
        raise ReportError("the JSON block is not inside a code fence")
    try:
        data = json.loads("\n".join(lines[1:-1]))
    except json.JSONDecodeError as exc:
        raise ReportError(f"the JSON block is damaged: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") not in SUPPORTED_FORMATS:
        raise ReportError(f"unsupported report format {data.get('format')!r}")
    return data


def replay(data: dict[str, Any], ctx: ToolContext) -> list[Replayed]:
    out: list[Replayed] = []
    detailed = data.get("format", 1) >= 2
    trace_sets = [(i, m.get("trace") or []) for i, m in enumerate(data.get("messages", []), 1)]
    last = data.get("last_error") or {}
    trace_sets.append((len(trace_sets) + 1, last.get("trace") or []))
    for turn, trace in trace_sets:
        for step in trace:
            recorded_evidence = step.get("evidence") if detailed else None
            if step.get("status") == "skipped" or step.get("repeated"):
                out.append(
                    Replayed(
                        turn,
                        step["name"],
                        step["arguments"],
                        step.get("status"),
                        step["for_model"],
                        "skipped",
                        step["for_model"],
                        skipped="not run in the original turn (repeated call or limit reached)",
                    )
                )
                continue
            result = execute(step["name"], step["arguments"], _context_for(step, ctx))
            out.append(
                Replayed(
                    turn=turn,
                    name=step["name"],
                    arguments=step["arguments"],
                    recorded_status=step.get("status"),
                    recorded=step["for_model"],
                    status=result.status,
                    now=result.for_model,
                    recorded_evidence=recorded_evidence,
                    evidence=result.evidence.model_dump(mode="json") if result.evidence else None,
                    compare_status=detailed,
                )
            )
    return out


def render(items: list[Replayed], *, full: bool = False) -> str:
    def cut(text: str) -> str:
        return text if full or len(text) <= 700 else text[:700] + f"... [{len(text) - 700} more]"

    lines: list[str] = []
    for r in items:
        if r.skipped:
            lines.append(f"[turn {r.turn}] {r.name}: skipped ({r.skipped})")
            continue
        verdict = "same" if r.same else "DIFFERENT"
        lines.append(
            f"[turn {r.turn}] {r.name}: {verdict} (recorded {r.recorded_status}, now {r.status})"
        )
        lines.append(f"  arguments: {json.dumps(r.arguments, ensure_ascii=False)}")
        lines.append(f"  recorded:  {cut(r.recorded)}")
        if not r.same_text:
            lines.append(f"  now:       {cut(r.now)}")
        if not r.same_status:
            lines.append(f"  status:    recorded {r.recorded_status}, now {r.status}")
        if not r.same_evidence:
            lines.append(f"  evidence:  recorded {json.dumps(r.recorded_evidence)}")
            lines.append(f"             now      {json.dumps(r.evidence)}")
    changed = sum(not r.same for r in items)
    run = sum(r.skipped is None for r in items)
    lines.append(f"{run} call(s) replayed, {changed} different, {len(items) - run} skipped.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- whole-turn replay
@dataclass
class TurnReplay:
    turn: int
    # Differences in what the engine did that missing context cannot explain: a different tool, argument,
    # status, verdict, stop reason, mark or reply for a call that does not depend on the missing part.
    differences: list[str] = field(default_factory=list)
    # Differences that the missing context may explain (listed apart, never counted as a verdict change).
    context_differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    message: dict[str, Any] | None = None
    # What the live chat would have restored for this turn that the report cannot give back (category -> why).
    # Empty means the replay was given the same context. Never treat a replay with entries here as equivalent.
    missing: dict[str, str] = field(default_factory=dict)
    # The equation-number high-water mark the replay started the turn from (see `_counter_for`).
    counter: int = 0

    @property
    def faithful(self) -> bool:
        return not self.missing

    @property
    def same(self) -> bool:
        return self.faithful and not self.differences and not self.context_differences


def prose_text(segments: list[dict[str, Any]]) -> str:
    """Rebuild the model's text from segments, putting the formula delimiters back."""
    parts = []
    for s in segments:
        value = s["value"]
        parts.append(
            f"$${value}$$"
            if s["type"] == "display_math"
            else f"${value}$"
            if s["type"] == "math"
            else value
        )
    return "".join(parts)


def rebuild_replies(
    message: dict[str, Any],
) -> tuple[list[list[Chunk]], dict[int, LLMError], list[str]]:
    """The model's replies of one recorded turn: one list of chunks per model call.

    Returns the replies, the calls that failed (by 0-based index), and notes about anything that could
    not be reproduced.
    """
    debug = message.get("debug") or {}
    events = debug.get("events") or []
    if not events:
        raise ReportError("this turn has no trace events; --llm needs a report made with tracing")
    trace = {step["id"]: step for step in message.get("trace") or []}
    stop = message.get("stop_reason") or debug.get("stop_reason")
    prose = list(message.get("prose") or [])
    sentence = (
        STOP_SENTENCES.get(StopReason(stop)) if stop in {s.value for s in StopReason} else None
    )
    if prose and sentence and prose[-1] == sentence:
        prose.pop()  # the loop adds this sentence itself
    notes: list[str] = []
    by_step: dict[int, list[str]] = {}
    for e in events:
        if e["event"] in ("tool.started", "tool.completed") and e.get("step") is not None:
            ids = by_step.setdefault(e["step"], [])
            if e["call_id"] not in ids:
                ids.append(e["call_id"])
    replies: list[list[Chunk]] = []
    failures: dict[int, LLMError] = {}
    responses = [e for e in events if e["event"] == "llm.response"]
    for index, e in enumerate(responses):
        finish = e.get("finish_reason")
        if finish == "timeout":
            notes.append(f"model call {index + 1} timed out; a timeout cannot be replayed")
        if finish in ("error", "timeout"):
            failures[index] = LLMError("The model host is not reachable.")
            replies.append([])
            continue
        chunks: list[Chunk] = []
        if e.get("text_chars", 0) and prose:
            chunks.append(Chunk(text=prose.pop(0)))
        calls = []
        for call_id in by_step.get(e.get("step") or -1, []):
            step = trace.get(call_id)
            if step is not None:
                calls.append(
                    ToolCall(step["name"], step.get("model_arguments") or step["arguments"])
                )
        if calls:
            chunks.append(Chunk(tool_calls=calls))
        replies.append(chunks)
    if prose:
        notes.append(
            f"{len(prose)} piece(s) of recorded prose could not be assigned to a model call"
        )
    return replies, failures, notes


def _verdict(evidence: dict[str, Any] | None) -> tuple[Any, ...] | None:
    """The part of an evidence record that says what was established (never its sampled points)."""
    if not evidence:
        return None
    return (evidence.get("check"), evidence.get("accepted"), evidence.get("reason"))


FIXTURE_SCHEMA = 1


def to_fixture(data: dict[str, Any], turn: int | None = None) -> dict[str, Any]:
    """A regression fixture from one recorded turn: the executed calls and what the engine said, no prose.

    The arguments are still the user's mathematics: the developer replaces them with a minimal example of
    their own that shows the same failure before committing (the fixture is written outside the repository
    and `contains_user_mathematics` stays true until they do).
    """
    messages = data.get("messages", [])
    if turn is None:
        candidates = [
            i for i, m in enumerate(messages, 1) if m["role"] == "assistant" and m.get("trace")
        ]
        if not candidates:
            raise ReportError("no assistant turn with tool calls in this report")
        turn = candidates[-1]
    if not 1 <= turn <= len(messages) or messages[turn - 1]["role"] != "assistant":
        raise ReportError(f"message {turn} is not an assistant turn")
    recorded = messages[turn - 1]
    debug = recorded.get("debug") or {}
    calls = []
    for step in recorded.get("trace") or []:
        if step.get("status") == "skipped" or step.get("repeated"):
            continue
        verdict = _verdict(step.get("evidence"))
        calls.append(
            {
                "name": step["name"],
                "arguments": step["arguments"],
                "expect": {
                    "status": step.get("status"),
                    "evidence": dict(zip(("check", "accepted", "reason"), verdict, strict=True))
                    if verdict
                    else None,
                },
            }
        )
    return {
        "schema": FIXTURE_SCHEMA,
        "id": "replace-me",
        "provenance": {
            "origin": "debug-report",
            "report_format": data.get("format"),
            "turn": turn,
            "prompt_sha": debug.get("prompt_sha"),
            "diagnosis": debug.get("diagnosis") or None,
            "stop_reason": debug.get("stop_reason"),
        },
        "contains_user_mathematics": True,
        "calls": calls,
    }


def compare_turn_tagged(recorded: dict[str, Any], now: dict[str, Any]) -> list[tuple[str, str]]:
    """What differs between the recorded turn and the replayed one, each with what kind of thing it is.

    The kinds: `tool` (a call, its arguments, status or evidence), `stop` (the stop reason), `marks` (the
    formula marks) and `reply` (the reply text). `replay_turn` uses the kind to say whether missing context could
    explain a difference.
    """
    out: list[tuple[str, str]] = []
    a_stop = recorded.get("stop_reason") or (recorded.get("debug") or {}).get("stop_reason")
    if a_stop != now.get("stop_reason"):
        out.append(("stop", f"stop reason: recorded {a_stop}, now {now.get('stop_reason')}"))
    a_steps = [(t["name"], t.get("status")) for t in recorded.get("trace") or []]
    b_steps = [(t["name"], t.get("status")) for t in now.get("trace") or []]
    if a_steps != b_steps:
        out.append(("tool", f"tool calls and statuses: recorded {a_steps}, now {b_steps}"))
    a_marks = [(f["value"], f["status"]) for f in recorded.get("formulas") or []]
    b_marks = [
        (s["value"], s["status"])
        for b in now.get("blocks") or []
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    ]
    # What the engine was asked and what it established, not only which tool: a changed argument or a
    # different verdict is a difference even when the tool names and statuses agree.
    for index, (old, new) in enumerate(
        zip(recorded.get("trace") or [], now.get("trace") or [], strict=False), 1
    ):
        if old["name"] != new["name"]:
            continue
        if old.get("arguments") != new.get("arguments"):
            out.append(("tool", f"step {index} ({old['name']}): the arguments executed differ"))
        if _verdict(old.get("evidence")) != _verdict(new.get("evidence")):
            out.append(
                (
                    "tool",
                    f"step {index} ({old['name']}): evidence recorded {_verdict(old.get('evidence'))}, "
                    f"now {_verdict(new.get('evidence'))}",
                )
            )
    if recorded.get("formulas") is not None and a_marks != b_marks:
        out.append(("marks", f"formula marks: recorded {a_marks}, now {b_marks}"))
    if recorded.get("text") != now.get("text"):
        out.append(("reply", "the reply text differs"))
    return out


def compare_turn(recorded: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """What differs between the recorded turn and the replayed one."""
    return [text for _, text in compare_turn_tagged(recorded, now)]


# ----------------------------------------------------------------------------- what the report holds
DISCOVERY_TOOLS = frozenset({"find_conserved_quantities"})
EQUATION_HANDLE = re.compile(r"@e\d+")
CAUSES = {
    "earlier messages": "the report holds only part of the conversation",
    "constants": "the verified constants of earlier discovery results come from their stored cards",
    "result handles": "an `@rN` result and its LaTeX come from the stored card of the call that made it",
    "restrictions": "the restrictions a result carries come from its stored card",
    "unverified inputs": "which unverified equations a result was computed from comes from its stored card",
    "equation table": "the equation numbers and what 'equation N' means come from the stored cards and messages",
}


def _stored_messages(messages: list[dict[str, Any]]) -> list[Any]:
    """The report's messages as the chat stored them, as far as the report holds them.

    Formats before 4 have no cards, no numbered user formulas and no references: those stay empty, which is
    exactly what `missing_context` then reports instead of pretending they were restored.
    """
    out: list[Any] = []
    for index, m in enumerate(messages):
        content: dict[str, Any] = {"text": m.get("text") or ""}
        if m["role"] == "user":
            content["equations"] = list(m.get("equations") or [])
            content["references"] = list(m.get("references") or [])
        else:
            content["trace"] = [
                {"id": f"step{index}.{i}", **step} for i, step in enumerate(m.get("trace") or [])
            ]
            content["blocks"] = list(m.get("blocks") or [])
            content["stop_reason"] = m.get("stop_reason")
        out.append(SimpleNamespace(role=m["role"], content=content))
    return out


def _card_ids(message: dict[str, Any]) -> set[Any]:
    return {
        (b.get("card") or {}).get("id")
        for b in message.get("blocks") or []
        if b.get("kind") == "card" and isinstance(b.get("card"), dict)
    }


def missing_context(data: dict[str, Any], upto: int) -> dict[str, str]:
    """What the live chat would have restored for the turn after message `upto` (0-based, exclusive) that this
    report cannot give back: category -> why. Empty means every category the earlier messages feed is in the
    report; a category nothing earlier feeds is not listed (a first turn misses nothing).
    """
    out: dict[str, str] = {}
    window = data.get("conversation") or {}
    skipped = int(window.get("first_message_index") or 0)
    if skipped > 0:
        out["earlier messages"] = (
            f"this report starts at message {skipped + 1} of {window.get('messages_total', '?')}: "
            "everything before it is missing: constants, results, restrictions and equation numbers are unknown"
        )
    complete = int(data.get("format") or 1) >= 4
    wording = (
        "this report format has no stored cards"
        if not complete
        else "its card is not in the report"
    )
    for index, m in enumerate(data.get("messages", [])[:upto]):
        if m["role"] == "user":
            if not complete and "$$" in (m.get("text") or "") and "equation table" not in out:
                out["equation table"] = (
                    "a displayed formula of yours was numbered, and this format keeps no numbers"
                )
            continue
        cards = _card_ids(m)
        for step in m.get("trace") or []:
            if step.get("status") == "skipped" or step.get("repeated"):
                continue
            has_card = step.get("id") in cards
            if step.get("name") in DISCOVERY_TOOLS and not has_card:
                out.setdefault("constants", f"message {index + 1}: a discovery result, {wording}")
            if not step.get("result_id") or has_card:
                continue
            note = f"message {index + 1}: result {step['result_id']}, {wording}"
            out.setdefault("result handles", note)
            if step.get("restrictions") or (step.get("evidence") or {}).get("restrictions"):
                out.setdefault("restrictions", note)
            if EQUATION_HANDLE.search(
                json.dumps(step.get("model_arguments") or step.get("arguments") or {})
            ):
                out.setdefault("unverified inputs", note)
            out.setdefault("equation table", f"message {index + 1}: a numbered result, {wording}")
        if (
            not complete
            and "equation table" not in out
            and any("$$" in piece for piece in m.get("prose") or [])
        ):
            out["equation table"] = (
                f"message {index + 1}: displayed formulas were numbered, and this format keeps no numbers"
            )
    return out


def _depends_on_context(recorded: dict[str, Any], now: dict[str, Any]) -> bool:
    """Could a missing category change what a *tool call* of this turn did? Only a call that refers to an
    earlier result or equation (`@rN`, `@eN`), uses a constant by name or is the total derivative can."""
    for message in (recorded, now):
        for step in message.get("trace") or []:
            if step.get("name") == "time_derivative":
                return True
            text = json.dumps(step.get("model_arguments") or step.get("arguments") or {})
            if re.search(r"@[re]\d+", text):
                return True
    return False


def _user_numbering(
    stored: list[Any], text: str, counter: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """What the chat shows under a user message: its numbered displayed formulas and what its references meant."""
    table = build_table(stored, counter)
    shown, _ = resolve_references(find_references(text), table)
    return number_user_text(text, table), shown


def _counter_for(
    stored: list[Any], text: str, recorded_user: dict[str, Any], reported: int | None
) -> tuple[int, bool]:
    """The equation-number high-water mark before this turn, and whether it reproduces what the chat recorded.

    Numbers given out once are never reused, so the mark can be above every number still stored. The report holds
    the mark after the whole conversation; the one before this turn lies between the highest stored number and
    that. It is the smallest value under which the numbers and references recorded on the user's message come
    out again. A report without those (formats before 4) cannot tell: the floor is used and `False` returned.
    """
    floor = build_table(stored, 0).counter
    if "equations" not in recorded_user and "references" not in recorded_user:
        return floor, False
    want = (recorded_user.get("equations") or [], recorded_user.get("references") or [])
    ceiling = max(floor, reported or 0)
    fits = [c for c in range(floor, ceiling + 1) if _user_numbering(stored, text, c) == want]
    if not fits:
        return floor, False
    # An unknown number is worded "no longer exists" when the mark had reached it and "does not exist" when not.
    # If the candidates disagree on that, the report cannot say which the chat showed: not reproduced.
    unknown = [int(r.get("number", 0)) for r in want[1] if r.get("source") == "unknown"]
    if any(len({n <= c for c in fits}) > 1 for n in unknown):
        return floor, False
    return fits[0], True


def replay_turn(data: dict[str, Any], ctx: ToolContext, turn: int | None = None) -> TurnReplay:
    """Replay one assistant turn through the real loop with a ReplayLLM built from the report.

    The earlier messages of the report are handed to `turn_context`, the function the chat route calls, so the
    turn is given what the live chat gave it. What a report cannot hold is named in `TurnReplay.missing`.
    """
    messages = data.get("messages", [])
    if turn is None:
        candidates = [
            i
            for i, m in enumerate(messages, 1)
            if m["role"] == "assistant" and (m.get("debug") or {}).get("events")
        ]
        if not candidates:
            raise ReportError("no assistant turn with trace events in this report")
        turn = candidates[-1]
    if not 1 <= turn <= len(messages) or messages[turn - 1]["role"] != "assistant":
        raise ReportError(f"message {turn} is not an assistant turn")
    recorded = messages[turn - 1]
    replies, failures, notes = rebuild_replies(recorded)
    asked = messages[turn - 2] if turn >= 2 else None
    if asked is None or asked["role"] != "user" or not asked.get("text"):
        raise ReportError("the user message before this turn is missing from the report")
    stored = _stored_messages(messages)[: turn - 2]
    missing = missing_context(data, turn - 2)
    debug = recorded.get("debug") or {}
    window = data.get("conversation") or {}
    counter, reproduced = _counter_for(stored, asked["text"], asked, window.get("equation_counter"))
    if "equations" in asked and not reproduced:
        missing["equation table"] = (
            "the numbers the chat gave this message's formulas and references cannot be reproduced "
            "from the report's equation counter"
        )
    parameters = debug.get("params") or {}
    context = turn_context(
        stored,
        asked["text"],
        counter=counter,
        max_messages=int(parameters.get("history_messages") or HISTORY_MESSAGES),
        max_tokens=parameters.get("history_tokens") or None,
    )
    llm = ReplayLLM(replies)
    llm.fail_on = failures
    limits = AgentLimits(
        max_steps=int(debug.get("max_steps") or 6),
        max_seconds=float(debug.get("max_seconds") or 180.0),
        max_tool_calls=int(debug.get("max_tool_calls") or 12),
        retry_delay_seconds=0.0,
    )
    # This turn's own copy: the caller's context keeps no constants or messages of it. `replace` is shallow: its
    # mutable fields (`matrices`, `notation`, ...) are shared; only whole fields of the copy are assigned.
    turn_ctx = replace(ctx)
    context.apply_to(turn_ctx)
    if missing:
        notes.append(
            "the follow-up cannot be faithfully replayed: the report lacks "
            + ", ".join(sorted(missing))
            + " (see `missing`); a difference below may come from that, not from the engine"
        )

    async def go() -> dict[str, Any] | None:
        final = None
        async for event in run_agent(
            llm, context.history, turn_ctx, limits, **context.agent_kwargs()
        ):
            if event["event"] == "message":
                final = event["data"]
        return final

    now = asyncio.run(go())
    result = TurnReplay(turn=turn, notes=notes, message=now, missing=missing, counter=counter)
    if now is None:
        result.differences.append("the replay produced no message")
        return result
    # A difference that missing context could explain is kept apart from one that it cannot: a call that does
    # not refer to an earlier result or equation, run on the same arguments, gives the engine's answer either way.
    dependent = bool(missing) and _depends_on_context(recorded, now)
    for kind, text in compare_turn_tagged(recorded, now):
        explained = bool(missing) and (
            kind in ("marks", "reply") or (kind in ("tool", "stop") and dependent)
        )
        (result.context_differences if explained else result.differences).append(text)
    return result


def render_turn(r: TurnReplay) -> str:
    if r.differences:
        verdict = "DIFFERENT"
    elif r.missing:
        verdict = "NOT FAITHFUL" if not r.context_differences else "NOT COMPARABLE"
    else:
        verdict = "same"
    lines = [f"[turn {r.turn}] whole-turn replay: {verdict}"]
    lines += [f"  missing {name}: {why}" for name, why in sorted(r.missing.items())]
    lines += [f"  note: {n}" for n in r.notes]
    lines += [f"  difference: {d}" for d in r.differences]
    lines += [f"  difference that missing context may explain: {d}" for d in r.context_differences]
    if r.missing and not r.differences:
        lines.append(
            "  this is not an equivalent replay: nothing was found that the engine did differently, "
            "but the follow-up was not given the context the live chat gave it"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="path to the saved report, or - for standard input")
    parser.add_argument("--full", action="store_true", help="do not shorten long results")
    parser.add_argument("--llm", action="store_true", help="replay a whole turn with a ReplayLLM")
    parser.add_argument("--turn", type=int, help="with --llm: the message number to replay")
    parser.add_argument(
        "--to-fixture",
        metavar="PATH",
        help="write the executed calls of a turn as a regression fixture (review before committing)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="seconds each tool may take (default 30)"
    )
    args = parser.parse_args(argv)
    text = sys.stdin.read() if args.report == "-" else Path(args.report).read_text("utf-8")
    try:
        data = extract_data(text)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.to_fixture:
        try:
            fixture = to_fixture(data, args.turn)
        except ReportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        Path(args.to_fixture).write_text(json.dumps(fixture, indent=2) + "\n", "utf-8")
        print(
            f"wrote {args.to_fixture} ({len(fixture['calls'])} call(s)). It still holds the user's "
            "mathematics: replace the arguments with a minimal example of your own, set an id and "
            "`contains_user_mathematics` to false, then add it to services/api/tests/fixtures/failures/."
        )
        return 0
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=args.timeout, max_concurrency=2)),
        kb=KnowledgeBase.load(),
    )
    if args.llm:
        try:
            outcome = replay_turn(data, ctx, args.turn)
        except ReportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(render_turn(outcome))
        if outcome.differences:
            return 1
        return 0 if outcome.same else 3
    items = replay(data, ctx)
    print(render(items, full=args.full))
    return 1 if any(not r.same for r in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
