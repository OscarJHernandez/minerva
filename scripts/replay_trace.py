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
reasons. It reproduces what the agent did with those replies; it cannot reproduce the model.

Exit status 1 means at least one call or turn now differs, which is what you want to see after a fix and
what a regression test pins. 2 means the report could not be read.

Reports contain the user's mathematics: keep them out of the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minerva_api.agent.llm import Chunk, LLMError, ToolCall
from minerva_api.agent.loop import STOP_SENTENCES, AgentLimits, run_agent
from minerva_api.agent.replay import ReplayLLM
from minerva_api.agent.tools import ToolContext, execute
from minerva_api.agent.tracing import StopReason
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

JSON_START = "<!-- minerva-debug-json -->"
JSON_END = "<!-- /minerva-debug-json -->"
SUPPORTED_FORMATS = (1, 2)
SUPPORTED_FORMAT = SUPPORTED_FORMATS[-1]


class ReportError(ValueError):
    """The text is not a debug report this script understands."""


def _context_for(step: dict[str, Any], ctx: ToolContext) -> ToolContext:
    """The context a recorded call ran in: a distribution card run with the user's answer carries it."""
    notation = step["arguments"].get("notation") if step["name"] == "distribution" else None
    if isinstance(notation, dict):
        return ToolContext(engine=ctx.engine, kb=ctx.kb, notation=dict(notation))
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
    differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    message: dict[str, Any] | None = None

    @property
    def same(self) -> bool:
        return not self.differences


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


def compare_turn(recorded: dict[str, Any], now: dict[str, Any]) -> list[str]:
    """What differs between the recorded turn and the replayed one."""
    out: list[str] = []
    a_stop = recorded.get("stop_reason") or (recorded.get("debug") or {}).get("stop_reason")
    if a_stop != now.get("stop_reason"):
        out.append(f"stop reason: recorded {a_stop}, now {now.get('stop_reason')}")
    a_steps = [(t["name"], t.get("status")) for t in recorded.get("trace") or []]
    b_steps = [(t["name"], t.get("status")) for t in now.get("trace") or []]
    if a_steps != b_steps:
        out.append(f"tool calls and statuses: recorded {a_steps}, now {b_steps}")
    a_marks = [(f["value"], f["status"]) for f in recorded.get("formulas") or []]
    b_marks = [
        (s["value"], s["status"])
        for b in now.get("blocks") or []
        if b["kind"] == "text"
        for s in b["segments"]
        if s["type"] != "text"
    ]
    if recorded.get("formulas") is not None and a_marks != b_marks:
        out.append(f"formula marks: recorded {a_marks}, now {b_marks}")
    if recorded.get("text") != now.get("text"):
        out.append("the reply text differs")
    return out


def replay_turn(data: dict[str, Any], ctx: ToolContext, turn: int | None = None) -> TurnReplay:
    """Replay one assistant turn through the real loop with a ReplayLLM built from the report."""
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
    history = [
        {"role": m["role"], "content": m.get("text", "")}
        for m in messages[: turn - 1]
        if m.get("text")
    ]
    if not history or history[-1]["role"] != "user":
        raise ReportError("the user message before this turn is missing from the report")
    llm = ReplayLLM(replies)
    llm.fail_on = failures
    debug = recorded.get("debug") or {}
    limits = AgentLimits(
        max_steps=int(debug.get("max_steps") or 6),
        max_seconds=float(debug.get("max_seconds") or 180.0),
        max_tool_calls=int(debug.get("max_tool_calls") or 12),
        retry_delay_seconds=0.0,
    )
    if turn > 1:
        notes.append(
            "earlier turns are replayed as plain text: their verified marks and results are not restored"
        )

    async def go() -> dict[str, Any] | None:
        final = None
        async for event in run_agent(llm, history, ctx, limits):
            if event["event"] == "message":
                final = event["data"]
        return final

    now = asyncio.run(go())
    result = TurnReplay(turn=turn, notes=notes, message=now)
    if now is None:
        result.differences.append("the replay produced no message")
        return result
    # A recorded turn that was the first of its conversation can be compared in full; with earlier
    # turns, marks of formulas from those turns may legitimately differ, so only report what is certain.
    result.differences = compare_turn(recorded, now)
    return result


def render_turn(r: TurnReplay) -> str:
    lines = [f"[turn {r.turn}] whole-turn replay: {'same' if r.same else 'DIFFERENT'}"]
    lines += [f"  note: {n}" for n in r.notes]
    lines += [f"  difference: {d}" for d in r.differences]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="path to the saved report, or - for standard input")
    parser.add_argument("--full", action="store_true", help="do not shorten long results")
    parser.add_argument("--llm", action="store_true", help="replay a whole turn with a ReplayLLM")
    parser.add_argument("--turn", type=int, help="with --llm: the message number to replay")
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
        return 0 if outcome.same else 1
    items = replay(data, ctx)
    print(render(items, full=args.full))
    return 1 if any(not r.same for r in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
