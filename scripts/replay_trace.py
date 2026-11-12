#!/usr/bin/env python3
"""Re-run the tool calls recorded in a chat debug report and show what changed.

In the chat, switch on "Debug mode", press "Copy debug report" (or "Copy this turn"), and save the
text to a file. Then:

    uv run python scripts/replay_trace.py report.md        # or: ... - < report.md

Each recorded tool call (name and the arguments the model sent) is executed again, through the same
sandboxed `tools.execute` the API uses, and its text is compared with what the model read back at the
time. No model and no network are involved; tools are pure. Exit status 1 means at least one call now
gives a different result, which is what you want to see after a fix and what a regression test pins.

Reports contain the user's mathematics: keep them out of the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

JSON_START = "<!-- minerva-debug-json -->"
JSON_END = "<!-- /minerva-debug-json -->"
SUPPORTED_FORMAT = 1


class ReportError(ValueError):
    """The text is not a debug report this script understands."""


@dataclass
class Replayed:
    turn: int  # 1-based index of the message in the report
    name: str
    arguments: dict[str, Any]
    recorded_status: str | None
    recorded: str
    status: str
    now: str

    @property
    def same(self) -> bool:
        return self.recorded == self.now


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
    if not isinstance(data, dict) or data.get("format") != SUPPORTED_FORMAT:
        raise ReportError(f"unsupported report format {data.get('format')!r}")
    return data


def replay(data: dict[str, Any], ctx: ToolContext) -> list[Replayed]:
    out: list[Replayed] = []
    trace_sets = [(i, m.get("trace") or []) for i, m in enumerate(data.get("messages", []), 1)]
    last = data.get("last_error") or {}
    trace_sets.append((len(trace_sets) + 1, last.get("trace") or []))
    for turn, trace in trace_sets:
        for step in trace:
            result = execute(step["name"], step["arguments"], ctx)
            out.append(
                Replayed(
                    turn=turn,
                    name=step["name"],
                    arguments=step["arguments"],
                    recorded_status=step.get("status"),
                    recorded=step["for_model"],
                    status=result.status,
                    now=result.for_model,
                )
            )
    return out


def render(items: list[Replayed], *, full: bool = False) -> str:
    def cut(text: str) -> str:
        return text if full or len(text) <= 700 else text[:700] + f"... [{len(text) - 700} more]"

    lines: list[str] = []
    for r in items:
        verdict = "same" if r.same else "DIFFERENT"
        lines.append(
            f"[turn {r.turn}] {r.name}: {verdict} (recorded {r.recorded_status}, now {r.status})"
        )
        lines.append(f"  arguments: {json.dumps(r.arguments, ensure_ascii=False)}")
        lines.append(f"  recorded:  {cut(r.recorded)}")
        if not r.same:
            lines.append(f"  now:       {cut(r.now)}")
    changed = sum(not r.same for r in items)
    lines.append(f"{len(items)} call(s) replayed, {changed} different.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report", help="path to the saved report, or - for standard input")
    parser.add_argument("--full", action="store_true", help="do not shorten long results")
    args = parser.parse_args(argv)
    text = sys.stdin.read() if args.report == "-" else Path(args.report).read_text("utf-8")
    try:
        data = extract_data(text)
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    items = replay(data, ctx)
    print(render(items, full=args.full))
    return 1 if any(not r.same for r in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
