"""Write the trace of an evaluation turn as a debug report that `scripts/replay_trace.py` reads.

A live run keeps every turn's full trace and debug record in memory and then reduces it to scores. When a
case fails, the scores say *that* it failed; this writes *what happened* (the model's calls, the tool
results, the routing decision) to a local file, so the failure can be replayed against the engine, told
apart as "the model sent bad arguments" or "the tool misbehaved", and pinned as a regression test.

A trace holds the benchmark prompts, the model's expressions and its prose. It is written only to the
directory the caller names (git-ignored: `/eval-traces/`) and never into the JSON or Markdown report, a log
line or a test fixture. The shape is the `messages` part of `apps/web/src/lib/debug-report.ts` (format 3);
`test_evaluation_traces.py` round-trips it through the replay script so the two cannot drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .score import Observation

REPORT_FORMAT = 3
JSON_START = "<!-- minerva-debug-json -->"
JSON_END = "<!-- /minerva-debug-json -->"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _prose(final: dict[str, Any]) -> list[str]:
    """The model's text as it wrote it, one string per text block, formula delimiters restored."""
    blocks = []
    for block in final.get("blocks") or []:
        if block.get("kind") != "text":
            continue
        parts = []
        for seg in block.get("segments") or []:
            value = str(seg.get("value", ""))
            kind = seg.get("type")
            parts.append(
                f"$${value}$$"
                if kind == "display_math"
                else f"${value}$"
                if kind == "math"
                else value
            )
        blocks.append("".join(parts))
    return blocks


def _formulas(final: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "value": s.get("value"),
            "status": s.get("status"),
            "note": s.get("note"),
            "number": s.get("number"),
        }
        for block in final.get("blocks") or []
        if block.get("kind") == "text"
        for s in block.get("segments") or []
        if s.get("type") != "text" and s.get("status")
    ]


def report_data(prompt: str, obs: Observation, *, label: str = "") -> dict[str, Any]:
    """The machine-readable part of a debug report for one evaluated turn."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "created_at": None,
            "text": prompt,
            "trace": [],
            "debug": None,
            "truncated": False,
            "stop_reason": None,
            "prose": [],
            "formulas": [],
        }
    ]
    last_error: dict[str, Any] | None = None
    if obs.final is not None:
        final = obs.final
        messages.append(
            {
                "role": "assistant",
                "created_at": None,
                "text": final.get("text", ""),
                "trace": final.get("trace") or [],
                "debug": final.get("debug"),
                "truncated": bool(final.get("truncated")),
                "stop_reason": final.get("stop_reason")
                or (final.get("debug") or {}).get("stop_reason"),
                "prose": _prose(final),
                "formulas": _formulas(final),
            }
        )
    else:
        last_error = {"message": obs.error, "trace": obs.partial_trace, "debug": obs.partial_debug}
    return {
        "format": REPORT_FORMAT,
        "contains_conversation_content": True,
        "generated_at": None,
        "title": label or None,
        "app": {"mode": "evaluation", "model": None, "build": None},
        "messages": messages,
        "last_error": last_error,
    }


def render_report(prompt: str, obs: Observation, *, label: str = "") -> str:
    data = report_data(prompt, obs, label=label)
    return "\n".join(
        [
            f"# Minerva evaluation trace: {label or 'turn'}",
            "",
            "This report contains conversation content (the benchmark prompt, the model's expressions and tool "
            "results). Keep it out of the repository.",
            "",
            JSON_START,
            "```json",
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            "```",
            JSON_END,
            "",
        ]
    )


def trace_path(directory: Path, case_id: str, repeat: int, turn: int) -> Path:
    return directory / f"{_SAFE.sub('-', case_id)}-r{repeat}-t{turn}.md"


class TraceWriter:
    """Writes one report per turn that fails (or per turn, with `everything`) into `directory`."""

    def __init__(self, directory: str | Path, *, everything: bool = False) -> None:
        self.directory = Path(directory)
        self.everything = everything
        self.written: list[Path] = []

    def __call__(
        self, case_id: str, repeat: int, turn: int, prompt: str, obs: Observation, passed: bool
    ) -> None:
        if passed and not self.everything:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = trace_path(self.directory, case_id, repeat, turn)
        label = f"{case_id} (repeat {repeat}, turn {turn}, {'passed' if passed else 'FAILED'})"
        path.write_text(render_report(prompt, obs, label=label), encoding="utf-8")
        self.written.append(path)
