"""Score one run of one case: measurements, notes, and exactly one failure class when it fails.

The failure classes answer the question the benchmark exists for: when the agent fails, was it tool
selection, arguments, parsing, missing assumptions, verification, the claim checker, model behaviour,
or a resource limit? They mirror the symptom table of the `chat-trace-triage` skill.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from ..segments import normalize
from .cases import ASK, NO_TOOL, EvalCase
from .reference import card_meets, card_status

VERIFIED_MARKS = ("from_tool", "checked")
MAX_REPLY_WORDS = 220
REPLY_KEPT = 240


class FailureClass(StrEnum):
    TOOL_SELECTION = "tool_selection"  # wrong tool, no tool, or a tool that was forbidden
    ARGUMENTS = "arguments"  # invalid arguments, or arguments that do not say what the user asked
    PARSING = "parsing"  # the engine could not read what the model passed
    MISSING_ASSUMPTIONS = "missing_assumptions"  # assumptions dropped, or invented
    VERIFICATION = "verification"  # an unsupported or false statement shown as verified
    MODEL_HONESTY = (
        "model_honesty"  # the reply misstates what the tools said, or is not what was asked
    )
    CLAIM_CHECKER = "claim_checker_false_positive"  # a correct statement was marked contradicted
    RESOURCE_LIMIT = "resource_limit"  # step, call, time or sandbox limit
    MODEL_HOST = "model_host"  # the model host failed or was unreachable
    LOOP = "unproductive_loop"  # the agent kept calling tools that could not help, and was stopped
    NO_REPLY = "no_reply"  # the turn ended without an answer for another reason


class CaseResult(BaseModel):
    case_id: str
    category: str
    repeat: int = 0
    informational: bool = False
    passed: bool
    failure_class: FailureClass | None = None
    notes: list[str] = Field(default_factory=list)
    expected_tool: str
    tools: list[str] = Field(default_factory=list)  # every tool called, in order
    first_tool: str | None = None
    tool_selected: bool  # the first call (or its absence) is the expected behaviour
    tool_reached: bool  # an acceptable tool was called at some point
    calls: int = 0
    invalid_calls: int = 0  # unknown tool or invalid arguments
    repeated_calls: int = 0  # the same tool with the same arguments again
    unnecessary_calls: int = 0  # calls beyond what the case needs
    args_ok: bool | None = None  # None: no argument check applied
    result_ok: bool | None = None  # None: no result expectation applied
    completed: bool = False  # a reply arrived and no limit ended the turn
    false_verified: int = 0
    unchecked: int = 0
    contradicted: int = 0
    checker_false_positive: int = 0
    seconds: float = 0.0
    llm_ms: int = 0
    tool_ms: int = 0
    llm_calls: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    first_token_ms: int | None = None  # of the first model call
    cost_usd: float | None = None  # as the provider reported it
    error_kind: str | None = (
        None  # auth | credits | rate_limit | timeout | ... when the model host failed
    )
    tool_failures: int = 0  # tools that answered with an error
    unverified_results: int = 0  # tool results whose status was not `ok` (refused or not verified)
    stop_reason: str | None = None
    reply: str = ""


@dataclass
class Observation:
    """What a turn did, extracted from its event stream: the input of scoring."""

    seconds: float = 0.0
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    cards: dict[str, dict[str, Any]] = field(default_factory=dict)  # by call id
    call_ids: list[str] = field(default_factory=list)
    final: dict[str, Any] | None = None
    error: str = ""
    partial_trace: list[dict[str, Any]] = field(default_factory=list)
    partial_debug: dict[str, Any] | None = None

    @property
    def trace(self) -> list[dict[str, Any]]:
        if self.final is not None:
            return list(self.final.get("trace") or [])
        return self.partial_trace

    @property
    def debug(self) -> dict[str, Any]:
        if self.final is not None:
            return self.final.get("debug") or {}
        return self.partial_debug or {}


def is_invalid_call(step: dict[str, Any]) -> bool:
    """The tool refused the call itself: an unknown tool, or arguments that failed validation."""
    text = str(step.get("for_model", ""))
    return text.startswith(("error: invalid arguments", "error: unknown tool"))


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, default=str)


def _segments(final: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        s
        for block in final.get("blocks", [])
        if block.get("kind") == "text"
        for s in block.get("segments", [])
        if s.get("type") != "text"
    ]


def _matches_any(value: str, formulas: list[str]) -> bool:
    wanted = {normalize(f) for f in formulas}
    return normalize(value) in wanted


def classify_failure(case: EvalCase, notes: dict[str, bool], r: CaseResult) -> FailureClass:
    """One class per failed case, by the earliest stage that went wrong."""
    if notes["host_error"]:
        return FailureClass.MODEL_HOST
    if notes["no_reply"]:
        return FailureClass.NO_REPLY
    if notes["loop"]:
        return FailureClass.LOOP
    if notes["resource"]:
        return FailureClass.RESOURCE_LIMIT
    if r.false_verified:
        return FailureClass.VERIFICATION
    if notes["tool_selection"]:
        return FailureClass.TOOL_SELECTION
    if notes["parse_error"]:
        return FailureClass.PARSING
    if notes["arguments"]:
        if case.args is not None and case.args.about_assumptions:
            return FailureClass.MISSING_ASSUMPTIONS
        return FailureClass.ARGUMENTS
    if r.checker_false_positive:
        return FailureClass.CLAIM_CHECKER
    return FailureClass.MODEL_HONESTY


_HOST_ERRORS = ("model host", "not reachable", "unreadable reply")


def score_case(case: EvalCase, obs: Observation, repeat: int = 0) -> CaseResult:
    trace = obs.trace
    tools = [name for name, _ in obs.calls]
    acceptable = case.acceptable_tools
    first = tools[0] if tools else None
    forbidden = sorted(set(case.forbid) & set(tools))

    if case.expected_tool == NO_TOOL:
        selected = not tools
        reached = not tools
    elif case.expected_tool == ASK:
        selected = reached = not forbidden
    else:
        reached = bool(acceptable & set(tools)) or (case.tools_optional and not tools)
        selected = (first in acceptable) or (case.tools_optional and not tools)
        if forbidden:
            selected = reached = False

    seen: set[tuple[str, str]] = set()
    repeated = 0
    for name, arguments in obs.calls:
        key = (name, _canonical(arguments))
        repeated += key in seen
        seen.add(key)
    invalid = sum(1 for step in trace if is_invalid_call(step))
    # An identical call that the loop refused to run again is a repeat too, though it has no tool_call.
    refused_repeats = sum(1 for step in trace if step.get("repeated"))
    repeated += refused_repeats
    unnecessary = (
        max(0, len(obs.calls) - case.max_calls) + refused_repeats if case.expects_tool else 0
    )

    result = CaseResult(
        case_id=case.id,
        category=case.category,
        repeat=repeat,
        informational=case.informational,
        passed=False,
        expected_tool=case.expected_tool,
        tools=tools,
        first_tool=first,
        tool_selected=bool(selected),
        tool_reached=bool(reached),
        calls=len(obs.calls),
        invalid_calls=invalid,
        repeated_calls=repeated,
        unnecessary_calls=unnecessary,
        seconds=round(obs.seconds, 3),
    )
    debug = obs.debug
    result.llm_calls = len(debug.get("llm_calls") or [])
    result.llm_ms = sum(int(c.get("ms", 0)) for c in debug.get("llm_calls") or [])
    result.tool_ms = sum(int(s.get("duration_ms") or 0) for s in trace)
    ins = [
        c.get("prompt_tokens")
        for c in debug.get("llm_calls") or []
        if c.get("prompt_tokens") is not None
    ]
    outs = [
        c.get("completion_tokens")
        for c in debug.get("llm_calls") or []
        if c.get("completion_tokens") is not None
    ]
    result.tokens_in = sum(ins) if ins else None
    result.tokens_out = sum(outs) if outs else None
    firsts = [
        c["first_token_ms"]
        for c in debug.get("llm_calls") or []
        if c.get("first_token_ms") is not None
    ]
    result.first_token_ms = firsts[0] if firsts else None
    costs = [c["cost_usd"] for c in debug.get("llm_calls") or [] if c.get("cost_usd") is not None]
    result.cost_usd = sum(costs) if costs else None
    result.error_kind = debug.get("error_kind")
    result.tool_failures = sum(1 for s in trace if s.get("status") == "error")
    result.unverified_results = sum(
        1 for s in trace if s.get("status") not in ("ok", "error", "skipped", None)
    )

    flags = {
        "no_reply": False,
        "host_error": False,
        "resource": False,
        "loop": False,
        "tool_selection": False,
        "parse_error": False,
        "arguments": False,
    }
    notes: list[str] = []

    if obs.final is None:
        flags["no_reply"] = True
        flags["host_error"] = any(h in obs.error for h in _HOST_ERRORS)
        notes.append(f"no reply ({obs.error or 'unknown error'})")
        result.notes = notes
        result.stop_reason = "error"
        result.failure_class = classify_failure(case, flags, result)
        return result

    final = obs.final
    text = str(final.get("text", ""))
    segments = _segments(final)
    marks = [s.get("status") for s in segments]
    result.unchecked = marks.count("unchecked")
    result.false_verified = sum(
        1
        for s in segments
        if s.get("status") in VERIFIED_MARKS and _matches_any(str(s["value"]), case.wrong_formulas)
    )
    result.checker_false_positive = sum(
        1
        for s in segments
        if s.get("status") == "contradicted" and _matches_any(str(s["value"]), case.right_formulas)
    )
    result.contradicted = marks.count("contradicted") - result.checker_false_positive
    result.reply = " ".join(text.split())[:REPLY_KEPT]
    truncated = bool(final.get("truncated"))
    stop = final.get("stop_reason") or ("truncated" if truncated else "completed")
    result.stop_reason = stop
    result.completed = not truncated
    expects_limit = case.expect_card is not None and case.expect_card.status == "resource_limit"
    tool_limit = not expects_limit and any(
        s.get("status") in ("resource_limit", "internal_error") for s in trace
    )
    flags["host_error"] = stop == "model_error"
    flags["no_reply"] = stop == "internal_error"
    flags["loop"] = stop == "repeated_calls"
    flags["resource"] = (
        truncated
        and stop not in ("model_error", "internal_error", "repeated_calls", "malformed_output")
    ) or tool_limit
    if stop == "malformed_output":
        flags["arguments"] = True
    if truncated:
        notes.append(f"the turn ended early: {stop}")
    if tool_limit:
        notes.append("a tool hit a resource limit")

    if case.expected_tool == NO_TOOL and tools:
        flags["tool_selection"] = True
        notes.append(f"called tools unexpectedly: {sorted(set(tools))}")
    if forbidden:
        flags["tool_selection"] = True
        notes.append(f"called forbidden tool(s) {forbidden}")
    if case.expects_tool and not reached:
        flags["tool_selection"] = True
        notes.append(f"expected one of {sorted(acceptable)}, called {sorted(set(tools)) or 'none'}")
    if case.sequence and not _is_subsequence(case.sequence, tools):
        flags["tool_selection"] = True
        notes.append(f"expected the sequence {case.sequence}, called {tools}")

    expected_cards = [
        (cid, obs.cards.get(cid))
        for cid, (name, _) in zip(obs.call_ids, obs.calls, strict=False)
        if name == case.expected_tool
    ]
    real_cards = [(cid, c) for cid, c in expected_cards if c and c.get("type") != "error"]
    if case.expects_tool and reached and tools:
        if invalid and not real_cards:
            flags["arguments"] = True
            notes.append("every call to the tool had invalid arguments")
        elif expected_cards and not real_cards:
            statuses = {str(s.get("status")) for s in trace if s.get("name") == case.expected_tool}
            flags["parse_error"] = "parse_error" in statuses
            flags["arguments"] = not flags["parse_error"]
            notes.append("every call to the tool ended in an error")
        if case.args is not None and expected_cards:
            ok = any(case.args.holds(a) for n, a in obs.calls if n == case.expected_tool)
            result.args_ok = ok
            if not ok:
                flags["arguments"] = True
                notes.append("arguments do not say what the user asked")
        if case.expect_card is not None and real_cards:
            tool_status = {str(s.get("id")): s.get("status") for s in trace}
            met = [
                not card_meets(case.expect_card, c, tool_status.get(cid))
                for cid, c in real_cards
                if c is not None
            ]
            result.result_ok = any(met)
            if not result.result_ok:
                card_statuses = {
                    card_status(c.get("data", {})) for _, c in real_cards if c is not None
                }
                flags["parse_error"] = "parse_error" in card_statuses
                flags["arguments"] = not flags["parse_error"]
                notes.append(
                    "the tool's result is not the expected one (different input or status)"
                )

    if result.false_verified:
        notes.append(f"{result.false_verified} false statement(s) shown as verified")
    if result.checker_false_positive:
        notes.append(f"{result.checker_false_positive} correct statement(s) marked contradicted")
    if result.contradicted:
        notes.append(f"{result.contradicted} equation(s) that do not hold as written")
    if case.reply_matches and not re.search(case.reply_matches, text, re.IGNORECASE):
        notes.append("reply does not say what it should")
    if case.reply_forbids and re.search(case.reply_forbids, text, re.IGNORECASE):
        notes.append("reply says what it should not")
    if case.max_unchecked is not None and result.unchecked > case.max_unchecked:
        notes.append(f"{result.unchecked} unchecked formula(s) in the reply")
    if re.search(r"</?think>|^\s*(okay|alright),? (let'?s|so)", text, re.IGNORECASE):
        notes.append("reasoning leaked into the reply")
    if len(text.split()) > MAX_REPLY_WORDS:
        notes.append(f"reply too long ({len(text.split())} words)")

    result.notes = notes
    result.passed = not notes
    if not result.passed:
        result.failure_class = classify_failure(case, flags, result)
    return result


def _is_subsequence(wanted: list[str], called: list[str]) -> bool:
    it = iter(called)
    return all(any(c == w for c in it) for w in wanted)
