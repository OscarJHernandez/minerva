"""Which subsystem a failed turn points at, decided by fixed rules over what the turn recorded.

The answer is for the person reading a debug report and for the evaluation's failure classes: it names the
first step that went wrong and the subsystem that owns that kind of failure. It never guesses: when two
steps point at different subsystems, or when the record is incomplete, the stage is `undetermined` and the
candidates are listed. Closed vocabulary only (no expression, argument or message ever enters it).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .events import TraceStep, TurnDebug

# error category (tracing.ErrorCategory values) -> the subsystem that owns that kind of failure
STAGE_OF_CATEGORY: dict[str, str] = {
    "invalid_arguments": "model",  # what the model sent could not be used
    "unknown_tool": "model",
    "malformed_model_output": "model",
    "repeated_call": "model",
    "parse_error": "parser",  # the input was refused or misread by the strict parsers
    "conditions_not_satisfied": "engine",  # an honest answer: the engine established a refusal
    "unsupported": "engine",
    "tool_failure": "engine",  # an unexpected exception inside a tool
    "not_verified": "verifier",  # a candidate failed its independent check
    "limit_reached": "orchestrator",  # a turn or tool budget stopped the call
    "needs_user_choice": "orchestrator",
    "timeout": "infrastructure",
    "resource_limit": "infrastructure",
    "sandbox_killed": "infrastructure",
    "busy": "infrastructure",
    "model_host": "infrastructure",
}
STAGE_OF_STOP: dict[str, str] = {
    "model_error": "infrastructure",
    "malformed_output": "model",
    "output_limit": "model",
    "internal_error": "orchestrator",
    "time_limit": "infrastructure",
}
STAGES = frozenset(
    {
        "none",
        "model",
        "parser",
        "orchestrator",
        "engine",
        "verifier",
        "infrastructure",
        "undetermined",
    }
)


def diagnose(trace: Sequence[TraceStep], debug: TurnDebug) -> dict[str, Any]:
    """`{stage, step, first_failure_code, candidates, replayable, replayable_reason}` for one finished turn."""
    failures = [(i, t.error_category) for i, t in enumerate(trace) if t.error_category]
    stop = debug.stop_reason or "completed"
    candidates: list[str] = []
    for _, category in failures:
        stage = STAGE_OF_CATEGORY.get(category or "")
        if stage and stage not in candidates:
            candidates.append(stage)
    if stop in STAGE_OF_STOP and not failures:
        candidates.append(STAGE_OF_STOP[stop])
    first_index, first_code = failures[0] if failures else (None, None)
    if not candidates:
        stage = "none" if stop == "completed" else "undetermined"
    elif debug.events_dropped or len(candidates) > 1:
        stage = "undetermined"  # the record is incomplete or several subsystems are involved
    else:
        stage = candidates[0]
    if any(t.error_category == "timeout" for t in trace):
        reason = "timing_dependent"
    elif debug.events_dropped:
        reason = "events_dropped"
    elif not trace:
        reason = "no_tool_calls"
    else:
        reason = "yes"
    return {
        "stage": stage,
        "step": first_index,
        "first_failure_code": first_code,
        "candidates": candidates,
        "replayable": reason == "yes",
        "replayable_reason": reason,
    }
