"""Structured JSON logging.

Rules: log event names and counters, never request bodies, expressions, emails, passwords,
tokens, cookies or full headers. Only add fields from the allow-list below.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

ALLOWED_FIELDS = frozenset(
    {
        "event",
        "level",
        "timestamp",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
        "operation",
        "status",
        "user_id",
        "steps",
        "tool_calls",
        "tool_errors",
        "logger",
        # Agent trace events (agent/tracing.py): identifiers, counts, durations and fixed-vocabulary
        # values only. A test requires this list to cover every event field and nothing that could hold
        # content; adding a field here is a reviewed decision.
        "trace_id",
        "seq",
        "t_ms",
        "step",
        "model",
        "prompt_sha",
        "mode",
        "api_version",
        "max_steps",
        "max_seconds",
        "max_tool_calls",
        "history_messages",
        "message_count",
        "tool_count",
        "latency_ms",
        "first_token_ms",
        "finish_reason",
        "proposed_tool",
        "text_chars",
        "tokens_in",
        "tokens_out",
        "tool",
        "call_id",
        "arguments_valid",
        "repeated",
        "references",
        "engine_ms",
        "sandbox_calls",
        "error_category",
        "result_id",
        "scope",
        "check",
        "accepted",
        "reason",
        "points_checked",
        "points_agreed",
        "symbolic_zero",
        "restrictions",
        "restrictions_complete",
        "verify_ms",
        "claims",
        "claims_holding",
        "claims_conditional",
        "claims_contradicted",
        "claims_unchecked",
        "attempt",
        "total_ms",
        "invalid_calls",
        "repeated_calls",
        "llm_ms",
        "tool_ms",
        "formulas",
        "formulas_with_evidence",
        "evidence_pct",
        "stop_reason",
        "events_dropped",
    }
)


def _allow_list(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    return {k: v for k, v in event_dict.items() if k in ALLOWED_FIELDS}


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _allow_list,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=False,
    )


def get_logger() -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger("minerva")  # type: ignore[no-any-return]
