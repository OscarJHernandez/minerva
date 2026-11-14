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
        "reason",
        "steps",
        "tool_calls",
        "tool_errors",
        "provider",
        "model",
        "first_token_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "error_kind",
        "dimension",  # geometry: counters and enumerated names only, never content
        "stage",
        "ops",
        "failure",
        "quantities",
        "logger",
        # Transformation diagnostics: durations, counts and enum-like codes only, never expressions.
        "parse_ms",
        "candidates_ms",
        "transform_ms",
        "verify_ms",
        "candidates",
        "rejections",
        "verification_method",
        "verification_outcome",
        "restrictions_outcome",
        # conserved-quantity search events (search_log.py): counters and fixed labels only
        "tier",
        "at_ms",
        "family",
        "unknowns",
        "equations",
        "rows",
        "cols",
        "rank",
        "nullity",
        "degree",
        "points",
        "passed",
        "numeric_passed",
        "symbolic_zero",
        "stop",
        "cause",
        "tiers",
        "constants",
        "new",
        "rejected",
        "independent",
        "rank_lower",
        "rank_upper",
        "proved",
        "relations",
        "peak_mb",
        "ms",
        "system",
        "budget",
        "time_dependent",
        "parameters",
        # Agent trace events (agent/tracing.py): identifiers, counts, durations and fixed-vocabulary
        # values only. A test requires this list to cover every event field and nothing that could hold
        # content; adding a field here is a reviewed decision.
        "trace_id",
        "seq",
        "t_ms",
        "step",
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
        "points_checked",
        "points_agreed",
        "restrictions",
        "restrictions_complete",
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
        # matrix calculus: enums only, never names, entries or expressions
        "matrix_method",
        "result_kind",
        "commutation",
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
        # stdout is looked up when a logger is made, not when logging is configured: a stream that was
        # replaced or closed since (a test capture, a redirect) must not make a log call raise inside a tool.
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger() -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger("minerva")  # type: ignore[no-any-return]
