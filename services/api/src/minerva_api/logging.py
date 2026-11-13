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
