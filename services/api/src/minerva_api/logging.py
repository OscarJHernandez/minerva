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
        "provider",
        "model",
        "first_token_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
        "error_kind",
        "logger",
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
