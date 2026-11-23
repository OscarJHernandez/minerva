"""Structured JSON logging.

Rules: log event names and counters, never request bodies, expressions, emails, passwords,
tokens, cookies or full headers. Only add fields from the allow-list below.
"""

from __future__ import annotations

import logging
import sys
import traceback
from collections.abc import Mapping, MutableMapping
from typing import Any

import structlog

ALLOWED_FIELDS = frozenset(
    {
        "thinking_chars",
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
        # vector calculus (vector_log.py): enums and counters only
        "coordinate_system",
        "vector_action",
        "operator_depth",
        "points",
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
        # differential equations (ode_log.py): counters, durations and fixed vocabulary only
        "ode_method",
        "methods_tried",
        "order",
        "classes",
        "verified",
        "dropped",
        "solution_kind",
        "evaluations",
        "max_residual",
        # Agent trace events (agent/tracing.py): identifiers, counts, durations and fixed-vocabulary
        # values only. A test requires this list to cover every event field and nothing that could hold
        # content; adding a field here is a reviewed decision.
        "trace_id",
        "seq",
        "t_ms",
        "step",
        "prompt_sha",
        "request_sha",
        "mode",
        "api_version",
        "max_steps",
        "max_seconds",
        "max_tool_calls",
        "history_messages",
        "message_count",
        "tool_count",
        "prompt_estimate_tokens",
        "window",
        "dropped_messages",
        "groups_loaded",
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
        "decision",
        "total_ms",
        "invalid_calls",
        "repeated_calls",
        "llm_ms",
        "tool_ms",
        "formulas",
        "formulas_with_evidence",
        "equation_labels_removed",
        "equations_reused",
        "equation_aliases_resolved",
        "evidence_pct",
        "stop_reason",
        "events_dropped",
        # variation (functionals): the action name from a fixed set and counters only
        "action",
        "functions",
        "highest_order",
        "stages",
        # matrix calculus: enums only, never names, entries or expressions
        "matrix_method",
        "result_kind",
        "commutation",
        # statistical mechanics: counts and enumerated kinds only, never energies or expressions
        "levels",
        "system_kind",
        # approximation: the method name from a fixed set and counters only
        "approx_method",
        "checks",
        "checks_passed",
        "warnings",
        # Tool outcome lines and where an error was raised (agent/tracing.py, `error_fields`): names from
        # fixed sets (the tool registry, an argument model's own field names, enums), counts and code
        # locations. Never an argument value, a user's symbol name or an exception message.
        "outcome_kind",
        "reason_code",
        "arg_fields",
        "invalid_fields",
        "tools_called",
        "failed_tools",
        "declined_tools",
        "first_failure_code",
        "error_type",
        "error_module",
        "error_line",
    }
)


# Exception classes whose names are fixed by code we ship or depend on; any other class name could be
# chosen by data (a dynamically created class), so it is reported as `other`.
_KNOWN_ERROR_ROOTS = frozenset(
    {
        "builtins",
        "sympy",
        "mpmath",
        "sqlalchemy",
        "pydantic",
        "pydantic_core",
        "httpx",
        "asyncio",
        "json",
        "concurrent",
        "multiprocessing",
    }
)


def error_fields(exc: BaseException) -> dict[str, Any]:
    """Where an exception came from, for the log: class name, module and line. Never its message.

    The message of an exception can hold an expression, a symbol or an input value; the class and the
    innermost frame of our own code (`minerva_*`) are fixed by the source and say where to look.
    """
    cls = type(exc)
    root = cls.__module__.split(".")[0]
    known = root in _KNOWN_ERROR_ROOTS or root.startswith("minerva_")
    out: dict[str, Any] = {"error_type": cls.__name__ if known else "other"}
    innermost: tuple[str, int] | None = None
    for frame, line in traceback.walk_tb(exc.__traceback__):
        module = str(frame.f_globals.get("__name__", ""))
        if module.startswith("minerva_"):
            innermost = (module, line)
    if innermost is not None:
        out["error_module"], out["error_line"] = innermost
    return out


def _allow_list(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    return {k: v for k, v in event_dict.items() if k in ALLOWED_FIELDS}


# Loggers of the libraries we run under. Their records are never written as they come: the message of
# a record (uvicorn's "Exception in ASGI application", a SQLAlchemy statement with its parameters, an
# HTTP client's URL) can hold a user's expression, e-mail address or input, and structlog would
# otherwise promote it to the `event` field, which the allow-list keeps.
_FOREIGN_LOGGERS = (
    "uvicorn",
    "sqlalchemy",
    "httpx",
    "httpcore",
    "asyncio",
    "multiprocessing",
    "alembic",
    "fastapi",
    "starlette",
)


def _foreign_record(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """Reduce a record from another library to its logger, level and, for an exception, where it
    came from. The event name becomes a fixed value built from the logger's name (a name from a fixed
    list, else `other`); the message, the arguments and the traceback text are dropped."""
    record = event_dict.get("_record")
    name = str(getattr(record, "name", ""))
    known = any(name == root or name.startswith(root + ".") for root in _FOREIGN_LOGGERS)
    out: dict[str, Any] = {
        key: event_dict[key]
        for key in ("level", "timestamp", "_record", "_from_structlog")
        if key in event_dict
    }
    out["event"] = f"foreign:{name}" if known else "foreign_log"
    out["logger"] = name if known else "other"
    exc_info = getattr(record, "exc_info", None)
    if exc_info and exc_info[1] is not None:
        out.update(error_fields(exc_info[1]))
    return out


def _stdlib_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _allow_list,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                _foreign_record,
            ],
        )
    )
    handler._minerva = True  # type: ignore[attr-defined]  # ours: replaced, never duplicated
    return handler


def _install_stdlib_logging(level: str) -> None:
    """One path for everything that is written: records of the libraries go through the same
    reduction and allow-list as our own events, as one JSON line each."""
    handler = _stdlib_handler()
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_minerva", False)] + [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):  # uvicorn gives these their own plain-text handlers
        library = logging.getLogger(name)
        library.handlers = [handler]
        library.propagate = False
    # Our `request` line replaces the access log, without the client's address or the query string.
    logging.getLogger("uvicorn.access").disabled = True


def configure_logging(level: str = "INFO") -> None:
    _install_stdlib_logging(level)
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
