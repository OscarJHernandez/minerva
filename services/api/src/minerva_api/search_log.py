"""Server-side logging of the conserved-quantity search.

The search returns structured events made of counters and fixed labels (family code, matrix shape, rank,
nullity, stage times, stop reason). They are logged here, one line each, with the request id already bound
by the middleware as the trace id. Only the fields below pass: never an expression, a coordinate name, a
parameter or any user content. The full detail that does carry expressions stays in the card stored with
the user's own conversation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .logging import get_logger

# Counter and label fields an event may carry (see `ALLOWED_FIELDS` in logging.py).
SEARCH_FIELDS = frozenset(
    {
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
        "reason",
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
    }
)

log = get_logger()


class _Event(Protocol):
    name: str
    tier: int | None
    at_ms: int
    data: dict[str, int | float | str | bool]


def log_search_events(events: Iterable[_Event]) -> None:
    for e in events:
        fields = {
            k: v
            for k, v in e.data.items()
            if k in SEARCH_FIELDS and (not isinstance(v, str) or _is_label(v))
        }
        log.info(e.name, tier=e.tier, at_ms=e.at_ms, **fields)


def _is_label(value: str) -> bool:
    """A fixed label such as a family code or a stop reason: short, no mathematical punctuation."""
    return len(value) <= 40 and all(ch.isalnum() or ch in "_-" for ch in value)
