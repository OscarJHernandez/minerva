"""Shared wall-clock budget for the test suites (engine, knowledge and API tests import it).

A test that expects a mathematical result is not a latency test. The CI runners are shared (five test shards,
the API pod and other jobs run side by side, with no CPU limits), so a computation that takes two seconds on a
workstation can take ten on a runner. With the engine's default 10 s limit that turned correct results into
`resource_limit` / `not_verified` and failed the build. Every such test builds its limits here, so the margin is
one number in one place instead of a constant raised after each failure.

This is a wall-clock cap, never a mathematical tolerance: no check in the engine reads it. Tests that exercise
the time limit itself (they expect `resource_limit`) keep their own short limits and must not use this.
`MINERVA_TEST_TIMEOUT_SECONDS` raises it for a slower machine.
"""

from __future__ import annotations

import os

from minerva_engine.models import Limits

DEFAULT_TEST_TIMEOUT_SECONDS = 60.0


def shared_timeout_seconds() -> float:
    raw = os.environ.get("MINERVA_TEST_TIMEOUT_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_TEST_TIMEOUT_SECONDS
    value = float(raw)
    if value <= 0:
        raise ValueError("MINERVA_TEST_TIMEOUT_SECONDS must be positive")
    return value


# Not named `test_*` on purpose: pytest would collect it from any module that imports it.
def budget_limits(**overrides: int | float) -> Limits:
    """`Limits` for a test that expects a result; `overrides` (e.g. `max_concurrency=2`) are passed through.

    `timeout_seconds` is never lower than the shared budget: asking for less is a mistake in a test that expects
    `ok`, and a test that wants a short limit builds `Limits(timeout_seconds=...)` itself.
    """
    values: dict[str, int | float] = dict(overrides)
    values["timeout_seconds"] = max(
        float(values.get("timeout_seconds", 0.0)), shared_timeout_seconds()
    )
    return Limits.model_validate(values)
