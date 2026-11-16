import os

import pytest
from hypothesis import HealthCheck, settings
from minerva_engine import Engine, Limits

# Two Hypothesis profiles, chosen with MINERVA_HYPOTHESIS_PROFILE:
#   pr        small and derandomized: the same examples on every run, so a pull request is deterministic.
#   extended  larger and random, with the failing example printed as a reproduction blob; the scheduled
#             workflow passes --hypothesis-seed and echoes it, so a failure can be replayed.
# A failure found by the extended profile is pinned with @example so the pr profile keeps it.
settings.register_profile(
    "pr",
    max_examples=8,
    deadline=None,
    derandomize=True,
    database=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "extended",
    max_examples=100,
    deadline=None,
    derandomize=False,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
_PROFILE = os.environ.get("MINERVA_HYPOTHESIS_PROFILE") or "pr"
if _PROFILE not in ("pr", "extended"):
    raise pytest.UsageError(f"MINERVA_HYPOTHESIS_PROFILE must be pr or extended, got {_PROFILE!r}")
settings.load_profile(_PROFILE)


@pytest.fixture(scope="session")
def engine() -> Engine:
    # A wall-clock cap, not a mathematical tolerance: the heaviest derivations here take a few seconds
    # alone and several times that on a loaded CI runner (five shards plus other jobs), where a 20 s cap
    # turned them into `resource_limit` / `not_verified`. Tests that exercise the timeout itself build
    # their own short-limit engine.
    return Engine(Limits(timeout_seconds=60, max_concurrency=2))
