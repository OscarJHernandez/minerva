"""Repository-wide pytest hooks.

`MINERVA_TEST_SHARD=i/n` (1 <= i <= n) runs only the i-th of n disjoint slices of the collected tests, so CI can
spread the suite over several runner pods. A test belongs to the shard `crc32(node id) % n`: no timing file to
keep fresh, the same answer in every xdist worker, and every test lands in exactly one shard. Unset, everything
runs. A malformed value is an error, never "run nothing" (a silently empty shard would pass CI).

Benchmark modules are the exception (`is_whole_module`): their tests share one module-scoped fixture that runs a
whole corpus (one to three minutes). Split by test id, that fixture ran again in every shard, and again in every
xdist worker of a shard. Such a module is kept together: all its tests go to one shard. The modules are dealt
heaviest first onto the lightest shard, by the seconds in `scripts/shard_weights.json` (a module not listed counts
as `DEFAULT_WEIGHT`); the deal is a function of the collected set and that file only, so every shard and every
xdist worker computes the same partition. CI runs `--dist loadfile`, so one worker
owns each file and the corpus runs once per CI run. (`--dist loadgroup` with an `xdist_group` mark would keep the
other files spread over the workers, but xdist 3.8 sets an attribute that pytest 9 no longer reads, so the mark is
silently ignored.)
"""

from __future__ import annotations

import json
import os
import zlib
from functools import cache
from pathlib import Path

import pytest

SHARD_ENV = "MINERVA_TEST_SHARD"
TIER_ENV = "MINERVA_TEST_TIER"
TIERS = ("pr", "extended")
WEIGHTS_FILE = Path(__file__).resolve().parent / "scripts" / "shard_weights.json"
DEFAULT_WEIGHT = 60.0  # seconds assumed for a whole module that has no entry yet


def parse_tier(value: str | None) -> str:
    """`MINERVA_TEST_TIER`: `pr` (default) skips tests marked `extended`; `extended` runs everything."""
    if value is None or value == "":
        return "pr"
    if value not in TIERS:
        raise ValueError(f"{TIER_ENV} must be one of {', '.join(TIERS)}, got {value!r}")
    return value


def parse_shard(value: str) -> tuple[int, int]:
    """`"2/5"` -> (2, 5). Raises `ValueError` for anything else."""
    head, sep, tail = value.partition("/")
    if not sep or not head.isdecimal() or not tail.isdecimal():
        raise ValueError(f"{SHARD_ENV} must look like 'i/n', got {value!r}")
    index, count = int(head), int(tail)
    if not 1 <= index <= count:
        raise ValueError(f"{SHARD_ENV} needs 1 <= i <= n, got {value!r}")
    return index, count


def shard_of(nodeid: str, count: int) -> int:
    """The 1-based shard a test belongs to."""
    return zlib.crc32(nodeid.encode()) % count + 1


def module_of(nodeid: str) -> str:
    """`a/b/test_x.py::test_y[z]` -> `a/b/test_x.py`."""
    return nodeid.split("::", 1)[0]


def is_whole_module(nodeid: str) -> bool:
    """A module whose tests must stay together: benchmark corpora and the capability record."""
    name = module_of(nodeid).rsplit("/", 1)[-1]
    return "benchmark" in name or name == "test_capabilities.py"


@cache
def load_weights() -> dict[str, float]:
    """Seconds per whole-module file from `scripts/shard_weights.json` (keys starting with `_` are comments)."""
    data = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def assign_shards(
    nodeids: list[str], count: int, weights: dict[str, float] | None = None
) -> dict[str, int]:
    """The 1-based shard of every collected test: whole modules by weight, the rest by `shard_of`.

    Whole modules go heaviest first (ties by name) onto the shard with the least weight so far (ties: the lowest
    number). A function of the collected set and the weights only, so every shard and xdist worker agrees.
    """
    table = load_weights() if weights is None else weights
    whole = sorted(
        {module_of(n) for n in nodeids if is_whole_module(n)},
        key=lambda module: (-table.get(module, DEFAULT_WEIGHT), module),
    )
    load = [0.0] * count
    module_shard: dict[str, int] = {}
    for module in whole:
        lightest = load.index(min(load))
        load[lightest] += table.get(module, DEFAULT_WEIGHT)
        module_shard[module] = lightest + 1
    return {
        n: module_shard[module_of(n)] if is_whole_module(n) else shard_of(n, count) for n in nodeids
    }


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Tier first, so the shard split is computed over the tests that will actually run.
    try:
        tier = parse_tier(os.environ.get(TIER_ENV))
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    if tier == "pr":
        slow = [item for item in items if item.get_closest_marker("extended")]
        if slow:
            config.hook.pytest_deselected(items=slow)
            items[:] = [item for item in items if not item.get_closest_marker("extended")]
    value = os.environ.get(SHARD_ENV)
    if value is None:
        return
    try:
        index, count = parse_shard(value)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    shards = assign_shards([item.nodeid for item in items], count)
    keep = [item for item in items if shards[item.nodeid] == index]
    dropped = [item for item in items if shards[item.nodeid] != index]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep
