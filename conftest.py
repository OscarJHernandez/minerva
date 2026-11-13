"""Repository-wide pytest hooks.

`MINERVA_TEST_SHARD=i/n` (1 <= i <= n) runs only the i-th of n disjoint slices of the collected tests, so CI can
spread the suite over several runner pods. A test belongs to the shard `crc32(node id) % n`: no timing file to
keep fresh, the same answer in every xdist worker, and every test lands in exactly one shard. Unset, everything
runs. A malformed value is an error, never "run nothing" (a silently empty shard would pass CI).
"""

from __future__ import annotations

import os
import zlib

import pytest

SHARD_ENV = "MINERVA_TEST_SHARD"


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    value = os.environ.get(SHARD_ENV)
    if value is None:
        return
    try:
        index, count = parse_shard(value)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from None
    keep = [item for item in items if shard_of(item.nodeid, count) == index]
    dropped = [item for item in items if shard_of(item.nodeid, count) != index]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep
