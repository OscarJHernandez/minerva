"""The CI shard selector (`conftest.py`, `MINERVA_TEST_SHARD=i/n`) must split tests exactly, never drop one."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "root_conftest", Path(__file__).resolve().parents[2] / "conftest.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

IDS = [f"packages/engine/tests/test_{i % 17}.py::test_case[{i}]" for i in range(2000)]


@pytest.mark.parametrize("count", [1, 2, 5, 7])
def test_every_test_is_in_exactly_one_shard(count: int) -> None:
    shards = [{n for n in IDS if MODULE.shard_of(n, count) == i} for i in range(1, count + 1)]
    assert sum(len(s) for s in shards) == len(IDS)
    assert set().union(*shards) == set(IDS)


def test_shards_are_balanced() -> None:
    sizes = [sum(MODULE.shard_of(n, 5) == i for n in IDS) for i in range(1, 6)]
    assert min(sizes) > 0.8 * len(IDS) / 5
    assert max(sizes) < 1.2 * len(IDS) / 5


def test_assignment_is_stable() -> None:
    # Different processes (xdist workers) must agree, so the answer may not depend on hash randomisation.
    assert MODULE.shard_of("a/b.py::test_x", 5) == MODULE.shard_of("a/b.py::test_x", 5)
    assert MODULE.shard_of("a/b.py::test_x", 5) == 2


@pytest.mark.parametrize(
    "value", ["", "3", "0/5", "6/5", "a/5", "1/b", "1/", "/5", "-1/5", "1/0", "1/5/2"]
)
def test_malformed_values_are_refused(value: str) -> None:
    with pytest.raises(ValueError):
        MODULE.parse_shard(value)


def test_valid_value() -> None:
    assert MODULE.parse_shard("2/5") == (2, 5)


def test_tier_defaults_to_pr_and_rejects_unknown_values() -> None:
    assert MODULE.parse_tier(None) == "pr"
    assert MODULE.parse_tier("") == "pr"
    assert MODULE.parse_tier("extended") == "extended"
    with pytest.raises(ValueError, match="MINERVA_TEST_TIER"):
        MODULE.parse_tier("nightly")  # a typo must not silently run (or skip) the wrong set


# -- benchmark modules stay together ----------------------------------------------------------------
BENCH = [
    f"packages/{pkg}/tests/test_{name}_benchmark.py::test_each_case[{i}]"
    for pkg, name in [("engine", "vector"), ("engine", "geometry"), ("knowledge", "stat")]
    for i in range(40)
] + [f"scripts/tests/test_capabilities.py::test_{i}" for i in range(4)]


@pytest.mark.parametrize("count", [1, 2, 5, 7])
def test_the_assignment_is_an_exact_partition_with_whole_benchmark_modules(count: int) -> None:
    ids = IDS + BENCH
    shards = MODULE.assign_shards(ids, count)
    assert set(shards) == set(ids) and all(1 <= s <= count for s in shards.values())
    for module in {MODULE.module_of(n) for n in BENCH}:
        assert len({shards[n] for n in BENCH if MODULE.module_of(n) == module}) == 1
    # ordinary tests keep the hash split
    assert all(shards[n] == MODULE.shard_of(n, count) for n in IDS)


def test_benchmark_modules_are_dealt_evenly_and_ordinary_modules_are_not_grouped() -> None:
    shards = MODULE.assign_shards(BENCH, 4)
    assert sorted({shards[n] for n in BENCH}) == [1, 2, 3, 4]  # four modules, four shards
    assert MODULE.is_whole_module("packages/knowledge/tests/test_benchmark.py::test_x")
    assert not MODULE.is_whole_module("packages/engine/tests/test_simplify.py::test_benchmarks[1]")


def test_the_assignment_does_not_depend_on_collection_order() -> None:
    ids = IDS + BENCH
    assert MODULE.assign_shards(ids, 5) == MODULE.assign_shards(list(reversed(ids)), 5)


# -- whole modules are dealt by weight, heaviest first ------------------------------------------------
def _module_ids(names: list[str]) -> list[str]:
    return [f"packages/x/tests/{name}::test_a" for name in names]


def test_whole_modules_are_dealt_heaviest_first_onto_the_lightest_shard() -> None:
    weights = {
        "packages/x/tests/test_a_benchmark.py": 100,
        "packages/x/tests/test_b_benchmark.py": 60,
        "packages/x/tests/test_c_benchmark.py": 50,
        "packages/x/tests/test_d_benchmark.py": 40,
        "packages/x/tests/test_e_benchmark.py": 10,
    }
    ids = _module_ids([Path(m).name for m in weights])
    shards = MODULE.assign_shards(ids, 2, weights)
    load = {1: 0, 2: 0}
    for n in ids:
        load[shards[n]] += weights[MODULE.module_of(n)]
    # 100 | 60+50 -> 100 | 110, then 40 joins the 100 side, 10 the lighter again: 150 | 110 -> best possible split
    assert abs(load[1] - load[2]) <= 40
    assert shards[ids[0]] == 1  # the heaviest goes to the first, empty shard


def test_a_module_without_a_weight_counts_as_the_default_and_is_still_assigned() -> None:
    ids = _module_ids(["test_new_benchmark.py", "test_other_benchmark.py"])
    shards = MODULE.assign_shards(ids, 2, {})
    assert sorted(shards.values()) == [1, 2]  # equal default weights: one each
    assert MODULE.DEFAULT_WEIGHT > 0


def test_the_committed_weights_cover_only_existing_files_and_are_positive() -> None:
    root = Path(__file__).resolve().parents[2]
    weights = MODULE.load_weights()
    assert weights
    for module, seconds in weights.items():
        assert (root / module).is_file(), f"stale entry in scripts/shard_weights.json: {module}"
        assert seconds > 0
        assert MODULE.is_whole_module(module)


def test_every_whole_module_in_the_repository_is_balanced_across_five_shards() -> None:
    root = Path(__file__).resolve().parents[2]
    ids = [
        f"{p.relative_to(root).as_posix()}::test_a"
        for p in root.glob("**/test_*.py")
        if ".venv" not in p.parts and "node_modules" not in p.parts
    ]
    whole = [n for n in ids if MODULE.is_whole_module(n)]
    shards = MODULE.assign_shards(whole, 5)
    weights = MODULE.load_weights()
    load = [0.0] * 5
    for n in whole:
        load[shards[n] - 1] += weights.get(MODULE.module_of(n), MODULE.DEFAULT_WEIGHT)
    assert max(load) < 1.25 * sum(load) / 5
