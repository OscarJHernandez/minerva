"""The vector benchmark: every corpus problem is answered correctly, and the stored results are current."""

import json
from pathlib import Path

import pytest
from minerva_engine.vector import benchmark as B

ROOT = Path(__file__).resolve().parents[3]
STORED = ROOT / "docs" / "benchmarks" / "vector.json"
REPORT = ROOT / "docs" / "vector-benchmark.md"


@pytest.fixture(scope="module")
def fresh() -> dict:  # type: ignore[type-arg]
    return B.run(1)


def test_every_corpus_problem_is_answered_correctly(fresh) -> None:  # type: ignore[no-untyped-def]
    cases = fresh["deterministic"]["cases"]
    assert set(cases) == {c.id for c in B.CASES}
    assert [cid for cid, e in cases.items() if not e["correct"]] == []


def test_the_corpus_is_not_only_easy_problems() -> None:
    groups = {g: sum(1 for c in B.CASES if c.group == g) for g in B.GROUPS}
    assert all(n >= 1 for n in groups.values()), groups
    ids = {c.id for c in B.CASES}
    # curvilinear systems, nesting, singular fields and the refusals that matter
    assert {"cylindrical-vector-laplacian", "spherical-curl-curl", "point-charge-field"} <= ids
    assert {"curl-grad-no-smoothness", "two-vectors-side-by-side", "convert-abstract"} <= ids
    assert sum(1 for c in B.CASES if c.status.value != "ok") >= 10


def test_refusals_never_carry_a_result(fresh) -> None:  # type: ignore[no-untyped-def]
    for case in B.CASES:
        if case.status.value != "ok":
            assert fresh["deterministic"]["cases"][case.id]["status"] == case.status.value


def test_the_stored_results_are_current(fresh) -> None:  # type: ignore[no-untyped-def]
    stored = json.loads(STORED.read_text())
    assert B.compare(stored, fresh) == []
    assert REPORT.read_text().startswith("# Vector calculus benchmark")
