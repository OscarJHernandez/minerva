"""The reference problems (answers established outside Minerva) stay as expected, and the judge cannot pass a wrong one."""

import json
from pathlib import Path

import pytest
from minerva_engine import Engine, Limits
from minerva_engine.reference_benchmark import (
    PROBLEMS,
    _verdict,
    deterministic,
    markdown,
    run_problems,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, object]]:
    return run_problems(Engine(Limits(timeout_seconds=60, max_concurrency=2)))


def test_every_problem_comes_out_as_expected(rows: list[dict[str, object]]) -> None:
    bad = [
        (r["id"], r["outcome"], r["detail"])
        for r in rows
        if r["outcome"] not in ("correct", "refused_as_expected")
    ]
    assert not bad


def test_outcomes_match_the_committed_report(rows: list[dict[str, object]]) -> None:
    committed = json.loads((ROOT / "docs/benchmarks/reference.json").read_text())["problems"]
    assert deterministic(rows) == committed, "run: python -m minerva_engine.reference_benchmark"
    assert (ROOT / "docs/reference-benchmark.md").read_text().split("\n", 4)[0] == markdown(
        rows
    ).split("\n", 4)[0]


def test_the_corpus_covers_the_requested_domains() -> None:
    assert {p.category for p in PROBLEMS} >= {"matrix", "distribution", "substitution", "mechanics"}
    assert all(p.source for p in PROBLEMS) and len({p.id for p in PROBLEMS}) == len(PROBLEMS)


def test_a_wrong_answer_a_wrong_result_and_a_lost_answer_are_each_caught() -> None:
    assert _verdict("correct", True, False, "d")[0] == "wrong"  # disagrees with the oracle
    assert _verdict("refused", True, None, "d")[0] == "wrong"  # an answer where a refusal was due
    assert _verdict("correct", False, None, "d")[0] == "unsupported_regression"
    assert _verdict("correct", True, True, "d")[0] == "correct"
    assert _verdict("refused", False, None, "d")[0] == "refused_as_expected"
