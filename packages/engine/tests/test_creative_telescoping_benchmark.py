"""The creative-telescoping benchmark is a gate: its outcomes are committed, and a wrong answer fails."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import mpmath as mp
import pytest
from minerva_engine.creative_telescoping import benchmark
from minerva_engine.models import Limits
from minerva_engine.service import Engine

ROOT = Path(__file__).resolve().parents[3]
CATEGORIES = {
    "gaussian",
    "laplace",
    "rational_exponential",
    "higher_order",
    "special_function",
    "nonzero_boundary",
    "convergence_restriction",
    "singular_divergent",
    "beyond_class",
    "forged_certificate",
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:  # type: ignore[type-arg]
    return benchmark.run_cases(Engine(Limits(timeout_seconds=120, max_concurrency=2)))


def test_no_case_is_wrong_and_the_outcomes_match_the_committed_file(rows: list[dict]) -> None:  # type: ignore[type-arg]
    assert [r["id"] for r in rows if r["outcome"] in ("wrong", "unsupported_regression")] == []
    committed = json.loads((ROOT / "docs/benchmarks/creative_telescoping.json").read_text())
    assert committed["problems"] == benchmark.deterministic(rows)
    lines = (ROOT / "docs/creative-telescoping-benchmark.md").read_text().splitlines()
    assert lines[4].startswith(f"{len(rows)} of {len(rows)} as expected")


def test_every_requested_category_is_covered_and_failures_are_classified(rows: list[dict]) -> None:  # type: ignore[type-arg]
    assert {r["category"] for r in rows} == CATEGORIES
    refused = [r for r in rows if r["expected"] == "refused"]
    assert len(refused) >= 6 and all(r["failure_class"] for r in refused)
    assert {r["order"] for r in rows if r["order"] is not None} >= {0, 1, 2}


def test_a_wrong_reference_or_a_missing_refusal_is_caught() -> None:
    engine = Engine(Limits(timeout_seconds=60, max_concurrency=2))
    case = next(c for c in benchmark.CASES if c.id == "gaussian-a")
    assert benchmark.judge_case(case, engine)[0] == "correct"
    wrong = dataclasses.replace(case, closed=lambda a: mp.sqrt(mp.pi / a) * (1 + 1e-6))
    assert benchmark.judge_case(wrong, engine)[0] == "wrong"
    divergent = next(c for c in benchmark.CASES if c.id == "divergent-at-zero")
    as_value = dataclasses.replace(divergent, expect="evaluated")
    assert benchmark.judge_case(as_value, engine)[0] == "unsupported_regression"
    wrong_reason = dataclasses.replace(divergent, reasons=("not_hyperexponential",))
    assert benchmark.judge_case(wrong_reason, engine)[0] == "wrong"


def test_a_forged_certificate_that_passes_would_be_flagged() -> None:
    engine = Engine(Limits(timeout_seconds=60, max_concurrency=2))
    control = next(c for c in benchmark.CASES if c.id == "forged-control")
    swapped = dataclasses.replace(control, expect="rejected")
    assert benchmark.judge_case(swapped, engine)[0] == "wrong"
