"""The strategy corpus stays as expected, and its judge cannot pass a wrong answer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from minerva_engine import Engine, Limits
from minerva_engine.integration_strategy import benchmark

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=60, max_concurrency=2))


@pytest.fixture(scope="module")
def rows(engine: Engine) -> list[dict[str, Any]]:
    return benchmark.run_cases(engine)


def test_no_case_is_wrong_and_outcomes_match_the_committed_file(rows: list[dict[str, Any]]) -> None:
    assert [r["id"] for r in rows if r["outcome"] == "wrong"] == []
    committed = json.loads((ROOT / "docs/benchmarks/integration_strategies.json").read_text())
    assert committed["cases"] == benchmark.deterministic(rows)
    assert (
        (ROOT / "docs/integration-strategies-benchmark.md")
        .read_text()
        .splitlines()[4]
        .startswith(f"{len(rows)} of {len(rows)} as expected")
    )


def test_every_technique_has_a_result_case_and_the_hard_ones_have_refusals(
    rows: list[dict[str, Any]],
) -> None:
    results = {r["technique"] for r in rows if r["expected"] == "result"}
    assert results >= {
        "substitution",
        "parts",
        "trigonometric",
        "partial_fractions",
        "parity",
        "reflection",
        "frullani",
        "schwinger",
        "parameter_differentiation",
        "numerical",
    }
    assert {r["technique"] for r in rows if r["expected"] == "refused"} >= {
        "frullani",
        "parity",
        "numerical",
    }
    # integrals that only the strategies solve: the default call did not use SymPy's route
    assert {r["id"] for r in rows if r["default"] in ("reflection", "frullani")} >= {"king-rule"}


def test_the_judge_catches_a_wrong_value_and_a_result_where_a_refusal_is_due(
    engine: Engine,
) -> None:
    case = next(c for c in benchmark.CASES if c.id == "parity-even")
    data = benchmark.ExpressionInput(
        input=case.expression, format="text", assumptions=case.assumptions
    )
    result = engine.integrate(data, "x", case.lower, case.upper, "parity")
    assert benchmark.judge(case, result)[0] == "correct"
    wrong = result.model_copy(update={"result_text": "1/3", "numeric": None})
    assert benchmark.judge(case, wrong)[0] == "wrong"
    must_refuse = dataclasses.replace(case, expect="refused")
    assert benchmark.judge(must_refuse, result)[0] == "wrong"
