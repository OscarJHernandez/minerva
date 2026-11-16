"""The offline orchestration suite (agent/evaluation/offline.py) against its stored reports.

`docs/benchmarks/orchestration-offline.json` is the current deterministic result; the `-baseline` file is the
same trajectories run on the code before the reliability work. A change that moves a deterministic field
(calls, estimated tokens, marks, correctness) fails here until the report is regenerated and the diff
reviewed: `uv run python scripts/eval_offline.py --json-out docs/benchmarks/orchestration-offline.json
--out docs/benchmarks/orchestration-offline.md`. Timings are never compared.
"""

import json
from pathlib import Path

import pytest
from minerva_api.agent.evaluation import offline
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

DOCS = Path(__file__).resolve().parents[3] / "docs" / "benchmarks"


@pytest.fixture(scope="module")
def report() -> dict:  # type: ignore[type-arg]
    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    return offline.run_suite(ctx)


def test_the_stored_report_is_current(report: dict) -> None:  # type: ignore[type-arg]
    stored = json.loads((DOCS / "orchestration-offline.json").read_text())
    assert offline.deterministic_view(report) == offline.deterministic_view(stored)


def test_every_trajectory_ends_mathematically_correct(report: dict) -> None:  # type: ignore[type-arg]
    wrong = [r["id"] for r in report["results"] if not r["deterministic"]["math_ok"]]
    assert wrong == []


def test_nothing_got_worse_than_the_baseline(report: dict) -> None:  # type: ignore[type-arg]
    baseline = json.loads((DOCS / "orchestration-offline-baseline.json").read_text())
    regressions = [
        line for line in offline.compare(baseline, report) if line.startswith("REGRESSION")
    ]
    assert regressions == []


def test_the_recorded_failure_now_costs_at_most_half_the_input_tokens(report: dict) -> None:  # type: ignore[type-arg]
    """The supplied example: 58 009 input tokens over 4 model calls, 0 of the equations backed."""
    baseline = json.loads((DOCS / "orchestration-offline-baseline.json").read_text())
    before = next(r for r in baseline["results"] if r["id"] == "pendulum-original-failure")
    after = next(r for r in report["results"] if r["id"] == "pendulum-routed-latex")
    assert after["deterministic"]["model_calls"] == 1 and after["deterministic"]["tool_calls"] == 1
    assert after["deterministic"]["est_tokens_in"] <= before["deterministic"]["est_tokens_in"] / 2
    assert after["deterministic"]["evidence_pct"] == 100.0
    assert before["deterministic"]["evidence_pct"] == 0.0
