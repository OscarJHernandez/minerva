"""The landing page's recorded research sessions are real, current and engine-backed.

apps/site/src/data/landing-sessions.json is written by scripts/record_landing_sessions.py from live runs. These tests
pin what the page may show: every tool result in it is what the engine returns today for the recorded call, the
recording belongs to the current prompts and to a record in which every run passed, and nothing in it names the model
host or carries a trace.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.card_equations import listed_equations
from minerva_api.agent.evaluation.cases import load_suite
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

ROOT = Path(__file__).resolve().parents[3]
SESSIONS = ROOT / "apps" / "site" / "src" / "data" / "landing-sessions.json"
RECORD = ROOT / "docs" / "benchmarks" / "landing-live.json"
CASES = ("landing-kepler", "landing-schwarzschild")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _cards(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for s in data["sessions"]
        for turn in s["turns"]
        for block in turn["assistant"]["blocks"]
        if block["kind"] == "card"
    ]


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def test_the_page_shows_the_first_passing_run_and_the_true_pass_count() -> None:
    sessions, record = _load(SESSIONS), _load(RECORD)
    runs = record["runs"]
    assert {r["case"] for r in runs} == set(CASES)
    assert len(runs) == len(CASES) * record["runs_per_conversation"] >= len(CASES) * 5
    assert sessions["prompts_digest"] == record["prompts_digest"]
    assert [s["id"] for s in sessions["sessions"]] == list(CASES)
    for shown in sessions["sessions"]:
        own = sorted((r for r in runs if r["case"] == shown["id"]), key=lambda r: r["run"])
        passed = [r["run"] for r in own if r["passed"] and all(r["turns"])]
        assert passed, f"{shown['id']}: no run passed, so nothing may be shown"
        assert (
            shown["shown_run"] == passed[0]
        )  # the first that passed, never a later one picked by hand
        assert shown["passed_runs"] == len(passed) == record["passed_runs"][shown["id"]]


def test_the_page_names_the_model_and_the_settings_of_the_run() -> None:
    sessions, record = _load(SESSIONS), _load(RECORD)
    assert sessions["settings"] == record["settings"]
    for key in ("temperature", "context_window_tokens", "reply_budget_tokens", "thinking"):
        assert key in sessions["settings"]
    model = sessions["model"]
    assert model == record["model"] and model["name"] and model["digest"]
    assert model.get("family") and model.get("parameter_size") and model.get("quantization")
    assert record["model_unchanged_during_run"] is True


def test_the_recording_is_of_the_current_prompts() -> None:
    cases = [c for c in load_suite("explore") if c.id in CASES]
    digest = hashlib.sha256(
        json.dumps([[c.prompt, *[f.prompt for f in c.followups]] for c in cases]).encode()
    ).hexdigest()[:12]
    assert _load(SESSIONS)["prompts_digest"] == digest, (
        "the prompts changed: re-record the sessions"
    )
    shown = {s["id"]: [t["user"]["text"] for t in s["turns"]] for s in _load(SESSIONS)["sessions"]}
    for case in cases:
        assert shown[case.id] == [case.prompt, *[f.prompt for f in case.followups]]


@pytest.mark.parametrize("path", [SESSIONS, RECORD], ids=["sessions", "record"])
def test_no_host_address_and_no_trace_is_published(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"https?://|\b\d{1,3}(\.\d{1,3}){3}\b|:11434|:11440|\.local\b", text)

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | set().union(*(keys(v) for v in node.values()))
        if isinstance(node, list):
            return set().union(*(keys(v) for v in node)) if node else set()
        return set()

    assert not keys(_load(path)) & {"trace", "debug", "llm_calls", "base_url", "host"}

    def reasoning(
        node: Any,
    ) -> bool:  # the model's thinking text; `thinking: true` is only the setting
        if isinstance(node, dict):
            return any(
                (k == "thinking" and isinstance(v, str)) or reasoning(v) for k, v in node.items()
            )
        return isinstance(node, list) and any(reasoning(v) for v in node)

    assert not reasoning(_load(path))


def test_every_recorded_result_is_what_the_engine_returns_for_its_call(ctx: ToolContext) -> None:
    cards = [c for c in _cards(_load(SESSIONS)) if c["sections"]]
    assert cards, "a recording without a single engine result shows nothing worth showing"
    for card in cards:
        call = card["call"]
        out = execute(call["tool"], call["arguments"], ctx)
        data = out.card.data
        inner = data.get("result") or data.get("report") or data
        assert inner.get("status") == card["status"], call["tool"]
        recorded = {row["latex"] for s in card["sections"] for row in s["rows"]}
        fresh = {latex for _, latex in listed_equations(out.card.type, data)}
        assert recorded <= fresh, (
            f"{call['tool']}: a recorded formula is not what the engine returns now"
        )


def test_every_shown_formula_of_a_result_has_its_number() -> None:
    for card in _cards(_load(SESSIONS)):
        for section in card["sections"]:
            for row in section["rows"]:
                assert isinstance(card["numbers"].get(row["key"]), int), row["key"]
