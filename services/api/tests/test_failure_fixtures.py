"""Every committed failure fixture is replayed against the current code and must still give its outcome.

A fixture is the executed calls of a failed turn and what the engine said (`scripts/replay_trace.py
--to-fixture` writes one from a debug report). Fixtures are reviewed before they are committed: the
arguments are a minimal example, never a user's conversation (`contains_user_mathematics` is false), and a
fixture records provenance: where it came from and which defect it pins.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "failures").glob("*.json"))


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_there_are_fixtures() -> None:
    assert FIXTURES


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_a_fixture_is_reviewed_and_keeps_its_outcome(path: Path, ctx: ToolContext) -> None:
    fixture: dict[str, Any] = json.loads(path.read_text("utf-8"))
    assert fixture["schema"] == 1 and fixture["id"] == path.stem
    assert fixture["contains_user_mathematics"] is False, "review and minimise before committing"
    assert fixture["provenance"]["origin"] in {"synthetic", "debug-report"}
    assert fixture["calls"]
    for recorded in fixture["calls"]:
        result = execute(recorded["name"], recorded["arguments"], ctx)
        assert result.status == recorded["expect"]["status"], path.stem
        expected = recorded["expect"]["evidence"]
        if expected is not None:
            assert result.evidence is not None
            for key, value in expected.items():
                assert getattr(result.evidence, key) == value, (path.stem, key)
