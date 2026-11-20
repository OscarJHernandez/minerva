"""The "Explore in Minerva" buttons are tied to the evaluation suite and to the landing example they continue.

`packages/ui/fixtures/explorations.json` is the one list of explorations (the site and the web app import it).
Each one names the landing-page example it continues, an evaluation case (suite `explore`) that runs its exact
prompt with an engine-checked reference call, and may be `approved` (its button shown) only with a stored record
of five correct, engine-backed live runs of that exact prompt at the deployed 10 s compute limit. The registry
holds no result: every number on the page comes from the engine when the user runs the prompt."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from minerva_api.agent.evaluation.cases import load_suite

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = json.loads((ROOT / "packages" / "ui" / "fixtures" / "explorations.json").read_text())
EXPLORATIONS = REGISTRY["explorations"]
LIVE = ROOT / "docs" / "benchmarks" / "exploration-live.json"
# The `landing-*` cases are the landing page's recorded research sessions, pinned by test_landing_sessions.py, not
# "Explore in Minerva" buttons.
CASES = {c.id: c for c in load_suite("explore") if not c.id.startswith("landing-")}
CAPABILITIES = {
    c["id"]: c
    for c in yaml.safe_load((ROOT / "docs" / "capabilities.yaml").read_text())["capabilities"]
}
RUNS_REQUIRED = 5


def sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def squash(text: str) -> str:
    return re.sub(r"\\,|\s+", "", text)


def landing_example(exploration: dict[str, Any]) -> dict[str, Any]:
    capability = CAPABILITIES[exploration["capability"]]
    found = [e for e in capability["examples"] if e["id"] == exploration["example"]]
    assert found, f"{exploration['id']}: no example {exploration['example']!r} in the capability"
    return found[0]


def test_ids_prompts_and_cases_are_unique() -> None:
    for key in ("id", "prompt", "title", "eval_case"):
        values = [e[key] for e in EXPLORATIONS]
        assert len(set(values)) == len(values), key


def test_every_exploration_case_belongs_to_an_exploration() -> None:
    assert set(CASES) == {e["eval_case"] for e in EXPLORATIONS}


@pytest.mark.parametrize("exploration", EXPLORATIONS, ids=lambda e: e["id"])
def test_a_button_puts_exactly_the_prompt_of_its_evaluation_case_in_the_composer(
    exploration: dict[str, Any],
) -> None:
    case = CASES[exploration["eval_case"]]
    assert case.prompt == exploration["prompt"]
    assert case.history == [] and not case.followups  # an exploration starts a new conversation
    assert case.reference is not None and case.expect_card is not None
    assert case.expected_tool == case.reference.tool == "geometry"
    assert case.reply_matches and case.reply_forbids and not case.informational and case.source


@pytest.mark.parametrize("exploration", EXPLORATIONS, ids=lambda e: e["id"])
def test_the_prompt_carries_the_metric_coordinates_and_assumptions_of_the_landing_example(
    exploration: dict[str, Any],
) -> None:
    """The model may not invent a metric or an assumption, so the prompt must hold every one of them."""
    example = landing_example(exploration)
    prompt = squash(exploration["prompt"])
    for coordinate in example["coordinates"]:
        assert squash(coordinate).lstrip("\\") in prompt, coordinate
    for row in example["metric"]:
        for entry in row:
            if entry != "0":
                assert squash(entry) in prompt, f"metric entry {entry!r} missing from the prompt"
    for symbol, kinds in example["assumptions"].items():
        for kind in kinds:
            assert kind in exploration["prompt"] and f"{symbol}" in exploration["prompt"]
    # and the reference call the case checks is the same metric
    reference = CASES[exploration["eval_case"]].reference
    assert reference is not None
    arguments = reference.arguments
    assert arguments["metric"] == example["metric"]
    assert arguments["assumptions"] == example["assumptions"]
    assert [c.replace("\\", "") for c in example["coordinates"]] == arguments["coordinates"]
    assert set(arguments["quantities"]) <= {"christoffel", *example["quantities"]}


@pytest.mark.parametrize("exploration", EXPLORATIONS, ids=lambda e: e["id"])
def test_each_exploration_names_a_real_capability_and_the_prompt_gives_no_result_away(
    exploration: dict[str, Any],
) -> None:
    assert exploration["capability"] in CAPABILITIES
    prompt = exploration["prompt"]
    assert "48" not in prompt and "Ricci-flat" not in prompt and "vanishes" not in prompt
    assert len(prompt) <= 1500  # far below the 4000-character message limit


@pytest.mark.parametrize("exploration", EXPLORATIONS, ids=lambda e: e["id"])
def test_a_button_is_shown_only_with_five_correct_live_runs_of_its_exact_prompt(
    exploration: dict[str, Any],
) -> None:
    if not exploration["approved"]:
        return
    assert LIVE.exists(), f"{exploration['id']} is approved but {LIVE.name} does not exist"
    record = json.loads(LIVE.read_text())["explorations"].get(exploration["id"])
    assert record is not None, f"{exploration['id']} is approved without a live record"
    assert record["prompt_sha256"] == sha(exploration["prompt"]), "the record is for another prompt"
    assert record["tool_time_limit_seconds"] <= 10.0
    runs = record["runs"]
    assert len(runs) >= RUNS_REQUIRED
    for run in runs:
        assert run["passed"] is True and run["failure_class"] is None
        assert run["engine_backed"] is True and run["false_verification_claims"] == 0


def test_the_live_record_never_names_the_model_host() -> None:
    if not LIVE.exists():
        return
    text = LIVE.read_text()
    assert "http://" not in text and "https://" not in text and "11440" not in text
