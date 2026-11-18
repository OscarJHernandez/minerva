"""The example cards of the chat home page are tied to the agent evaluation suite.

`packages/ui/fixtures/homepage-examples.json` is the one list of cards (the web app imports it). Every card must name
an evaluation case that runs the exact prompt the card submits, with a reference call the engine reproduces (that is
checked for every case by `test_evaluation.py`), and a card may be `approved` (shown) only with a stored record of five
live runs of that exact prompt that were all correct and engine-backed. The registry holds no expected answer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from minerva_api.agent.evaluation.cases import load_cases

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = json.loads(
    (ROOT / "packages" / "ui" / "fixtures" / "homepage-examples.json").read_text()
)
EXAMPLES = REGISTRY["examples"]
LIVE = ROOT / "docs" / "benchmarks" / "homepage-live.json"
CASES = {c.id: c for c in load_cases()}
CAPABILITIES = {
    c["id"]
    for c in yaml.safe_load((ROOT / "docs" / "capabilities.yaml").read_text())["capabilities"]
}
RUNS_REQUIRED = 5


def sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def test_the_registry_has_six_examples_with_unique_ids_prompts_and_cases() -> None:
    assert len(EXAMPLES) == 6
    for key in ("id", "prompt", "title", "eval_case", "latex"):
        values = [e[key] for e in EXAMPLES]
        assert len(set(values)) == len(values), key


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_each_card_submits_exactly_the_prompt_of_its_evaluation_case(example: dict) -> None:
    case = CASES.get(example["eval_case"])
    assert case is not None, f"{example['id']}: no evaluation case {example['eval_case']!r}"
    assert case.prompt == example["prompt"], "the card and its case must carry the identical prompt"
    assert case.history == [] and not case.followups  # a card starts a new conversation


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_each_case_has_an_engine_checked_reference_and_a_reply_check(example: dict) -> None:
    case = CASES[example["eval_case"]]
    assert case.reference is not None and case.expect_card is not None
    assert (
        case.expected_tool == case.reference.tool
    )  # the card is answered by the tool its reference calls
    assert (
        case.reply_matches and case.reply_forbids
    )  # the explanation is scored, not only the tool call
    assert not case.informational  # a card is never an unscored measurement
    assert case.source  # where the expected result comes from, outside Minerva


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_each_card_names_a_real_capability_and_no_case_is_shared(example: dict) -> None:
    assert example["capability"] in CAPABILITIES
    owners = [e["id"] for e in EXAMPLES if e["eval_case"] == example["eval_case"]]
    assert owners == [example["id"]]


def test_every_homepage_case_belongs_to_a_card() -> None:
    """No untested card and no orphan case: the `homepage-` cases are exactly the cards' cases."""
    assert {c for c in CASES if c.startswith("homepage-")} == {e["eval_case"] for e in EXAMPLES}


def test_the_displayed_expression_and_description_give_no_result_away() -> None:
    for e in EXAMPLES:
        assert "=" not in e["description"] and "pi/4" not in e["description"].lower()
        assert not e["prompt"].count("$") and "\\" not in e["prompt"]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e["id"])
def test_a_card_is_shown_only_with_five_correct_live_runs_of_its_exact_prompt(
    example: dict,
) -> None:
    if not example["approved"]:
        return
    assert LIVE.exists(), f"{example['id']} is approved but {LIVE.name} does not exist"
    record = json.loads(LIVE.read_text())["cards"].get(example["id"])
    assert record is not None, f"{example['id']} is approved without a live record"
    assert record["prompt_sha256"] == sha(example["prompt"]), "the record is for a different prompt"
    runs = record["runs"]
    assert len(runs) >= RUNS_REQUIRED
    for run in runs:
        assert run["passed"] is True and run["failure_class"] is None
        assert run["engine_backed"] is True and run["false_verification_claims"] == 0


def test_the_live_record_never_names_the_model_host() -> None:
    """Everything committed is public: a record may name the model, never the address it ran on."""
    if not LIVE.exists():
        return
    text = LIVE.read_text()
    assert "http://" not in text and "https://" not in text and "11440" not in text
