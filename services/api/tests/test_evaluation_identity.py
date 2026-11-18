"""A benchmark says which model, with what configuration, was measured (never the host's address)."""

import json
from typing import Any

import httpx
import pytest
from minerva_api.agent.evaluation import cli
from minerva_api.agent.evaluation.cli import identity_unchanged, model_identity
from minerva_api.agent.evaluation.report import build_document, compare, render_markdown

from .test_evaluation import META, result

HOST = "http://model-host.invalid:11440"

SHOW = {
    "details": {
        "parent_model": "gemma4:26b",
        "format": "gguf",
        "family": "gemma4",
        "parameter_size": "25.2B",
        "quantization_level": "Q4_K_M",
    },
    "capabilities": ["tools", "completion", "thinking"],
    "modified_at": "2026-11-15T16:30:28Z",
    "parameters": "top_k 64\nnum_ctx 16384\nnum_predict 1200\ntemperature 0.2\nstop <x>\nweird\n",
    "license": "do not copy this text into a report",
}
TAGS = {"models": [{"name": "minerva:latest", "digest": "149a0ff8a1ca" + "0" * 52}]}
PS = {"models": [{"name": "minerva:latest", "context_length": 32768}]}


class _Response:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload, self.status_code = payload, status

    def json(self) -> dict[str, Any]:
        return self.payload


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, **_: Any) -> _Response:
        return _Response(SHOW)

    def get(url: str, **_: Any) -> _Response:
        return _Response(TAGS if url.endswith("/api/tags") else PS)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)


def test_identity_is_what_the_host_reports_and_never_contains_the_host(host: None) -> None:
    identity = model_identity("ollama", "minerva", HOST)
    assert identity is not None
    assert identity["digest"] == "149a0ff8a1ca"
    assert (identity["family"], identity["parameter_size"], identity["quantization"]) == (
        "gemma4",
        "25.2B",
        "Q4_K_M",
    )
    assert identity["base_model"] == "gemma4:26b" and "thinking" in identity["capabilities"]
    # Only known keys, typed; unknown lines and licence text are not copied.
    assert identity["host_defaults"] == {
        "top_k": 64,
        "num_ctx": 16384,
        "num_predict": 1200,
        "temperature": 0.2,
    }
    assert identity["loaded_context"] == 32768
    dumped = json.dumps(identity)
    assert "model-host" not in dumped and "11440" not in dumped and "do not copy" not in dumped


def test_an_unreachable_host_is_not_recorded_rather_than_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", fail)
    monkeypatch.setattr(httpx, "get", fail)
    assert model_identity("ollama", "minerva", HOST) is None
    assert model_identity("ollama", "minerva", None) is None


def test_a_hosted_model_says_its_weights_are_not_pinned() -> None:
    identity = model_identity("openrouter", "vendor/model", None)
    assert identity and identity["hosted"] and "not recorded" in identity["note"]


def test_a_model_that_changed_during_the_run_is_flagged(host: None) -> None:
    before = model_identity("ollama", "minerva", HOST)
    after = dict(before or {})
    assert identity_unchanged(before, after) is True
    after["digest"] = "ffffffffffff"
    assert identity_unchanged(before, after) is False
    assert identity_unchanged(None, after) is None


def _doc(label: str, identity: dict[str, Any] | None, **meta: Any) -> dict[str, Any]:
    params = {"num_ctx": 32768, "num_predict": None, "temperature": 0.2, "think": False}
    return build_document(
        {**META, "label": label, "params": params, **meta},
        {"minerva": [result("a")]},
        {"minerva": identity},
    )


def test_the_report_names_the_model_its_weights_and_the_host_defaults(host: None) -> None:
    identity = model_identity("ollama", "minerva", HOST)
    assert identity is not None
    identity["unchanged_during_run"] = True
    text = render_markdown(_doc("run", identity))
    section = text.split("## Model under test")[1].split("## Headline")[0]
    for expected in (
        "minerva",
        "gemma4",
        "25.2B",
        "Q4_K_M",
        "digest 149a0ff8a1ca",
        "num_predict 1200",
        "32768",
        "off (model can think)",
        "yes",
    ):
        assert expected in section, expected
    # num_predict left to the host is said to be the host's value, not "None".
    assert "num_predict is left to the host: 1200" in section


def test_a_report_without_identity_says_so() -> None:
    section = render_markdown(_doc("run", None)).split("## Model under test")[1]
    assert "not recorded" in section.split("## Headline")[0]


def test_a_comparison_of_different_weights_is_loudly_not_the_same_setup(host: None) -> None:
    a = model_identity("ollama", "minerva", HOST)
    b = dict(a or {})
    b["digest"] = "abcabcabcabc"
    b["host_defaults"] = {**(a or {})["host_defaults"], "num_predict": 4000}
    text = compare(_doc("old", a), _doc("new", b))
    assert "NOT THE SAME MODEL SETUP" in text
    assert "| digest | 149a0ff8a1ca | abcabcabcabc |" in text and "num_predict" in text
    assert "NOT THE SAME MODEL SETUP" not in compare(_doc("old", a), _doc("new", a))


def test_a_comparison_without_identity_cannot_claim_the_same_weights(host: None) -> None:
    a = model_identity("ollama", "minerva", HOST)
    assert "Model identity not recorded" in compare(_doc("old", None), _doc("new", a))


def test_a_changed_run_configuration_is_listed(host: None) -> None:
    a = model_identity("ollama", "minerva", HOST)
    text = compare(_doc("old", a, seed=7, routing=True), _doc("new", a, seed=8, routing=False))
    assert "The run configuration differs" in text
    assert "| seed | 7 | 8 |" in text and "| routing | True | False |" in text
    assert "The run configuration differs" not in compare(_doc("o", a), _doc("n", a))


def test_cli_exposes_the_helpers() -> None:
    assert cli.identity_unchanged is identity_unchanged
