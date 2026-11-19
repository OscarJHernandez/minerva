"""The site's Benchmarks page shows what the committed reports say, and nothing it was not allowed to say.

If `test_the_generated_file_is_up_to_date` fails, a report in docs/benchmarks changed (or the generator did) without
regenerating the site data: run `uv run python scripts/benchmarks_site.py` and review the diff.
"""

import copy
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks_site.py"
SPEC = importlib.util.spec_from_file_location("benchmarks_site", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    return MODULE.build()  # type: ignore[no-any-return]


def test_the_generated_file_is_up_to_date() -> None:
    # compared as data: a formatter may lay the file out differently without changing a number
    assert json.loads(MODULE.OUT.read_text(encoding="utf-8")) == json.loads(MODULE.render())
    assert MODULE.main(["--check"]) == 0


def test_every_suite_adds_up_and_links_to_things_that_exist(data: dict[str, Any]) -> None:
    assert data["suites"]
    for s in data["suites"]:
        assert s["correct"] + s["refused"] + s["rejected"] + s["pending"] == s["total"], s["id"]
        assert (ROOT / s["doc"]).exists(), s["doc"]
        assert all((ROOT / "docs" / "benchmarks" / f).exists() for f in s["sources"])
        assert s["command"].startswith("uv run "), s["id"]


def test_the_known_gaps_are_shown_not_dropped(data: dict[str, Any]) -> None:
    # the reference suite has a pending problem and the verification corpus a known open weakness
    assert next(s for s in data["suites"] if s["id"] == "reference")["pending"] >= 1
    assert data["verification"]["known_open"] >= 1


def test_the_chat_numbers_never_appear_without_the_model(data: dict[str, Any]) -> None:
    for row in data["chat"]["rows"]:
        m = row["model"]
        assert m["digest"] and m["base_model"] and m["parameter_size"] and m["quantization"]
        assert isinstance(m["thinking_requested"], bool)
        assert m["reply_limit"]
        assert row["run"]["cases_digest"] and row["run"]["code"]


def test_targets_are_measured_never_adjusted(data: dict[str, Any]) -> None:
    by = {t["id"]: t for t in data["chat"]["targets"]}
    assert by["pass_rate"]["target"] == 0.85 and by["tool_selection"]["target"] == 0.95
    assert by["valid_arguments"]["target"] == 0.99 and by["completion"]["target"] == 0.95
    assert by["false_verified"]["target"] == 0
    for key, t in by.items():
        ok = (
            t["measured"] <= t["target"]
            if key == "false_verified"
            else t["measured"] >= t["target"]
        )
        assert t["met"] is ok, key


def test_no_address_or_credential_is_written(data: dict[str, Any]) -> None:
    text = json.dumps(data)
    assert not re.search(r"https?://", text)
    assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text)
    assert not re.search(r"\.(local|lan|internal)\b", text)
    assert "api_key" not in text.lower() and "base_url" not in text.lower()


# ---------------------------------------------------------------------------- wrong twins: refuse to publish a lie
def patched(monkeypatch: pytest.MonkeyPatch, name: str, change: Any) -> None:
    real = MODULE.load

    def load(report: str) -> dict[str, Any]:
        doc = copy.deepcopy(real(report))
        if report == name:
            change(doc)
        return doc  # type: ignore[no-any-return]

    monkeypatch.setattr(MODULE, "load", load)


def test_a_failing_case_in_a_report_stops_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    patched(monkeypatch, "statistics", lambda d: d["cases"][0].update(passed=False))
    with pytest.raises(SystemExit, match="failing case"):
        MODULE.statistics()


def test_counts_that_do_not_add_up_stop_the_build() -> None:
    with pytest.raises(SystemExit, match="do not add up"):
        MODULE.outcome(10, correct=5, refused=4)


def test_a_lost_refusal_stops_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    patched(
        monkeypatch,
        "geometry",
        lambda d: d["deterministic"]["coverage"].update(refusals_as_expected=7),
    )
    with pytest.raises(SystemExit, match="refusal was lost"):
        MODULE.geometry()


def test_a_wrong_verdict_on_the_checker_stops_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    patched(
        monkeypatch,
        "verification-summary",
        lambda d: d["deterministic"]["totals"].update(false_accepts=1),
    )
    with pytest.raises(SystemExit, match="wrong verdict"):
        MODULE.verification()


def test_a_chat_report_without_the_models_identity_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def drop(d: dict[str, Any]) -> None:
        for entry in d["models"].values():
            entry["identity"] = None

    patched(monkeypatch, "baseline-2026-11-19-all", drop)
    with pytest.raises(SystemExit, match="identity"):
        MODULE.chat_row("all", "baseline-2026-11-19-all", "x")


def test_an_unknown_outcome_is_not_guessed(monkeypatch: pytest.MonkeyPatch) -> None:
    patched(monkeypatch, "reference", lambda d: d["problems"][0].update(outcome="mostly_right"))
    with pytest.raises(SystemExit, match="unknown outcome"):
        MODULE.problems("reference")()
