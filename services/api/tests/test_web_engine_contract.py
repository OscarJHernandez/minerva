"""Contracts between what the engine reports and what the browser can display or must split on.

A field the engine returns but the browser type does not declare is silently dropped by the panel (this
happened to the on-motion conclusion of a total derivative). These tests fail instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from minerva_api.agent import refs
from minerva_engine.differentiate import DifferentiateResult

ROOT = Path(__file__).resolve().parents[3]
TYPES_TS = ROOT / "apps/web/src/lib/types.ts"
PROVENANCE = ROOT / "packages/ui/fixtures/input-provenance.json"

# Fields of the engine result the derivative panel's conclusion depends on: each must be declared.
DISPLAY_CRITICAL = {
    "on_motion_latex",
    "zero_on_motion",
    "symbolic_zero_on_motion",
    "definitions_used",
}
# Engine fields the browser deliberately does not declare (typed input diagnostics are shown elsewhere).
NOT_DISPLAYED = {"diagnostic", "reading"}


def _interface_fields(name: str) -> set[str]:
    source = TYPES_TS.read_text()
    start = source.index(f"export interface {name} {{")
    body = source[start : source.index("\n}\n", start)]
    # Top-level members only: two-space indentation.
    return set(re.findall(r"^  ([a-z_]+)\??:", body, flags=re.MULTILINE))


def test_the_browser_type_declares_every_display_critical_derivative_field() -> None:
    declared = _interface_fields("DifferentiateResult")
    assert set(DifferentiateResult.model_fields) >= DISPLAY_CRITICAL  # the list itself stays true
    assert declared >= DISPLAY_CRITICAL, DISPLAY_CRITICAL - declared


def test_every_engine_derivative_field_is_declared_in_the_browser_or_listed_as_not_displayed() -> (
    None
):
    declared = _interface_fields("DifferentiateResult")
    missing = set(DifferentiateResult.model_fields) - declared - NOT_DISPLAYED
    assert not missing, f"engine fields the browser type does not declare: {sorted(missing)}"


def test_the_component_suffix_the_web_app_splits_on_is_the_one_refs_writes() -> None:
    fixture = json.loads(PROVENANCE.read_text())
    assert fixture["component_suffix"] == refs.COMPONENT_SUFFIX
    # The fixture's examples are what the note builder separates, in the same way.
    note = refs.unverified_note(
        [fixture["examples"]["model_written"], fixture["examples"]["list_component"]]
    )
    assert "Input not verified" in note and "Input taken from a list" in note
    assert fixture["examples"]["list_component"].endswith(refs.COMPONENT_SUFFIX)
    assert not fixture["examples"]["model_written"].endswith(refs.COMPONENT_SUFFIX)


def test_the_no_result_id_class_is_shared_and_has_its_own_sentence() -> None:
    fixture = json.loads(PROVENANCE.read_text())
    assert fixture["no_result_id_suffix"] == refs.NO_ID_SUFFIX
    example = fixture["examples"]["no_result_id"]
    assert example.endswith(refs.NO_ID_SUFFIX)
    assert not example.endswith(refs.COMPONENT_SUFFIX)
    assert refs.NO_ID_SUFFIX != refs.COMPONENT_SUFFIX
    everything = [
        fixture["examples"]["model_written"],
        fixture["examples"]["list_component"],
        example,
    ]
    note = refs.unverified_note(everything)
    # three separate sentences, each naming only its own entry
    sentences = {
        "Input not verified": fixture["examples"]["model_written"],
        "Input taken from a list": fixture["examples"]["list_component"],
        "Input without a result id": example,
    }
    for head in sentences:
        assert note.count(head) == 1
    for head, entry in sentences.items():
        start = note.index(head)
        ends = [note.index(h) for h in sentences if note.index(h) > start]
        chunk = note[start : min(ends)] if ends else note[start:]
        assert entry in chunk
        assert all(other not in chunk for h, other in sentences.items() if h != head)


def test_the_premise_class_is_shared_and_has_its_own_sentence() -> None:
    fixture = json.loads(PROVENANCE.read_text())
    assert fixture["premise_suffix"] == refs.PREMISE_SUFFIX
    example = fixture["examples"]["premise"]
    assert example.endswith(refs.PREMISE_SUFFIX)
    assert not example.endswith((refs.COMPONENT_SUFFIX, refs.NO_ID_SUFFIX))
    note = refs.unverified_note([fixture["examples"]["model_written"], example])
    assert note.count("Input given, not proved") == 1 and note.count("Input not verified") == 1
    # the premise is never described as written by the model
    chunk = note[note.index("Input given, not proved") :]
    assert "model wrote" not in chunk
