"""The capability record must match what the engine does today.

If this fails, behaviour changed (or docs/capabilities.yaml did) without regenerating the record:
run `uv run python scripts/capabilities.py` and review the diff in docs/capabilities.md.
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "capabilities", Path(__file__).resolve().parents[1] / "capabilities.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    # Every example runs through the engine (minutes): built once for the module, not once per test.
    built: dict[str, Any] = MODULE.build()
    return built


def test_generated_files_are_up_to_date(data: dict[str, Any]) -> None:
    assert MODULE.stale_outputs(data) == [], "run: uv run python scripts/capabilities.py"


def test_planned_capabilities_have_no_examples_and_refusals_are_recorded(
    data: dict[str, Any],
) -> None:
    for cap in data["capabilities"]:
        if cap["status"] == "planned":
            assert cap["examples"] == []
    statuses = {ex["status"] for cap in data["capabilities"] for ex in cap["examples"]}
    # What Minerva declines to do is part of what it does: at least one refusal of each kind.
    assert {"parse_error", "unsupported", "conditions_not_satisfied", "unchanged"} <= statuses


def test_every_capability_has_a_known_group_and_no_group_is_empty(data: dict[str, Any]) -> None:
    group_ids = [g["id"] for g in data["groups"]]
    assert len(set(group_ids)) == len(group_ids)
    used = {cap["group"] for cap in data["capabilities"]}
    assert used <= set(group_ids), "a capability names a group that is not declared"
    assert set(group_ids) <= used, "a declared group has no capability"
    # Group ids become page anchors next to capability ids, so they must not collide.
    assert not set(group_ids) & {cap["id"] for cap in data["capabilities"]}


def test_formula_image_examples_use_the_strict_parser(data: dict[str, Any]) -> None:
    cap = next(c for c in data["capabilities"] if c["id"] == "formula-image")
    statuses = {ex["id"]: ex["status"] for ex in cap["examples"]}
    assert statuses == {"image-read-clean": "ok", "image-read-ambiguous": "parse_error"}
