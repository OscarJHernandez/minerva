"""The capability record must match what the engine does today.

If this fails, behaviour changed (or docs/capabilities.yaml did) without regenerating the record:
run `uv run python scripts/capabilities.py` and review the diff in docs/capabilities.md.
"""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "capabilities", Path(__file__).resolve().parents[1] / "capabilities.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_generated_files_are_up_to_date() -> None:
    assert MODULE.main(["--check"]) == 0


def test_planned_capabilities_have_no_examples_and_refusals_are_recorded() -> None:
    data = MODULE.build()
    for cap in data["capabilities"]:
        if cap["status"] == "planned":
            assert cap["examples"] == []
    statuses = {ex["status"] for cap in data["capabilities"] for ex in cap["examples"]}
    # What Minerva declines to do is part of what it does: at least one refusal of each kind.
    assert {"parse_error", "unsupported", "conditions_not_satisfied", "unchanged"} <= statuses
