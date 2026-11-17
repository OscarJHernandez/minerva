"""The evaluation matrix names every registered tool, and only registered tools."""

import re
from pathlib import Path

from minerva_api.agent.tools import TOOLS

MATRIX = Path(__file__).resolve().parents[3] / "docs" / "reliability" / "evaluation-matrix.md"


def _rows() -> set[str]:
    return set(re.findall(r"^\| `([a-z_]+)` \|", MATRIX.read_text(encoding="utf-8"), re.M))


def test_every_registered_tool_has_a_row() -> None:
    assert set(TOOLS) - _rows() == set()


def test_no_row_names_a_tool_that_does_not_exist() -> None:
    assert _rows() - set(TOOLS) == set()
