"""scripts/diagnostic_codes.py counts the closed fields of the tool outcome log lines and nothing else."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "diagnostic_codes", Path(__file__).resolve().parents[1] / "diagnostic_codes.py"
)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def line(**fields: object) -> str:
    return json.dumps({"timestamp": "2026-11-16T10:00:00Z", **fields})


LOG = [
    line(
        event="tool.failed",
        tool="simplify",
        reason_code="text_syntax_in_latex",
        error_category="parse_error",
    ),
    line(
        event="tool.failed",
        tool="simplify",
        reason_code="text_syntax_in_latex",
        error_category="parse_error",
    ),
    line(event="tool.declined", tool="differentiate", error_category="unsupported"),
    line(event="agent.completed", tools_called=["simplify"]),
    "plain text that is not json",
    "{broken json",
    line(
        event="tool.failed",
        tool="analyze_lagrangian",
        reason_code="undeclared_dependent",
        error_category="parse_error",
    ),
]


def test_failures_and_declines_are_counted_and_everything_else_is_ignored() -> None:
    counts = tool.count(LOG)
    assert counts[("tool.failed", "simplify", "text_syntax_in_latex", "parse_error")] == 2
    assert counts[("tool.declined", "differentiate", "-", "unsupported")] == 1
    assert sum(counts.values()) == 4


def test_the_most_frequent_combination_comes_first() -> None:
    rendered = tool.render(tool.count(LOG), 2)
    first = rendered.splitlines()[1]
    assert "text_syntax_in_latex" in first and first.strip().startswith("2")


def test_since_ignores_earlier_lines() -> None:
    assert sum(tool.count(LOG, since="2026-12-01").values()) == 0


def test_an_empty_log_says_so() -> None:
    assert "no tool failures" in tool.render(tool.count([]), 5)
