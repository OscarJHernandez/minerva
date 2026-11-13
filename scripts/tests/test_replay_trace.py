import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

SPEC = importlib.util.spec_from_file_location(
    "replay_trace", Path(__file__).resolve().parents[1] / "replay_trace.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["replay_trace"] = MODULE  # dataclasses look their module up here
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def report(trace: list[dict[str, Any]], fence: str = "```") -> str:
    data = {
        "format": 1,
        "messages": [
            {"role": "user", "text": "q", "trace": []},
            {"role": "assistant", "text": "a", "trace": trace},
        ],
        "last_error": None,
    }
    body = f"# report\n\n{MODULE.JSON_START}\n{fence}json\n{json.dumps(data)}\n{fence}\n{MODULE.JSON_END}\n"
    return body


def step(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    out = execute(name, arguments, ctx)
    return {"name": name, "arguments": arguments, "for_model": out.for_model, "status": out.status}


def test_report_of_a_real_call_replays_without_differences(ctx: ToolContext) -> None:
    trace = [step(ctx, "simplify", {"latex": r"\frac{x^2-1}{x-1}"})]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert len(items) == 1 and items[0].turn == 2 and items[0].same
    assert "DIFFERENT" not in MODULE.render(items)


def test_changed_result_is_reported(ctx: ToolContext) -> None:
    recorded = step(ctx, "simplify", {"latex": r"\frac{x^2-1}{x-1}"})
    recorded["for_model"] = "status=ok; result=$something old$"
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert not items[0].same
    assert "DIFFERENT" in MODULE.render(items) and "now:" in MODULE.render(items)


def test_bare_function_names_are_visible_in_the_replay(ctx: ToolContext) -> None:
    """The case that motivated the report: `cosx` without a backslash is read as letters."""
    trace = [step(ctx, "simplify", {"latex": "e^{cosx}"})]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert items[0].same
    assert r"\cos" not in items[0].recorded  # whatever the engine did, the report shows it


def test_longer_fence_and_failed_turn_trace(ctx: ToolContext) -> None:
    data = MODULE.extract_data(report([], fence="````"))
    data["last_error"] = {"trace": [step(ctx, "get_entry", {"id": "nope"})]}
    items = MODULE.replay(data, ctx)
    assert [i.name for i in items] == ["get_entry"] and items[0].recorded_status == "error"


@pytest.mark.parametrize(
    "text",
    ["nothing here", f"{MODULE.JSON_START}\nnot fenced\n{MODULE.JSON_END}"],
)
def test_malformed_reports_are_rejected(text: str) -> None:
    with pytest.raises(MODULE.ReportError):
        MODULE.extract_data(text)


def test_unknown_format_is_rejected() -> None:
    text = f'{MODULE.JSON_START}\n```json\n{{"format": 99}}\n```\n{MODULE.JSON_END}'
    with pytest.raises(MODULE.ReportError, match="format"):
        MODULE.extract_data(text)


GEOMETRY_ARGS = {
    "coordinates": ["theta", "phi"],
    "metric": [["a^2", "0"], ["0", "a^2 \\sin^2\\theta"]],
    "assumptions": {"a": ["positive"]},
    "quantities": ["christoffel", "ricci_scalar"],
}


def test_a_geometry_call_replays_exactly_from_its_arguments(ctx: ToolContext) -> None:
    # The geometry travels in the arguments (nothing is stored server-side), and the text the model read
    # has no timings in it, so a recorded call reproduces byte for byte.
    trace = [step(ctx, "geometry", GEOMETRY_ARGS)]
    assert "status=ok" in trace[0]["for_model"] and "ms" not in trace[0]["for_model"]
    items = MODULE.replay(MODULE.extract_data(report(trace)), ctx)
    assert len(items) == 1 and items[0].name == "geometry" and items[0].same


def test_a_geometry_call_whose_result_changed_is_reported(ctx: ToolContext) -> None:
    recorded = step(ctx, "geometry", GEOMETRY_ARGS)
    recorded["for_model"] = recorded["for_model"].replace("2}{a^{2}}", "3}{a^{2}}")
    items = MODULE.replay(MODULE.extract_data(report([recorded])), ctx)
    assert not items[0].same and "DIFFERENT" in MODULE.render(items)
