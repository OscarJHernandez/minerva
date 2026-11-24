"""A coordinate named like the time symbol is refused with a typed, recoverable diagnostic (#161).

Regression: `analyze_lagrangian` on the Schwarzschild geodesic Lagrangian in `(t, r, theta, phi)` answered
`not_verified` with the default `time: t` and `ok` with `time: s`, and the model had no typed reason to repair
the call. Every tool that takes both `coordinates` and `time` now refuses the clash with the code
`time_symbol_is_coordinate`; nothing is renamed, and the corrected call returns its verified result.
"""

import json
from typing import Any

import pytest
from minerva_api.agent.recovery import canonical_key
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_api.agent.tracing import REASON_CODES
from minerva_engine import Engine
from minerva_engine.lagrangian import analyze_lagrangian
from minerva_engine.models import Status
from minerva_engine.notation.models import InputCode
from minerva_engine.notation.time_symbol import CANDIDATES, time_clash
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

TWO = r"\frac12 m \dot{t}^2 + \frac12 m \dot{r}^2 - \frac12 k r^2"
ONE = r"\frac12 m \dot{t}^2 - \frac12 k t^2 - \frac14 \lambda t^4"
SCHWARZSCHILD = (
    "1/2*(-(1-2*M/r)*t_dot^2 + r_dot^2/(1-2*M/r) + r^2*theta_dot^2 + r^2*sin(theta)^2*phi_dot^2)"
)

# tool -> arguments with a coordinate `t` and the default time symbol.
CLASH: dict[str, dict[str, Any]] = {
    "analyze_lagrangian": {"lagrangian": TWO, "coordinates": ["t", "r"]},
    "find_conserved_quantities": {"lagrangian": TWO, "coordinates": ["t", "r"]},
    "search_invariants": {"lagrangian": TWO, "coordinates": ["t", "r"]},
    "expand_motion": {"lagrangian": TWO, "coordinates": ["t", "r"]},
    "perturbation_solve": {
        "lagrangian": ONE,
        "coordinates": ["t"],
        "small_parameter": "lambda",
        "assumptions": {"m": ["positive"], "k": ["positive"], "lambda": ["positive"]},
    },
    "time_derivative": {"latex": r"m \dot{t}", "coordinates": ["t"]},
    "routh_reduce": {"lagrangian": TWO, "coordinates": ["t", "r"], "cyclic": ["t"]},
    "integrate_motion": {
        "lagrangian": TWO,
        "coordinates": ["t", "r"],
        "parameters": {"m": 1.0, "k": 1.0},
        "initial_positions": [0.0, 1.0],
        "initial_velocities": [1.0, 0.0],
        "t_end": 2.0,
    },
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def test_every_tool_with_a_time_field_is_covered() -> None:
    with_both = {
        name
        for name, tool in TOOLS.items()
        if {"time", "coordinates"} <= set(tool.args_model.model_fields)
    }
    assert with_both == set(CLASH)


@pytest.mark.parametrize("tool", sorted(CLASH))
def test_the_clash_is_a_typed_recoverable_refusal_and_the_fix_runs(
    tool: str, ctx: ToolContext
) -> None:
    out = execute(tool, CLASH[tool], ctx)
    assert out.status == "parse_error", out.for_model[:300]
    assert out.reason_code == "time_symbol_is_coordinate"
    assert "time: s" in out.for_model and "not a coordinate" in out.for_model
    (line,) = [x for x in out.for_model.splitlines() if x.startswith("diagnostic: ")]
    diagnostic = json.loads(line.removeprefix("diagnostic: "))
    assert diagnostic["code"] == "time_symbol_is_coordinate"
    assert diagnostic["field"] == "time" and diagnostic["recoverable"] is True
    assert diagnostic["symbols"] == ["t"] and diagnostic["suggestions"][0] == "s"
    assert not out.known_latex  # nothing was computed

    fixed = {**CLASH[tool], "time": "s"}
    assert canonical_key(tool, fixed) != canonical_key(tool, CLASH[tool])
    ok = execute(tool, fixed, ctx)
    assert ok.status == "ok", ok.for_model[:400]
    assert ok.reason_code is None


def test_schwarzschild_with_the_default_time_is_refused_and_with_s_verified(
    ctx: ToolContext,
) -> None:
    arguments = {
        "lagrangian": SCHWARZSCHILD,
        "coordinates": ["t", "r", "theta", "phi"],
        "format": "text",
    }
    refused = execute("analyze_lagrangian", arguments, ctx)
    assert refused.status == "parse_error" and refused.reason_code == "time_symbol_is_coordinate"
    ok = execute("analyze_lagrangian", {**arguments, "time": "s"}, ctx)
    assert ok.status == "ok", ok.for_model[:300]
    assert "p_{t}" in ok.for_model and "p_{\\phi}" in ok.for_model


def test_a_hamiltonian_with_a_momentum_named_like_the_time_symbol_is_refused(
    ctx: ToolContext,
) -> None:
    out = execute(
        "find_conserved_quantities",
        {
            "hamiltonian": "p_x^2/2 + x^2/2",
            "coordinates": ["x"],
            "momenta": ["t"],
            "format": "text",
        },
        ctx,
    )
    assert out.reason_code == "time_symbol_is_coordinate"


def test_a_system_without_the_clash_is_unchanged(ctx: ToolContext) -> None:
    out = execute(
        "analyze_lagrangian",
        {"lagrangian": r"\frac12 m \dot{x}^2 - \frac12 k x^2", "coordinates": ["x"]},
        ctx,
    )
    assert out.status == "ok" and out.reason_code is None


def test_the_engine_refuses_a_direct_call_and_keeps_the_user_names() -> None:
    report = analyze_lagrangian(TWO, ["t", "r"], "t", {}, budget_limits())
    assert report.status is Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.TIME_SYMBOL_IS_COORDINATE
    assert report.diagnostic.symbols == ["t"] and report.diagnostic.recoverable
    assert not report.equations and not report.momenta


def test_suggestions_skip_names_in_use_and_are_deterministic() -> None:
    first = time_clash(["t", "s"], "t")
    assert first is not None and first.suggestions[0] == "tau"
    in_source = time_clash(["t"], "t", source=r"\frac12 s \dot{t}^2")
    assert in_source is not None and "s" not in in_source.suggestions
    assert in_source == time_clash(["t"], "t", source=r"\frac12 s \dot{t}^2")
    assert time_clash(["x", "y"], "t") is None
    exhausted = time_clash([*CANDIDATES, "t"], "t")
    assert exhausted is not None and exhausted.suggestions == []


def test_the_code_is_in_the_log_vocabulary() -> None:
    assert InputCode.TIME_SYMBOL_IS_COORDINATE.value in REASON_CODES
