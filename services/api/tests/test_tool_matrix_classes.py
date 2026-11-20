"""Every tool answers one question before it ships: what does a declared matrix do to it (A-009).

`check_identity`, `rewrite_part`, `substitute`, `sums` and the Lagrangian tools each read `A` and `B` as commuting numbers
and showed a wrong result as verified, because nobody had asked the question of them. The classification is therefore
a registry: a new tool without a class fails here, so the answer cannot be forgotten, and a class cannot go stale.
"""

import pytest
from minerva_api.agent.tools import (
    MATRIX_ONLY_TOOLS,
    MATRIX_SAFE,
    NONCOMMUTATIVE_AWARE_TOOLS,
    SCALAR_TOOLS,
    TOOLS,
    ToolContext,
    execute,
)
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

CLASSES = {
    "scalar (refused for a declared matrix or vector)": set(SCALAR_TOOLS),
    "noncommutative-aware (refused unless the call declares it)": set(NONCOMMUTATIVE_AWARE_TOOLS),
    "matrix-only (refused for a declared matrix, not a vector)": set(MATRIX_ONLY_TOOLS),
    "matrix-safe (with a stated reason)": set(MATRIX_SAFE),
}


def test_every_tool_is_in_exactly_one_matrix_class() -> None:
    seen: dict[str, str] = {}
    for label, names in CLASSES.items():
        for name in names:
            assert name not in seen, f"{name} is in two classes: {seen[name]} and {label}"
            seen[name] = label
    missing = sorted(set(TOOLS) - set(seen))
    assert not missing, (
        f"{missing} has no class: say how a declared matrix affects it (tools.py: SCALAR_TOOLS, "
        "NONCOMMUTATIVE_AWARE_TOOLS, MATRIX_ONLY_TOOLS, or MATRIX_SAFE with a reason)"
    )


def test_no_class_names_a_tool_that_does_not_exist() -> None:
    stale = sorted({n for names in CLASSES.values() for n in names} - set(TOOLS))
    assert not stale, f"{stale} is classified but is not a tool any more"


def test_a_safe_tool_states_a_reason() -> None:
    for name, reason in MATRIX_SAFE.items():
        assert len(reason) >= 25, (
            f"{name}: say why a declared matrix cannot make it show a wrong result as verified"
        )


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(budget_limits(max_concurrency=2)),
        kb=KnowledgeBase.load(),
        matrices=frozenset({"A", "B"}),
    )


# What "safe" means, run for the safe tools that take an expression: with `A` and `B` declared as matrices they never
# present sampled equality of commuting numbers as a result about them.
SAFE_CALLS = [
    (
        "check_units",
        {
            "equation": "A*B = B*A",
            "quantities": [{"name": "A", "unit": "m"}, {"name": "B", "unit": "m"}],
        },
    ),
    (
        "dimensional_analysis",
        {"quantities": [{"name": "A", "unit": "m"}, {"name": "B", "unit": "s"}], "unknown": "A"},
    ),
    (
        "tensor_canonicalize",
        {
            "expression": "A(a,b)*B(-a,-b) - A(b,a)*B(-a,-b)",
            "tensors": {"A": {"rank": 2}, "B": {"rank": 2}},
            "dimension": 3,
        },
    ),
]


@pytest.mark.parametrize(("tool", "arguments"), SAFE_CALLS, ids=[c[0] for c in SAFE_CALLS])
def test_a_safe_tool_never_shows_sampled_equality_for_a_declared_matrix(
    ctx: ToolContext, tool: str, arguments: dict[str, object]
) -> None:
    assert tool in MATRIX_SAFE
    out = execute(tool, arguments, ctx)
    text = out.for_model.lower()
    for claim in ("symbolic_zero=true", "checked numerically", "proved_equal", "numerically_equal"):
        assert claim not in text, (
            f"{tool} claims `{claim}` about a declared matrix: {out.for_model[:200]}"
        )
