"""Equivalent spellings of one Lagrangian give the same result.

Regression: `theta_dot^2 + sin(theta)^2*phi_dot^2` was read as LaTeX (a product of letters) and the analysis
said the Lagrangian had no velocities; the model had to retry with `\\dot{\\theta}`.
"""

import pytest
from minerva_api.agent.tools import (
    AnalyzeLagrangianArgs,
    ToolContext,
    resolve_syntax,
    run_lagrangian,
    strip_assignment,
)
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

LATEX = r"\frac12 m R^2 (\dot{\theta}^2 + \sin^2(\theta) \dot{\phi}^2) - m g R \cos(\theta)"
SPELLINGS = {
    "latex": LATEX,
    "latex_assignment": "L = " + LATEX,
    "text": "1/2*m*R^2*(theta_dot^2 + sin(theta)^2*phi_dot^2) - m*g*R*cos(theta)",
    "text_assignment": "L = 1/2*m*R**2*(theta_dot**2 + sin(theta)**2*phi_dot**2) - m*g*R*cos(theta)",
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def _run(ctx: ToolContext, source: str, **extra: str):  # type: ignore[no-untyped-def]
    return run_lagrangian(
        AnalyzeLagrangianArgs(lagrangian=source, coordinates=["theta", "phi"], **extra), ctx
    )


def test_every_spelling_gives_the_same_verified_result(ctx: ToolContext) -> None:
    outputs = {name: _run(ctx, src) for name, src in SPELLINGS.items()}
    for name, out in outputs.items():
        assert out.status == "ok", (name, out.for_model[:300])
    reference = outputs["latex"]
    for name, out in outputs.items():
        assert sorted(out.known_latex) == sorted(reference.known_latex), name


def test_wrong_explicit_format_is_a_clear_parse_error_not_a_misreading(ctx: ToolContext) -> None:
    out = _run(ctx, SPELLINGS["text"], format="latex")
    assert out.status == "parse_error"
    assert "write" in out.for_model


def test_a_misspelled_coordinate_is_named(ctx: ToolContext) -> None:
    out = run_lagrangian(AnalyzeLagrangianArgs(lagrangian=LATEX, coordinates=["theta", "psi"]), ctx)
    assert out.status == "parse_error"
    assert "psi" in out.for_model


def test_helpers() -> None:
    assert resolve_syntax(r"\dot{x}^2", "auto") == "latex"
    assert resolve_syntax("x_dot^2/2", "auto") == "text"
    assert resolve_syntax("1/2*m*x_dot^2", "auto") == "text"
    assert resolve_syntax("x_dot^2", "latex") == "latex"
    assert strip_assignment("L = T - V", "L") == "T - V"
    assert strip_assignment("L = T = 1", "L") == "L = T = 1"
    assert strip_assignment("F = ma", "L") == "F = ma"


def test_every_main_expression_field_says_which_syntax_it_takes() -> None:
    """A model that is not told the syntax guesses it, and a wrong guess costs a failed call."""
    import re

    from minerva_api.agent.tools import TOOLS

    main = {"latex", "expression", "equation", "lagrangian", "hamiltonian", "integrand"}
    missing = [
        f"{name}.{field}"
        for name, tool in TOOLS.items()
        for field, info in tool.args_model.model_fields.items()
        if field in main and not re.search(r"latex|text|syntax", info.description or "", re.I)
    ]
    assert missing == []
