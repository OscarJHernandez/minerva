"""`@eN` of the wrong kind for a tool field is refused with a typed diagnostic, never rewritten (issue #133, part 3).

A numbered equation is a relation (`y = x^2`) or an expression (`x^2`). A tool that reads an expression and gets
`@e3` of a relation used to fail in the parser ("unexpected '='"), or, embedded, read `\\left(x = ...\\right)`.
The table of (tool, field) is closed: a field is classified only where the real tool refuses (or requires) a
relation, and this file runs every entry against the real tool. A kind that is not known is never refused.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from minerva_api.agent import refs
from minerva_api.agent.card_equations import listed_equations
from minerva_api.agent.equations import Equation, EquationTable, build_table, relation_kind
from minerva_api.agent.events import Block, Card
from minerva_api.agent.mathinput import MATH_FIELDS
from minerva_api.agent.reference_types import FIELD_POLICY, policy_of, refusal_diagnostic
from minerva_api.agent.refs import ResultStore
from minerva_api.agent.replay import call, say
from minerva_api.agent.tools import TOOLS, GeometryArgs, ToolContext, execute, run_geometry
from minerva_api.agent.tracing import REASON_CODES
from minerva_engine import Engine
from minerva_engine.notation.models import InputCode
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_total_derivative_handles import Row, card_of, play


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


# --- the kind of a formula ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("latex", "kind"),
    [
        ("y = x^2", "equation"),
        (r"\Gamma^{\theta}_{\phi\phi} = -\sin\theta\cos\theta", "equation"),
        (r"\ddot{x} = - \frac{k x}{m}", "equation"),
        ("x^2 + 1", "expression"),
        (r"\sum_{n=1}^{N} a_n", "expression"),  # `=` inside a subscript is not a relation
        (r"f\left(a=1\right) + 2", "expression"),
        (r"\begin{pmatrix} a = 1 & b \\ c & d \end{pmatrix}", "expression"),
        (r"x_{n=1} y", "expression"),
        (r"\left\{ x = 1 \right\} + 2", "expression"),
        ("x \\leq 3", None),  # another relation: undecided, never an expression
        ("a < b", None),
        (r"a \\ b", None),  # a row separator at the top level
        ("a & b", None),
        ("(x + 1", None),  # unbalanced
        (r"\int_0^1 x\,dx", "expression"),  # `\in` must not match `\int`
        ("A := B", "equation"),
    ],
)
def test_the_kind_is_read_from_the_text_by_one_rule(latex: str, kind: str | None) -> None:
    assert relation_kind(latex) == kind


def test_a_numbered_formula_records_its_kind_and_a_rebuilt_table_agrees() -> None:
    table = EquationTable()
    a = table.number_for("y = x^2", "written_by_model", mode="model")
    b = table.number_for("x^2", "from_tool", mode="none")
    assert a is not None and b is not None
    assert (table.items[a].kind, table.items[b].kind) == ("equation", "expression")
    stored = Row(
        "assistant",
        {
            "blocks": [
                {
                    "kind": "text",
                    "segments": [
                        {"type": "display_math", "value": "y = x^2", "number": a, "status": ""},
                        {"type": "display_math", "value": "x^2", "number": b, "status": ""},
                    ],
                }
            ]
        },
    )
    rebuilt = build_table([stored])
    assert [rebuilt.items[n].kind for n in (a, b)] == ["equation", "expression"]


def test_the_rows_of_the_existing_list_cards_are_classified(ctx: ToolContext) -> None:
    sphere = GeometryArgs(
        coordinates=["theta", "phi"],
        metric=[["a^2", "0"], ["0", r"a^2 \sin^2\theta"]],
        assumptions={"a": ["positive"]},
        quantities=["christoffel", "kretschmann"],
    )
    out = run_geometry(sphere, ctx)
    kinds = {
        key: relation_kind(latex) for key, latex in listed_equations("geometry", out.card.data)
    }
    assert kinds["kretschmann"] == "expression"
    assert {k for k, v in kinds.items() if k.startswith("christoffel")} and all(
        v == "equation" for k, v in kinds.items() if k.startswith("christoffel")
    )
    report = {
        "equations": [{"latex": r"\ddot{x} = - x"}],
        "momenta": [{"latex": r"p_{x} = m \dot{x}"}],
        "conserved": [{"expression_latex": r"\frac{m \dot{x}^{2}}{2}"}],
    }
    rows = dict(listed_equations("lagrangian", {"report": report}))
    assert [relation_kind(rows[k]) for k in ("eq:0", "mom:0", "cons:0")] == [
        "equation",
        "equation",
        "expression",
    ]


# --- the table ---------------------------------------------------------------------------------------------
def test_every_tool_with_a_formula_field_is_classified_with_a_reason() -> None:
    missing = [
        (name, f)
        for name, tool in TOOLS.items()
        for f in tool.args_model.model_fields
        if f in MATH_FIELDS and (name, f) not in FIELD_POLICY
    ]
    assert not missing, f"classify (or mark unclassified, with a reason): {missing}"
    assert all(reason.strip() for _, reason in FIELD_POLICY.values())
    stale = [key for key in FIELD_POLICY if key[1] not in TOOLS[key[0]].args_model.model_fields]
    assert not stale  # an entry for a field the tool does not have


# A valid call of each classified tool, with the formula field left to the test.
BASE: dict[str, dict[str, Any]] = {
    "simplify": {},
    "differentiate": {"variables": [{"name": "x"}]},
    "series": {"variable": "x", "order": 3, "point": "0"},
    "integrate": {"variable": "x"},
    "distribution": {},
    "time_derivative": {"coordinates": ["x"], "latex": "x"},
    "find_identities": {},
    "sums": {},
    "expectation": {"random": ["X"]},
    "creative_telescoping": {"variable": "x", "parameter": "a"},
    "approximation": {},
    "substitute": {"relations": ["x = 2"]},
    "rewrite_part": {"action": "expand", "path": []},
    "variation": {"functions": ["y"], "action": "first"},
    "solve_polynomial": {},
    "isolate_variable": {"variable": "x"},
    "check_identity": {},
    "check_units": {"quantities": [{"name": "x", "unit": "m"}, {"name": "y", "unit": "m^2"}]},
    "expand_motion": {"coordinates": ["x"]},
    "perturbation_solve": {"coordinates": ["x"]},
    "integrate_motion": {
        "coordinates": ["x"],
        "initial_positions": [1],
        "initial_velocities": [0],
        "t_end": 1,
    },
    "tensor_canonicalize": {
        "tensors": {
            "S": {"rank": 2, "symmetry": "symmetric"},
            "A": {"rank": 2, "symmetry": "none"},
        }
    },
}
FIELD_OF = {
    "variation": "integrand",
    "solve_polynomial": "equation",
    "isolate_variable": "equation",
    "check_units": "equation",
    "tensor_canonicalize": "expression",
}
# Tools with a formula field that is not the one `FIELD_OF` names, and the text each one reads for it.
OTHER_FIELDS: dict[tuple[str, str], tuple[str, str]] = {
    # (tool, field) -> (an expression the field accepts, a relation of the field's own syntax)
    ("time_derivative", "lagrangian"): (r"\dot{x}^2", r"y = \dot{x}^2"),
    ("expand_motion", "lagrangian"): (r"\dot{x}^2 - x^2", r"y = \dot{x}^2 - x^2"),
    ("perturbation_solve", "lagrangian"): (
        r"\frac{1}{2}\dot{x}^2 - \frac{1}{2}x^2",
        r"y = \dot{x}^2",
    ),
    ("integrate_motion", "lagrangian"): (
        r"\frac{1}{2}\dot{x}^2 - \frac{1}{2}x^2",
        r"y = \dot{x}^2",
    ),
    ("tensor_canonicalize", "expression"): ("S(a,b)*A(-a,-b)", "S(a,b) = A(a,b)"),
}


def _field(tool: str) -> str:
    return FIELD_OF.get(tool, "latex")


def _keys(policy: str) -> list[tuple[str, str]]:
    return sorted(k for k, (p, _) in FIELD_POLICY.items() if p == policy)


def _kinds(tool: str, field: str) -> tuple[str, str]:
    """(an expression, a relation) in the syntax this field reads."""
    return OTHER_FIELDS.get((tool, field), ("x^2", "y = x^2"))


def test_the_classified_tools_are_all_covered_by_the_runs_below() -> None:
    classified = {t for (t, _), (p, _) in FIELD_POLICY.items() if p != "unclassified"}
    assert classified == set(BASE)
    # every classified field is either the tool's main one or listed with its own examples
    for (tool, field), (policy, _) in FIELD_POLICY.items():
        assert policy == "unclassified" or field == _field(tool) or (tool, field) in OTHER_FIELDS


@pytest.mark.parametrize(("tool", "field"), _keys("expression_only"))
def test_a_field_marked_expression_only_really_refuses_a_relation(
    tool: str, field: str, ctx: ToolContext
) -> None:
    expression, relation = _kinds(tool, field)
    out = execute(tool, {**BASE[tool], field: relation}, ctx)
    assert out.status == "parse_error", (tool, field, out.status, out.for_model[:200])
    if (
        tool,
        field,
    ) in OTHER_FIELDS:  # the accepting twin of the fields classified with their own examples
        ok = execute(tool, {**BASE[tool], field: expression}, ctx)
        assert ok.status in ("ok", "unchanged"), (tool, field, ok.status, ok.for_model[:200])


@pytest.mark.parametrize(("tool", "field"), _keys("equation_required"))
def test_a_field_marked_equation_required_really_refuses_an_expression(
    tool: str, field: str, ctx: ToolContext
) -> None:
    out = execute(tool, {**BASE[tool], field: "x^2"}, ctx)
    assert out.status == "parse_error", (tool, field, out.status, out.for_model[:200])


@pytest.mark.parametrize(
    ("tool", "field"),
    [
        ("time_derivative", "lagrangian"),
        ("expand_motion", "lagrangian"),
        ("perturbation_solve", "lagrangian"),
        ("integrate_motion", "lagrangian"),
        ("tensor_canonicalize", "expression"),
    ],
)
def test_a_relation_reference_is_refused_in_the_newly_classified_fields(
    tool: str, field: str
) -> None:
    store = _store(("y = x^2", "from_tool"), ("x^2", "from_tool"))
    refused = store.resolve({field: "@e1"}, tool)
    assert refused.diagnostic is not None
    assert refused.diagnostic.code is InputCode.EQUATION_REFERENCE_NOT_EXPRESSION
    assert refused.diagnostic.field == field and refused.diagnostic.symbols == ["@e1"]
    assert refused.arguments == {field: "@e1"}  # nothing was rewritten
    assert store.resolve({field: "@e2"}, tool).error is None  # an expression is passed on


def test_fields_that_accept_a_leading_name_or_several_actions_stay_unclassified(
    ctx: ToolContext,
) -> None:
    # `L = ...` / `H = ...` is read off by these tools, so refusing every relation would refuse a working call.
    for tool, field in (
        ("analyze_lagrangian", "lagrangian"),
        ("search_invariants", "lagrangian"),
        ("find_conserved_quantities", "lagrangian"),
        ("find_conserved_quantities", "hamiltonian"),
        ("matrix_calculus", "latex"),
        ("vector_calculus", "expression"),
    ):
        assert policy_of(tool, field) == "unclassified"
    out = execute("analyze_lagrangian", {"lagrangian": r"L = \dot{x}^2", "coordinates": ["x"]}, ctx)
    assert out.status == "ok"


def test_the_two_new_codes_can_be_logged_and_are_in_the_closed_vocabulary() -> None:
    for code in (
        InputCode.EQUATION_REFERENCE_NOT_EXPRESSION,
        InputCode.EXPRESSION_REFERENCE_NOT_EQUATION,
    ):
        assert code.value in REASON_CODES


# --- resolution --------------------------------------------------------------------------------------------
def _store(*formulas: tuple[str, str]) -> ResultStore:
    table = EquationTable()
    for latex, source in formulas:
        table.number_for(latex, source, mode="none")  # type: ignore[arg-type]
    return ResultStore.from_prior({}, {}, table.items, unverified={})


def test_a_relation_for_an_expression_field_is_refused_whole_or_embedded_and_names_the_equation() -> (
    None
):
    store = _store(("y = x^2", "from_tool"))
    for latex in ("@e1", "2 @e1 + 1"):
        r = store.resolve({"latex": latex, "variables": []}, "differentiate")
        assert r.error is not None and "equation (1)" in r.error
        assert r.diagnostic is not None
        assert r.diagnostic.code is InputCode.EQUATION_REFERENCE_NOT_EXPRESSION
        assert r.diagnostic.field == "latex" and r.diagnostic.symbols == ["@e1"]
        assert r.arguments == {"latex": latex, "variables": []}  # nothing was rewritten
        assert "x^2" not in r.error  # the message holds the number, not the formula


def test_an_expression_for_an_equation_field_is_refused_when_it_is_the_whole_argument() -> None:
    store = _store(("x^2 - 1", "from_tool"))
    r = store.resolve({"equation": "@e1"}, "solve_polynomial")
    assert r.diagnostic is not None
    assert r.diagnostic.code is InputCode.EXPRESSION_REFERENCE_NOT_EQUATION
    assert store.resolve({"equation": "@e1 = 0"}, "solve_polynomial").error is None


def test_what_the_field_takes_still_works_exactly_as_before() -> None:
    store = _store(("y = x^2", "from_tool"), ("x^2 + 1", "from_tool"))
    assert store.resolve({"equation": "@e1"}, "solve_polynomial").arguments["equation"] == "y = x^2"
    assert store.resolve({"equation": "@e1"}, "isolate_variable").error is None
    assert store.resolve({"latex": "@e1"}, "check_identity").error is None
    ok = store.resolve({"latex": "@e2"}, "differentiate")
    assert ok.error is None and ok.arguments["latex"] == "x^2 + 1" and ok.diagnostic is None
    embedded = store.resolve({"latex": "3 @e2"}, "simplify")
    assert embedded.arguments["latex"] == r"3 \left(x^2 + 1\right)"


def test_an_unclassified_field_and_an_unknown_tool_are_never_refused() -> None:
    store = _store(("y = x^2", "from_tool"))
    for tool, field in (("matrix_calculus", "latex"), ("analyze_lagrangian", "lagrangian")):
        assert policy_of(tool, field) == "unclassified"
        assert store.resolve({field: "@e1"}, tool).error is None
    assert store.resolve({"latex": "@e1"}, "a_tool_nobody_has").error is None
    assert store.resolve({"latex": "@e1"}).error is None  # no tool given: the old behaviour


def test_a_formula_whose_kind_is_unknown_is_never_refused() -> None:
    store = ResultStore.from_prior({}, {}, {1: Equation(1, "y = x^2", "from_tool")}, unverified={})
    assert store.equations[1].kind is None  # an old stored equation: no kind recorded
    assert store.resolve({"latex": "@e1"}, "simplify").error is None
    undecided = _store((r"x \leq 3", "from_tool"))
    assert undecided.equations[1].kind is None
    assert undecided.resolve({"latex": "@e1"}, "simplify").error is None


def test_a_relation_inside_a_field_the_table_does_not_resolve_is_left_alone() -> None:
    store = _store(("y = x^2", "from_tool"))
    args = {"latex": "x", "relations": ["@e1"]}
    r = store.resolve(args, "substitute")
    assert r.error is None and r.arguments["relations"] == ["@e1"]


def test_a_geometry_row_is_a_relation_and_a_scalar_is_an_expression(ctx: ToolContext) -> None:
    sphere = GeometryArgs(
        coordinates=["theta", "phi"],
        metric=[["a^2", "0"], ["0", r"a^2 \sin^2\theta"]],
        assumptions={"a": ["positive"]},
        quantities=["christoffel", "ricci_scalar"],
    )
    out = run_geometry(sphere, ctx)
    out.card.id = "g1"
    table = EquationTable()
    from minerva_api.agent.card_equations import listed_equations as rows

    number_card = Block(kind="card", card=Card.model_validate(out.card.model_dump(mode="json")))
    from minerva_api.agent.equations import number_blocks

    number_blocks([number_card], table)
    card = number_card.card
    assert card is not None
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    christoffel = card.numbers[
        next(k for k, _ in rows("geometry", card.data) if k.startswith("christoffel"))
    ]
    scalar = card.numbers["ricci_scalar"]
    assert store.resolve({"latex": f"@e{scalar}"}, "differentiate").error is None  # still works
    assert store.resolve({"latex": f"@e{christoffel}"}, "check_identity").error is None
    refused = store.resolve({"latex": f"@e{christoffel}"}, "differentiate")
    assert refused.diagnostic is not None


# --- the loop ----------------------------------------------------------------------------------------------
def test_the_model_sees_a_typed_refusal_and_an_equivalent_retry_is_not_run(
    ctx: ToolContext,
) -> None:
    table = EquationTable()
    table.number_for("y = x^2", "written_by_model", mode="model")
    same = {"latex": "@e1", "variables": [{"name": "x"}]}
    message, _ = play(
        ctx,
        [call("differentiate", same), call("differentiate", same), say("which side do you mean?")],
        equations=table,
    )
    first, second = message["trace"]
    assert first["status"] == "error" and first["arguments_valid"] is False
    line = next(x for x in first["for_model"].splitlines() if x.startswith("diagnostic: "))
    payload = json.loads(line.removeprefix("diagnostic: "))
    assert payload["code"] == "equation_reference_not_expression"
    assert payload["field"] == "latex" and payload["symbols"] == ["@e1"]
    assert "equation (1)" in first["for_model"] and "x^2" not in first["for_model"]
    assert (
        second["status"] == "error" and second["arguments_valid"] is False
    )  # refused again, never run
    assert second["for_model"] == first["for_model"]
    assert card_of(message)["data"]["diagnostic"]["code"] == "equation_reference_not_expression"


def test_an_expression_row_is_still_used_by_number_everywhere_it_was(ctx: ToolContext) -> None:
    table = EquationTable()
    table.number_for("x^2", "written_by_model", mode="model")
    message, _ = play(
        ctx,
        [call("differentiate", {"latex": "@e1", "variables": [{"name": "x"}]}), say("done")],
        equations=table,
    )
    assert message["trace"][0]["status"] == "ok"
    assert refs  # the module under test is the one the loop uses


def test_a_refusal_carries_no_formula_and_the_diagnostic_is_the_users_name_only() -> None:
    refused = refusal_diagnostic("integrate", "latex", 12, "equation")
    assert refused is not None
    diagnostic, message = refused
    assert diagnostic.symbols == ["@e12"] and "12" in message
    assert refusal_diagnostic("integrate", "latex", 12, "expression") is None
    assert refusal_diagnostic("solve_polynomial", "equation", 3, "equation") is None
    assert refusal_diagnostic("matrix_calculus", "latex", 3, "equation") is None
