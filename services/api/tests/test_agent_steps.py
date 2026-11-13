"""The agent's controlled-step tools: premises are the user's, refusals are relayed, nothing partial."""

import pytest
from minerva_api.agent.llm import Chunk, ToolCall
from minerva_api.agent.tools import TOOLS, ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_agent_units import FakeLLM, cards_of, collect, final, segments_of


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def known(out: object) -> set[str]:
    from minerva_api.agent.segments import normalize

    return {normalize(k) for k in out.known_latex}  # type: ignore[attr-defined]


def test_substitute_tells_the_model_the_result_holds_given_the_users_relation(
    ctx: ToolContext,
) -> None:
    out = execute("substitute", {"latex": "x^2 + y^2", "relations": ["y^2 = 1 - x^2"]}, ctx)
    assert out.status == "ok" and out.card.type == "step"
    assert out.card.data["operation"] == "substitute"
    text = out.for_model
    assert text.startswith("status=ok")
    assert "evidence=conditional" in text and "HOLDS GIVEN" in text
    assert "never 'proved'" in text and "result=$1$" in text
    # What the model may quote: the premise, its sides, the input and the result.
    from minerva_api.agent.segments import normalize

    for formula in ("1", "x^{2} + y^{2}", "y^{2} = 1 - x^{2}"):
        assert normalize(formula) in known(out), formula


def test_a_refused_substitution_asks_for_the_order_and_gives_no_result(ctx: ToolContext) -> None:
    out = execute("substitute", {"latex": "x + y", "relations": ["x = y", "y = 1"]}, ctx)
    assert out.status == "unsupported"
    assert "result=none" in out.for_model and "order_dependent" in out.for_model
    assert "Ask the user" in out.for_model
    given = execute(
        "substitute", {"latex": "x + y", "relations": ["x = y", "y = 1"], "order": "given"}, ctx
    )
    assert given.status == "ok" and "ANOTHER ORDER" in given.for_model


def test_the_model_cannot_pass_more_than_four_relations_or_none(ctx: ToolContext) -> None:
    assert execute("substitute", {"latex": "x", "relations": []}, ctx).card.type == "error"
    assert (
        execute("substitute", {"latex": "x", "relations": ["x = 1"] * 5}, ctx).card.type == "error"
    )


def test_rewrite_part_applies_to_the_named_part_only(ctx: ToolContext) -> None:
    out = execute(
        "rewrite_part",
        {"latex": r"(x+1)^2 (y+1)^2", "action": "expand", "part": r"(x+1)^2"},
        ctx,
    )
    assert out.status == "ok" and out.card.data["operation"] == "rewrite"
    assert "applied sympy.expand to the part" in out.for_model
    assert "checked numerically" in out.for_model


def test_an_ambiguous_part_is_returned_as_a_question_not_guessed(ctx: ToolContext) -> None:
    out = execute(
        "rewrite_part",
        {
            "latex": r"\sin(2x) + y \sin(2x)",
            "action": "rule",
            "rule_id": "trig.double_sin",
            "part": r"\sin(2x)",
        },
        ctx,
    )
    assert out.status == "parse_error" and "occurrence 1 of 2" in out.for_model
    assert "ask the user which occurrence" in out.for_model.lower()
    assert "result=none" in out.for_model
    chosen = execute(
        "rewrite_part",
        {
            "latex": r"\sin(2x) + y \sin(2x)",
            "action": "rule",
            "rule_id": "trig.double_sin",
            "part": r"\sin(2x)",
            "occurrence": 2,
        },
        ctx,
    )
    assert chosen.status == "ok"


def test_an_identity_with_unproven_conditions_is_not_applied_and_the_model_is_told_to_ask(
    ctx: ToolContext,
) -> None:
    out = execute(
        "rewrite_part",
        {
            "latex": r"\log(x y) + z",
            "action": "rule",
            "rule_id": "log.product",
            "part": r"\log(x y)",
        },
        ctx,
    )
    assert out.status == "conditions_not_satisfied"
    assert "NOT APPLIED" in out.for_model and "Ask whether they hold" in out.for_model


def test_an_unknown_identity_is_an_error_card(ctx: ToolContext) -> None:
    out = execute("rewrite_part", {"latex": "x", "action": "rule", "rule_id": "no.such.rule"}, ctx)
    assert out.card.type == "error" and out.status == "error"


def test_isolate_lists_every_solution_with_conditions_and_the_degenerate_case(
    ctx: ToolContext,
) -> None:
    out = execute(
        "isolate_variable", {"equation": r"s = u t + \frac{1}{2} a t^2", "variable": "t"}, ctx
    )
    assert out.status == "ok" and out.card.data["operation"] == "isolate"
    text = out.for_model
    assert text.count("- solution ") == 2
    assert r"ALL solutions need $a \neq 0$" in text
    assert "NOT COVERED when $a = 0$" in text
    assert "complete=True" in text
    from minerva_api.agent.segments import normalize

    assert normalize(r"a \neq 0") in known(out)


def test_isolate_reports_no_solution_and_unsupported_without_a_partial_answer(
    ctx: ToolContext,
) -> None:
    none = execute("isolate_variable", {"equation": r"\sqrt{x} = -2", "variable": "x"}, ctx)
    assert none.status == "ok" and "NO SOLUTION" in none.for_model
    unsupported = execute("isolate_variable", {"equation": r"\sin x = a", "variable": "x"}, ctx)
    assert unsupported.status == "unsupported"
    assert "do not offer a partial answer" in unsupported.for_model
    assert "- solution" not in unsupported.for_model


def test_a_family_is_described_as_such(ctx: ToolContext) -> None:
    out = execute("isolate_variable", {"equation": r"N = M \exp(-r t)", "variable": "t"}, ctx)
    assert "family: k_1 any integer" in out.for_model


def test_hostile_arguments_become_an_error_card_not_a_crash(ctx: ToolContext) -> None:
    for name, args in (
        ("substitute", {"latex": "x", "relations": ["__import__('os').system('x') = 1"]}),
        ("isolate_variable", {"equation": "x = 1", "variable": "x; import os"}),
        ("rewrite_part", {"latex": "x", "action": "rm -rf"}),
    ):
        out = execute(name, args, ctx)
        assert out.status != "ok" and out.card.type in {"error", "step"}


def test_a_model_turn_with_substitute_marks_the_formulas_it_may_quote(ctx: ToolContext) -> None:
    llm = FakeLLM(
        [
            [
                Chunk(
                    tool_calls=[
                        ToolCall(
                            "substitute", {"latex": "x^2 + y^2", "relations": ["y^2 = 1 - x^2"]}
                        )
                    ]
                )
            ],
            [
                Chunk(
                    text="Given your relation, the result is $1$, and the invented value $x^2 = 7$ is not from the tool."
                )
            ],
        ]
    )
    msg = final(collect(llm, ctx, [{"role": "user", "content": "use my relation"}]))
    assert [c["type"] for c in cards_of(msg)] == ["step"]
    marks = {
        s["value"]: s["status"] for s in segments_of(msg) if s["type"] in ("math", "display_math")
    }
    assert marks["1"] == "from_tool"
    assert marks["x^2 = 7"] == "unchecked", "a formula the tool did not produce stays unchecked"


def test_the_tool_schemas_are_what_the_model_sees() -> None:
    for name in ("substitute", "rewrite_part", "isolate_variable"):
        schema = TOOLS[name].schema()["function"]["parameters"]
        assert schema["properties"], name
    assert (
        TOOLS["substitute"].schema()["function"]["parameters"]["properties"]["relations"][
            "maxItems"
        ]
        == 4
    )
