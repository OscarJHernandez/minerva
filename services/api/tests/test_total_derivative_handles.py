"""Handles for a total derivative (`time_derivative`): an `@rN` result id and an equation number for a result of
any length, and never a handle for the conclusion on the equations of motion (issue #149).

The one engine-backed total derivative here is genuinely long (over 2000 characters); the other cases change
only the registration policy and use that real output or small ones.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest
from minerva_api.agent.equations import (
    MAX_EQUATION_CHARS,
    EquationTable,
    build_table,
    number_blocks,
    unverified_inputs,
)
from minerva_api.agent.events import Block, Card
from minerva_api.agent.loop import run_agent
from minerva_api.agent.refs import (
    COMPONENT_SUFFIX,
    NO_ID_SUFFIX,
    UNLINKED_INPUT,
    UNNUMBERED_INPUT,
    UNVERIFIED_INPUT,
    ResultStore,
    quote_reason,
    unverified_note,
)
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import LONG_RESULT, ToolContext, ToolOutput, execute
from minerva_api.agent.turn_context import _history
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

LONG_INPUT = (
    r"(x \dot{y} - y \dot{x} + z \dot{z})^3 (x^2 + y^2 + z^2 + \dot{x}^2 + \dot{y}^2)^2 \sin(x y z)"
)
COORDS = ["x", "y", "z"]
KEPLER = r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}"
ASSUME = {"mu": ["positive"], "k": ["positive"], "r": ["positive"], "theta": ["real"]}


class Row:
    def __init__(self, role: str, content: dict[str, Any]) -> None:
        self.role, self.content = role, content


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def long_output(ctx: ToolContext) -> ToolOutput:
    out = execute("time_derivative", {"latex": LONG_INPUT, "coordinates": COORDS}, ctx)
    assert out.status == "ok" and out.evidence is not None and out.evidence.accepted
    assert len(out.card.data["result_latex"]) > LONG_RESULT
    return out


def play(
    ctx: ToolContext, replies: list[Any], text: str = "go", **kw: Any
) -> tuple[dict[str, Any], ReplayLLM]:
    llm = ReplayLLM(replies)

    async def go() -> list[dict[str, Any]]:
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx, **kw)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message"), llm


def card_of(message: dict[str, Any]) -> dict[str, Any]:
    return next(b["card"] for b in message["blocks"] if b["kind"] == "card")


# --- registration -------------------------------------------------------------------------------------
def test_a_long_verified_total_derivative_gets_a_result_id(long_output: ToolOutput) -> None:
    latex = long_output.card.data["result_latex"]
    store = ResultStore()
    assert store.register("time_derivative", long_output) == "r1"
    assert store.results["r1"] == latex  # exactly what the engine printed


def test_a_result_between_500_and_800_characters_gets_one_too(long_output: ToolOutput) -> None:
    medium = copy.deepcopy(long_output)
    medium.card.data["result_latex"] = "x + " * 150 + "1"  # 600 characters: over the equation limit
    assert MAX_EQUATION_CHARS < len(medium.card.data["result_latex"]) < LONG_RESULT
    assert ResultStore().register("time_derivative", medium) == "r1"


def test_the_model_is_told_what_the_id_names_and_that_the_on_motion_conclusion_has_none(
    long_output: ToolOutput,
) -> None:
    text = long_output.for_model
    assert "reuse it exactly by its result id" in text  # not the old "not repeated here" dead end
    assert "no result id" in text and "equations of motion" in text
    assert long_output.card.data["result_latex"] not in text  # still not copied into the prompt


def test_a_failed_or_unsupported_total_derivative_has_no_handle(ctx: ToolContext) -> None:
    store = ResultStore()
    refused = execute(
        "time_derivative",
        {"latex": r"r \dot{r}", "coordinates": ["r"], "lagrangian": "L = " + KEPLER},
        ctx,
    )
    assert refused.status == "parse_error"
    assert store.register("time_derivative", refused) is None
    accel = execute("time_derivative", {"latex": r"\ddot{x} + x", "coordinates": ["x"]}, ctx)
    assert accel.status != "ok" and store.register("time_derivative", accel) is None
    assert "result id" not in accel.for_model
    unverified = copy.deepcopy(accel)
    unverified.status = "not_verified"
    unverified.card.data["result_latex"] = "x"
    assert store.register("time_derivative", unverified) is None
    assert store.results == {}


# --- the two results cannot be confused ----------------------------------------------------------------
def test_the_on_motion_conclusion_is_never_registered(ctx: ToolContext) -> None:
    zero = execute(
        "time_derivative",
        {
            "latex": r"\mu r^2 \dot{\theta}",
            "coordinates": ["r", "theta"],
            "lagrangian": KEPLER,
            "assumptions": ASSUME,
        },
        ctx,
    )
    assert zero.status == "ok" and zero.card.data["zero_on_motion"] is True
    nonzero = execute(
        "time_derivative",
        {
            "latex": r"r \dot{r}",
            "coordinates": ["r", "theta"],
            "lagrangian": KEPLER,
            "assumptions": ASSUME,
        },
        ctx,
    )
    assert nonzero.card.data["zero_on_motion"] is False
    store = ResultStore()
    assert store.register("time_derivative", zero) == "r1"
    assert store.register("time_derivative", nonzero) == "r2"
    # each id is the unrestricted derivative; neither the plain 0 nor the on-motion formula is a result
    assert store.results["r1"] == zero.card.data["result_latex"] != "0"
    assert store.results["r2"] == nonzero.card.data["result_latex"]
    assert nonzero.card.data["on_motion_latex"] not in store.results.values()
    assert "0" not in store.results.values()
    assert nonzero.quoted_latex  # the on-motion formula stays quoted, never `known_latex`
    assert nonzero.card.data["on_motion_latex"] not in nonzero.known_latex


def test_only_the_derivative_is_numbered_not_the_on_motion_conclusion(ctx: ToolContext) -> None:
    msg, _ = play(
        ctx,
        [
            call(
                "time_derivative",
                {
                    "latex": r"r \dot{r}",
                    "coordinates": ["r", "theta"],
                    "lagrangian": KEPLER,
                    "assumptions": ASSUME,
                },
            ),
            say("done"),
        ],
        equations=EquationTable(),
    )
    card = card_of(msg)
    assert card["number"] == 1 and msg["trace"][0]["result_id"] == "r1"
    table = build_table([Row("assistant", msg)])
    assert set(table.items) == {1} and table.items[1].latex == card["data"]["result_latex"]
    assert card["data"]["on_motion_latex"] != table.items[1].latex


# --- numbering ------------------------------------------------------------------------------------------
def test_a_tool_cards_single_result_is_numbered_whatever_its_length() -> None:
    long = "x + " * 700 + "1"
    card = Card(
        id="c1",
        tool="time_derivative",
        type="differentiate",
        title="t",
        data={"result_latex": long},
    )
    table = EquationTable()
    number_blocks([Block(kind="card", card=card)], table, {"c1": "r1"})
    assert card.number == 1 and table.items[1].result_id == "r1" and table.items[1].latex == long
    # the limit stays for the model's prose and the user's formulas
    assert EquationTable().number_for(long, "written_by_model", mode="model") is None
    assert EquationTable().number_for(long, "yours", mode="user") is None


# --- reuse after reload, retry and trimming -------------------------------------------------------------
def stored_conversation(ctx: ToolContext, turns: int = 1) -> tuple[list[Row], dict[str, Any]]:
    rows: list[Row] = []
    first: dict[str, Any] = {}
    for n in range(turns):
        rows.append(Row("user", {"text": f"question {n}"}))
        if n == 0:
            msg, _ = play(
                ctx,
                [
                    call("time_derivative", {"latex": LONG_INPUT, "coordinates": COORDS}),
                    say("It is long."),
                ],
                equations=EquationTable(),
            )
            first = msg
        else:
            msg = {"blocks": [], "trace": [], "text": "ok"}
        rows.append(Row("assistant", msg))
    return rows, first


def test_the_handle_resolves_to_the_exact_expression_after_reload(ctx: ToolContext) -> None:
    rows, msg = stored_conversation(ctx)
    exact = card_of(msg)["data"]["result_latex"]
    assert len(exact) > LONG_RESULT and msg["trace"][0]["result_id"] == "r1"
    *_, results, restrictions = _history(rows)  # type: ignore[arg-type]
    assert results == {"r1": exact}
    table = build_table(rows)
    store = ResultStore.from_prior(results, restrictions, table.items)
    by_id = store.resolve({"latex": "@r1"})
    by_number = store.resolve({"latex": f"@e{card_of(msg)['number']}"})
    assert by_id.error is None and by_id.arguments["latex"] == exact
    assert by_number.error is None and by_number.arguments["latex"] == exact
    assert by_number.unverified == []  # it is the verified result, not an unlinked row


def test_the_handle_survives_context_trimming(ctx: ToolContext) -> None:
    rows, msg = stored_conversation(ctx, turns=4)
    exact = card_of(msg)["data"]["result_latex"]
    history, _, _, results, _ = _history(rows, max_messages=2)  # type: ignore[arg-type]
    assert all(
        "question 0" not in str(m.get("content")) for m in history
    )  # the first turn is out of the window
    assert results == {"r1": exact}  # read from the whole conversation, not the window


def test_the_handle_survives_a_retry_and_a_deleted_reference_fails(ctx: ToolContext) -> None:
    rows, msg = stored_conversation(ctx)
    exact = card_of(msg)["data"]["result_latex"]
    # retry: the reply is replaced by a new one that runs the same tool again; ids continue after r1
    *_, results, _ = _history(rows[:1])  # type: ignore[arg-type]
    assert results == {}  # the reply is gone: its handle no longer exists...
    gone = ResultStore.from_prior(results).resolve({"latex": "@r1"})
    assert gone.error and "unknown result reference @r1" in gone.error  # ...and says so
    again, _ = play(
        ctx, [call("time_derivative", {"latex": LONG_INPUT, "coordinates": COORDS}), say("again")]
    )
    assert card_of(again)["data"]["result_latex"] == exact  # the same expression
    *_, results2, _ = _history([rows[0], Row("assistant", again)])  # type: ignore[arg-type]
    assert results2 == {"r1": exact}
    missing = ResultStore.from_prior(results2).resolve({"latex": "@e1"})
    assert missing.error and "unknown equation reference @e1" in missing.error


# --- provenance travels with the handle -----------------------------------------------------------------
def test_inherited_restrictions_stay_attached(ctx: ToolContext) -> None:
    msg, _ = play(
        ctx,
        [call("time_derivative", {"latex": "@r1", "coordinates": ["x"]}), say("ok")],
        prior_results={"r1": r"\frac{x^{2}}{x - 1}"},
        prior_restrictions={"r1": [r"x \neq 1"]},
    )
    data = card_of(msg)["data"]
    assert data["inherited_restrictions_latex"] == [r"x \neq 1"]
    assert "Inherited from @r1" in msg["trace"][0]["for_model"]
    assert msg["trace"][0]["result_id"] == "r2"


def test_an_unverified_input_stays_attached_to_the_handle(ctx: ToolContext) -> None:
    table = EquationTable()
    table.number_for(r"x^{2} \dot{x}", "written_by_model", mode="model")
    msg, _ = play(
        ctx,
        [
            call("time_derivative", {"latex": "@e1", "coordinates": ["x"]}),
            say("ok"),
        ],
        equations=table,
    )
    (first,) = msg["trace"]
    assert first["result_id"] == "r1" and card_of(msg)["data"]["unverified_inputs"] == [
        "equation (1)"
    ]
    assert first["verified_latex"] == [] and first["quoted"]  # quoted, never `known_latex`
    assert "Input not verified" in first["for_model"]
    # rebuilt from the stored card, so a later `@r1` inherits it
    assert unverified_inputs([Row("assistant", msg)]) == {"r1": ["equation (1)"]}


def test_a_card_equation_without_a_result_id_is_marked_when_reused() -> None:
    table = EquationTable()
    table.number_for("x + 1", "from_tool", None, mode="none")  # a tool result that has no id
    r = ResultStore.from_prior({}, equations=table.items).resolve({"latex": "@e1"})
    assert r.arguments["latex"] == "x + 1"
    assert r.unverified == [f"equation (1){NO_ID_SUFFIX}"]


# --- downstream limits still apply ----------------------------------------------------------------------
def test_an_over_limit_handle_is_refused_by_the_receiving_tool(
    ctx: ToolContext, long_output: ToolOutput
) -> None:
    exact = long_output.card.data["result_latex"]
    msg, _ = play(
        ctx,
        [call("simplify", {"latex": "@r1"}), say("too big")],
        prior_results={"r1": exact * 2},  # beyond the 2000-character input limit
    )
    step = msg["trace"][0]
    assert step["status"] != "ok" and step["result_id"] is None
    assert step["verified_latex"] == []


def test_the_three_provenance_classes_have_their_own_sentences_and_reasons() -> None:
    written, part = "equation (2)", f"equation (3){COMPONENT_SUFFIX}"
    unnumbered = f"equation (4){NO_ID_SUFFIX}"
    assert "which the model wrote and Minerva never verified" in unverified_note([written])
    list_note = unverified_note([part])
    assert (
        list_note.startswith("Input taken from a list:")
        and "reusing one component by number" in list_note
    )
    no_id = unverified_note([unnumbered])
    assert no_id.startswith("Input without a result id:") and "taken from a list" not in no_id
    assert "component" not in no_id
    both = unverified_note([part, unnumbered])  # each class keeps its own sentence
    assert (
        both.count("Input taken from a list") == 1 and both.count("Input without a result id") == 1
    )
    assert "equation (4)" not in list_note + unverified_note([written])
    assert quote_reason([part]) == UNLINKED_INPUT
    assert quote_reason([unnumbered]) == UNNUMBERED_INPUT
    assert quote_reason([written, part, unnumbered]) == UNVERIFIED_INPUT
    assert quote_reason([part, unnumbered]) == UNLINKED_INPUT
    assert len({UNLINKED_INPUT, UNNUMBERED_INPUT, UNVERIFIED_INPUT}) == 3
