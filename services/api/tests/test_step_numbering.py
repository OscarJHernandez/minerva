"""Equation numbers and reusable handles for `substitute`, `rewrite_part` and the matrix card (issue #133, part 1).

Both step tools keep their result under `card.data["result"]`, so the numbering code that read only the top-level
`result_latex` never numbered them although `step` is classified as a single-result card, and `ResultStore` did not
list them. `card_equations.result_formula` is the one reader now. A result that rests on the user's premise keeps
that wherever it is reused; a result that was not accepted has neither a number nor a handle; a matrix result
that is not an evaluated result is never numbered as one.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    PREMISE_SUFFIX,
    SINGLE_RESULT,
    premise_labels,
    result_formula,
    result_restrictions,
)
from minerva_api.agent.equations import (
    EquationTable,
    build_table,
    number_blocks,
    number_stored,
    unverified_inputs,
)
from minerva_api.agent.events import Block, Card
from minerva_api.agent.refs import (
    PREMISE_INPUT,
    REFERENCEABLE_TOOLS,
    ResultStore,
    quote_reason,
    unverified_note,
)
from minerva_api.agent.replay import call, say
from minerva_api.agent.tools import ToolContext, ToolOutput, execute
from minerva_api.agent.turn_context import _history
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_total_derivative_handles import Row, card_of, play

GIVEN = {
    "latex": r"x^2 + y^2 + z",
    "relations": ["y^2 = 1 - x^2"],
}  # holds given the premise: z + 1
REWRITE = {"latex": r"(x+1)^2 (y+1)^2", "action": "expand", "part": r"(x+1)^2"}  # an identity
EVALUATED_MATRIX = {
    "latex": r"\exp(wA)\exp(-wB)",
    "matrices": [
        {"name": "A", "entries": [["1", "0"], ["0", "2"]]},
        {"name": "B", "entries": [["0", "1"], ["0", "0"]]},
    ],
    "action": "integrate",
    "variable": "w",
    "lower": "0",
    "upper": "1",
}
REPRESENTATION_MATRIX = {
    "latex": r"\exp(tP)\exp(tQ)",
    "matrices": [{"name": "P", "rows": 2}, {"name": "Q", "rows": 2}],
    "action": "integrate",
    "variable": "t",
    "lower": "0",
    "upper": "1",
}


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def conditional(ctx: ToolContext) -> ToolOutput:
    out = execute("substitute", GIVEN, ctx)
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "conditional"
    return out


@pytest.fixture(scope="module")
def identity(ctx: ToolContext) -> ToolOutput:
    out = execute("rewrite_part", REWRITE, ctx)
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "identity"
    return out


def numbered(out: ToolOutput, rid: str | None = None) -> tuple[Card, EquationTable]:
    table = EquationTable()
    number_blocks([Block(kind="card", card=out.card)], table, {out.card.id: rid} if rid else None)
    return out.card, table


# --- the one reader ------------------------------------------------------------------------------------
def test_the_step_and_matrix_cards_keep_their_result_below_the_top_level(
    conditional: ToolOutput,
) -> None:
    assert (
        "result_latex" not in conditional.card.data
    )  # the defect: the old readers looked only here
    nested = conditional.card.data["result"]["result_latex"]
    assert result_formula("step", conditional.card.data) == nested == "z + 1"
    # a top-level formula is read as it was; another card type does not read a nested one
    assert result_formula("differentiate", {"result_latex": "x"}) == "x"
    assert result_formula("differentiate", conditional.card.data) is None


def test_only_an_ok_step_result_has_a_formula(conditional: ToolOutput) -> None:
    for status in ("not_verified", "unsupported", "conditions_not_satisfied", "parse_error"):
        data = copy.deepcopy(conditional.card.data)
        data["result"]["status"] = status
        assert result_formula("step", data) is None, status
    blank = copy.deepcopy(conditional.card.data)
    blank["result"]["result_latex"] = "  "
    assert result_formula("step", blank) is None


def test_the_classification_still_names_each_card_type_once() -> None:
    assert "step" in SINGLE_RESULT and "matrix" in LIST_READERS
    assert not (set(LIST_READERS) & SINGLE_RESULT) and "step" not in LIST_NOT_YET_NUMBERED


# --- numbering ------------------------------------------------------------------------------------------
def test_a_successful_substitute_is_numbered_as_a_whole(conditional: ToolOutput) -> None:
    card, table = numbered(conditional, "r1")
    assert card.number == 1 and not card.numbers
    assert table.items[1].latex == "z + 1" == card.data["result"]["result_latex"]
    assert table.items[1].result_id == "r1" and table.items[1].source == "from_tool"


def test_a_successful_rewrite_part_is_numbered_as_a_whole(identity: ToolOutput) -> None:
    card, table = numbered(identity, "r1")
    assert card.number == 1 and table.items[1].latex == card.data["result"]["result_latex"]


def test_a_result_that_was_not_accepted_gets_no_number(ctx: ToolContext) -> None:
    ambiguous = execute("substitute", {"latex": "x + y", "relations": ["x = y", "y = 1"]}, ctx)
    assert ambiguous.status == "unsupported"
    unmet = execute(
        "rewrite_part",
        {
            "latex": r"\log(x y) + z",
            "action": "rule",
            "rule_id": "log.product",
            "part": r"\log(x y)",
        },
        ctx,
    )
    assert unmet.status == "conditions_not_satisfied"
    for out in (ambiguous, unmet):
        card, table = numbered(out)
        assert card.number is None and table.items == {} and table.counter == 0
        assert ResultStore().register(out.card.tool, out) is None


# --- handles --------------------------------------------------------------------------------------------
def test_substitute_and_rewrite_part_are_referenceable() -> None:
    assert {"substitute", "rewrite_part"} <= REFERENCEABLE_TOOLS
    assert "isolate_variable" not in REFERENCEABLE_TOOLS  # its card has solutions, not one result
    assert "matrix_calculus" not in REFERENCEABLE_TOOLS


def test_a_successful_step_gets_a_result_id_with_the_exact_formula(
    conditional: ToolOutput, identity: ToolOutput
) -> None:
    store = ResultStore()
    assert store.register("substitute", conditional) == "r1"
    assert store.register("rewrite_part", identity) == "r2"
    assert store.results == {
        "r1": "z + 1",
        "r2": identity.card.data["result"]["result_latex"],
    }


def test_a_step_that_was_not_accepted_gets_no_result_id(
    conditional: ToolOutput, identity: ToolOutput
) -> None:
    store = ResultStore()
    for source in (conditional, identity):
        for status in ("not_verified", "unsupported", "conditions_not_satisfied"):
            bad = copy.deepcopy(source)
            bad.status = status
            bad.card.data["result"]["status"] = status
            assert store.register(bad.card.tool, bad) is None, status
        # an accepted-looking result with no accepted evidence is not registered either
        unaccepted = copy.deepcopy(source)
        assert unaccepted.evidence is not None
        unaccepted.evidence.accepted = False
        assert store.register(unaccepted.card.tool, unaccepted) is None
    assert store.results == {}


def test_a_step_result_keeps_its_own_restrictions_on_the_handle(identity: ToolOutput) -> None:
    out = copy.deepcopy(identity)
    out.card.data["result"]["restrictions_latex"] = [r"x \neq 1"]
    assert result_restrictions("step", out.card.data) == [r"x \neq 1"]
    store = ResultStore()
    rid = store.register("rewrite_part", out, inherited=[r"y \neq 0"])
    assert rid is not None and store.restrictions[rid] == [r"x \neq 1", r"y \neq 0"]
    assert store.unverified == {}  # an identity rests on no premise


def test_a_premise_stays_with_the_handle_of_a_conditional_result(conditional: ToolOutput) -> None:
    store = ResultStore()
    rid = store.register("substitute", conditional)
    assert rid == "r1"
    (label,) = store.unverified[rid]
    assert label.endswith(PREMISE_SUFFIX) and "y^{2} = 1 - x^{2}" in label
    assert premise_labels("step", conditional.card.data) == [label]
    # a reuse resolves with the premise attached, however it is named
    for text in ("@r1", "@e1"):
        table = EquationTable()
        table.number_for("z + 1", "from_tool", "r1", mode="none")
        again = ResultStore.from_prior(
            store.results, store.restrictions, table.items, store.unverified
        ).resolve({"latex": text})
        assert again.error is None and again.arguments["latex"] == "z + 1"
        assert again.unverified == [label]


def test_a_premise_has_its_own_sentence_and_is_never_called_written_by_the_model() -> None:
    label = f"$y = 1${PREMISE_SUFFIX}"
    note = unverified_note([label])
    assert note.startswith("Input given, not proved")
    assert "model wrote" not in note and "never 'proved'" in note
    assert quote_reason([label]) == PREMISE_INPUT
    assert quote_reason([label, "equation (2)"]) != PREMISE_INPUT  # a model-written input wins


# --- through the agent loop, storage and reload ---------------------------------------------------------
def first_turn(ctx: ToolContext) -> dict[str, Any]:
    msg, _ = play(
        ctx,
        [call("substitute", GIVEN), call("rewrite_part", REWRITE), say("done")],
        equations=EquationTable(),
    )
    return msg


def test_the_loop_numbers_and_registers_both_steps_and_a_reload_agrees(ctx: ToolContext) -> None:
    msg = first_turn(ctx)
    cards = [b["card"] for b in msg["blocks"] if b["kind"] == "card"]
    assert [c["number"] for c in cards] == [1, 2]
    assert [s["result_id"] for s in msg["trace"]] == ["r1", "r2"]
    exact = [c["data"]["result"]["result_latex"] for c in cards]
    rows = [Row("user", {"text": "go"}), Row("assistant", msg)]
    # rebuilt from the stored rows: the same table, the same results, the same premise
    table = build_table(rows)
    assert {n: (e.latex, e.result_id) for n, e in table.items.items()} == {
        1: (exact[0], "r1"),
        2: (exact[1], "r2"),
    }
    *_, results, restrictions = _history(rows)  # type: ignore[arg-type]
    assert results == {"r1": exact[0], "r2": exact[1]}
    carried = unverified_inputs(rows)
    assert list(carried) == ["r1"] and carried["r1"][0].endswith(PREMISE_SUFFIX)
    assert restrictions == {}


def test_a_stored_conversation_without_numbers_is_numbered_the_same_way(ctx: ToolContext) -> None:
    msg = first_turn(ctx)
    old = copy.deepcopy(msg)
    for block in old["blocks"]:
        if block["kind"] == "card":
            block["card"].pop("number", None)
    table = EquationTable()
    assert number_stored([("user", {"text": "go"}), ("assistant", old)], table)
    assert [b["card"]["number"] for b in old["blocks"] if b["kind"] == "card"] == [1, 2]
    assert {n: e.latex for n, e in table.items.items()} == {
        n: e.latex for n, e in build_table([Row("assistant", msg)]).items.items()
    }


def test_numbers_are_not_reused_after_a_retry(ctx: ToolContext) -> None:
    msg = first_turn(ctx)
    rows = [Row("user", {"text": "go"}), Row("assistant", msg)]
    table = build_table(rows, counter=2)
    assert table.counter == 2
    # the reply is replaced: a new result takes number 3, and (1) and (2) are never handed out again
    retried = EquationTable(counter=table.counter)
    new, _ = play(ctx, [call("rewrite_part", REWRITE), say("again")], equations=retried)
    assert card_of(new)["number"] == 3
    assert ResultStore.from_prior({}, equations={}).resolve({"latex": "@e1"}).error


def test_unknown_references_fail_explicitly_after_the_reply_is_deleted(ctx: ToolContext) -> None:
    msg = first_turn(ctx)
    kept = [Row("user", {"text": "go"})]  # the assistant reply was deleted
    *_, results, _ = _history(kept)  # type: ignore[arg-type]
    store = ResultStore.from_prior(results, equations=build_table(kept).items)
    assert results == {} and "unknown result reference @r1" in (
        store.resolve({"latex": "@r1"}).error or ""
    )
    assert "unknown equation reference @e2" in (store.resolve({"latex": "@e2"}).error or "")
    assert msg  # the first reply existed and had numbers 1 and 2


def test_a_wrong_twin_a_premise_result_is_not_verified_after_reuse(ctx: ToolContext) -> None:
    """z + 1 holds only given y^2 = 1 - x^2; what is computed from it must say so and not count as verified."""
    msg = first_turn(ctx)
    rows = [Row("user", {"text": "go"}), Row("assistant", msg)]
    table = build_table(rows)
    *_, results, restrictions = _history(rows)  # type: ignore[arg-type]
    second, _ = play(
        ctx,
        [call("differentiate", {"latex": "@e1^2", "variables": [{"name": "z"}]}), say("ok")],
        text="differentiate equation 1",
        equations=table,
        prior_results=results,
        prior_restrictions=restrictions,
        prior_unverified=unverified_inputs(rows),
    )
    (step,) = second["trace"]
    card = card_of(second)
    assert step["status"] == "ok" and step["result_id"] == "r3"
    assert card["number"] == 3  # continues the conversation's counter
    assert card["data"]["unverified_inputs"][0].endswith(PREMISE_SUFFIX)
    assert "Input given, not proved" in step["for_model"]
    assert step["verified_latex"] == []  # never `known_latex`
    assert step["quoted"]  # quoted, with the premise as the reason
    assert PREMISE_INPUT in step["quoted"].values()
    # the premise travels on: the next reuse of @r3 inherits it too
    rows.append(Row("assistant", second))
    assert unverified_inputs(rows)["r3"][0].endswith(PREMISE_SUFFIX)


def test_an_identity_step_reused_by_number_carries_no_premise(ctx: ToolContext) -> None:
    msg = first_turn(ctx)
    rows = [Row("user", {"text": "go"}), Row("assistant", msg)]
    second, _ = play(
        ctx,
        [call("differentiate", {"latex": "@e2", "variables": [{"name": "x"}]}), say("ok")],
        equations=build_table(rows),
        prior_results=_history(rows)[3],  # type: ignore[arg-type]
        prior_unverified=unverified_inputs(rows),
    )
    (step,) = second["trace"]
    assert step["status"] == "ok" and "unverified_inputs" not in card_of(second)["data"]
    assert step["verified_latex"]  # the identity chain is verified: its reuse is too


def test_end_to_end_a_step_result_is_used_by_number_in_a_follow_up(ctx: ToolContext) -> None:
    """Turn 1: substitute. Turn 2 ("use equation 1"): the follow-up operation runs on exactly that formula."""
    first, _ = play(ctx, [call("substitute", GIVEN), say("done")], equations=EquationTable())
    exact = card_of(first)["data"]["result"]["result_latex"]
    rows = [Row("user", {"text": "substitute"}), Row("assistant", first)]
    table = build_table(rows)
    second, _ = play(
        ctx,
        [call("simplify", {"latex": r"\frac{@e1^2 - 1}{z}"}), say("ok")],
        text="simplify equation 1 plus z",
        equations=table,
        prior_results=_history(rows)[3],  # type: ignore[arg-type]
        prior_unverified=unverified_inputs(rows),
    )
    (step,) = second["trace"]
    assert step["status"] == "ok" and "@e1" not in step["arguments"]["latex"]
    assert (
        exact in step["arguments"]["latex"]
    )  # the server put in the exact formula, not a retyping
    assert card_of(second)["number"] == 2


# --- matrix: an evaluated result is named, a representation never is --------------------------------------
def test_an_evaluated_matrix_result_is_numbered_but_has_no_handle(ctx: ToolContext) -> None:
    out = execute("matrix_calculus", EVALUATED_MATRIX, ctx)
    result = out.card.data["result"]
    assert out.status == "ok" and result["evaluated"] is True and out.card.type == "matrix"
    card, table = numbered(out)
    assert card.number == 1 and table.items[1].latex == result["result_latex"]
    assert table.items[1].result_id is None  # a name, not a verified `@rN`
    assert ResultStore().register("matrix_calculus", out) is None
    # reused by number it says so: a tool result with no result id
    reused = ResultStore.from_prior({}, equations=table.items).resolve({"latex": "@e1"})
    assert reused.unverified and reused.unverified[0].startswith("equation (1)")


def test_a_matrix_representation_is_never_numbered_as_an_evaluated_result(
    ctx: ToolContext,
) -> None:
    out = execute("matrix_calculus", REPRESENTATION_MATRIX, ctx)
    result = out.card.data["result"]
    assert out.status == "ok" and result["evaluated"] is False
    assert result["result_kind"] in {"equation", "representation"}
    card, table = numbered(out)
    assert card.number is None and table.items == {} and table.counter == 0
    assert ResultStore().register("matrix_calculus", out) is None
    # a numerical approximation is no evaluated result either
    numeric = copy.deepcopy(out.card.data)
    numeric["result"].update(evaluated=True, result_kind="numeric", result_latex="1.5")
    assert result_formula("matrix", numeric) is None
    numeric["result"]["result_kind"] = "explicit"
    assert result_formula("matrix", numeric) == "1.5"


def test_the_matrix_number_survives_a_reload(ctx: ToolContext) -> None:
    msg, _ = play(
        ctx, [call("matrix_calculus", EVALUATED_MATRIX), say("done")], equations=EquationTable()
    )
    card = card_of(msg)
    assert card["number"] == 1
    table = build_table([Row("assistant", msg)])
    assert table.items[1].latex == card["data"]["result"]["result_latex"]
    assert msg["trace"][0]["result_id"] is None
