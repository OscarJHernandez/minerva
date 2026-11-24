"""Issue #172 (A-011 follow-up): the conversation-level coverage of reused and mistyped verified constants.

One conversation is driven through the evaluation harness (`runner.observe`, `score_case`, `run_case`) with the REAL
first-turn discovery card (`find_conserved_quantities` on the polar Kepler Lagrangian, run by the engine) and a
scripted model (`ReplayLLM`) for every turn after it. What each later turn is given is rebuilt from the stored first
answer by `turn_context`, the chat route's own function, so the constants, the prior evidence and the declarations
that reach the tools and the claim checker are the ones a user's conversation gets.

The names (`C_1` ...) are read from the card and each is matched to its role by an independent hand-written
reference (angular momentum, energy, the two Laplace-Runge-Lenz components) checked with `check_identity`; the order
the search returns them in is never assumed. Nothing here takes an expected answer from a model.

The live counterpart is the reliability case `kepler-discovery-then-derivative-of-a-relation-among-the-constants`;
it cannot depend on a model mistyping a constant, so the mistyped constant is owned by this module.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import ClaimMark, EvalCase, Followup
from minerva_api.agent.evaluation.runner import observe, run_case
from minerva_api.agent.evaluation.score import CaseResult, FailureClass, Observation, score_case
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_api.agent.turn_context import turn_context
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_claim_evidence_agent import C4_RETYPED, C4_RIGHT, POLAR, formulas

FAST = AgentLimits(retry_delay_seconds=0)
COORDINATES = ["r", "theta"]
FIRST_PROMPT = (
    r"Let $r = |\mathbf{r}|$ be the distance from the centre. Find the hidden conserved quantities of the "
    r"Kepler problem with Lagrangian $L = " + POLAR + r"$, coordinates r and theta."
)
# The model's reply names the Runge-Lenz *vector*: that must not make `r` a vector for the next tool.
FIRST_REPLY = (
    "Four constants were found within the families searched: the angular momentum, the energy and two "
    "components of the Laplace-Runge-Lenz vector."
)
SECOND_PROMPT = "Take the total time derivative of the relation among those constants and say whether it vanishes."
ASK = "What is the fourth constant?"

# Independent, hand-written definitions of the four constants for POLAR (Goldstein, ch. 3: the angular momentum, the
# energy and the two components of the Laplace-Runge-Lenz vector). The right side of `C4_RIGHT` is the sine component.
ROLES = {
    "momentum": r"\mu r^{2} \dot{\theta}",
    "energy": r"\frac{\mu}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) - \frac{G M \mu}{r}",
    "lrl_cos": r"- G M \cos\theta + r^3 \dot\theta^2 \cos\theta + r^2 \dot r \dot\theta \sin\theta",
    "lrl_sin": C4_RIGHT.split("=", 1)[1].strip(),
}
RETYPED = C4_RETYPED.split("=", 1)[1].strip()  # \dot\theta -> \ddot\theta in one factor
# The same function as ROLES["lrl_sin"], factorised: an algebraically equivalent way to write it.
EQUIVALENT = (
    r"G M \sin\theta - \left( r \dot\theta \sin\theta - \dot r \cos\theta \right) r^2 \dot\theta"
)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def case_for(prompt: str, tool: str = "time_derivative", **fields: Any) -> EvalCase:
    return EvalCase(
        id="multiturn",
        category="followup",
        prompt=prompt,
        expected_tool=tool,
        source="tests/test_conserved_quantity_multiturn.py",
        **fields,
    )


def drive(llm: ReplayLLM, ctx: ToolContext, case: EvalCase, stored: list[Any]) -> Observation:
    obs = asyncio.run(observe(llm, case, ctx, FAST, stored=stored))
    assert not obs.error, obs.error
    return obs


def stored_turn(prompt: str, final: dict[str, Any]) -> list[Any]:
    return [
        SimpleNamespace(role="user", content={"text": prompt}),
        SimpleNamespace(role="assistant", content=final),
    ]


@pytest.fixture(scope="module")
def stored(ctx: ToolContext) -> list[Any]:
    """Turn one: the real discovery on the engine, then the scripted summary, stored as the chat route stores it."""
    llm = ReplayLLM(
        [
            call(
                "find_conserved_quantities",
                {"lagrangian": POLAR, "coordinates": COORDINATES, "budget": "thorough"},
            ),
            say(FIRST_REPLY),
        ]
    )
    obs = drive(llm, ctx, case_for(FIRST_PROMPT, "find_conserved_quantities"), [])
    assert obs.final is not None
    assert [c["type"] for c in obs.cards.values()] == ["discovery"]
    return stored_turn(FIRST_PROMPT, obs.final)


def proved_equal(ctx: ToolContext, a: str, b: str) -> bool:
    out = execute("check_identity", {"latex": f"{a} = {b}"}, ctx)
    return out.status == "ok" and out.card.data.get("verdict") == "proved_equal"


@pytest.fixture(scope="module")
def names(ctx: ToolContext, stored: list[Any]) -> dict[str, str]:
    """Role -> the name the card gave it, matched by the independent references and never by position."""
    found = turn_context(stored, "x").constants
    assert len(found) == 4 and all(c.verified for c in found)
    mapping: dict[str, str] = {}
    for role, reference in ROLES.items():
        hits = [c.name for c in found if proved_equal(ctx, c.expression, reference)]
        assert len(hits) == 1, (role, hits, [(c.name, c.expression) for c in found])
        mapping[role] = hits[0]
    assert sorted(mapping.values()) == sorted(c.name for c in found)
    return mapping


def relation_sides(n: dict[str, str]) -> tuple[str, str]:
    """2 L^2 E = mu^3 (A^2 + B^2 - G^2 M^2) among the four constants, written with the names the card gave."""
    a, b, ell, energy = n["lrl_cos"], n["lrl_sin"], n["momentum"], n["energy"]
    return (
        rf"2 {ell}^{{2}} {energy}",
        rf"{a}^{{2}} \mu^{{3}} + {b}^{{2}} \mu^{{3}} - G^{{2}} M^{{2}} \mu^{{3}}",
    )


def relation(n: dict[str, str]) -> str:
    """The quantity whose total time derivative is asked for: left side minus right side."""
    left, right = relation_sides(n)
    return f"{left} - ({right})"


def td_call(latex: str) -> list[Any]:
    return call("time_derivative", {"latex": latex, "coordinates": COORDINATES})


def card_of(obs: Observation, kind: str | None = None) -> dict[str, Any]:
    cards = [c for c in obs.cards.values() if kind is None or c["type"] == kind]
    assert cards, list(obs.cards)
    return cards[-1]["data"]


# -- (a) a relation among the returned constants: zero ON the equations of motion, not unrestricted -----------------
def test_the_relation_is_an_identity_in_the_hand_written_definitions(
    ctx: ToolContext, names: dict[str, str]
) -> None:
    """The premise of the next test, from the engine's identity check on the hand-written definitions with the
    velocities as plain symbols u, w (an equation with `\\dot` is read as an equation of motion and not tested)."""
    sides = list(relation_sides(names))
    for index, text in enumerate(sides):
        for name, role in sorted(((n, r) for r, n in names.items()), key=lambda p: -len(p[0])):
            text = text.replace(name, f"({ROLES[role]})")
        for old, new in (
            (r"\dot{r}", "u"),
            (r"\dot r", "u"),
            (r"\dot{\theta}", "w"),
            (r"\dot\theta", "w"),
        ):
            text = text.replace(old, new)
        assert "dot" not in text, text
        sides[index] = text
    assert proved_equal(ctx, sides[0], sides[1]), sides


def test_a_relation_among_the_constants_is_zero_on_the_motion_and_not_unrestricted(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    turn = turn_context(stored, SECOND_PROMPT)
    # the conversation reaches the tool: the constants of the real card, and its verified result as evidence
    assert {c.name for c in turn.constants} == set(names.values())
    assert turn.prior_evidence and all(row["status"] == "ok" for row in turn.prior_evidence)
    assert any(row.get("verified_latex") for row in turn.prior_evidence)

    llm = ReplayLLM(
        [
            td_call(relation(names)),
            say(
                "The total derivative is long and not zero by itself; on the equations of motion of this "
                "Lagrangian it is zero."
            ),
        ]
    )
    case = case_for(SECOND_PROMPT, forbid=["differentiate"], reply_matches="equations of motion")
    obs = drive(llm, ctx, case, stored)
    data = card_of(obs, "differentiate")
    assert data["status"] == "ok"
    assert len(data["definitions_used"]) == 4  # every name was replaced by its verified expression
    assert data["zero_on_motion"] is True and data["on_motion_latex"] == "0"
    assert data["result_latex"].strip() != "0"  # the unrestricted derivative is not claimed zero
    assert (
        r"\ddot" in data["result_latex"]
    )  # it still carries the accelerations the equations of motion remove
    told = " ".join(str(m.get("content", "")) for m in llm.seen[-1])
    assert "zero on the equations of motion" in told and "cannot be reused" in told
    scored = score_case(case, obs)
    assert scored.passed, scored.notes


def test_without_the_earlier_cards_the_same_call_is_refused_and_nothing_is_guessed(
    ctx: ToolContext, names: dict[str, str]
) -> None:
    """The twin of the test above: no stored discovery, so no definitions (`no_dynamic_dependence`)."""
    llm = ReplayLLM([td_call(relation(names)), say("It could not be differentiated.")])
    data = card_of(drive(llm, ctx, case_for(SECOND_PROMPT), []))
    assert data["status"] == "unsupported" and not data.get("zero_on_motion")
    assert data["diagnostic"]["code"] == "no_dynamic_dependence"


def test_the_harness_runs_the_whole_conversation_and_scores_the_follow_up(
    ctx: ToolContext, names: dict[str, str]
) -> None:
    """`run_case` (what `eval_models.py` runs) over both turns: the real discovery, then the scripted follow-up."""
    case = case_for(
        FIRST_PROMPT,
        "find_conserved_quantities",
        followups=[
            Followup(
                prompt=SECOND_PROMPT,
                expected_tool="time_derivative",
                forbid=["differentiate"],
                reply_matches="equations of motion",
            )
        ],
    )
    llm = ReplayLLM(
        [
            call("find_conserved_quantities", {"lagrangian": POLAR, "coordinates": COORDINATES}),
            say(FIRST_REPLY),
            td_call(relation(names)),
            say("On the equations of motion of this Lagrangian the derivative is zero."),
        ]
    )
    result: CaseResult = asyncio.run(run_case(llm, ctx, case, FAST))
    assert result.passed and result.turns == 2, result.notes


# -- (b), (c) a constant the model re-types in a later turn ----------------------------------------------------
def retype(
    ctx: ToolContext, stored: list[Any], name: str, rhs: str, marks: list[ClaimMark]
) -> tuple[Observation, CaseResult, dict[str, Any]]:
    """Turn two: no tool call, the model writes `name = rhs` in its reply. The claim checker reads the stored turn."""
    case = case_for(ASK, "none", claim_marks=marks)
    obs = drive(ReplayLLM([say(f"The constant is $$ {name} = {rhs} $$")]), ctx, case, stored)
    assert obs.final is not None
    claim = next(s for s in formulas(obs.final) if s["value"].strip().startswith(name))
    return obs, score_case(case, obs), claim


def rules_of(obs: Observation) -> set[str | None]:
    assert obs.final is not None
    return {c["evidence_rule"] for c in obs.final["debug"]["claims"]}


def test_a_re_typed_constant_is_unchecked_with_the_mismatch_rule_never_verified_or_contradicted(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    name = names["lrl_sin"]
    marks = [ClaimMark(formula=f"{name} = {RETYPED}", status="unchecked", evidence_rule="mismatch")]
    obs, scored, claim = retype(ctx, stored, name, RETYPED, marks)
    assert claim["status"] == "unchecked", claim
    assert "verified result" in (
        claim["note"] or ""
    )  # it names the verified equation it differs from
    assert rules_of(obs) == {"mismatch"}
    assert not any(
        s["status"] in ("from_tool", "checked", "contradicted") for s in formulas(obs.final or {})
    )
    assert scored.passed and scored.contradicted == 0 and scored.false_verified == 0, scored.notes


@pytest.mark.parametrize(
    ("how", "rhs"), [("unchanged", ROLES["lrl_sin"]), ("equivalent", EQUIVALENT)]
)
def test_an_unchanged_or_equivalent_re_type_is_linked_to_the_verified_result(
    ctx: ToolContext, stored: list[Any], names: dict[str, str], how: str, rhs: str
) -> None:
    assert proved_equal(ctx, rhs, ROLES["lrl_sin"]), how  # the re-type really is the same function
    name = names["lrl_sin"]
    marks = [
        ClaimMark(formula=f"{name} = {rhs}", status="from_tool", evidence_rule="same_left_side")
    ]
    obs, scored, claim = retype(ctx, stored, name, rhs, marks)
    assert claim["status"] == "from_tool", (how, claim)
    assert rules_of(obs) == {"same_left_side"}
    assert scored.passed, scored.notes


def test_the_mark_checks_can_fail_and_say_why(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    """Wrong twins of the scoring itself: each pinned mark that is not what the checker did fails, with its class."""
    name = names["lrl_sin"]
    right = ROLES["lrl_sin"]
    # pinned `unchecked`, but the claim is the verified one: a false verification
    scored = retype(
        ctx, stored, name, right, [ClaimMark(formula=f"{name} = {right}", status="unchecked")]
    )[1]
    assert not scored.passed and scored.failure_class is FailureClass.VERIFICATION
    # pinned `from_tool`, but the re-typed claim is unchecked: the checker did something else
    scored = retype(
        ctx, stored, name, RETYPED, [ClaimMark(formula=f"{name} = {RETYPED}", status="from_tool")]
    )[1]
    assert not scored.passed and scored.failure_class is FailureClass.CLAIM_CHECKER
    # the right mark with the wrong rule
    wrong_rule = ClaimMark(
        formula=f"{name} = {RETYPED}", status="unchecked", evidence_rule="same_left_side"
    )
    scored = retype(ctx, stored, name, RETYPED, [wrong_rule])[1]
    assert not scored.passed and scored.failure_class is FailureClass.CLAIM_CHECKER
    # a formula the reply never wrote
    scored = retype(ctx, stored, name, right, [ClaimMark(formula="z = 1", status="from_tool")])[1]
    assert not scored.passed and "not in the reply" in " ".join(scored.notes)


# -- (d) a name that cannot be resolved ------------------------------------------------------------------------
def with_card_edit(
    stored: list[Any], edit: Callable[[dict[str, Any]], None], *, second_search: bool
) -> list[Any]:
    """The real first answer with its discovery card edited (a copy): in place, or kept as a second search."""
    changed = copy.deepcopy(stored[1].content)
    for block in changed["blocks"]:
        if block.get("kind") == "card":
            edit(block["card"])
    if not second_search:
        return [stored[0], SimpleNamespace(role="assistant", content=changed)]
    return [*stored, *stored_turn("Search again.", changed)]


def refused(obs: Observation) -> dict[str, Any]:
    """The refusal's typed diagnostic; nothing was computed."""
    data = card_of(obs)
    assert "zero_on_motion" not in data and not data.get("result_latex"), data
    return data["diagnostic"]


def test_two_meanings_of_a_name_are_ambiguous_and_nothing_is_computed(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    def second_meaning(c: dict[str, Any]) -> None:
        row = c["data"]["report"]["constants"][0]
        row["expression_latex"] = row["expression_latex"].split("=")[0] + r"= \mu r \dot{\theta}"

    both = with_card_edit(stored, second_meaning, second_search=True)
    assert (
        len(turn_context(both, SECOND_PROMPT).constants) == 8
    )  # both searches are kept: the clash is visible
    llm = ReplayLLM([td_call(relation(names)), say("It was refused.")])
    assert refused(drive(llm, ctx, case_for(SECOND_PROMPT), both))["code"] == "ambiguous_constant"
    # the same search stored twice is one meaning, not a clash
    twice = with_card_edit(stored, lambda c: None, second_search=True)
    llm = ReplayLLM([td_call(relation(names)), say("On the equations of motion it is zero.")])
    assert (
        card_of(drive(llm, ctx, case_for(SECOND_PROMPT), twice), "differentiate")["zero_on_motion"]
        is True
    )


def test_an_unverified_name_is_refused_with_its_own_code(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    def unverify(c: dict[str, Any]) -> None:
        for row in c["data"]["report"]["constants"]:
            row["verified"] = False

    edited = with_card_edit(stored, unverify, second_search=False)
    llm = ReplayLLM([td_call(relation(names)), say("It was refused.")])
    assert (
        refused(drive(llm, ctx, case_for(SECOND_PROMPT), edited))["code"] == "constant_not_verified"
    )


# -- (e) the scalar/vector declarations of the earlier turns ---------------------------------------------------
def test_r_stays_a_scalar_after_the_norm_definition_whatever_the_reply_says_about_vectors(
    stored: list[Any],
) -> None:
    declarations = turn_context(stored, SECOND_PROMPT).declarations
    assert "r" in declarations.scalar_names  # `r = |\mathbf{r}|` is a definition, not an opinion
    assert "r" not in declarations.prior_vectors and "r" not in declarations.prior_matrices
    assert "r" not in declarations.vector_kind_names


def test_the_follow_up_tool_takes_r_as_a_coordinate_in_that_conversation(
    ctx: ToolContext, stored: list[Any], names: dict[str, str]
) -> None:
    """The reply of turn one speaks of the Runge-Lenz vector; the derivative in `r` and `theta` still runs."""
    llm = ReplayLLM([td_call(relation(names)), say("On the equations of motion it is zero.")])
    assert (
        card_of(drive(llm, ctx, case_for(SECOND_PROMPT), stored), "differentiate")["status"] == "ok"
    )


def test_a_user_who_declares_r_a_vector_does_change_what_the_tool_sees(
    stored: list[Any],
) -> None:
    """The twin of the scalar test: the declaration comes from the user's words, not from the reply."""
    vector = [SimpleNamespace(role="user", content={"text": "Let r be a vector."}), stored[1]]
    declarations = turn_context(vector, SECOND_PROMPT).declarations
    assert "r" not in declarations.scalar_names
    assert "r" in declarations.prior_vectors or "r" in declarations.vector_kind_names


def test_claim_marks_change_the_digest_only_where_a_case_uses_them() -> None:
    """Adding the field to the schema left every existing case-set digest alone; a case that pins a mark has its own."""
    from minerva_api.agent.evaluation.cases import cases_digest, load_cases, load_suite

    assert cases_digest(load_cases()) == "b7d2202c3d87"  # the pinned core digest
    cases = load_suite("reliability")
    plain = cases_digest(cases)
    marked = [c.model_copy(deep=True) for c in cases]
    target = next(c for c in marked if c.followups)
    target.followups[0].claim_marks = [ClaimMark(formula="x = 1", status="unchecked")]
    assert cases_digest(marked) != plain
    target.followups[0].claim_marks = []
    assert cases_digest(marked) == plain
