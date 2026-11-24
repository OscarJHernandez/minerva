"""Issue #146: a verified tool result types its constants as scalars, and the model calling them "the
Laplace-Runge-Lenz vector" does not change that.

Reproduces the 2026-11-24 report: after `find_conserved_quantities` on the polar Kepler Lagrangian the reply wrote
bare right sides under headings such as "Component of Laplace-Runge-Lenz Vector ($C_3$)" and the exact relation
`2 C_1^2 C_2 - m C_3^2 - m C_4^2 + k^2 m = 0`. `C_3` had become a vector (the generous reading took every name of
every formula of the sentence), the vector test runs before the evidence match, and 0 of 5 formulas had evidence.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from minerva_api.agent.declarations import (
    declared_vectors,
    typed_scalar_names,
)
from minerva_api.agent.loop import run_agent
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_api.agent.turn_context import _declaration_inputs, declarations_of
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

from .test_claim_evidence_agent import formulas

KEPLER = r"\frac{m}{2}(\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}"
C1_RIGHT = r"m r^2 \dot{\theta}"
C3_RIGHT = (
    r"-k \cos{\theta} + m r^3 \dot{\theta}^2 \cos{\theta} + m r^2 \dot{r} \dot{\theta} \sin{\theta}"
)
RELATION = r"2 C_1^2 C_2 - m C_3^2 - m C_4^2 + k^2 m = 0"
HEADING = "### Component of Laplace-Runge-Lenz Vector ($C_3$)"

REPLY = (
    "The search found four constants.\n\n"
    f"**Angular Momentum ($C_1$):** ${C1_RIGHT}$\n\n"
    "$C_3$ and $C_4$ are the components of the Laplace-Runge-Lenz vector.\n\n"
    f"{HEADING}\n\n"
    f"$$C_3 = {C3_RIGHT}$$\n\n"
    f"They obey $${RELATION}$$"
)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=60, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def turn(ctx: ToolContext, answer: str) -> dict[str, Any]:
    llm = ReplayLLM(
        [
            call(
                "find_conserved_quantities",
                {"lagrangian": KEPLER, "coordinates": ["r", "theta"], "budget": "thorough"},
            ),
            say(answer),
        ]
    )

    async def go() -> list[dict[str, Any]]:
        text = "Find the hidden conserved quantities of the Kepler problem."
        return [e async for e in run_agent(llm, [{"role": "user", "content": text}], ctx)]

    return next(e["data"] for e in asyncio.run(go()) if e["event"] == "message")


def by_prefix(message: dict[str, Any], prefix: str) -> dict[str, Any]:
    return next(s for s in formulas(message) if s["value"].strip().startswith(prefix))


@pytest.fixture(scope="module")
def reply(ctx: ToolContext) -> dict[str, Any]:
    return turn(ctx, REPLY)


def test_the_discovered_constant_restated_with_braced_trig_keeps_its_evidence(
    reply: dict[str, Any],
) -> None:
    claim = by_prefix(reply, "C_3 =")
    assert claim["status"] == "from_tool", claim
    detail = " ".join(c["detail"] or "" for c in reply["debug"]["claims"])
    assert "vector notation" not in detail


def test_the_bare_right_side_and_the_relation_get_their_marks_without_a_vector_note(
    reply: dict[str, Any],
) -> None:
    assert by_prefix(reply, "m r^2 \\dot{\\theta}")["status"] == "from_tool"
    relation = by_prefix(reply, "2 C_1^2")
    assert relation["status"] == "from_tool", relation
    assert "vector" not in (relation["note"] or "")
    claims = {c["value"]: c for c in reply["debug"]["claims"]}
    assert claims[RELATION]["status"] == "supported"
    assert "vector" not in (claims[RELATION]["detail"] or "")


def test_a_reordered_bare_expression_stays_unchecked(ctx: ToolContext) -> None:
    # text match only: the energy is right, its terms are in another order than the tool printed them
    message = turn(
        ctx, r"The energy is $\frac{m}{2}(\dot{r}^2 + r^2\dot{\theta}^2) - \frac{k}{r}$."
    )
    assert by_prefix(message, r"\frac{m}{2}")["status"] == "unchecked"


@pytest.mark.parametrize(
    "wrong",
    [
        C3_RIGHT.replace("-k", "+k"),  # a sign
        C3_RIGHT.replace(r"\sin{\theta}", r"\sin{\phi}"),  # a changed argument
        C3_RIGHT.replace(r"+ m r^2 \dot{r} \dot{\theta} \sin{\theta}", ""),  # a dropped term
        C3_RIGHT.replace("-k", r"-1.000001 k"),  # a 1e-6 factor
        C3_RIGHT.replace(r"\dot{\theta}^2", r"\ddot{\theta}^2"),  # \dot changed to \ddot
    ],
)
def test_a_wrong_constant_stays_unchecked_under_the_same_heading(
    ctx: ToolContext, wrong: str
) -> None:
    message = turn(ctx, f"{HEADING}\n\n$$C_3 = {wrong}$$")
    assert by_prefix(message, "C_3 =")["status"] == "unchecked"
    assert all(c["status"] != "supported" for c in message["debug"]["claims"])


def test_a_wrong_bare_right_side_stays_unchecked(ctx: ToolContext) -> None:
    message = turn(ctx, r"Angular momentum: $m r^2 \ddot{\theta}$ and $m r \dot{\theta}$.")
    assert {s["status"] for s in formulas(message) if "theta" in s["value"]} == {"unchecked"}


def test_the_tool_typing_wins_over_a_vector_sentence_of_the_model(ctx: ToolContext) -> None:
    # a sentence the narrow generous reading still takes: a bare name after "vector"
    text = f"The vector $C_3$ is the third constant, and $C_4$ the fourth.\n\n$${RELATION}$$"
    relation = by_prefix(turn(ctx, text), "2 C_1^2")
    assert relation["status"] == "from_tool", relation
    assert "vector" not in (relation["note"] or "")


# -- the declarations ----------------------------------------------------------------------------


def test_one_over_r_in_a_vector_sentence_declares_nothing() -> None:
    text = (
        "The Laplace-Runge-Lenz vector is special to $1/r$ potentials and $\\cos{\\theta}$ terms."
    )
    assert declared_vectors([text]) == set()


def test_the_accepted_loss_is_pinned_with_its_declared_twin(ctx: ToolContext) -> None:
    statement = r"(u+v)^2 = u^2 + 2uv + v^2"
    undeclared = f"For vectors, ${statement}$."
    declared = f"Let $u$ and $v$ be vectors. Then ${statement}$."
    assert declared_vectors([undeclared]) == set()
    assert declared_vectors([declared]) == {"u", "v"}
    plain = ctx.engine.check_claims([statement], vectors=sorted(declared_vectors([undeclared])))
    vectors = ctx.engine.check_claims([statement], vectors=sorted(declared_vectors([declared])))
    assert plain[0].status == "holds"  # tested as scalars: the accepted loss
    assert vectors[0].status == "unverified"  # declared: never sampled


class Row:
    def __init__(self, role: str, text: str, trace: list[dict[str, Any]] | None = None) -> None:
        self.role, self.content = role, {"text": text, "trace": trace or []}


DISCOVERY_STEP = {
    "name": "find_conserved_quantities",
    "status": "ok",
    "repeated": False,
    "arguments": {},
    "formula_roles": {
        "C_{3} = - k \\cos{\\left(\\theta \\right)}": "conserved",
        "C_{4} = k \\sin{\\left(\\theta \\right)}": "conserved",
    },
    "variables": ["r", "theta"],
}
SAID = (
    "The vector $C_3$ is the Laplace-Runge-Lenz vector, and $C_4$ is a vector too. "
    "Let $r$ be a vector field."
)


def declared(*rows: Row) -> Any:
    return declarations_of(*_declaration_inputs(list(rows), ""))  # type: ignore[arg-type]


def test_the_model_prose_cannot_make_a_typed_scalar_a_vector() -> None:
    found = declared(Row("user", "Find the constants."), Row("assistant", SAID, [DISCOVERY_STEP]))
    assert not {"C_3", "C_4", "r"} & set(found.prior_vectors)
    assert not {"C_3", "C_4", "r"} & set(found.prior_matrices)


def test_the_same_prose_with_no_verified_result_still_declares_its_vectors() -> None:
    found = declared(Row("user", "Find the constants."), Row("assistant", SAID))
    assert {"C_3", "C_4"} <= set(found.prior_vectors)


def test_a_user_declaration_of_a_typed_scalar_still_wins() -> None:
    found = declared(
        Row("user", "Let $C_4$ be a vector."), Row("assistant", SAID, [DISCOVERY_STEP])
    )
    assert "C_4" in found.prior_vectors and "C_3" not in found.prior_vectors


def test_a_vector_calculus_field_declaration_is_never_removed() -> None:
    step = {
        "name": "vector_calculus",
        "status": "ok",
        "arguments": {"fields": [{"name": "C_3", "kind": "vector"}]},
    }
    found = declared(Row("user", "Go."), Row("assistant", SAID, [DISCOVERY_STEP, step]))
    assert "C_3" in found.prior_vectors


def test_typed_scalar_names_read_roles_and_variables_of_accepted_results_only() -> None:
    names = typed_scalar_names([DISCOVERY_STEP])
    assert {"C_3", "C_4", "r", "r_dot", "r_ddot", "theta", "theta_dot", "theta_ddot"} <= names
    assert typed_scalar_names([{**DISCOVERY_STEP, "status": "not_verified"}]) == set()
    assert typed_scalar_names([{**DISCOVERY_STEP, "repeated": True}]) == set()
    other = {**DISCOVERY_STEP, "formula_roles": {"x = 1": "equation_of_motion"}, "variables": []}
    assert typed_scalar_names([other]) == set()


def test_the_right_side_of_a_printed_row_is_known_even_with_left_right_brackets() -> None:
    from minerva_api.agent.tools import _right_side

    assert _right_side(r"C_{3} = - k \cos{\left(\theta \right)}") == [
        r"- k \cos{\left(\theta \right)}"
    ]
    assert _right_side(r"x \leq 2") == [] and _right_side(r"x \neq 2") == []
    assert _right_side(r"A = 0") == [] and _right_side(r"a = b = c") == []
