"""Statements about matrices are never "checked" by sampling numbers, however the matrices were declared."""

import asyncio

import pytest
from minerva_api.agent.declarations import declared_matrices
from minerva_api.agent.events import Block, Segment, TraceStep
from minerva_api.agent.loop import check_model_claims, matrix_names
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=30, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def claim(value: str) -> Block:
    return Block(kind="text", segments=[Segment(type="math", value=value, status="unchecked")])


def status_of(ctx: ToolContext, value: str, **kw: object) -> str:
    block = claim(value)
    asyncio.run(check_model_claims([block], ctx, [], **kw))  # type: ignore[arg-type]
    return block.segments[0].status


# -- declarations in words -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("Let A and B be 3x3 matrices.", {"A", "B"}),
        (r"Let $A$, $B$ be $n\times n$ matrices. Then", {"A", "B"}),
        ("X, Y are noncommuting operators.", {"X", "Y"}),
        ("Suppose A_1 and A_{2} do not commute", {"A_1", "A_2"}),
        (r"Let $\Omega$ and $\Lambda$ be matrices", {"Omega", "Lambda"}),
        ("A matrix is invertible if its determinant is nonzero.", set()),
        ("The solution of x^2 = 4 is x = 2.", set()),
    ],
)
def test_declarations_in_words(text: str, names: set[str]) -> None:
    assert declared_matrices([text]) == names


def test_a_matrix_declared_only_in_words_is_not_checked_as_numbers(ctx: ToolContext) -> None:
    formula = r"\exp(A)\exp(B) = \exp(A+B)"
    assert status_of(ctx, formula) == "checked"  # nothing says A, B are matrices: scalar claim
    assert status_of(ctx, formula, prior_matrices=["A", "B"]) == "unchecked"
    # the model's own prose in the same reply declares them too
    block = Block(
        kind="text",
        segments=[
            Segment(type="text", value="Let "),
            Segment(type="math", value="A", status="unchecked"),
            Segment(type="text", value=" and "),
            Segment(type="math", value="B", status="unchecked"),
            Segment(type="text", value=" be matrices. Then "),
            Segment(type="math", value=formula, status="unchecked"),
        ],
    )
    asyncio.run(check_model_claims([block], ctx, []))
    assert block.segments[-1].status == "unchecked"


@pytest.mark.parametrize(
    "formula",
    [
        r"AB - BA = 0",
        r"(A+B)^2 = A^2 + 2AB + B^2",
        r"(AB)^{-1} = A^{-1}B^{-1}",
        r"A_1 A_2 = A_2 A_1",
    ],
)
def test_wrong_matrix_identities_are_never_checked(ctx: ToolContext, formula: str) -> None:
    assert status_of(ctx, formula, prior_matrices=["A", "B"]) == "unchecked"


# -- noncommutative symbols declared in any tool call -----------------------------------------------
def test_noncommutative_assumptions_count_as_matrix_declarations() -> None:
    trace = [
        TraceStep(
            id="a",
            name="substitute",
            arguments={"assumptions": {"X": ["noncommutative"], "x": ["real"]}},
            for_model="",
        )
    ]
    assert matrix_names(trace) == {"X"}


def test_a_noncommutative_declaration_does_not_switch_off_scalar_checks(ctx: ToolContext) -> None:
    trace = [
        TraceStep(
            id="a",
            name="rewrite_part",
            arguments={"assumptions": {"X": ["noncommutative"]}},
            for_model="",
        )
    ]
    block = Block(
        kind="text",
        segments=[
            Segment(type="math", value=r"\sin^2 x + \cos^2 x = 1", status="unchecked"),
            Segment(type="math", value=r"XY = YX", status="unchecked"),
        ],
    )
    asyncio.run(check_model_claims([block], ctx, trace))
    assert [s.status for s in block.segments] == ["checked", "unchecked"]


# -- the whole conversation counts ---------------------------------------------------------------------
def test_prior_matrices_read_the_whole_conversation_not_the_history_window() -> None:
    from minerva_api.routers_chat import HISTORY_MESSAGES, _prior_matrices

    class Row:
        def __init__(self, role: str, content: dict[str, object]) -> None:
            self.role, self.content = role, content

    rows = [Row("user", {"text": "Let P and Q be matrices."})]
    rows += [Row("assistant", {"text": f"filler {i}"}) for i in range(HISTORY_MESSAGES + 5)]
    rows.append(
        Row(
            "assistant",
            {
                "text": "",
                "trace": [{"name": "matrix_calculus", "arguments": {"matrices": [{"name": "R"}]}}],
            },
        )
    )
    assert _prior_matrices(rows, "and $S$ is an operator")[:] == ["P", "Q", "R", "S"]  # type: ignore[arg-type]


# -- scalar tools refuse declared matrices ------------------------------------------------------------------
def test_scalar_tools_refuse_an_expression_with_a_declared_matrix(ctx: ToolContext) -> None:
    guarded = ToolContext(engine=ctx.engine, kb=ctx.kb, matrices=frozenset({"A", "B"}))
    out = execute("simplify", {"latex": r"(A+B)^2", "goal": "expand"}, guarded)
    assert out.status == "error" and out.evidence is not None and not out.evidence.accepted
    assert "matrix_calculus" in out.for_model
    assert out.known_latex == []
    # the same call about scalars still runs, and so does the call with no declaration
    assert execute("simplify", {"latex": r"(a+b)^2", "goal": "expand"}, guarded).status == "ok"
    assert execute("simplify", {"latex": r"(A+B)^2", "goal": "expand"}, ctx).status == "ok"
    # the matrix tool itself is never refused
    assert (
        execute(
            "matrix_calculus",
            {"latex": r"\exp(A)", "matrices": [{"name": "A", "rows": 2}], "action": "evaluate"},
            guarded,
        ).status
        != "error"
    )
