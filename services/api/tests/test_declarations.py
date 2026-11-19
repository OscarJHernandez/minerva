"""Which names a conversation declares as matrices or vectors, and where the refusal of a scalar
tool comes from. The strict reading gates tools; the generous one only leaves claims unchecked."""

from __future__ import annotations

import pytest
from minerva_api.agent.declarations import declared_matrices, declared_vectors, read_declarations
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

# The reply that made `r` a vector in the Kepler chat: three sentences that merely mention the words.
KEPLER_REPLY = (
    "3.  **Laplace-Runge-Lenz Vector component ($C_3$):**\n"
    r"    $$C_3 = - k \cos{\theta} + \mu r^3 \dot{\theta}^2 \cos{\theta}$$"
    "\n"
    '    This is a component of the Laplace-Runge-Lenz vector, which is a "hidden" symmetry '
    "unique to $1/r$ potentials.\n"
    "The tool also identified a dependent constant $C_4$ (the other component of the vector)."
)


def strict(*texts: str, role: str = "user") -> set[str]:
    return set(read_declarations([(role, t) for t in texts], strict=True))


def test_the_kepler_reply_declares_nothing_under_the_strict_reading() -> None:
    assert strict(KEPLER_REPLY, role="assistant") == set()
    # the generous reading (claims only) still takes the names, which is the cautious direction there
    assert {"r"} <= declared_vectors([KEPLER_REPLY])


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("Let A and B be 3x3 matrices.", {"A", "B"}),
        (r"Let $A$, $B$ be $n\times n$ matrices. Then", {"A", "B"}),
        ("X, Y are noncommuting operators.", {"X", "Y"}),
        ("Suppose A_1 and A_{2} do not commute", {"A_1", "A_2"}),
        (r"Let $\Omega$ and $\Lambda$ be matrices", {"Omega", "Lambda"}),
        ("and $S$ is an operator", {"S"}),
        ("matrices A and B", {"A", "B"}),
        (r"Let $\mathbf{E}$ and $\mathbf{B}$ be vector fields.", {"E", "B"}),
        (r"Here $a$ and $b_1$ are vectors. Also $s$ is a scalar.", {"a", "b_1"}),
        (r"Consider the vector $\mathbf{r}$ and a point.", {"r"}),
        ("a and b are vectors", {"a", "b"}),
        # a sentence that starts with a name: "A" here is not the article
        ("A and B are 2x2 matrices.", {"A", "B"}),
        ("A and B are matrices. Show that $(A+B)^2 = A^2 + 2AB + B^2$.", {"A", "B"}),
    ],
)
def test_real_declarations_are_still_read(text: str, names: set[str]) -> None:
    assert strict(text) == names


@pytest.mark.parametrize(
    "text",
    [
        "A matrix is invertible if its determinant is nonzero.",
        "A matrix and a vector are different objects.",  # the article, then a noun
        "The solution of x^2 = 4 is x = 2.",
        "Vectors are useful.",
        r"No keyword here: $x \times y$.",
        r"Take the Hamiltonian operator $H = T + V$ as given.",
        r"The vector $1/r$ potential",
        "I think the operator I used was fine.",
        "r is a scalar and the vector v is given.",
    ],
)
def test_mentions_and_expressions_are_not_declarations(text: str) -> None:
    assert not (strict(text) - ({"v"} if "vector v" in text else set()))


def test_a_user_sentence_clears_a_name_and_a_later_one_restores_it() -> None:
    rows = [("user", "Let A and B be matrices."), ("user", "Actually A is a scalar.")]
    assert set(read_declarations(rows, strict=True)) == {"B"}
    rows.append(("user", "Let A be a matrix again."))
    assert set(read_declarations(rows, strict=True)) == {"A", "B"}


def test_the_models_own_text_never_clears_a_declaration() -> None:
    rows = [("user", "Let A and B be matrices."), ("assistant", "Here A is a scalar.")]
    assert set(read_declarations(rows, strict=True)) == {"A", "B"}


def test_the_source_sentence_and_who_wrote_it_are_kept() -> None:
    found = read_declarations(
        [("user", "Hello. Let E and B be vector fields."), ("assistant", "Noted.")], strict=True
    )
    assert found["E"].role == "user" and found["E"].kind == "vector"
    assert found["E"].quote == "Let E and B be vector fields."


@pytest.fixture(scope="module")
def engine_ctx() -> ToolContext:
    return ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )


def test_the_refusal_quotes_the_sentence_and_tells_how_to_clear_it(engine_ctx: ToolContext) -> None:
    ctx = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"A", "B"}),
        matrix_sources={"A": "your message: «Let A and B be matrices.»"},
    )
    out = execute("simplify", {"latex": r"(A+B)^2", "goal": "expand"}, ctx)
    assert out.status == "error"
    assert "«Let A and B be matrices.»" in out.for_model
    assert "scalar" in out.for_model  # the way out: the user can say it is one
    assert execute("simplify", {"latex": r"(a+b)^2", "goal": "expand"}, ctx).status == "ok"


def test_generous_and_strict_agree_on_plain_declarations() -> None:
    text = "Let A and B be matrices."
    assert declared_matrices([text]) == strict(text) == {"A", "B"}


class _Row:
    def __init__(self, role: str, content: dict[str, object]) -> None:
        self.role, self.content = role, content


def test_the_kepler_conversation_does_not_block_scalar_tools_on_r() -> None:
    from minerva_api.routers_chat import _declared_for_tools, _prior_matrices

    rows = [
        _Row("user", {"text": "Can you find the conserved quantities?"}),
        _Row("assistant", {"text": KEPLER_REPLY, "trace": []}),
    ]
    current = "Please differentiate $C_1 = \\mu r^2 \\dot{\\theta}$ with respect to t"
    assert _declared_for_tools(rows, current) == {}  # type: ignore[arg-type]
    # claims keep the cautious reading: the formulas about the "vector" stay unchecked
    assert "r" in _prior_matrices(rows, current)  # type: ignore[arg-type]


def test_a_declared_name_is_refused_with_its_sentence_until_the_user_clears_it() -> None:
    from minerva_api.routers_chat import _declared_for_tools, _prior_matrices

    rows = [_Row("user", {"text": "Let A and B be matrices."}), _Row("assistant", {"text": "Ok."})]
    declared = _declared_for_tools(rows, "Simplify (A+B)^2")  # type: ignore[arg-type]
    assert set(declared) == {"A", "B"}
    assert "your message" in declared["A"] and "Let A and B be matrices." in declared["A"]
    after = _declared_for_tools(rows, "A is a scalar. Simplify (A+B)^2")  # type: ignore[arg-type]
    assert set(after) == {"B"}
    assert "A" not in _prior_matrices(rows, "A is a scalar. Simplify (A+B)^2")  # type: ignore[arg-type]


def test_a_tool_call_declaration_is_cleared_by_the_user_too() -> None:
    from minerva_api.routers_chat import _declared_for_tools

    rows = [
        _Row(
            "assistant",
            {
                "text": "",
                "trace": [{"name": "matrix_calculus", "arguments": {"matrices": [{"name": "R"}]}}],
            },
        )
    ]
    assert set(_declared_for_tools(rows, "Simplify R+R")) == {"R"}  # type: ignore[arg-type]
    assert _declared_for_tools(rows, "R is a scalar here") == {}  # type: ignore[arg-type]


def test_the_model_writing_c_underscore_1_is_marked_from_tool(engine_ctx: ToolContext) -> None:
    # Kepler discovery: the tool printed C1 = ..., the model wrote C_1 = ... and got "unchecked".
    from minerva_api.agent.segments import normalize, split_segments

    out = execute(
        "find_conserved_quantities",
        {
            "lagrangian": r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}",
            "coordinates": ["r", "theta"],
            "budget": "thorough",  # the search's own wall-clock limit must not depend on runner load
        },
        engine_ctx,
    )
    assert out.status == "ok"
    known = {normalize(x) for x in out.known_latex}
    marks = {
        s.value: s.status
        for s in split_segments(
            r"Angular momentum $C_1 = \mu r^2 \dot{\theta}$ and energy "
            r"$C_2 = \frac{1}{2} \mu \dot{r}^2 + \frac{1}{2} \mu r^2 \dot{\theta}^2 - \frac{k}{r}$.",
            known,
        )
        if s.type == "math"
    }
    assert marks[r"C_1 = \mu r^2 \dot{\theta}"] == "from_tool"


@pytest.mark.parametrize(
    "text",
    [
        "A and B are 2x2 matrices. Show that $(A+B)^2 = A^2 + 2AB + B^2$.",
        "A and B are matrices.",
        "A or B is a matrix.",
    ],
)
def test_a_sentence_that_starts_with_a_name_declares_it_in_the_generous_reading_too(
    text: str,
) -> None:
    assert declared_matrices([text]) == {"A", "B"}
