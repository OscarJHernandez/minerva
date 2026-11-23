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


# -- A-009: `check_identity` decides by sampling commuting numbers, so it must refuse a declared matrix -----------
def test_check_identity_refuses_a_declared_matrix_and_still_answers_for_scalars(
    engine_ctx: ToolContext,
) -> None:
    ctx = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"A", "B"}),
        matrix_sources={"A": "a matrix_calculus call earlier in this conversation"},
    )
    false_for_matrices = {"latex": "(A+B)^2 = A^2 + 2*A*B + B^2"}
    refused = execute("check_identity", false_for_matrices, ctx)
    assert refused.status == "error" and "matrix_calculus" in refused.for_model
    assert "proved" not in refused.for_model.lower()  # a refusal is never a verdict
    # the same equation about numbers is a true identity and is still answered
    scalar = {"latex": "(a+b)^2 = a^2 + 2*a*b + b^2"}
    assert execute("check_identity", scalar, ctx).status == "ok"
    # and without any declaration the tool is unchanged (nothing says A and B are matrices)
    plain = ToolContext(engine=engine_ctx.engine, kb=engine_ctx.kb)
    assert execute("check_identity", false_for_matrices, plain).status == "ok"


def test_rewrite_part_and_substitute_refuse_a_declared_matrix_unless_the_call_says_noncommutative(
    engine_ctx: ToolContext,
) -> None:
    """`rewrite_part` expanded `(A+B)^2` to `A^2 + 2AB + B^2` for declared matrices and showed it checked at 20/20
    points. These tools do handle noncommuting symbols, but only when the call declares them so (A-009)."""
    ctx = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"A", "B"}),
        matrix_sources={"A": "a matrix_calculus call earlier in this conversation"},
    )
    wrong = execute("rewrite_part", {"latex": "(A+B)^2", "action": "expand"}, ctx)
    assert wrong.status == "error" and "noncommutative" in wrong.for_model
    sub = execute("substitute", {"latex": "A*B", "relations": ["A = B"]}, ctx)
    assert sub.status == "error" and "noncommutative" in sub.for_model
    # declared noncommutative in the call: the tool is exact and the answer keeps both orders
    nc = {"A": ["noncommutative"], "B": ["noncommutative"]}
    right = execute(
        "rewrite_part", {"latex": "(A+B)^2", "action": "expand", "assumptions": nc}, ctx
    )
    assert right.status == "ok"
    assert "B A" in right.for_model and "A B" in right.for_model
    # a name the conversation did not declare is untouched
    assert execute("rewrite_part", {"latex": "(a+b)^2", "action": "expand"}, ctx).status == "ok"


# -- A-009 (continued): `sums` and the Lagrangian tools treat symbols as commuting numbers, but only a declared MATRIX
#    makes them wrong; a vector's coordinate or components are scalars ---------------------------------------------
LAGRANGIAN = r"\frac12 A \dot{x}^2 - \frac12 B x^2"
SUM_INTS = {"n": ["integer", "positive"]}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            "sums",
            {"latex": r"\sum_{i=1}^{n} (A_i + B_i)^2", "action": "expand", "assumptions": SUM_INTS},
        ),
        ("analyze_lagrangian", {"lagrangian": LAGRANGIAN, "coordinates": ["x"]}),
        (
            "expand_motion",
            {"lagrangian": LAGRANGIAN, "coordinates": ["x"], "equilibrium": {"x": "0"}},
        ),
        ("find_conserved_quantities", {"lagrangian": LAGRANGIAN, "coordinates": ["x"]}),
        ("search_invariants", {"lagrangian": LAGRANGIAN, "coordinates": ["x"]}),
    ],
)
def test_the_sums_and_lagrangian_tools_refuse_a_declared_matrix(
    engine_ctx: ToolContext, tool: str, arguments: dict[str, object]
) -> None:
    matrices = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"A", "B"}),
        matrix_sources={"A": "your message: «Let A and B be matrices.»"},
    )
    refused = execute(tool, arguments, matrices)
    assert refused.status == "error" and "matrix_calculus" in refused.for_model, tool
    # the same call with nothing declared is unchanged (it is a valid question about numbers)
    assert execute(tool, arguments, engine_ctx).status in {
        "ok",
        "conditions_not_satisfied",
        "unchanged",
    }


def test_a_declared_vector_does_not_block_the_tools_only_a_matrix_makes_wrong(
    engine_ctx: ToolContext,
) -> None:
    """ "The vector r" is common in a Kepler conversation, and `r` is also the radial coordinate of its Lagrangian:
    declaring a vector must not stop `analyze_lagrangian` or `sums` over its components `r_i`."""
    ctx = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"r"}),
        vector_names=frozenset({"r"}),
        matrix_sources={"r": "your message: «the position vector r»"},
    )
    kepler = r"\frac12 m (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}"
    assert (
        execute(
            "analyze_lagrangian", {"lagrangian": kepler, "coordinates": ["r", "theta"]}, ctx
        ).status
        == "ok"
    )
    ints = {"n": ["integer", "positive"]}
    squared = execute(
        "sums",
        {"latex": r"\sum_{i=1}^{n} (r_i + 1)^2", "action": "expand", "assumptions": ints},
        ctx,
    )
    assert squared.status == "ok"
    # the scalar tools that a vector does break are still refused for it
    assert execute("simplify", {"latex": r"(r+1)^2", "goal": "expand"}, ctx).status == "error"


# -- issue #132: the position vector and its length are two different names ---------------------------

KEPLER_VECTOR_REPLY = (
    r"Take the relative position vector $\mathbf{r}$, where $r = |\mathbf{r}|$ is the distance. "
    r"In polar form $L = \frac{\mu}{2}(\dot{r}^2 + r^2\dot{\theta}^2) + \frac{GM\mu}{r}$."
)
POLAR_LAGRANGIAN = r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{G M \mu}{r}"


def test_a_norm_definition_makes_the_bare_letter_a_scalar_whoever_wrote_it() -> None:
    for role in ("assistant", "user"):
        assert strict(KEPLER_VECTOR_REPLY, role=role) == set()
        assert read_declarations([(role, KEPLER_VECTOR_REPLY)], strict=False) == {}


@pytest.mark.parametrize(
    ("text", "names"),
    [
        # the same sentence without the definition still declares the vector
        (r"Take the position vector $\mathbf{r}$ of the body.", {"r"}),
        # the norm of another letter, or of a bare letter, is not a definition of `r`
        (r"The vector $\mathbf{r}$, where $s = |\mathbf{r}|$ is its length.", {"r"}),
        (r"The vector $\mathbf{r}$, where $r = |x|$.", {"r"}),
        # a bare letter named after the keyword stays a declaration
        (r"The vector $r$, where $r = |\mathbf{r}|$.", set()),
    ],
)
def test_only_the_norm_of_the_same_letter_defines_a_scalar(text: str, names: set[str]) -> None:
    assert strict(text, role="assistant") == names


def _rows(*pairs: tuple[str, str]) -> list[_Row]:
    return [_Row(role, {"text": text, "trace": []}) for role, text in pairs]


def test_the_polar_kepler_conversation_declares_no_vector_and_no_matrix() -> None:
    from minerva_api.agent.turn_context import _declaration_inputs, declarations_of

    rows = _rows(
        ("user", "Give the Kepler Lagrangian in vector and polar form."),
        ("assistant", KEPLER_VECTOR_REPLY),
    )
    found = declarations_of(*_declaration_inputs(rows, "Differentiate $C_1$ in time"))  # type: ignore[arg-type]
    assert found.declared_for_tools == {} and not found.vector_kind_names
    assert "r" not in found.prior_matrices and "r" not in found.prior_vectors
    assert "r" in found.scalar_names


def test_a_vector_name_goes_to_the_vector_list_not_to_the_matrix_list() -> None:
    from minerva_api.agent.turn_context import _declaration_inputs, declarations_of

    rows = _rows(("user", r"Let $\mathbf{E}$ be a vector field and let A and B be matrices."))
    found = declarations_of(*_declaration_inputs(rows, ""))  # type: ignore[arg-type]
    assert "E" in found.prior_vectors and "E" not in found.prior_matrices
    assert {"A", "B"} <= set(found.prior_matrices) and not {"A", "B"} & set(found.prior_vectors)


def test_a_scalar_the_user_named_stays_a_scalar_when_the_model_repeats_the_vector() -> None:
    from minerva_api.agent.turn_context import _declaration_inputs, declarations_of

    again = r"Here the position vector $\mathbf{r}$ is used again."
    rows = _rows(
        ("assistant", r"Let the position vector $\mathbf{r}$ be given."),
        ("user", "Fix your tool call. r is a scalar."),
        ("assistant", again),
    )
    found = declarations_of(*_declaration_inputs(rows, ""))  # type: ignore[arg-type]
    assert "r" not in found.declared_for_tools and "r" not in found.prior_vectors
    # ... and only the user can bring it back
    rows.append(_Row("user", {"text": r"Let the vector $r$ be a field again.", "trace": []}))
    back = declarations_of(*_declaration_inputs(rows, ""))  # type: ignore[arg-type]
    assert "r" in back.declared_for_tools


def test_a_user_sentence_about_one_name_does_not_hold_the_other_names_of_the_sentence() -> None:
    from minerva_api.agent.turn_context import _declaration_inputs, declarations_of

    rows = _rows(
        ("user", "Let A and B be matrices."),
        ("user", "Treat x as a scalar, and keep B in mind."),
        ("assistant", "Let A and B be matrices again."),
    )
    found = declarations_of(*_declaration_inputs(rows, ""))  # type: ignore[arg-type]
    assert {"A", "B"} <= set(found.declared_for_tools)


def test_a_declared_vector_in_the_lagrangian_does_not_block_time_derivative(
    engine_ctx: ToolContext,
) -> None:
    system = {"lagrangian": POLAR_LAGRANGIAN, "coordinates": ["r", "theta"]}
    vector_r = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"r"}),
        vector_names=frozenset({"r"}),
        matrix_sources={"r": "my earlier reply: «the vector r»"},
    )
    # the quantity has no `r` in it: the Lagrangian's coordinates are not the quantity
    ok = execute("time_derivative", {"latex": r"\mu \dot{\theta}", **system}, vector_r)
    assert ok.status == "ok", ok.for_model
    # a quantity that does contain the declared vector is still refused
    refused = execute("time_derivative", {"latex": r"\mu r^2 \dot{\theta}", **system}, vector_r)
    assert refused.status == "error" and "declared as a matrix" in refused.for_model


def test_a_declared_matrix_in_the_lagrangian_still_blocks_time_derivative(
    engine_ctx: ToolContext,
) -> None:
    matrix_r = ToolContext(
        engine=engine_ctx.engine,
        kb=engine_ctx.kb,
        matrices=frozenset({"r"}),
        matrix_sources={"r": "your message: «Let r be a matrix.»"},
    )
    out = execute(
        "time_derivative",
        {
            "latex": r"\mu \dot{\theta}",
            "lagrangian": POLAR_LAGRANGIAN,
            "coordinates": ["r", "theta"],
        },
        matrix_r,
    )
    assert out.status == "error" and "declared as a matrix" in out.for_model


def test_a_vector_name_gets_the_vector_wording_in_the_claim_check(engine_ctx: ToolContext) -> None:
    prose = r"For the field, $(r + 1)^2 = r^2 + 2r + 1$."
    as_vector = engine_ctx.engine.check_claims([r"(r+1)^2 = r^2 + 2 r + 1"], {}, [], [], ["r"], [])
    as_matrix = engine_ctx.engine.check_claims([r"(r+1)^2 = r^2 + 2 r + 1"], {}, [], ["r"], [], [])
    assert prose and as_vector[0].status == as_matrix[0].status == "unverified"
    assert "vector" in as_vector[0].detail and "matrix" not in as_vector[0].detail
    assert "matrix" in as_matrix[0].detail
