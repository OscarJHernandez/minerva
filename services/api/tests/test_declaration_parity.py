"""The evaluation harness gives the loop the declarations the chat route would (A-008).

A single-turn case has no stored messages, so the harness reads its declarations from the plain history. The chat route
reads them from the stored conversation. Both go through `routers_chat.declarations_of`; this test makes sure that no
conversation can tell them apart, so a measurement made by the harness is a measurement of what users get.
"""

import pytest
from minerva_api.agent.evaluation.runner import plain_declarations
from minerva_api.agent.turn_context import _declaration_inputs, declarations_of


class Row:
    def __init__(self, role: str, text: str) -> None:
        self.role = role
        self.content: dict[str, object] = {"text": text}
        if role == "assistant":
            self.content["trace"] = []


CONVERSATIONS: list[tuple[str, list[tuple[str, str]], str]] = [
    (
        "matrices in words, same message",
        [],
        "A and B are 2x2 matrices. Show that $(A+B)^2 = A^2 + 2AB + B^2$.",
    ),
    ("matrices earlier, question later", [("user", "Let A and B be matrices.")], "Is $AB = BA$?"),
    (
        "a name the user later calls a scalar",
        [("user", "Let A and B be matrices."), ("user", "Actually A is a scalar.")],
        "Expand $(A+B)^2$.",
    ),
    (
        "a vector in words",
        [("user", "The position vector r points from the origin.")],
        "Differentiate $r^2$.",
    ),
    (
        "the assistant's reply mentions a vector",
        [
            ("user", "Find the conserved quantities."),
            ("assistant", "The Laplace-Runge-Lenz vector is conserved."),
        ],
        "And its magnitude?",
    ),
    (
        "matrices and a vector together",
        [("user", "A and B do not commute."), ("user", "Let v be a vector.")],
        "Simplify $A B v$.",
    ),
    ("nothing declared", [("user", "What is 2+2?")], "Differentiate $x^2$."),
]


@pytest.mark.parametrize(
    ("label", "earlier", "current"), CONVERSATIONS, ids=[c[0] for c in CONVERSATIONS]
)
def test_the_harness_and_the_chat_route_read_the_same_declarations(
    label: str, earlier: list[tuple[str, str]], current: str
) -> None:
    rows = [Row(role, text) for role, text in earlier]
    route = declarations_of(*_declaration_inputs(rows, current))._asdict()  # type: ignore[arg-type]
    history = [{"role": role, "content": text} for role, text in earlier]
    harness = plain_declarations([*history, {"role": "user", "content": current}])
    assert harness == route, label


def test_the_parity_is_not_vacuous() -> None:
    harness = plain_declarations(
        [{"role": "user", "content": "A and B are 2x2 matrices. Expand $(A+B)^2$."}]
    )
    assert set(harness["prior_matrices"]) == {"A", "B"} and set(harness["declared_for_tools"]) == {
        "A",
        "B",
    }
    vector = plain_declarations([{"role": "user", "content": "The position vector r is given."}])
    assert vector["vector_kind_names"] == frozenset({"r"})
