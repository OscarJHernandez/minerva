"""ODE: the assistant lists the applicable methods and asks which to run, unless the user named one."""

from __future__ import annotations

import pytest
from minerva_api.agent.choice import names_a_method


@pytest.mark.parametrize(
    "text",
    [
        "y''(x) = y(x).",
        "Solve this ODE: $y' = y$",
        "solve y'' + y = 0",
        "Can you help with this differential equation?",
        "what is the solution",  # "the solution" names no method
    ],
)
def test_a_bare_equation_or_a_plain_solve_names_no_method(text: str) -> None:
    assert not names_a_method(text)


@pytest.mark.parametrize(
    "text",
    [
        "Find the closed form of this ODE",
        "closed-form solution please",
        "solve it exactly",
        "Solve this ODE numerically, y(0)=1",
        "give me the power series",
        "Taylor expansion about 0",
        "do a perturbation expansion in epsilon",
        "draw the phase plane",
        "find the equilibria and their stability",
        "Give the general solution",
        "use Runge-Kutta",
    ],
)
def test_naming_a_method_is_recognised(text: str) -> None:
    assert names_a_method(text)


@pytest.mark.parametrize("text", ["the first one", "2", "option 3", "2.", "let's do the last one"])
def test_a_pick_by_position_counts_only_as_a_reply_to_a_list(text: str) -> None:
    assert not names_a_method(text)
    assert names_a_method(text, replying=True)


def test_a_number_inside_a_sentence_is_not_a_pick() -> None:
    assert not names_a_method("y(0)=2 and y'(0)=1", replying=True)
