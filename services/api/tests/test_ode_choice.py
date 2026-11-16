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


# --- numbers: the user hands the choice to the assistant --------------------------------------
from minerva_api.agent.choice import delegates_numbers, delegation_active, numbers_in  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "Yes just choose something reasonable here",
        "you choose",
        "Pick reasonable values please",
        "use your example",
        "use those values",
        "whatever you like",
        "I don't mind, up to you",
        "choose some sensible numbers",
        "Go with your suggested values",
    ],
)
def test_a_user_who_hands_the_numbers_over_is_recognised(text: str) -> None:
    assert delegates_numbers(text)


@pytest.mark.parametrize(
    "text",
    [
        "Which values are reasonable?",
        "What numbers should I choose?",
        "how do I pick values for a pendulum?",
        "I want to solve this numerically",
        "choose the first method",
        "G=1, M=1, m=1",
        "Yes",
        "try again",
    ],
)
def test_a_question_or_a_plain_message_is_not_a_hand_over(text: str) -> None:
    assert not delegates_numbers(text)


def test_numbers_in_a_text_skip_digits_inside_names() -> None:
    assert numbers_in("y1(0)=1, r_0 = 0.5, k=1e-3, 3/4, x^2, -2") == [0.0, 1.0, 0.5, 0.001, 0.75, -2.0]


def test_a_hand_over_lasts_two_more_messages_and_ends_when_the_user_gives_numbers() -> None:
    assert delegation_active(["choose something reasonable"])
    assert delegation_active(["choose something reasonable", "try again"])
    assert delegation_active(["choose something reasonable", "try again", "go on"])
    assert not delegation_active(["choose something reasonable", "a", "b", "c"])
    assert not delegation_active(["choose something reasonable", "ok", "k = 2"])
    assert not delegation_active(["ok", "x"])
    assert not delegation_active([])
