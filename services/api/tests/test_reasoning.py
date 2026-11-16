"""Reasoning that leaks into a reply: dropped from what is shown, stored and replayed."""

from __future__ import annotations

import pytest
from minerva_api.agent.reasoning import ReasoningGate, clean_reply


def run(*chunks: str) -> str:
    gate = ReasoningGate()
    return "".join(gate.feed(c) for c in chunks) + gate.finish()


def test_a_leading_reasoning_block_is_dropped_with_its_marker() -> None:
    assert run("thought\nI should call a tool.\n<channel|>The answer is 4.") == "The answer is 4."
    assert run("\n\nthought\n<channel|>Sorry about that.") == "Sorry about that."
    # the second leak of the report: an empty reasoning block, twice, then the reply
    assert run("thought\n<channel|>", "thought\n<channel|>Done.") == "thought\nDone."


def test_reasoning_that_never_reaches_its_marker_shows_nothing() -> None:
    # the call ended at the length limit inside the reasoning (the report's "ERm? what happened?")
    assert run("thought\n导出的结果 is f'(r). Wait, I should check", " the rules...\nActually,") == ""


def test_the_marker_may_arrive_split_across_chunks() -> None:
    assert run("thou", "ght\nreasoning <chan", "nel|>Hello", " there") == "Hello there"
    assert run("Hello <chan", "nel|>there") == "Hello there"
    assert run("a <", "b") == "a <b"  # a lone "<" is text


def test_ordinary_text_streams_and_is_untouched() -> None:
    gate = ReasoningGate()
    assert gate.feed("The Kepler Lagrangian is") == "The Kepler Lagrangian is"
    assert gate.feed(" $L = T - V$.") == " $L = T - V$."
    assert gate.finish() == ""
    assert run("thoughtful people", " agree") == "thoughtful people agree"
    assert run("thought about it\nand more") == "thought about it\nand more"
    assert run("tho", "ugh it is") == "though it is"
    assert run("thought") == "thought"


def test_a_leak_is_held_for_only_a_few_characters() -> None:
    gate = ReasoningGate()
    assert gate.feed("I will compute") == "I will compute"  # decided at once: not a leak
    assert not gate.leaked
    gate = ReasoningGate()
    assert gate.feed("thought\nabc") == "" and gate.leaked


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer.\n\nthought\nreasoning here\n<channel|>Final.", "Answer.\n\nFinal."),
        ("Answer.\n\nthought\nreasoning that was cut off", "Answer."),
        ("thought\nx\n<channel|>Real text <channel|>more", "Real text more"),
        ("A thought on r: it is $r_0$.", "A thought on r: it is $r_0$."),
        ("No leak at all.", "No leak at all."),
        ("", ""),
    ],
)
def test_a_whole_text_is_cleaned(text: str, expected: str) -> None:
    assert clean_reply(text) == expected


def test_an_earlier_reply_that_leaked_is_replayed_without_the_leak() -> None:
    from minerva_api.agent.equations import replay_prose

    stored = {
        "text": "To linearise we need r_0.\n\nthought\nWait, rule 1 says call a tool.\nActually,",
        "blocks": [],
    }
    assert replay_prose(stored) == "To linearise we need r_0."
