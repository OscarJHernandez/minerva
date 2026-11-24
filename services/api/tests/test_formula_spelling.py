"""A correct restatement of a verified result is not shown as unchecked because of how it is spelled.

Baseline live run (2026-11-18): `Compute $\\int_0^\\pi \\sin x \\, dx$` was answered correctly, with a verified tool result,
in all three repeats and failed all three: the reply's `\\int_0^\\pi \\sin x \\, dx = 2` did not match the tool's
`\\int\\limits_{0}^{\\pi} \\sin{\\left(x \\right)}\\, dx = 2` (spaced argument before `dx`, a braced command in a script), the
claim checker does not read integrals, so the formula stayed "unchecked". Spelling is not meaning; the wrong twins
below must still differ."""

from __future__ import annotations

import pytest
from minerva_api.agent.segments import normalize, split_segments

PRINTED = r"\int\limits_{0}^{\pi} \sin{\left(x \right)}\, dx = 2"


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (PRINTED, r"\int_0^\pi \sin x \, dx = 2"),
        (PRINTED, r"\int_0^{\pi} \sin(x)\,dx = 2"),
        (r"x^{\pi}", r"x^\pi"),
        (r"a_{\theta}", r"a_\theta"),
        (r"\sin x \, dx", r"\sin(x)dx"),
        (r"\cos^2 t \; dt", r"\cos^2(t)dt"),
        # issue #146: one Greek-letter argument, however the function is written
        (r"\cos{\left(\theta \right)}", r"\cos\theta"),
        (r"\cos{\theta}", r"\cos\theta"),
        (r"\cos(\theta)", r"\cos{\theta}"),
        (r"\sin^2{\theta}", r"\sin^{2}\theta"),
        (r"\sin\theta\cos\theta", r"\sin{\left(\theta \right)} \cos{\left(\theta \right)}"),
        (r"r^3 \dot\theta^2 \cos{\theta}", r"r^{3} \dot\theta^{2} \cos{\left(\theta \right)}"),
    ],
)
def test_one_reading_in_two_spellings_normalises_alike(left: str, right: str) -> None:
    assert normalize(left) == normalize(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (PRINTED, r"\int_0^\pi \sin x \, dx = 3"),  # a different value
        (PRINTED, r"\int_0^{2\pi} \sin x \, dx = 2"),  # a different limit
        (r"\sin x y", r"\sin(xy)"),  # the argument is x, not xy
        (r"\sin xy", r"\sin(xy)"),
        (r"x^{\pi x}", r"x^\pi x"),  # braces around more than one token do matter
        (r"x^{\pi}y", r"x^{\pi y}"),
        (r"\sin x \, dx", r"\sin(xd)x"),
        # issue #146: wrong twins of the Greek-letter rule
        (r"\cos{\theta}", r"\cos{\phi}"),  # another letter
        (r"\cos\theta", r"-\cos\theta"),  # a sign
        (r"\cos\theta + \sin\theta", r"\cos\theta"),  # a dropped term
        (
            r"\cos\theta\phi",
            r"\cos(\theta)\phi",
        ),  # `\phi` is not part of the argument, and not unified
        (r"\cos 2\theta", r"\cos(2\theta)"),
        (
            r"\cos\theta^2",
            r"\cos(\theta)^2",
        ),  # a power of the argument is not a power of the function
        (r"\cos\theta", r"\cos 2\theta"),
        (r"\dot{\theta}", r"\ddot{\theta}"),  # \dot is not \ddot
        (r"\cos{\theta}", r"\sin{\theta}"),
    ],
)
def test_different_readings_stay_different(left: str, right: str) -> None:
    assert normalize(left) != normalize(right)


def test_a_restated_integral_is_marked_from_the_tool_and_a_wrong_one_is_not() -> None:
    known = {normalize(PRINTED), normalize("2")}
    right = split_segments(r"The value is $$\int_0^\pi \sin x \, dx = 2$$", known)
    assert [s.status for s in right if s.type != "text"] == ["from_tool"]
    wrong = split_segments(r"The value is $$\int_0^\pi \sin x \, dx = 3$$", known)
    assert [s.status for s in wrong if s.type != "text"] == ["unchecked"]


def test_a_conversation_numbered_before_the_greek_rule_keeps_every_number_and_gains_no_alias() -> (
    None
):
    """`normalize` also keys the equation table: formulas stored under the old keys are restored under their own
    numbers, and none becomes an alias of another (they were different formulas before and still are)."""
    from minerva_api.agent.equations import EquationTable

    stored = [
        (1, r"E = m c^2"),
        (
            2,
            r"C_{3} = - k \cos{\left(\theta \right)} + m r^{3} \dot{\theta}^{2} \cos{\left(\theta \right)}",
        ),
        (3, r"C_{4} = k \sin{\left(\theta \right)}"),
        (4, r"y = \cos{\phi}"),
        (5, r"y = \cos\theta"),
        (6, r"\ddot{\theta} = 0"),
        (7, r"\dot{\theta} = 0"),
    ]
    table = EquationTable(counter=7)
    for number, latex in stored:
        table.restore(number, latex, "written_by_model", mode="none")
    assert sorted(table.items) == [n for n, _ in stored]
    assert all(item.alias_of is None for item in table.items.values())
    # and a restated formula is found under its own number whatever the spelling
    assert table.number_for(r"y = \cos{\theta}", "written_by_model", mode="model") == 5
    assert table.counter >= 7
