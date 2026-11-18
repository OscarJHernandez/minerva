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
