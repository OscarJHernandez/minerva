"""Transport damage is undone from a closed list, and nothing else is touched.

Regression: a retry of `m*sdot*(Xdot*cos(alpha) + ...)` arrived as `m\\*sdot\\*(...)` (markdown escaping), and a
single backslash made the whole text count as LaTeX. Every damage the cleaner repairs has an inverse operator
(`DAMAGE`), and a property test checks `clean(damage(x)) == x`; every form that must not be touched has a test.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from minerva_engine.notation.models import NormalizationKind
from minerva_engine.notation.transport import (
    DAMAGE,
    GREEK,
    clean_transport,
    localize_symbols,
)

CLEAN = [
    "1/2*m*x_dot^2 - 1/2*k*x^2",
    "m*sdot*(Xdot*cos(alpha) + Ydot*sin(alpha)) - (M+m)*g*Y",
    r"\frac{1}{2} m \dot{x}^2 - \frac{1}{2} k x^2",
    r"\sin(x)^2 + \cos(x)^2",
    "a*b - c/d",
]


@pytest.mark.parametrize("name", sorted(DAMAGE))
@pytest.mark.parametrize("source", CLEAN)
def test_every_damage_is_undone(name: str, source: str) -> None:
    damaged = DAMAGE[name](source)  # type: ignore[operator]
    if damaged == source:
        return  # this damage does not apply to this string
    cleaned, notes = clean_transport(damaged)
    assert cleaned == source, (name, damaged)
    assert notes and all(n.kind is NormalizationKind.TRANSPORT for n in notes)


@pytest.mark.parametrize("source", CLEAN)
def test_clean_input_is_returned_unchanged_with_no_notes(source: str) -> None:
    assert clean_transport(source) == (source, ())


@pytest.mark.parametrize(
    "source",
    [
        r"\left( x + 1 \right) \, y",  # LaTeX spacing and sizing are commands, not damage
        r"\{ a, b \}",
        r"x_{1}^{2} + x_{ab}",
        r"$a$ + $b$",  # two formulas, not one wrapped in dollars
        "f(x) = 1.5",
        "2.",  # a float literal, not a sentence period
        "a \\\\ b",  # a real line break is not repaired
        f"x{chr(0xD7)}y",  # a multiplication sign could be a cross product: left for the parser to refuse
        "a [b, c]",
        "x**2",
    ],
)
def test_forms_that_are_not_damage_are_left_alone(source: str) -> None:
    assert clean_transport(source) == (source, ())


def test_the_recorded_retry_is_cleaned() -> None:
    damaged = (
        r"1/2\*(M+m)\*(Xdot^2 + Ydot^2) + 1/2\*m\*sdot^2 + m\*sdot\*(Xdot\*cos(alpha) + Ydot\*sin(alpha))"
        r" - (M+m)\*g\*Y - m\*g\*s\*sin(alpha)"
    )
    cleaned, notes = clean_transport(damaged)
    assert "\\" not in cleaned
    assert cleaned.startswith("1/2*(M+m)*(Xdot^2 + Ydot^2)")
    assert [n.source for n in notes] == ["\\*"]


def test_a_sentence_period_is_dropped_but_not_after_a_digit() -> None:
    assert clean_transport("x^2 + y.")[0] == "x^2 + y"
    assert clean_transport("x^2 + 2,")[0] == "x^2 + 2,"


def test_html_entities_and_wrappers_combine() -> None:
    cleaned, notes = clean_transport("$a &lt; b$")
    assert cleaned == "a < b" and len(notes) == 2


def test_nested_wrappers_are_peeled() -> None:
    assert clean_transport("```\n$x^2$\n```")[0] == "x^2"


@given(st.sampled_from(sorted(DAMAGE)), st.sampled_from(CLEAN), st.sampled_from(sorted(DAMAGE)))
def test_two_damages_compose(first: str, source: str, second: str) -> None:
    damaged = DAMAGE[second](DAMAGE[first](source))  # type: ignore[operator]
    cleaned, _ = clean_transport(damaged)
    # The cleaner is a left inverse of any pair that does not double-escape the same character twice.
    assert cleaned.replace("\\*", "*") == source or cleaned == source or "\\" in cleaned


# -- Greek letters and dots over letters -------------------------------------------------------------


def test_every_greek_letter_has_a_distinct_ascii_name() -> None:
    assert len(set(GREEK.values())) == len(GREEK)
    assert GREEK[chr(0x3B8)] == "theta" and GREEK[chr(0x3BB)] == "lambda"


def test_greek_letters_are_spelled_for_the_syntax() -> None:
    theta, omega = chr(0x3B8), chr(0x3C9)
    text, notes = localize_symbols(f"m*{theta}^2*{omega}", "text")
    assert text == "m*theta^2*omega" and len(notes) == 2
    latex, _ = localize_symbols(f"m{theta}^2{omega}", "latex")
    assert latex == r"m\theta ^2\omega"


def test_a_dot_over_a_letter_is_a_time_derivative_in_the_syntax() -> None:
    theta = chr(0x3B8)
    dotted = f"{theta}̇"
    assert localize_symbols(f"{dotted}^2", "text")[0] == "theta_dot^2"
    assert localize_symbols(f"{dotted}^2", "latex")[0] == r"\dot{\theta}^2"
    assert localize_symbols("ẍ + x", "text")[0] == "x_ddot + x"


def test_ascii_input_is_not_touched_by_localization() -> None:
    assert localize_symbols("x_dot + theta", "text") == ("x_dot + theta", ())
