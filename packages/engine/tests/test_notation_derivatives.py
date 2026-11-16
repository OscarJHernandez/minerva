"""Derivative spellings are read by one reader, uniquely, or refused with a typed diagnostic.

Regression: a model wrote the velocities of a Lagrangian as `Xdot`, `Ydot`, `sdot`; the engine knew only
`X_dot`, so the analysis failed and a retry cycle began. Every spelling the reader generates must give the same
expression as the canonical one; every ambiguity must be refused with the candidates (never guessed); and a
spelling must never turn an ordinary symbol into a velocity when the name is declared as something else.
"""

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st
from minerva_engine.errors import ParseError
from minerva_engine.lagrangian import analyze_lagrangian
from minerva_engine.models import Limits, Status
from minerva_engine.notation.derivatives import (
    canonical,
    join_spaced_keywords,
    orders_of,
    refuse_latex_letter_runs,
    resolve_derivatives,
    spellings,
)
from minerva_engine.notation.models import InputCode
from minerva_engine.textparse import parse_text

STEMS = ["x", "X", "q", "q1", "theta", "theta_1", "s", "alpha"]
LIMITS = Limits(timeout_seconds=60)


def read(source: str, dependents: list[str], assumptions: dict[str, list[str]] | None = None):  # type: ignore[no-untyped-def]
    return resolve_derivatives(parse_text(source, assumptions), dependents, "t", assumptions or {})


@pytest.mark.parametrize("stem", STEMS)
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_every_generated_spelling_is_read_as_the_canonical_symbol(stem: str, order: int) -> None:
    target = canonical(stem, order)
    for name in spellings(stem, order):
        assert order in orders_of(name, stem, "t"), name
        got = read(f"{name}**2 + {stem}", [stem])
        want = parse_text(f"{target}**2 + {stem}")
        assert got.expr == want, name
        assert [n.target for n in got.normalizations] == [target]


def test_the_stem_is_matched_exactly() -> None:
    """`xdot` is the velocity of `x`, never of `X`: the stem keeps its case."""
    assert orders_of("xdot", "x", "t") == {1}
    assert orders_of("xdot", "X", "t") == set()
    resolved = read("Xdot + x", ["x"])
    # `Xdot` is not a spelling of any declared stem: it stays an ordinary symbol (here a parameter).
    assert resolved.expr == parse_text("Xdot + x")
    assert resolved.normalizations == ()


def test_the_canonical_spelling_is_untouched() -> None:
    resolved = read("x_dot**2 + x_ddot", ["x"])
    assert resolved.expr == parse_text("x_dot**2 + x_ddot")
    assert resolved.normalizations == ()


def test_a_declared_name_is_never_rewritten() -> None:
    """With `x` and `xdot` both declared, `xdot` is its own coordinate, not the velocity of `x`."""
    resolved = read("xdot**2 + x", ["x", "xdot"])
    assert resolved.expr == parse_text("xdot**2 + x")
    assert resolved.normalizations == ()


def test_two_readings_are_refused_with_the_candidates() -> None:
    with pytest.raises(ParseError) as info:
        read("xddot", ["x", "xd"])
    assert info.value.code is InputCode.AMBIGUOUS_DERIVATIVE
    assert set(info.value.suggestions) == {"x_ddot", "xd_dot"}


def test_two_spellings_of_one_velocity_are_refused() -> None:
    for source in ("xdot + x_dot", "xdot + x_DOT", "xdot*dx_dt"):
        with pytest.raises(ParseError) as info:
            read(source, ["x"])
        assert info.value.code is InputCode.MIXED_DERIVATIVE_SPELLINGS, source
        assert info.value.suggestions == ("x_dot",)


def test_a_name_with_declared_assumptions_is_not_silently_a_velocity() -> None:
    with pytest.raises(ParseError) as info:
        read("xdot", ["x"], {"xdot": ["positive"]})
    assert info.value.code is InputCode.AMBIGUOUS_DERIVATIVE


def test_a_velocity_of_an_undeclared_coordinate_is_refused_not_a_parameter() -> None:
    with pytest.raises(ParseError) as info:
        read("x_dot**2 + y_dot**2", ["x"])
    assert info.value.code is InputCode.UNDECLARED_DEPENDENT
    assert info.value.suggestions == ("y",)
    # An alias is refused only when the stem itself also appears (a coordinate that was forgotten).
    with pytest.raises(ParseError):
        read("x_dot**2 + ydot**2 + y", ["x"])
    assert (
        read("x_dot**2 + ydot**2", ["x"]).normalizations == ()
    )  # `ydot` alone: an ordinary symbol


def test_assumptions_of_the_canonical_name_are_carried_to_the_alias() -> None:
    resolved = read("xdot", ["x"], {"x_dot": ["real"]})
    (symbol,) = resolved.expr.free_symbols
    assert symbol == sp.Symbol("x_dot", real=True)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("x dot**2", "x_dot**2"),
        ("1/2*m*(x dot)**2", "1/2*m*(x_dot)**2"),
        ("theta ddot + theta", "theta_ddot + theta"),
        ("dot x + y", "x_dot + y"),
    ],
)
def test_two_adjacent_names_are_joined_only_for_a_declared_dependent(
    source: str, expected: str
) -> None:
    out, notes = join_spaced_keywords(source, ["x", "theta"])
    assert out == expected
    assert notes and all(n.target.endswith(("_dot", "_ddot")) for n in notes)


def test_spaced_names_for_an_undeclared_name_are_left_alone() -> None:
    assert join_spaced_keywords("y dot + 1", ["x"]) == ("y dot + 1", ())
    assert join_spaced_keywords("a b", ["a"]) == ("a b", ())


@pytest.mark.parametrize("source", ["Xdot**2", "1/2 m Xdot^2", "X_dot + 1", r"Xddot"])
def test_a_latex_letter_run_for_a_declared_dependent_is_refused_with_the_fix(source: str) -> None:
    with pytest.raises(ParseError) as info:
        refuse_latex_letter_runs(source, ["X"])
    assert info.value.code is InputCode.UNRECOGNIZED_DERIVATIVE
    assert any("\\" in s for s in info.value.suggestions)


@pytest.mark.parametrize("source", [r"\dot{X}^2 + X", r"\frac12 m \dot{X}^2", r"Xdot", r"m Y dot"])
def test_other_latex_is_not_inspected(source: str) -> None:
    refuse_latex_letter_runs(source, ["Y" if source == "Xdot" else "Z"])


def test_a_greek_coordinate_run_is_refused_with_its_command() -> None:
    with pytest.raises(ParseError) as info:
        refuse_latex_letter_runs(r"\frac12 thetadot^2", ["theta"])
    assert r"\dot{\theta}" in info.value.suggestions


# -- end to end: the recorded failure ----------------------------------------------------------------

WEDGE = (
    "1/2*(M+m)*(Xdot^2 + Ydot^2) + 1/2*m*sdot^2 + m*sdot*(Xdot*cos(alpha) + Ydot*sin(alpha))"
    " - (M+m)*g*Y - m*g*s*sin(alpha)"
)
WEDGE_CANONICAL = (
    "1/2*(M+m)*(X_dot^2 + Y_dot^2) + 1/2*m*s_dot^2 + m*s_dot*(X_dot*cos(alpha) + Y_dot*sin(alpha))"
    " - (M+m)*g*Y - m*g*s*sin(alpha)"
)


def _report(source: str, coordinates: list[str], fmt: str = "text"):  # type: ignore[no-untyped-def]
    return analyze_lagrangian(source, coordinates, "t", {}, LIMITS, fmt)  # type: ignore[arg-type]


def test_the_recorded_failure_is_analysed_and_the_reading_is_reported() -> None:
    report = _report(WEDGE, ["X", "Y", "s"])
    assert report.status is Status.OK
    # Derived by hand from the mass matrix [[M+m,0,m cos a],[0,M+m,m sin a],[m cos a,m sin a,m]]
    # (determinant M m (M+m)): the block and wedge do not interact in this Lagrangian.
    assert [a.latex for a in report.accelerations] == [
        r"\ddot{X} = 0",
        r"\ddot{Y} = - g",
        r"\ddot{s} = 0",
    ]
    assert report.reading is not None and report.reading.format == "text"
    assert {(n.source, n.target) for n in report.reading.normalizations} == {
        ("Xdot", "X_dot"),
        ("Ydot", "Y_dot"),
        ("sdot", "s_dot"),
    }


def test_the_spellings_give_the_same_report_as_the_canonical_input() -> None:
    plain, canonical_report = (
        _report(WEDGE, ["X", "Y", "s"]),
        _report(WEDGE_CANONICAL, ["X", "Y", "s"]),
    )
    assert [e.latex for e in plain.equations] == [e.latex for e in canonical_report.equations]
    assert [m.latex for m in plain.momenta] == [m.latex for m in canonical_report.momenta]
    assert plain.restrictions_latex == canonical_report.restrictions_latex
    assert canonical_report.reading is not None and canonical_report.reading.normalizations == []


def test_the_failure_diagnostic_is_typed_for_the_latex_letter_run() -> None:
    report = _report(WEDGE, ["X", "Y", "s"], "latex")
    assert report.status is Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.UNRECOGNIZED_DERIVATIVE
    assert report.diagnostic.field == "lagrangian" and report.diagnostic.recoverable


def test_a_forgotten_coordinate_is_named_with_its_code() -> None:
    report = _report("1/2*m*(x_dot^2 + y_dot^2)", ["x"])
    assert report.status is Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.UNDECLARED_DEPENDENT
    assert report.diagnostic.suggestions == ["y"]


def test_a_coordinate_that_is_absent_gets_the_declared_name_code() -> None:
    report = _report("1/2*m*x_dot^2", ["x", "z"])
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.DECLARED_NAME_ABSENT
    assert report.diagnostic.symbols == ["z"]


def test_an_equation_is_refused_with_a_code_and_the_fix() -> None:
    report = _report("L = 1/2*m*x_dot^2 - 1/2*k*x^2", ["x"])
    assert report.status is Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.EQUATION_NOT_EXPRESSION


def test_a_declared_coordinate_named_like_a_spelling_is_not_rewritten_end_to_end() -> None:
    """Coordinates `x` and `xdot`: the false twin of the rewrite. `xdot` is not the velocity of `x`."""
    report = _report("1/2*m*(x_dot^2 + xdot_dot^2) - 1/2*k*(x^2 + xdot^2)", ["x", "xdot"])
    assert report.status is Status.OK
    assert report.reading is not None and report.reading.normalizations == []
    assert len(report.accelerations) == 2


# -- property: a name is read one way or refused ---------------------------------------------------


@given(
    st.lists(st.sampled_from(STEMS), min_size=1, max_size=3, unique=True),
    st.data(),
)
def test_a_spelling_resolves_to_one_tree_or_is_refused(
    stems: list[str], data: st.DataObject
) -> None:
    terms: list[str] = []
    wanted: list[str] = []
    for stem in stems:
        order = data.draw(st.integers(1, 2))
        name = data.draw(st.sampled_from(spellings(stem, order)))
        terms.append(f"{name}**2")
        wanted.append(f"{canonical(stem, order)}**2")
    source = " + ".join([*terms, *stems])
    try:
        got = resolve_derivatives(parse_text(source), stems, "t", {})
    except ParseError as exc:
        assert exc.code is not None  # refused with a typed reason, never an untyped failure
        return
    assert got.expr == parse_text(" + ".join([*wanted, *stems]))


# -- the same reader in time_derivative ---------------------------------------------------------------


def _time_derivative(expression: str, lagrangian: str | None = None, fmt: str = "text"):  # type: ignore[no-untyped-def]
    from minerva_engine.models import ExpressionInput
    from minerva_engine.time_derivative import time_derivative

    data = ExpressionInput(input=expression, format=fmt)  # type: ignore[arg-type]
    return time_derivative(data, ["x"], "t", lagrangian, LIMITS)


def test_time_derivative_reads_spelled_velocities_and_reports_it() -> None:
    energy = "1/2*m*xdot^2 + 1/2*k*x^2"
    result = _time_derivative(energy, "1/2*m*x dot^2 - 1/2*k*x^2")
    assert result.status is Status.OK
    assert result.zero_on_motion is True  # energy is conserved for this Lagrangian
    assert result.reading is not None
    assert {(n.source, n.target) for n in result.reading.normalizations} == {("xdot", "x_dot")}
    clean = _time_derivative("1/2*m*x_dot^2 + 1/2*k*x^2")
    assert _time_derivative(energy).result_latex == clean.result_latex


def test_time_derivative_names_a_velocity_of_an_undeclared_coordinate() -> None:
    result = _time_derivative("x_dot*y_dot")
    assert result.status is Status.UNSUPPORTED
    assert result.diagnostic is not None
    assert result.diagnostic.code is InputCode.UNDECLARED_DEPENDENT
    assert result.diagnostic.suggestions == ["y"]


def test_time_derivative_refuses_a_latex_letter_run_for_a_coordinate() -> None:
    result = _time_derivative(r"\frac12 m xdot^2", fmt="latex")
    assert result.status is Status.PARSE_ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.code is InputCode.UNRECOGNIZED_DERIVATIVE
