"""The two places where input refused before is now accepted, and the adversarial cases that keep them honest.

1. A geometry grid under `format: auto` whose cells are not all one syntax (the Schwarzschild chat of 2026-11-26:
   `-(1-2M/r)` next to `r^2*sin(theta)^2`). Each cell is read by exactly one strict parser, and only when that is
   unambiguous: a cell that only one parser reads goes to it, a cell both read must read *identically*, and a cell
   that both read differently (`r0^2` is `0` in LaTeX and `r0**2` in text; `pi` is `i*p` and pi; `oo` is `o**2`
   and infinity; `ab` is `a*b` and `ab`) makes the whole grid fall back to the single syntax of before.
2. A declared coordinate spelt differently from the Lagrangian: `\\theta_1`, `\\theta_{1}`, `x_{1}` are LaTeX
   spellings of `theta_1`, `x_1`; `x1` (LaTeX only, where `x1` cannot be one symbol) is read as `x_1` when exactly
   one symbol read is spelt alike and its velocity is there too. Never in text syntax, never with two candidates.

A misread here would verify perfectly, so the tests pin what was read, not only the status.
"""

import pytest
from minerva_engine import Limits, Status
from minerva_engine.geometry import GeometryInput, geometry
from minerva_engine.lagrangian import DerivationError, analyze_lagrangian, derive
from minerva_engine.notation.models import InputCode, NormalizationKind
from minerva_engine.notation.names import alias_in_latex

LIMITS = Limits(timeout_seconds=60)

# --- 1. mixed geometry grids --------------------------------------------------------------------------------

CHAT = [
    ["-(1-2M/r)", "0", "0", "0"],
    ["0", "(1-2M/r)^{-1}", "0", "0"],
    ["0", "0", "r^2", "0"],
    ["0", "0", "0", "r^2*sin(theta)^2"],
]
TEXT = [
    ["-(1-2*M/r)", "0", "0", "0"],
    ["0", "(1-2*M/r)^(-1)", "0", "0"],
    ["0", "0", "r^2", "0"],
    ["0", "0", "0", "r^2*sin(theta)^2"],
]
SCHWARZSCHILD = ["t", "r", "theta", "phi"]


def run(metric, fmt, coordinates=SCHWARZSCHILD, quantities=("christoffel",)):  # type: ignore[no-untyped-def]
    data = GeometryInput(
        coordinates=coordinates, metric=metric, format=fmt, quantities=list(quantities)
    )
    return geometry(data, LIMITS)


def test_the_chat_metric_is_read_cell_by_cell_and_equals_the_text_reading() -> None:
    mixed = run(CHAT, "auto")
    reference = run(TEXT, "text")
    assert mixed.status is Status.OK, mixed.message
    assert reference.status is Status.OK
    assert mixed.metric_latex == reference.metric_latex
    assert [q.model_dump() for q in mixed.quantities] == [
        q.model_dump() for q in reference.quantities
    ]
    assert mixed.cell_syntax == [
        ["latex", "latex", "latex", "latex"],
        ["latex", "latex", "latex", "latex"],
        ["latex", "latex", "latex", "latex"],
        ["latex", "latex", "latex", "text"],
    ]


@pytest.mark.parametrize("fmt", ["latex", "text"])
def test_an_explicit_format_stays_strict_for_the_whole_grid(fmt: str) -> None:
    assert run(CHAT, fmt).status is Status.PARSE_ERROR


def test_a_uniform_grid_reads_exactly_as_before() -> None:
    result = run(TEXT, "auto")
    assert result.status is Status.OK and result.cell_syntax is None
    assert result.metric_latex == run(TEXT, "text").metric_latex


@pytest.mark.parametrize(
    "ambiguous",
    [
        "r0^2",  # LaTeX 0, text r0**2
        "pi*r^2",  # has `*`: text pi r^2 (only text reads it): fine on its own, see the next test
        "oo",  # LaTeX o**2, text infinity
        "ab",  # LaTeX a*b, text ab
    ],
)
def test_a_cell_two_parsers_read_differently_is_never_split(ambiguous: str) -> None:
    metric = [row[:] for row in CHAT]
    metric[2][2] = ambiguous
    result = run(metric, "auto")
    if ambiguous == "pi*r^2":
        # Only text reads `pi*r^2`? LaTeX reads it as i*p*r^2: both read it, differently, so no split either.
        assert result.status is Status.PARSE_ERROR
        return
    assert result.status is Status.PARSE_ERROR, (ambiguous, result.metric_latex)
    assert result.cell_syntax is None


def test_a_bare_greek_word_in_a_latex_cell_is_still_refused() -> None:
    metric = [row[:] for row in CHAT]
    metric[3][3] = r"r^2 \sin^2 theta"
    result = run(metric, "auto")
    assert result.status is Status.PARSE_ERROR
    assert result.diagnostic is not None and "theta" in result.diagnostic.symbols


def test_the_split_is_deterministic() -> None:
    first, second = run(CHAT, "auto"), run(CHAT, "auto")
    assert first.metric_latex == second.metric_latex and first.cell_syntax == second.cell_syntax


# --- 2. declared coordinate names ---------------------------------------------------------------------------

THETA = r"\frac12 m \dot{\theta}_1^2 + \frac12 m \dot{\theta}_2^2 - k (\theta_2 - \theta_1)^2"
CARTESIAN = r"\frac12 m (\dot{x}_1^2 + \dot{y}_1^2) - k (x_1^2 + y_1^2)"


def equations(lag: str, coordinates: list[str], fmt: str = "latex") -> list[str]:
    report = analyze_lagrangian(lag, coordinates, "t", {}, LIMITS, fmt)  # type: ignore[arg-type]
    assert report.status is Status.OK, report.message
    return [e.latex for e in report.equations]


@pytest.mark.parametrize(
    "declared",
    [
        [r"\theta_1", r"\theta_2"],
        [r"\theta_{1}", r"\theta_{2}"],
        ["theta_{1}", "theta_{2}"],
        ["theta1", "theta2"],
    ],
)
def test_latex_spellings_of_a_coordinate_read_as_the_exact_name(declared: list[str]) -> None:
    assert equations(THETA, declared) == equations(THETA, ["theta_1", "theta_2"])


@pytest.mark.parametrize("declared", [["x_{1}", "y_{1}"], ["x1", "y1"]])
def test_cartesian_spellings_read_as_the_exact_name(declared: list[str]) -> None:
    assert equations(CARTESIAN, declared) == equations(CARTESIAN, ["x_1", "y_1"])


def test_a_name_read_by_alias_is_reported_in_the_reading() -> None:
    d = derive(CARTESIAN, ["x1", "y1"], "t", {}, LIMITS)
    notes = [n for n in d.normalizations if n.kind is NormalizationKind.DECLARED_NAME]
    assert [(n.source, n.target) for n in notes] == [("x1", "x_1"), ("y1", "y_1")]


def test_text_syntax_never_aliases_x1_to_x_1() -> None:
    with pytest.raises(DerivationError) as info:
        derive("1/2*m*(x_1_dot^2) - k*x_1^2", ["x1"], "t", {}, LIMITS, "text")
    assert info.value.diagnostic is not None
    assert info.value.diagnostic.code is InputCode.DECLARED_NAME_ABSENT
    assert info.value.diagnostic.suggestions == ["x_1"]


@pytest.mark.parametrize(
    ("declared", "read", "expected"),
    [
        (["x1"], {"x_1", "x_1_dot"}, {"x1": "x_1"}),
        (["xy1"], {"x_y1", "x_y1_dot", "xy_1", "xy_1_dot"}, {}),  # two candidates
        (["m1"], {"m_1", "x", "x_dot"}, {}),  # no velocity: a parameter, not a coordinate
        (["x1", "x_1"], {"x_1", "x_1_dot"}, {}),  # the target is declared already
        (["x_2"], {"x_1", "x_1_dot"}, {}),  # has an underscore: a different name, never aliased
        (["x1", "x01"], {"x_1", "x_1_dot"}, {"x1": "x_1"}),  # x01 is not spelt like x_1
    ],
)
def test_the_alias_rule_is_narrow(
    declared: list[str], read: set[str], expected: dict[str, str]
) -> None:
    aliases, notes = alias_in_latex(declared, read)
    assert aliases == expected
    assert [(n.source, n.target) for n in notes] == list(expected.items())


def test_a_parameter_spelt_like_the_declared_name_is_not_taken_for_a_coordinate() -> None:
    lag = r"\frac12 m_1 \dot{x}^2 - k x^2"
    with pytest.raises(DerivationError) as info:
        derive(lag, ["x", "m1"], "t", {}, LIMITS)
    assert info.value.diagnostic is not None
    assert info.value.diagnostic.suggestions == ["m_1"]


def test_an_alias_onto_a_name_already_declared_is_refused() -> None:
    with pytest.raises(DerivationError) as info:
        derive(CARTESIAN, ["x_1", "x1", "y_1"], "t", {}, LIMITS)
    assert info.value.status is Status.PARSE_ERROR


def test_two_spellings_of_one_coordinate_are_not_distinct() -> None:
    with pytest.raises(DerivationError) as info:
        derive(THETA, ["theta_1", r"\theta_{1}"], "t", {}, LIMITS)
    assert "distinct" in info.value.message
