"""`\\delta(...)` and `theta(...)` are distributions only after the user says so.

Everywhere else they stay ordinary symbols. In the distribution operation an unanswered name is
asked about (`notation_unconfirmed`) and nothing is computed; a confirmed reading goes through exactly the
same exact conditions and test-function pairing as the explicit `DiracDelta` / `Heaviside` names.
"""

from typing import Any

import pytest
import sympy as sp
from minerva_engine import (
    DistributionInput,
    DistributionResult,
    Engine,
    ExpressionInput,
    Limits,
    Status,
    parse_latex,
    parse_text,
)
from minerva_engine.build import NotationLog
from pydantic import ValidationError


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=30, max_concurrency=2))


def run(
    engine: Engine,
    source: str,
    action: str,
    syntax: str,
    notation: dict[str, str] | None = None,
    **extra: Any,
) -> DistributionResult:
    return engine.distribution(
        DistributionInput(
            expression=ExpressionInput(
                input=source,
                format=syntax,  # type: ignore[arg-type]
                assumptions={"x": ["real"]},
            ),
            action=action,  # type: ignore[arg-type]
            variable="x",
            notation=notation or {},  # type: ignore[arg-type]
            **extra,
        )
    )


def same_outcome(a: DistributionResult, b: DistributionResult) -> None:
    assert a.status is b.status is Status.OK
    assert a.result_latex == b.result_latex
    assert [s.entry_id for s in a.steps] == [s.entry_id for s in b.steps]
    assert a.verification is not None and b.verification is not None
    assert a.verification.passed and b.verification.passed
    assert a.verification.conditions_established and b.verification.conditions_established


CASES = [
    # (typed in LaTeX, typed in text, explicit name in text, action, extra, notation name)
    (
        r"x^2 \delta(x^2 - 4)",
        "x^2*delta(x^2 - 4)",
        "x^2*DiracDelta(x^2 - 4)",
        "integrate",
        {"lower": "-oo", "upper": "oo"},
        "delta",
    ),
    (r"\delta(2x - 6)", "delta(2*x - 6)", "DiracDelta(2*x - 6)", "simplify", {}, "delta"),
    (r"\theta(2x)", "theta(2*x)", "Heaviside(2*x)", "differentiate", {}, "theta"),
    (r"\Theta(x - 1)", "Theta(x - 1)", "Heaviside(x - 1)", "differentiate", {}, "Theta"),
]


@pytest.mark.parametrize(("latex", "text", "explicit", "action", "extra", "name"), CASES)
def test_without_an_answer_the_engine_asks_and_computes_nothing(
    engine: Engine,
    latex: str,
    text: str,
    explicit: str,
    action: str,
    extra: dict[str, str],
    name: str,
) -> None:
    for source, syntax in ((latex, "latex"), (text, "text")):
        r = run(engine, source, action, syntax, **extra)
        assert r.status is Status.CONDITIONS_NOT_SATISFIED
        assert r.reason == "notation_unconfirmed"
        assert [q.name for q in r.questions] == [name]
        assert r.result_latex is None and r.steps == [] and r.verification is None
        assert r.diagnostics.reason == "notation_unconfirmed"


@pytest.mark.parametrize(("latex", "text", "explicit", "action", "extra", "name"), CASES)
def test_a_confirmed_reading_gives_the_same_result_as_the_explicit_name(
    engine: Engine,
    latex: str,
    text: str,
    explicit: str,
    action: str,
    extra: dict[str, str],
    name: str,
) -> None:
    reading = "dirac_delta" if name == "delta" else "heaviside"
    baseline = run(engine, explicit, action, "text", **extra)
    for source, syntax in ((latex, "latex"), (text, "text")):
        r = run(engine, source, action, syntax, {name: reading}, **extra)
        same_outcome(r, baseline)
        assert r.notation == {name: reading}
        assert any("as you confirmed" in n for n in r.notes)
        assert r.questions == []


def test_the_integral_of_a_confirmed_delta_shell_is_two(engine: Engine) -> None:
    r = run(
        engine,
        r"x^2 \delta(x^2 - 4)",
        "integrate",
        "latex",
        {"delta": "dirac_delta"},
        lower="-oo",
        upper="oo",
    )
    assert r.status is Status.OK and r.result_text == "2"
    assert (
        r.input_latex is not None and "DiracDelta" in r.input_latex
    )  # printed with the explicit name


def test_reading_it_as_a_symbol_is_todays_refusal_not_a_guess(engine: Engine) -> None:
    r = run(engine, r"x^2 \delta(x^2 - 4)", "simplify", "latex", {"delta": "symbol"})
    assert r.status is Status.UNSUPPORTED and r.reason == "unsupported_distribution"
    assert r.questions == []


def test_a_wrong_candidate_still_fails_the_same_checks(engine: Engine) -> None:
    # The confirmed reading changes how the text is read, not what is checked: a delta of a
    # repeated root is refused exactly as `DiracDelta(x**2)` is.
    explicit = run(engine, "DiracDelta(x^2)", "simplify", "text")
    r = run(engine, r"\delta(x^2)", "simplify", "latex", {"delta": "dirac_delta"})
    assert r.status is explicit.status and r.reason == explicit.reason == "singular_root"
    assert r.result_latex is None


@pytest.mark.parametrize(
    "source",
    [r"\delta_{ij} x", r"\delta x", r"\theta_1 (x)", r"\delta_{1}(x - 1)"],
)
def test_subscripts_and_bare_names_are_never_asked_about(engine: Engine, source: str) -> None:
    log = NotationLog()
    parse_latex(source, notation=log)
    assert log.unconfirmed == {} and log.used == {}


def test_a_name_not_followed_by_an_argument_is_not_a_candidate() -> None:
    log = NotationLog()
    parse_text("delta*x + theta", notation=log)
    assert log.unconfirmed == {}


def test_every_other_parse_is_unchanged() -> None:
    # No log: the names are symbols, as `test_greek_letters_stay_symbols` pins for LaTeX.
    d, t, y = sp.symbols("delta theta x")
    assert parse_latex(r"\delta(x)") == d * y and parse_latex(r"\theta(x)") == t * y
    with pytest.raises(Exception, match="unsupported function"):
        parse_text("delta(x)")


@pytest.mark.parametrize(
    "notation",
    [{"delta": "heaviside"}, {"theta": "dirac_delta"}, {"Theta": "dirac_delta"}, {"phi": "symbol"}],
)
def test_a_reading_must_fit_the_name(notation: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        DistributionInput(
            expression=ExpressionInput(input="x"),
            action="simplify",
            variable="x",
            notation=notation,  # type: ignore[arg-type]
        )


def test_the_other_operations_still_refuse_a_delta_even_if_written_with_the_explicit_name(
    engine: Engine,
) -> None:
    # `\delta(x)` outside the distribution operation is delta times x: no distribution appears.
    r = engine.simplify(ExpressionInput(input=r"\delta(x-1)", assumptions={"x": ["real"]}))
    assert "DiracDelta" not in (r.result_latex or "")
