"""`check_units` with `unknown`: the dimension of a quantity found from the equation it is in.

Every expected dimension below was worked out by hand (the comment says how); none is produced by running
the solver. The solver only proposes: the dimensions it finds are shown only if the ordinary checker,
`dimensions.dimension_of`, accepts the equation with them put in.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from minerva_engine import Engine, pi_groups
from minerva_engine.dimension_solver import Solution
from minerva_engine.models import Status
from minerva_engine.notation.models import InputCode
from minerva_engine.pi_groups import QuantitySpec, check_units
from minerva_engine.testing import budget_limits


def q(name: str, unit: str) -> QuantitySpec:
    return QuantitySpec(name=name, unit=unit)


def run(
    equation: str, quantities: list[QuantitySpec], unknown: list[str]
) -> pi_groups.UnitsCheckReport:
    return check_units(equation, quantities, budget_limits(), unknown)


def derived(report: pi_groups.UnitsCheckReport) -> dict[str, str]:
    return {d.name: d.dimension_text for d in report.derived}


KEPLER_C3 = (
    r"C_3 = - k \cos{\left(\theta \right)} + m r^{3} \dot{\theta}^{2} \cos{\left(\theta \right)}"
    r" + m r^{2} \dot{r} \dot{\theta} \sin{\left(\theta \right)}"
)
KEPLER_KNOWN = [q("m", "kg"), q("r", "m"), q("theta", "rad")]


def test_the_report_of_issue_184_gives_both_the_same_dimension() -> None:
    # m r^3 thetadot^2 = kg m^3 s^-2 (thetadot = rad/s); the sum makes k and C_3 the same.
    report = run(KEPLER_C3, KEPLER_KNOWN, ["C_3", "k"])
    assert report.status is Status.OK and report.outcome == "determined"
    assert derived(report) == {"C_3": "kg m^3 s^-2", "k": "kg m^3 s^-2"}
    assert report.consistent is True
    by_name = {x.name: x for x in report.quantities}
    assert by_name["theta_dot"].origin == "time_derivative"
    assert by_name["theta_dot"].dimension_text == "s^-1"  # rad is 1; divided by s
    assert by_name["r_dot"].dimension_text == "m s^-1"
    assert by_name["m"].origin == "given"


def test_one_unknown_alone_on_the_left() -> None:
    report = run("F = m a", [q("m", "kg"), q("a", "m/s^2")], ["F"])
    assert derived(report) == {"F": "kg m s^-2"}  # newton, by hand


def test_two_unknowns_fixed_by_a_sum() -> None:
    # d = v t + a t^2 / 2: v t is a length, so v is m/s; a t^2 is a length, so a is m/s^2.
    report = run(r"d = v t + \frac{1}{2} a t^2", [q("d", "m"), q("t", "s")], ["v", "a"])
    assert derived(report) == {"v": "m s^-1", "a": "m s^-2"}


def test_a_rational_power_gives_a_fractional_exponent() -> None:
    # W = k sqrt(x): k = J / m^(1/2) = kg m^(3/2) s^-2
    report = run(r"W = k \sqrt{x}", [q("W", "J"), q("x", "m")], ["k"])
    assert derived(report) == {"k": "kg m^(3/2) s^-2"}


def test_an_unknown_inside_a_function_is_made_dimensionless_with_its_factor() -> None:
    # y = cos(k x): k x is dimensionless, so k is 1/m.
    report = run(r"y = \cos(k x)", [q("y", "1"), q("x", "m")], ["k"])
    assert derived(report) == {"k": "m^-1"}


def test_an_underdetermined_pair_stays_free_and_nothing_is_guessed() -> None:
    report = run("a b = F", [q("F", "N")], ["a", "b"])
    assert report.status is Status.OK and report.outcome == "underdetermined"
    assert report.undetermined == ["a", "b"]
    assert report.derived == [] and report.consistent is None


def test_an_inconsistent_sum_names_the_conflict() -> None:
    # F = a + m: a would have to be a force (the left side) and a mass (the other term).
    report = run("F = a + m", [q("F", "N"), q("m", "kg")], ["a"])
    assert report.outcome == "inconsistent" and report.consistent is False
    assert report.derived == []
    assert report.problem_terms and "no dimensions for the unknowns" in (report.problem or "")


@pytest.mark.parametrize("equation", [r"y = 2^{a}", r"y = x^{a}"])
def test_an_unknown_in_an_exponent_is_unsupported(equation: str) -> None:
    report = run(equation, [q("y", "1"), q("x", "1")], ["a"])
    assert report.status is Status.UNSUPPORTED and report.derived == []


def test_a_dimension_off_by_one_exponent_is_rejected_at_the_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = pi_groups.solve_unknown_dimensions

    def off_by_one(*args, **kwargs):  # type: ignore[no-untyped-def]
        solution = real(*args, **kwargs)
        name, dim = next(iter(solution.dimensions.items()))
        twin = (dim[0] + Fraction(1), *dim[1:])  # one extra power of kg
        return Solution("determined", {**solution.dimensions, name: twin})

    monkeypatch.setattr(pi_groups, "solve_unknown_dimensions", off_by_one)
    report = run("F = m a", [q("m", "kg"), q("a", "m/s^2")], ["F"])
    assert report.status is Status.NOT_VERIFIED
    assert report.derived == [] and report.sides == []


def test_a_declared_unit_wins_over_the_time_derivative_reading() -> None:
    quantities = [q("m", "kg"), q("x", "m")]
    read = run(r"W = \frac12 m \dot{x}^2", quantities, ["W"])
    assert derived(read) == {"W": "kg m^2 s^-2"}  # x_dot read as m/s
    declared = run(r"W = \frac12 m \dot{x}^2", [*quantities, q("x_dot", "m")], ["W"])
    assert derived(declared) == {"W": "kg m^2"}  # the caller's x_dot is a length: theirs wins
    assert all(r.name != "x_dot" or r.origin == "given" for r in declared.quantities)


def test_a_velocity_of_an_undeclared_coordinate_is_not_read() -> None:
    report = run(r"W = \frac12 m \dot{x}^2", [q("m", "kg")], ["W"])
    assert report.status is Status.PARSE_ERROR and "x_dot" in report.message


def test_the_old_checks_are_unchanged_without_unknowns() -> None:
    ok = run("F = m a", [q("F", "N"), q("m", "kg"), q("a", "m/s^2")], [])
    assert ok.consistent is True and ok.derived == [] and ok.outcome is None
    bad = run("W = m v", [q("W", "J"), q("m", "kg"), q("v", "m/s")], [])
    assert bad.consistent is False and bad.missing_factor_text == "m s^-1"


def test_refusals_of_the_call_itself() -> None:
    assert run("F = m a", [q("m", "kg")], ["F"]).status is Status.PARSE_ERROR  # a has no unit
    assert run("F = m a", [q("m", "kg"), q("a", "m/s^2")], ["z"]).status is Status.PARSE_ERROR
    assert run("F = m a", [q("m", "kg"), q("F", "N")], ["F"]).status is Status.PARSE_ERROR
    many = ["a", "b", "c", "d", "e"]
    assert run("a b c d e = x", [q("x", "m")], many).status is Status.UNSUPPORTED


def test_an_unknown_quantity_kind_has_a_typed_code_that_names_both_ways_out() -> None:
    kind = QuantitySpec(name="k", kind="Kepler parameter")
    report = run(KEPLER_C3, [*KEPLER_KNOWN, kind], ["C_3"])
    assert report.status is Status.PARSE_ERROR
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.UNKNOWN_QUANTITY_KIND
    assert report.diagnostic.symbols == ["k"] and report.diagnostic.recoverable
    assert "unit" in report.message and "`unknown`" in report.message


def test_the_solver_runs_in_the_sandbox_like_every_operation() -> None:
    engine = Engine(budget_limits(max_concurrency=1))
    report = engine.check_units("F = m a", [q("m", "kg"), q("a", "m/s^2")], ["F"])
    assert report.status is Status.OK and derived(report) == {"F": "kg m s^-2"}


def test_a_name_listed_as_unknown_is_not_read_as_a_time_derivative() -> None:
    report = run(r"W = \frac12 m \dot{x}^2", [q("m", "kg"), q("x", "m"), q("W", "J")], ["x_dot"])
    assert derived(report) == {"x_dot": "m s^-1"}
    assert all(r.origin == "given" for r in report.quantities)


def test_a_conflict_among_known_terms_is_not_blamed_on_the_unknowns() -> None:
    report = run("F c = m + t", [q("F", "N"), q("m", "kg"), q("t", "s")], ["c"])
    assert report.outcome == "inconsistent"
    assert "whatever the unknowns are" in (report.problem or "")
