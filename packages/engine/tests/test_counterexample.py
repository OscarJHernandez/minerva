"""The counterexample search: it must find the identities `verify_equal` is blind to, and never invent one.

A counterexample disproves an identity; finding none proves nothing. The first group of tests is the false
identities that passed 20 of 20 points in `verify_equal` (docs/verification-defects.md, V-001..V-013), each run
with only the probe family that is meant to catch it. The second group is true identities, which must stay free
of counterexamples whatever the probes try (false rejection is as bad as false acceptance).
"""

import pytest
import sympy as sp
from minerva_engine.counterexample import (
    KINDS,
    TARGETED,
    Budget,
    check_counterexample,
    find_counterexample,
)

x = sp.Symbol("x")
xr = sp.Symbol("x", real=True)
xp = sp.Symbol("x", positive=True)
y = sp.Symbol("y", real=True)
z = sp.Symbol("z", real=True)
n = sp.Symbol("n", integer=True)
A = sp.Symbol("A", commutative=False)
B = sp.Symbol("B", commutative=False)

# (id, a, b, the one probe family that must find it)
FALSE_IDENTITIES = [
    ("V-001", sp.Abs(xr - 60), 60 - xr, "boundary"),
    ("V-002", sp.sqrt((xr - 60) ** 2), 60 - xr, "boundary"),
    ("V-003", sp.Max(xr, 49), sp.Integer(49), "boundary"),
    ("V-004", sp.Piecewise((1, xr < 48), (2, True)), sp.Integer(1), "boundary"),
    ("V-005", sp.floor(xp / 100), sp.Integer(0), "boundary"),
    ("V-006", sp.Abs(xr - 5) + y + z, 5 - xr + y + z, "boundary"),
    ("V-007", sp.log(sp.exp(x / 8)), x / 8, "branch"),
    ("V-008", sp.asin(sp.sin(x / 40)), x / 40, "scale"),
    ("V-009", sp.KroneckerDelta(n, 7), sp.Integer(0), "boundary"),
    ("V-010", sp.prod([n - k for k in range(-6, 7)]), sp.Integer(0), "special"),
    ("V-011", xr + sp.Rational(1, 10**11), xr, "random"),
    ("V-012", xr**2 + sp.Rational(1, 10**10), xr**2, "random"),
    ("V-013", sp.Piecewise((1, xp > sp.Rational(1, 1000)), (2, True)), sp.Integer(1), "boundary"),
]

TRUE_IDENTITIES = [
    ("removable singularity", (x**2 - 3025) / (x - 55), x + 55),
    ("gamma recurrence", sp.gamma(x + 1), x * sp.gamma(x)),
    ("pythagoras", sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)),
    ("product of roots", sp.sqrt(1 - x) * sp.sqrt(1 + x), sp.sqrt(1 - x**2)),
    ("acosh of cosh, x>0", sp.acosh(sp.cosh(xp)), xp),
    ("abs is sqrt of square, x real", sp.Abs(xr), sp.sqrt(xr**2)),
    ("expansion", sp.expand((x + 1) ** 5), (x + 1) ** 5),
    (
        "log of product, positives",
        sp.log(xp * sp.Symbol("w", positive=True)),
        sp.log(xp) + sp.log(sp.Symbol("w", positive=True)),
    ),
    ("abs on its region, x>0", sp.Abs(xp - 60) + 0 * xp, sp.Abs(xp - 60)),
    ("exp splits", sp.exp(x + sp.Symbol("w")), sp.exp(x) * sp.exp(sp.Symbol("w"))),
    ("sqrt of square, x>0", sp.sqrt(xp**2), xp),
    ("floor of integer", sp.floor(n), n),
    ("piecewise equal branches", sp.Piecewise((x, xr < 5), (x, True)), x),
]


@pytest.mark.parametrize(
    ("name", "a", "b", "kind"), FALSE_IDENTITIES, ids=[c[0] for c in FALSE_IDENTITIES]
)
def test_the_family_meant_to_catch_each_false_identity_does(
    name: str, a: sp.Expr, b: sp.Expr, kind: str
) -> None:
    result = find_counterexample(a, b, kinds=(kind,))  # type: ignore[arg-type]
    assert result.counterexample is not None, f"{name}: {kind} probes found nothing"
    found = result.counterexample
    assert found.probe == kind
    # The reported point is a real disagreement, re-checked independently of the search...
    assert check_counterexample(a, b, found.values)
    # ... and exact arithmetic agrees: the two sides differ there at 50 digits.
    difference = sp.N((a - b).xreplace(found.values), 50)
    assert abs(complex(difference)) > 1e-12 or abs(difference) > sp.Float("1e-12", 50)


@pytest.mark.parametrize(("name", "a", "b"), TRUE_IDENTITIES, ids=[c[0] for c in TRUE_IDENTITIES])
def test_a_true_identity_never_yields_a_counterexample(name: str, a: sp.Expr, b: sp.Expr) -> None:
    result = find_counterexample(a, b, budget=Budget(points=200, seconds=20.0))
    assert result.counterexample is None, f"{name}: invented {result.counterexample}"
    assert result.probes_tried > 0  # it did look


def test_numerical_cancellation_is_not_a_counterexample() -> None:
    # sin^2 + cos^2 at a large imaginary part loses all digits at 30 digits; higher precision shows it is 1.
    result = find_counterexample(sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1), kinds=("branch",))
    assert result.counterexample is None
    assert result.cancellation_suspected > 0


def test_a_point_defined_on_one_side_only_is_counted_not_reported() -> None:
    # sin(x)/x against sin(x)*(1/x) agree wherever either is defined; against a version with the pole
    # filled in (Piecewise) the sides differ only AT x = 0. A removable singularity is not a counterexample.
    a = sp.sin(xr) / xr
    b = sp.Piecewise((sp.sin(xr) / xr, sp.Ne(xr, 0)), (1, True))
    result = find_counterexample(a, b, budget=Budget(points=200, seconds=20.0))
    assert result.counterexample is None


def test_the_counterexample_is_minimised_and_records_its_reproduction() -> None:
    result = find_counterexample(xr + sp.Rational(1, 10**11), xr, kinds=("random",), seed=7)
    found = result.counterexample
    assert found is not None
    assert found.minimised
    assert found.substitution == {"x": "0"}
    assert found.seed == 7
    assert found.dps == 120
    assert found.assumptions == {"x": ["real"]} or "real" in found.assumptions["x"]
    assert found.expression_a and found.expression_b and found.residual.startswith("1.0e-11")


def test_the_search_is_deterministic_for_a_seed() -> None:
    first = find_counterexample(sp.Abs(xr - 60), 60 - xr, seed=3)
    second = find_counterexample(sp.Abs(xr - 60), 60 - xr, seed=3)
    assert first.counterexample is not None and second.counterexample is not None
    assert first.counterexample.substitution == second.counterexample.substitution
    assert first.probes_tried == second.probes_tried


def test_points_respect_the_declared_assumptions() -> None:
    positive = sp.Symbol("p", positive=True)
    result = find_counterexample(sp.Abs(positive - 60), 60 - positive)
    assert result.counterexample is not None
    assert sp.sympify(result.counterexample.values[positive]).is_positive
    integer = sp.Symbol("k", integer=True, nonnegative=True)
    found = find_counterexample(sp.KroneckerDelta(integer, 7), sp.Integer(0)).counterexample
    assert found is not None
    assert found.values[integer].is_integer and found.values[integer] >= 0


def test_a_small_budget_is_reported_as_exhausted() -> None:
    result = find_counterexample(sp.expand((x + 1) ** 5), (x + 1) ** 5, budget=Budget(points=3))
    assert result.counterexample is None
    assert result.exhausted
    assert result.probes_tried == 3


def test_every_family_runs_when_there_is_nothing_to_find() -> None:
    result = find_counterexample(xr + 1, 1 + xr, budget=Budget(points=500, seconds=20.0))
    assert result.counterexample is None
    assert not result.exhausted
    assert set(result.kinds_tried) <= set(KINDS) and "random" in result.kinds_tried


def test_a_constant_identity_is_checked_once() -> None:
    assert find_counterexample(sp.Integer(2) + 2, sp.Integer(5)).counterexample is not None
    assert find_counterexample(sp.Integer(2) + 2, sp.Integer(4)).counterexample is None


def test_distributions_and_noncommuting_symbols_are_refused_not_searched() -> None:
    delta = find_counterexample(sp.DiracDelta(x), sp.Integer(0))
    assert (
        delta.unsupported == "distribution"
        and delta.counterexample is None
        and delta.probes_tried == 0
    )
    matrices = find_counterexample((A + B) ** 2, A**2 + 2 * A * B + B**2)
    assert matrices.unsupported == "noncommutative" and matrices.counterexample is None


# -- V-017: the clock stop is told apart from the point cap ----------------------------------------------------------
def test_the_clock_stop_is_reported_apart_from_the_point_cap() -> None:
    z = sp.Symbol("z")
    a, b = sp.log(1 / z), -sp.log(z)
    by_clock = find_counterexample(a, b, budget=Budget(points=96, seconds=0.0), kinds=TARGETED)
    assert by_clock.exhausted and by_clock.clock_cut
    by_cap = find_counterexample(a, b, budget=Budget(points=1, seconds=60.0), kinds=TARGETED)
    assert by_cap.exhausted and not by_cap.clock_cut
    found = find_counterexample(a, b, budget=Budget(points=96, seconds=60.0), kinds=TARGETED)
    assert found.counterexample is not None and not found.clock_cut


# -- V-014: the centre of a quadratic exponent -----------------------------------------------------------------------
@pytest.mark.parametrize("centre", [sp.Rational(1, 7), sp.Rational(22, 7), -sp.Rational(355, 113)])
@pytest.mark.parametrize("scale", [10**4, 10**8])
def test_the_extremum_probe_finds_a_narrow_peak_at_its_exact_centre(
    centre: sp.Rational, scale: int
) -> None:
    result = find_counterexample(
        sp.exp(-scale * (xr - centre) ** 2), sp.Integer(0), kinds=("extremum",)
    )
    assert result.counterexample is not None
    assert result.counterexample.probe == "extremum"
    if scale >= 10**8:  # a peak this narrow is only hit at its centre (minimisation keeps it)
        assert result.counterexample.values[xr] == centre


def test_the_extremum_probe_never_rejects_the_same_peak_written_expanded() -> None:
    c = sp.Rational(1, 7)
    a = sp.exp(-(10**6) * (xr - c) ** 2)
    b = sp.exp(-(10**6) * xr**2 + 2 * 10**6 * c * xr - 10**6 * c**2)
    assert find_counterexample(a, b).counterexample is None


def test_the_extremum_probe_skips_other_shapes_and_stays_within_its_cap() -> None:
    from minerva_engine.counterexample import EXTREMUM_DRAWS, EXTREMUM_NODES, EXTREMUM_OFFSETS

    quartic = sp.exp(-(10**12) * (xr - sp.Rational(1, 7)) ** 4)
    assert find_counterexample(quartic, sp.Integer(0), kinds=("extremum",)).probes_tried == 0
    many = sum(sp.exp(-(10**6) * (xr - sp.Rational(k, 7)) ** 2) for k in range(1, 9))
    tried = find_counterexample(many, many + 1, kinds=("extremum",)).probes_tried
    assert tried <= EXTREMUM_NODES * EXTREMUM_DRAWS * (1 + EXTREMUM_OFFSETS)
