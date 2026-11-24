"""A verified Taylor series is not "contradicted" by the checker that reads the prose around it (issue #167).

The reported chat: the series tool returned `ok`, the reply restated `f(x) = ... + O(x^7)` and sampled the error at
`|x| = 0.5`, and the claim checker marked all of it contradicted. Three causes, one test group each:

1. `x = 0` ("about x = 0") was learned as a definition and substituted into later formulas;
2. Big-O (`O(t^3)`) and an undeclared `g(t)` were read as products (O*t^3, g*t);
3. nothing linked the restated series to the tool's result.

Every wrong twin (changed coefficient, order, centre, function, a label nobody defined) must stay unlinked and must
not be called contradicted; a real definition must still contradict a wrong value.
"""

import pytest
from minerva_engine.claim_evidence import SeriesEvidence, SeriesTermEvidence
from minerva_engine.claims import check_claims
from minerva_engine.testing import budget_limits

SIN = SeriesEvidence(
    source="r1",
    expression_latex=r"\sin{\left(x \right)}",
    variable="x",
    centre_latex="0",
    order=5,
    terms=[
        SeriesTermEvidence(power=1, coefficient_latex="1"),
        SeriesTermEvidence(power=3, coefficient_latex=r"- \frac{1}{6}"),
        SeriesTermEvidence(power=5, coefficient_latex=r"\frac{1}{120}"),
    ],
    distances=[0.5, 0.1, 0.01],
)
SIN_POLY = r"x - \frac{x^{3}}{6} + \frac{x^{5}}{120}"
SIN_CLAIM = rf"\sin x = {SIN_POLY} + O\left(x^{{6}}\right)"

# e^x about 1, order 2: the centre is not zero
EXP = SeriesEvidence(
    source="r2",
    expression_latex=r"\mathrm{e}^{x}",
    variable="x",
    centre_latex="1",
    order=2,
    terms=[
        SeriesTermEvidence(power=0, coefficient_latex=r"\mathrm{e}"),
        SeriesTermEvidence(power=1, coefficient_latex=r"\mathrm{e}"),
        SeriesTermEvidence(power=2, coefficient_latex=r"\frac{\mathrm{e}}{2}"),
    ],
    distances=[0.5],
)
EXP_POLY = r"\mathrm{e} + \mathrm{e}\left(x - 1\right) + \frac{\mathrm{e}}{2}\left(x - 1\right)^{2}"


def run(
    claims: list[str],
    relations: list[str] | None = None,
    series: list[SeriesEvidence] | None = None,
    assumptions: dict[str, list[str]] | None = None,
):  # type: ignore[no-untyped-def]
    return check_claims(
        claims, assumptions or {}, relations or [], budget_limits(), series=series or []
    )


def statuses(results) -> list[str]:  # type: ignore[no-untyped-def]
    return [r.status for r in results]


# --- the issue's two synthetic reproductions ---------------------------------------------------------------


def test_an_undeclared_function_with_big_o_is_unverified_not_contradicted() -> None:
    results = run([r"t=0", r"g(t)=1+t+O(t^3)"])
    assert results[1].status == "unverified"
    assert results[1].given == []


def test_pointwise_agreement_never_establishes_a_big_o_claim() -> None:
    """`1 + t = 1 + O(t^2)` was `conditional` ("holds given t=0"): the product O*t^2 vanishes at t = 0."""
    results = run([r"t=0", r"1+t=1+O(t^2)"])
    assert results[1].status == "unverified"
    assert "Big-O" in results[1].detail
    assert results[1].given == []


@pytest.mark.parametrize(
    "remainder",
    [
        r"O(t^2)",
        r"O\left(t^{2}\right)",
        r"\mathcal{O}(t^2)",
        r"\mathcal{O}\left(t^2\right)",
        r"\operatorname{O}(t^2)",
        r"\mathrm{O}(t^2)",
        r"o(t)",
    ],
)
def test_every_spelling_of_big_o_stays_unverified_after_a_point_was_stated(remainder: str) -> None:
    results = run([r"t=0", rf"1+t=1+{remainder}", rf"1 = 2+{remainder}"])
    assert statuses(results)[1:] == ["unverified", "unverified"]


def test_a_big_o_statement_is_never_learned_as_a_definition() -> None:
    # `y = 1 + O(h)` read as a product would define y; a later `y = 7` is then judged against it
    results = run([r"y = 1 + O(h)", r"2 y = 14"])
    assert results[1].status == "unverified"


# --- undeclared function notation --------------------------------------------------------------------------


def test_an_undeclared_applied_function_is_not_read_as_multiplication() -> None:
    results = run([r"t = 0", r"g(t) = 1 + t"])
    assert results[1].status == "unverified"
    assert "function" in results[1].detail


def test_an_applied_function_is_never_learned_as_a_product() -> None:
    # `y = g(t)` read as y = g*t would make `2y = 12` hold given g = 2, t = 3
    results = run([r"y = g(t)", r"g = 2", r"t = 3", r"2 y = 12"])
    assert results[3].status == "unverified"
    # the same with a product written as one: the definition is learned and used
    results = run([r"y = g t", r"g = 2", r"t = 3", r"2 y = 12"])
    assert results[3].status == "conditional"


@pytest.mark.parametrize(
    "claim",
    [
        r"x(x+1) = x^2 + x",
        r"a(b+c) = ab + ac",
        r"n(n+1) = n^2 + n",
        r"2(x+1) = 2x + 2",
        r"(x+1)(x-1) = x^2 - 1",
        r"m (a+b) = ma + mb",
        r"\sin(x)\cos(x) = \frac{\sin(2x)}{2}",
        r"\omega(\omega+1) = \omega^2 + \omega",
    ],
)
def test_products_written_without_a_space_are_still_tested(claim: str) -> None:
    assert run([claim])[0].status == "holds"


def test_a_false_product_is_still_not_an_identity() -> None:
    assert run([r"a(b+c) = ab + 2ac"])[0].status == "not_identity"


# --- the centre and the places the error was measured are parameters, not definitions ----------------------


def test_the_expansion_point_and_the_sample_distances_do_not_contradict_the_expansion() -> None:
    reply = [r"x=0", r"|x|=0.5", r"|x|=0.1", r"x=0.01", SIN_CLAIM, r"|x| = 0.01"]
    results = run(reply, series=[SIN])
    assert "fails" not in statuses(results)
    assert "conditional" not in statuses(results)
    assert all(r.given == [] for r in results)
    assert results[4].status == "supported"


def test_an_earlier_statement_of_the_centre_does_not_leak_into_this_expansion() -> None:
    """The user's "about x = 0" arrives as a relation of the conversation, not as a claim of this reply."""
    results = run([r"|x| = 0.5", SIN_CLAIM], relations=[r"x = 0"], series=[SIN])
    assert results[0].status == "unverified"
    assert "parameter" in results[0].detail
    assert results[1].status == "supported"


def test_a_centre_other_than_zero_and_the_distance_from_it_are_parameters_too() -> None:
    results = run(
        [r"x = 1", r"|x - 1| = 0.5", r"x = 1.5", r"x = 0.5", r"\mathrm{e}^{x} = " + EXP_POLY],
        relations=[r"x = 1"],
        series=[EXP],
    )
    assert statuses(results)[:4] == ["unverified"] * 4
    assert all("parameter" in r.detail for r in results[:4])


def test_a_real_definition_still_contradicts_a_wrong_value_while_a_series_is_known() -> None:
    results = run(
        [r"2x = 7", r"y = 3", r"2 y = 5"],
        relations=[r"x = 3"],  # not the centre (0) and not a measured place: a definition
        series=[SIN],
    )
    assert results[0].status == "fails"
    assert results[0].given == ["x = 3"]


def test_a_horizon_definition_still_contradicts_a_wrong_k_with_a_series_known() -> None:
    """The A-010 case: `K = 48 M^2 / r^6` at `r = 2M` is `3 / (4 M^4)`; a wrong value must be contradicted."""
    relations = [r"K = \frac{48 M^2}{r^6}"]
    wrong = run([r"r = 2M", r"K = \frac{48 M^2}{(2M)^6} = \frac{3}{5M^4}"], relations, series=[SIN])
    right = run([r"r = 2M", r"K = \frac{48 M^2}{(2M)^6} = \frac{3}{4M^4}"], relations, series=[SIN])
    assert wrong[1].status == "fails"
    assert right[1].status == "conditional"


def test_a_location_the_tool_did_not_use_is_still_a_definition() -> None:
    # the centre is 0 and the measured distances are 0.5, 0.1, 0.01: `x = 0.2` is the reply's own value
    results = run([r"x = 0.2", r"2x = 1"], series=[SIN])
    assert results[1].status == "fails"


def test_without_a_series_result_the_point_is_a_definition_as_before() -> None:
    """Documented limit (docs/verification-defects.md A-010): with no tool record, `x = 0` cannot be told from a
    definition, so a different value of x is still compared with it."""
    assert run([r"x = 0", r"|x| = 0.5"])[1].status == "fails"


# --- the restated series is linked to the verified one only if every part agrees ---------------------------


@pytest.mark.parametrize(
    "claim",
    [
        SIN_CLAIM,
        rf"\sin(x) = {SIN_POLY} + O(x^6)",
        r"\sin x = \frac{x^5}{120} - \frac{x^3}{6} + x + \mathcal{O}\left(x^6\right)",
        rf"{SIN_POLY} + O\left(x^{{6}}\right) = \sin x",
        rf"\sin x = {SIN_POLY} + \operatorname{{O}}\left(x^{{6}}\right) \quad (x \to 0)",
    ],
)
def test_the_same_series_in_another_arrangement_is_supported(claim: str) -> None:
    result = run([claim], series=[SIN])[0]
    assert result.status == "supported"
    assert result.evidence_rule == "series"
    assert result.evidence_source == "r1"
    assert result.evidence_basis == "numeric"
    assert "not a proof or a bound" in result.detail


WRONG_TWINS = {
    "a coefficient": r"\sin x = x - \frac{x^3}{5} + \frac{x^5}{120} + O(x^6)",
    "a coefficient off by 1e-9": r"\sin x = x - \frac{x^3}{6} + \frac{1.000000001 x^5}{120} + O(x^6)",
    "a dropped term": r"\sin x = x - \frac{x^3}{6} + O(x^6)",
    "an extra term": rf"\sin x = {SIN_POLY} + \frac{{x^6}}{{720}} + O(x^6)",
    "a lower remainder order": rf"\sin x = {SIN_POLY} + O(x^5)",
    "a higher remainder order": rf"\sin x = {SIN_POLY} + O(x^7)",
    "a remainder in another variable": rf"\sin x = {SIN_POLY} + O(y^6)",
    "a shifted centre": rf"\sin x = {SIN_POLY} + O((x-1)^6)",
    "another function": rf"\cos x = {SIN_POLY} + O(x^6)",
    "a scaled function": rf"2\sin x = {SIN_POLY} + O(x^6)",
    "a label nobody defined": rf"f(x) = {SIN_POLY} + O(x^6)",
    "a sum of two remainders": rf"\sin x = {SIN_POLY} + O(x^6) + O(x^7)",
}


@pytest.mark.parametrize("claim", WRONG_TWINS.values(), ids=WRONG_TWINS.keys())
def test_a_changed_series_is_neither_linked_nor_contradicted(claim: str) -> None:
    for relations in ([], [r"x = 0"]):
        result = run([claim], relations=relations, series=[SIN])[0]
        assert result.status == "unverified", (result.status, result.detail)
        assert result.evidence_rule is None


def test_a_changed_centre_is_not_linked() -> None:
    # the tool expanded about 1; the claim writes the expansion about 0 (and the other way round)
    about_zero = r"\mathrm{e}^{x} = " + EXP_POLY.replace("x - 1", "x") + r" + O(x^3)"
    mixed = r"\mathrm{e}^{x} = " + EXP_POLY + r" + O(x^3)"
    assert run([about_zero], series=[EXP])[0].status == "unverified"
    assert run([mixed], series=[EXP])[0].status == "unverified"
    assert run([SIN_CLAIM], series=[EXP])[0].status == "unverified"


def test_a_series_about_a_nonzero_centre_is_linked() -> None:
    claim = r"\mathrm{e}^{x} = " + EXP_POLY + r" + O\left(\left(x - 1\right)^{3}\right)"
    result = run([claim], series=[EXP])[0]
    assert result.status == "supported"
    assert result.evidence_source == "r2"


def test_a_label_is_linked_only_through_a_definition_of_that_label() -> None:
    claim = rf"f(x) = {SIN_POLY} + O(x^6)"
    assert run([claim], relations=[r"f(x) = \sin x"], series=[SIN])[0].status == "supported"
    # the reply may define the label itself, before the claim
    assert run([r"f(x) = \sin x", claim], series=[SIN])[1].status == "supported"
    # a label defined as another function, or by an expression that only looks alike, is not linked
    assert run([claim], relations=[r"f(x) = \cos x"], series=[SIN])[0].status == "unverified"
    assert run([claim], relations=[r"f(x) = 2 \sin x"], series=[SIN])[0].status == "unverified"
    # the latest definition wins
    assert (
        run([claim], relations=[r"f(x) = \sin x", r"f(x) = \cos x"], series=[SIN])[0].status
        == "unverified"
    )
    # a label for another argument is another function
    assert run([claim], relations=[r"f(t) = \sin t"], series=[SIN])[0].status == "unverified"


def test_a_two_turn_conversation_defines_in_turn_one_and_expands_in_turn_two() -> None:
    turn_one = [r"f(x) = \sin x", r"x = 0"]  # the user's message: a function and "about x = 0"
    claim = rf"f(x) = {SIN_POLY} + O\left(x^{{6}}\right)"
    results = run([r"|x| = 0.5", claim], relations=turn_one, series=[SIN])
    assert statuses(results) == ["unverified", "supported"]
    assert results[0].given == []
    # wrong twin: turn one defined another function
    twin = run([claim], relations=[r"f(x) = \cos x", r"x = 0"], series=[SIN])
    assert twin[0].status == "unverified"
    # wrong twin: the same turn two, but the tool expanded another function
    other = run([claim], relations=turn_one, series=[EXP])
    assert other[0].status == "unverified"


def test_the_assumptions_of_the_call_are_the_ones_the_claim_is_read_under() -> None:
    positive = SeriesEvidence(
        source="r3",
        expression_latex=r"\sqrt{x^{2}}",
        variable="x",
        centre_latex="1",
        order=1,
        terms=[
            SeriesTermEvidence(power=0, coefficient_latex="1"),
            SeriesTermEvidence(power=1, coefficient_latex="1"),
        ],
        distances=[],
        assumptions={"x": ["positive"]},
    )
    claim = r"\sqrt{x^2} = 1 + \left(x - 1\right) + O\left(\left(x - 1\right)^{2}\right)"
    assert run([claim], series=[positive])[0].status == "supported"
    # without the declaration the series record would not describe sqrt(x^2) (it is |x|): it is read as declared
    assert "x positive" in run([claim], series=[positive])[0].detail


def test_a_series_claim_is_never_called_proved_or_a_bound() -> None:
    detail = run([SIN_CLAIM], series=[SIN])[0].detail
    assert "proof" in detail and "bound" in detail
    assert "proved" not in detail.replace("not a proof", "")


def test_a_series_that_was_never_verified_links_nothing() -> None:
    assert run([SIN_CLAIM], series=[])[0].status == "unverified"
