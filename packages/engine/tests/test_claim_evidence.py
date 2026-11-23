"""An equation the model wrote is linked to a verified result by mathematics, not by spelling.

Regression: the spherical-pendulum answer had 0 of 12 formulas with evidence although the tool had verified
every equation, because `\\ddot\\theta = \\sin\\theta\\cos\\theta\\,\\dot\\phi^2 + ...` is not printed like the engine's
form. Each wrong variant (sign, 1e-6 factor, a different conserved quantity, another name for a value) must stay
unmatched: a match is "the same statement a verified result made", never a new verification.
"""

import pytest
from minerva_engine.claim_evidence import ClaimEvidence
from minerva_engine.claims import check_claims
from minerva_engine.testing import budget_limits

V = ["theta", "phi"]
THETA = (
    r"\ddot{\theta} = \frac{\left(R \dot{\phi}^{2} \cos{\left(\theta \right)} + g\right)"
    r" \sin{\left(\theta \right)}}{R}"
)
PHI = r"\ddot{\phi} = - \frac{2 \dot{\phi} \dot{\theta}}{\tan{\left(\theta \right)}}"
EOM = (
    r"R m \left(- \frac{R \dot{\phi}^{2} \sin{\left(2 \theta \right)}}{2} + R \ddot{\theta}"
    r" - g \sin{\left(\theta \right)}\right) = 0"
)
P_PHI = r"p_{\phi} = R^{2} m \dot{\phi} \sin^{2}{\left(\theta \right)}"
ENERGY = (
    r"h = \frac{R^{2} m \dot{\phi}^{2} \sin^{2}{\left(\theta \right)}}{2}"
    r" + \frac{R^{2} m \dot{\theta}^{2}}{2} + R g m \cos{\left(\theta \right)}"
)
EVIDENCE = [
    ClaimEvidence(
        latex=THETA,
        source="r1",
        role="acceleration",
        variables=V,
        restrictions=[r"\sin\theta \neq 0"],
    ),
    ClaimEvidence(latex=PHI, source="r1", role="acceleration", variables=V),
    ClaimEvidence(latex=EOM, source="r1", role="equation_of_motion", variables=V),
    ClaimEvidence(latex=P_PHI, source="r1", role="momentum", variables=V),
    ClaimEvidence(latex=ENERGY, source="r1", role="conserved", variables=V, basis="symbolic"),
]


def run(claims: list[str], evidence: list[ClaimEvidence] = EVIDENCE):  # type: ignore[no-untyped-def]
    return check_claims(claims, {}, [], budget_limits(), evidence=evidence)


@pytest.mark.parametrize(
    ("claim", "rule"),
    [
        (
            r"\ddot\theta = \sin\theta\cos\theta\,\dot\phi^2 + \frac{g}{R}\sin\theta",
            "same_left_side",
        ),
        (r"\ddot\phi = -2\cot\theta\,\dot\theta\dot\phi", "same_left_side"),
        (r"p_\phi = mR^2\sin^2\theta\,\dot\phi", "same_left_side"),
        (
            r"E = \frac12 mR^2(\dot\theta^2+\sin^2\theta\,\dot\phi^2)+mgR\cos\theta",
            "named_quantity",
        ),
        (
            r"mR^2\ddot\theta - mR^2\sin\theta\cos\theta\,\dot\phi^2 - mgR\sin\theta = 0",
            "proportional_residual",
        ),
        (
            r"\sin\theta\cos\theta\,\dot\phi^2 + \frac{g}{R}\sin\theta = \ddot\theta",
            "same_left_side",
        ),
    ],
)
def test_equivalent_forms_of_verified_results_are_supported(claim: str, rule: str) -> None:
    (result,) = run([claim])
    assert result.status == "supported", result
    assert result.evidence_rule == rule
    assert result.evidence_source == "r1"


def test_the_restrictions_of_the_result_travel_with_the_match() -> None:
    (result,) = run([r"\ddot\theta = \sin\theta\cos\theta\,\dot\phi^2 + \frac{g}{R}\sin\theta"])
    assert result.restrictions == [r"\sin\theta \neq 0"]
    (scaled,) = run([r"mR^2\ddot\theta - mR^2\sin\theta\cos\theta\,\dot\phi^2 - mgR\sin\theta = 0"])
    assert "m \\neq 0" in scaled.restrictions and "R \\neq 0" in scaled.restrictions


def test_the_basis_is_symbolic_only_when_the_result_and_the_link_both_are() -> None:
    (energy,) = run([r"E = \frac12 mR^2(\dot\theta^2+\sin^2\theta\,\dot\phi^2)+mgR\cos\theta"])
    assert energy.evidence_basis in ("symbolic", "numeric")
    (acceleration,) = run([r"\ddot\phi = -2\cot\theta\,\dot\theta\dot\phi"])
    assert acceleration.evidence_basis == "numeric"  # the tool's own check was sampled


@pytest.mark.parametrize(
    "claim",
    [
        r"\ddot\theta = \sin\theta\cos\theta\,\dot\phi^2 - \frac{g}{R}\sin\theta",  # wrong sign
        r"\ddot\theta = 1.000001\sin\theta\cos\theta\,\dot\phi^2 + \frac{g}{R}\sin\theta",  # 1e-6
        r"\ddot\phi = -2\tan\theta\,\dot\theta\dot\phi",  # tan for cot
        r"p_\phi = mR^2\sin\theta\,\dot\phi",  # missing square
        r"E = \frac12 mR^2(\dot\theta^2+\sin^2\theta\,\dot\phi^2)-mgR\cos\theta",  # wrong sign of V
        r"E = \frac12 mR^2\dot\theta^2 + mgR\cos\theta",  # drops a term
        r"mR^2\ddot\theta - mR^2\sin\theta\cos\theta\,\dot\phi^2 + mgR\sin\theta = 0",
        r"y = 2",
        r"\theta",
    ],
)
def test_a_wrong_or_unrelated_claim_is_never_supported(claim: str) -> None:
    (result,) = run([claim])
    assert result.status != "supported", result


def test_a_new_name_for_a_value_is_not_supported_by_a_non_named_result() -> None:
    evidence = [ClaimEvidence(latex=r"x = 2", source="r1", role="")]
    assert run([r"y = 2"], evidence)[0].status != "supported"
    # a role that names quantities allows the model's own name for it
    named = [ClaimEvidence(latex=r"h = m g", source="r1", role="conserved")]
    assert run([r"W = m g"], named)[0].status == "supported"
    assert run([r"W = 2 m g"], named)[0].status != "supported"


def test_terms_moved_across_the_equals_sign_still_match() -> None:
    (result,) = run([r"\frac{g}{R}\sin\theta = \ddot\theta - \sin\theta\cos\theta\,\dot\phi^2"])
    assert result.status == "supported" and result.evidence_rule == "proportional_residual"


def test_a_true_form_multiplied_by_a_coordinate_dependent_factor_is_not_matched() -> None:
    """Known limit, pinned: the factor must not depend on the coordinates, velocities or accelerations."""
    claim = (
        r"\sin\theta\,\ddot\theta = \sin^2\theta\cos\theta\,\dot\phi^2 + \frac{g}{R}\sin^2\theta"
    )
    (result,) = run([claim])
    assert result.status != "supported"


def test_without_evidence_nothing_changes() -> None:
    (result,) = run([r"\ddot\phi = -2\cot\theta\,\dot\theta\dot\phi"], [])
    assert result.status == "unverified" and result.evidence_source is None


def test_matrices_distributions_and_sums_are_never_linked() -> None:
    evidence = [ClaimEvidence(latex=r"h = m g", source="r1", role="conserved")]
    (result,) = check_claims(
        [r"W = m g"], {}, [], budget_limits(), matrices=["m"], evidence=evidence
    )
    assert result.status == "unverified"


# -- issue #132: the model re-typed a verified constant of motion ----------------------------------------------
KEPLER_V = ["r", "theta"]
C4 = (
    r"C_{4} = G M \sin{\left(\theta \right)} - r^{3} \dot{\theta}^{2} \sin{\left(\theta \right)}"
    r" + r^{2} \dot{r} \dot{\theta} \cos{\left(\theta \right)}"
)
KEPLER_EVIDENCE = [ClaimEvidence(latex=C4, source="r2", role="conserved", variables=KEPLER_V)]


def kepler(claims: list[str], evidence: list[ClaimEvidence] = KEPLER_EVIDENCE):  # type: ignore[no-untyped-def]
    return run(claims, evidence)


def test_a_re_typed_constant_with_one_term_changed_is_reported_as_different_not_false() -> None:
    (result,) = kepler(
        [r"C_4 = G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\,\ddot\theta\cos\theta"]
    )
    assert result.status == "unverified"  # never `fails`, never `supported`
    assert result.evidence_rule == "mismatch" and result.evidence_source == "r2"
    assert "different expression" in result.detail and r"\dot{\theta} \cos" in result.detail


def test_the_same_constant_in_another_spelling_is_supported_not_a_mismatch() -> None:
    (result,) = kepler(
        [r"C_4 = r^2\dot r\dot\theta\cos\theta + G M\sin\theta - r^3\dot\theta^2\sin\theta"]
    )
    assert result.status == "supported" and result.evidence_rule == "same_left_side"


@pytest.mark.parametrize(
    "wrong",
    [
        r"C_4 = -G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\dot\theta\cos\theta",  # a sign
        r"C_4 = G M \sin\theta - r^3\dot\theta^2\sin\theta",  # a dropped term
        r"C_4 = 1.000001 (G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\dot\theta\cos\theta)",
    ],
)
def test_a_wrong_constant_is_a_mismatch_and_never_supported(wrong: str) -> None:
    (result,) = kepler([wrong])
    assert result.status == "unverified" and result.evidence_rule == "mismatch"


@pytest.mark.parametrize(
    "claim",
    [
        # the model's own name for a quantity: no mismatch with a result that has another name
        r"K = G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\,\ddot\theta\cos\theta",
        # the same name for something nobody verified (no evidence under that name)
        r"C_9 = r^2\dot\theta",
        # a relation that is not a definition of the name
        r"C_4 + 1 = r\dot r",
    ],
)
def test_only_the_tools_own_name_can_mismatch(claim: str) -> None:
    (result,) = kepler([claim])
    assert result.status != "supported"
    assert result.evidence_rule != "mismatch"


def test_evidence_without_a_role_is_never_compared_by_name() -> None:
    # results stored before roles existed (and every other tool's) cannot make a claim "different"
    plain = [ClaimEvidence(latex=C4, source="r2", role="", variables=KEPLER_V)]
    (result,) = kepler([r"C_4 = r\dot r"], plain)
    assert result.evidence_rule != "mismatch"


def test_the_models_own_name_for_a_verified_constant_still_needs_the_right_expression() -> None:
    right = r"K = G M \sin\theta - r^3\dot\theta^2\sin\theta + r^2\dot r\dot\theta\cos\theta"
    assert kepler([right])[0].evidence_rule == "named_quantity"
    for wrong in (
        right.replace("- r^3", "+ r^3"),
        right.replace(r"\cos\theta", r"\cos\theta\,(1+10^{-6})"),
        right.replace("G M ", ""),
    ):
        assert kepler([wrong])[0].status != "supported"
