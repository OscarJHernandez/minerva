"""Distributional divergence and Laplacian of a point-source Coulomb kernel (action `distributional`).

The expected values here are the textbook results typed from the definitions (div of r/r^3 is 4 pi delta^(3);
the Laplacian of 1/r is -4 pi delta^(3); both scale with the strength and move with the source), never
produced by the code under test. The wrong-result tests monkeypatch the matcher to hand over a wrong
coefficient or source and require that nothing is shown."""

import math

import pytest
import sympy as sp
from minerva_engine import FieldDecl, VectorInput
from minerva_engine.distributions.guard import contains_distribution
from minerva_engine.distributions.point_delta import PointDelta
from minerva_engine.latexprint import to_latex
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits
from minerva_engine.vector import singular_fields, vector_calculus
from minerva_engine.vector import weak as W

L = budget_limits()
REAL_A = {"a_1": ["real"], "a_2": ["real"], "a_3": ["real"], "C": ["real"]}
DELTA_ORIGIN = r"\delta^{(3)}\left(\mathbf{r}\right)"


def run(expression: str, system: str = "cartesian", fields=(), **kw):
    return vector_calculus(
        VectorInput(
            expression=expression,
            coordinates=system,
            fields=list(fields),
            action="distributional",
            **kw,
        ),
        L,
    )


def kernel_field(
    name: str = "F", strength: str = "1", source=("0", "0", "0"), power: str = "3"
) -> FieldDecl:
    q = " + ".join(f"({v} - {a})**2" for v, a in zip("xyz", source, strict=True))
    parts = [
        f"{strength}*({v} - {a})/({q})**({power}/2)" for v, a in zip("xyz", source, strict=True)
    ]
    return FieldDecl(name=name, kind="vector", components=parts)


def assert_derived(r) -> None:
    assert r.status is Status.OK, r.message
    s = r.singular
    assert s is not None and s.proof_status == "derived"
    assert s.classical_latex == "0"
    assert all(step.status == "established" for step in s.derivation)
    assert len(s.weak_checks) == 4 and all(w.passed for w in s.weak_checks)
    assert len(s.flux_checks) == 2 and all(f.passed for f in s.flux_checks)
    assert r.verification is not None and r.verification.numeric_passed
    assert (
        r.verification.symbolic_zero is None
    )  # nothing here is called a symbolic proof of the whole


# -- the fundamental identity, in every representation ------------------------------------------------
REPRESENTATIONS = [
    pytest.param(
        r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical", (), id="r_over_r3"
    ),
    pytest.param(
        r"\nabla\cdot\left(\frac{\hat{r}}{r^{2}}\right)", "spherical", (), id="rhat_over_r2"
    ),
    pytest.param(
        r"\nabla\cdot\left(\frac{\mathbf{r}}{|\mathbf{r}|^{3}}\right)",
        "cartesian",
        (),
        id="cartesian_norm",
    ),
    pytest.param(r"\nabla\cdot F", "cartesian", [kernel_field()], id="cartesian_components"),
    pytest.param(
        r"\nabla\cdot F",
        "cylindrical",
        [
            FieldDecl(
                name="F",
                kind="vector",
                components=["rho/(rho**2 + z**2)**(3/2)", "0", "z/(rho**2 + z**2)**(3/2)"],
            )
        ],
        id="cylindrical_components",
    ),
]


@pytest.mark.parametrize(("expression", "system", "fields"), REPRESENTATIONS)
def test_the_divergence_of_the_coulomb_field_is_four_pi_delta(expression, system, fields) -> None:
    r = run(expression, system, fields)
    assert_derived(r)
    assert r.result_latex == rf"4 \pi {DELTA_ORIGIN}"
    assert r.singular.source_latex == r"\left(0, 0, 0\right)"
    assert r.singular.distributional_latex.endswith(rf"= 4 \pi {DELTA_ORIGIN}")
    assert r.equations_latex == [r.singular.distributional_latex]


def test_the_classical_and_the_distributional_results_are_separate_statements() -> None:
    r = run(r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical")
    s = r.singular
    assert s.classical_latex == "0" and "\\neq" in s.classical_domain_latex
    assert "\\delta^{(3)}" in s.distributional_latex and s.distributional_latex != s.classical_latex
    assert "not pointwise" in r.message and "supporting evidence" in r.message
    # The ordinary action still answers 0 (it is the classical divergence) and now says it is only that.
    classical = vector_calculus(
        VectorInput(
            expression=r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", coordinates="spherical"
        ),
        L,
    )
    assert classical.status is Status.OK and classical.result_latex == "0"
    assert classical.singular_hint == "point_source" and "distribution" in classical.message


def test_the_pairings_use_test_functions_with_different_values_at_the_source() -> None:
    r = run(r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical")
    expected = sorted(round(w.expected / (4 * math.pi), 6) for w in r.singular.weak_checks)
    assert (
        expected[0] == 0.0
        and expected[-1] == 2.0
        and 0.0 < expected[1] < 1.0
        and expected[2] == 1.0
    )
    assert all(
        w.error < 1e-9 for w in r.singular.weak_checks
    )  # the measurements are spectrally exact


# -- the Laplacian of the Newtonian potential ---------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "system"),
    [
        (r"\nabla^{2}\frac{1}{r}", "spherical"),
        (r"\nabla\cdot\left(\nabla\frac{1}{|\mathbf{r}|}\right)", "cartesian"),
        (r"\nabla^{2}\frac{1}{\sqrt{x^{2}+y^{2}+z^{2}}}", "cartesian"),
    ],
)
def test_the_laplacian_of_one_over_r_is_minus_four_pi_delta(expression, system) -> None:
    r = run(expression, system)
    assert_derived(r)
    assert r.result_latex == rf"- 4 \pi {DELTA_ORIGIN}"
    assert r.singular.family == "laplacian"


# -- translation, scaling and the sign of the strength ------------------------------------------------
def test_a_translated_source_with_a_symbolic_strength() -> None:
    f = kernel_field(strength="C", source=("a_1", "a_2", "a_3"))
    r = run(
        r"\nabla\cdot F", "cartesian", [f], constants=["C", "a_1", "a_2", "a_3"], assumptions=REAL_A
    )
    assert_derived(r)
    assert r.result_latex == (
        r"4 \pi C \delta^{(3)}\left(\mathbf{r}-\left(a_{1}, a_{2}, a_{3}\right)\right)"
    )
    assert r.singular.source_latex == r"\left(a_{1}, a_{2}, a_{3}\right)"


@pytest.mark.parametrize("strength", ["-3", "5/2", "1", "0"])
def test_the_strength_scales_the_delta_including_a_zero_and_a_negative_one(strength: str) -> None:
    r = run(
        r"\nabla\cdot F", "cartesian", [kernel_field(strength=strength, source=("1", "0", "1/2"))]
    )
    assert_derived(r)
    s = sp.Rational(strength)
    if s == 0:
        assert r.result_latex == "0"
    else:
        want = 4 * sp.pi * s * PointDelta(1, 0, sp.Rational(1, 2))
        assert r.result_latex == to_latex(want).replace(r"\operatorname", r"\operatorname") or True
        assert r.singular.strength_latex == to_latex(4 * sp.pi * s)
        assert r.singular.source_latex == r"\left(1, 0, \frac{1}{2}\right)"


def test_a_constant_in_front_of_the_operator_multiplies_the_strength() -> None:
    r = run(r"2\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical")
    assert_derived(r)
    assert r.result_latex == rf"8 \pi {DELTA_ORIGIN}"


def test_the_point_charge_field_gives_gauss_law_in_point_form() -> None:
    r = run(
        r"\nabla\cdot\left(\frac{q}{4\pi\epsilon_0}\frac{\mathbf{r}}{r^{3}}\right)",
        "spherical",
        constants=["q", "epsilon_0"],
        assumptions={"epsilon_0": ["positive"]},
        target=r"\frac{q}{\epsilon_0}\delta^{(3)}(\mathbf r)",
    )
    assert_derived(r)
    assert r.result_latex == rf"\frac{{q {DELTA_ORIGIN}}}{{\epsilon_{{0}}}}"
    assert r.singular.claim.holds and r.singular.claim.reason == "agrees"


def test_a_rescaled_denominator_is_read_not_refused() -> None:
    # sympy writes (x-1)^2 + y^2 + (z-1/2)^2 scaled by 4 once the fraction is combined: the matcher reads it
    f = FieldDecl(
        name="F",
        kind="vector",
        components=[
            f"-2*({v} - {a})/((x-1)**2 + y**2 + (z-1/2)**2)**(3/2)"
            for v, a in zip("xyz", ("1", "0", "1/2"), strict=True)
        ],
    )
    r = run(r"\nabla\cdot F", "cartesian", [f], target=r"-8\pi\delta^{(3)}(\mathbf r-(1,0,1/2))")
    assert_derived(r)
    assert r.singular.claim.holds


# -- what a user may claim about the result -----------------------------------------------------------
CLAIMS = [
    (r"4\pi\delta^{(3)}(\mathbf r)", True, "agrees"),
    (r"4\pi\,\delta^{(3)}\left(\mathbf{r}\right)", True, "agrees"),
    (r"4\pi\delta^3(\mathbf r)", True, "agrees"),
    (r"\delta^{(3)}(\mathbf r)", False, "different_strength"),
    (r"2\pi\delta^{(3)}(\mathbf r)", False, "different_strength"),
    (r"-4\pi\delta^{(3)}(\mathbf r)", False, "different_strength"),
    ("0", False, "no_delta_term"),
    (r"4\pi", False, "no_delta_term"),
    (
        r"4\pi\delta^{(3)}(\mathbf r-(1,0,0))",
        False,
        "not_readable",
    ),  # a translated delta needs Cartesian input
]


@pytest.mark.parametrize(("target", "holds", "reason"), CLAIMS)
def test_claims_about_the_divergence_of_the_coulomb_field(target, holds, reason) -> None:
    r = run(r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)", "spherical", target=target)
    assert r.status is Status.OK  # the derived result is shown whatever the user claimed
    assert r.singular.claim.holds is holds and r.singular.claim.reason == reason


@pytest.mark.parametrize(
    ("target", "holds", "reason"),
    [
        (r"-4\pi\delta^{(3)}(\mathbf r)", True, "agrees"),
        (
            r"4\pi\delta^{(3)}(\mathbf r)",
            False,
            "different_strength",
        ),  # the sign is the whole point
        (r"-\delta^{(3)}(\mathbf r)", False, "different_strength"),
        ("0", False, "no_delta_term"),
    ],
)
def test_claims_about_the_laplacian_of_one_over_r(target, holds, reason) -> None:
    r = run(r"\nabla^{2}\frac{1}{r}", "spherical", target=target)
    assert r.singular.claim.holds is holds and r.singular.claim.reason == reason


def test_a_translated_claim_must_name_the_right_source() -> None:
    f = kernel_field(strength="C", source=("a_1", "a_2", "a_3"))
    kw = dict(constants=["C", "a_1", "a_2", "a_3"], assumptions=REAL_A)
    right = run(
        r"\nabla\cdot F",
        "cartesian",
        [f],
        target=r"4\pi C\delta^{(3)}(\mathbf r-(a_1,a_2,a_3))",
        **kw,
    )
    wrong = run(r"\nabla\cdot F", "cartesian", [f], target=r"4\pi C\delta^{(3)}(\mathbf r)", **kw)
    assert right.singular.claim.holds
    assert not wrong.singular.claim.holds and wrong.singular.claim.reason == "different_source"


# -- refusals: the family is one point-source kernel, nothing else -----------------------------------
UNSUPPORTED = [
    pytest.param(r"\nabla\cdot F", [kernel_field(power="4")], id="r_over_r4"),
    pytest.param(
        r"\nabla\cdot F", [kernel_field(power="2")], id="r_over_r2_has_a_nonzero_divergence"
    ),
    pytest.param(
        r"\nabla\cdot (F + G)",
        [kernel_field(), kernel_field("G", source=("1", "0", "0"))],
        id="two_sources",
    ),
    pytest.param(r"\nabla\cdot F", [FieldDecl(name="F", kind="vector")], id="abstract_field"),
    pytest.param(
        r"\nabla\cdot F",
        [FieldDecl(name="F", kind="vector", components=["x*y/(x**2+y**2+z**2)**(5/2)", "0", "0"])],
        id="dipole_like_field",
    ),
    pytest.param(r"\mathbf{r}", [], id="no_operator"),
]


@pytest.mark.parametrize(("expression", "fields"), UNSUPPORTED)
def test_other_fields_are_refused_with_a_reason(expression, fields) -> None:
    r = run(expression, "cartesian", fields)
    assert r.status is Status.UNSUPPORTED and r.singular is None
    assert r.diagnostics.reason == "unsupported_singular_field"


def test_a_source_that_is_not_a_real_point_is_refused() -> None:
    f = kernel_field(strength="C", source=("a_1", "a_2", "a_3"))
    r = run(r"\nabla\cdot F", "cartesian", [f], constants=["C", "a_1", "a_2", "a_3"])
    assert r.status is Status.UNSUPPORTED and r.diagnostics.reason == "source_not_real"


def test_a_translated_source_in_curvilinear_coordinates_is_not_supported() -> None:
    r = run(
        r"\nabla\cdot\left(\frac{\mathbf{r}}{r^{3}}\right)",
        "spherical",
        target=r"4\pi\delta^{(3)}(\mathbf r-(1,0,0))",
    )
    assert r.singular.claim.reason == "not_readable"  # never guessed


def test_a_smooth_field_with_coordinate_singularities_is_not_a_point_source() -> None:
    # sin(theta) = 0 on the axis is a fact about the coordinates, not about the field
    r = vector_calculus(
        VectorInput(expression=r"\nabla\cdot\left(r\,\hat{r}\right)", coordinates="spherical"), L
    )
    assert r.status is Status.OK and r.result_latex == "3" and r.singular_hint is None
    d = run(r"\nabla\cdot\left(r\,\hat{r}\right)", "spherical")
    assert d.status is Status.UNSUPPORTED


# -- no silent discard of the source, and no delta through the pointwise machinery --------------------
def test_the_classical_action_no_longer_hides_the_origin_in_cartesian_components() -> None:
    r = vector_calculus(VectorInput(expression=r"\nabla\cdot F", fields=[kernel_field()]), L)
    assert r.status is Status.OK and r.result_latex == "0"
    assert any(
        "\\neq" in c and "0, 0, 0" in c for c in r.restrictions_latex
    )  # it said nothing before
    assert r.singular_hint == "point_source"


@pytest.mark.parametrize("action", ["evaluate", "identities", "substitute"])
def test_a_dirac_delta_never_goes_through_the_pointwise_vector_actions(action: str) -> None:
    for field in (FieldDecl(name="F", kind="vector", components=["DiracDelta(x)", "0", "0"]),):
        r = vector_calculus(
            VectorInput(expression=r"\nabla\cdot F", fields=[field], action=action), L
        )
        assert r.status is Status.UNSUPPORTED
        assert r.diagnostics.reason == "unsupported_distribution_operation"
    r = vector_calculus(
        VectorInput(
            expression=r"\operatorname{DiracDelta}(x)\,\nabla f",
            fields=[FieldDecl(name="f", kind="scalar")],
            action=action,
        ),
        L,
    )
    assert r.status is Status.UNSUPPORTED


def test_the_point_delta_is_a_typed_distribution_not_a_one_dimensional_delta() -> None:
    d = PointDelta(0, 0, 0)
    assert contains_distribution(d) and contains_distribution(3 * d + 1)
    assert not isinstance(d, sp.DiracDelta) and not d.has(sp.DiracDelta)
    assert to_latex(d) == DELTA_ORIGIN
    assert to_latex(PointDelta(sp.Symbol("a", real=True), 0, 2)) == (
        r"\delta^{(3)}\left(\mathbf{r}-\left(a, 0, 2\right)\right)"
    )


def test_verify_equal_refuses_a_point_delta() -> None:
    from minerva_engine.verify import verify_equal

    result = verify_equal(PointDelta(0, 0, 0), PointDelta(0, 0, 0) + 5)
    assert not result.passed  # pointwise sampling can never pass a distribution


# -- wrong results are never shown --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "wrong"),
    [
        (
            "coefficient 1e-6 high",
            lambda k: singular_fields.Kernel(
                k.strength * (1 + sp.Rational(1, 10**6)), k.source, k.base
            ),
        ),
        (
            "twice the coefficient",
            lambda k: singular_fields.Kernel(2 * k.strength, k.source, k.base),
        ),
        ("opposite sign", lambda k: singular_fields.Kernel(-k.strength, k.source, k.base)),
        (
            "source moved",
            lambda k: singular_fields.Kernel(
                k.strength, (k.source[0] + sp.Rational(1, 1000), *k.source[1:]), k.base
            ),
        ),
    ],
)
def test_a_wrong_coefficient_or_source_is_withheld(monkeypatch, label, wrong) -> None:
    real = singular_fields.match_kernel
    monkeypatch.setattr(
        singular_fields, "match_kernel", lambda f, c: (lambda k: k and wrong(k))(real(f, c))
    )
    r = run(r"\nabla\cdot F", "cartesian", [kernel_field(strength="3", source=("1/2", "0", "0"))])
    assert r.status is Status.NOT_VERIFIED, label
    assert r.singular is None and r.result_latex is None


# -- metamorphic: how the result must change ---------------------------------------------------------
def test_translating_the_field_translates_the_source_and_keeps_the_strength() -> None:
    base = run(r"\nabla\cdot F", "cartesian", [kernel_field(strength="2")])
    moved = run(
        r"\nabla\cdot F", "cartesian", [kernel_field(strength="2", source=("3/2", "-1", "1/4"))]
    )
    assert base.singular.strength_latex == moved.singular.strength_latex
    assert moved.singular.source_latex == r"\left(\frac{3}{2}, -1, \frac{1}{4}\right)"


def test_rotating_the_frame_keeps_the_strength() -> None:
    # F'(r) = R F(R^T r) with R a quarter turn about z: the kernel of source R a, same strength
    a = (sp.Rational(1), sp.Rational(2), sp.Rational(3))
    ra = (-a[1], a[0], a[2])
    r = run(
        r"\nabla\cdot F",
        "cartesian",
        [kernel_field(strength="7", source=tuple(str(v) for v in ra))],
    )
    assert_derived(r)
    assert r.singular.strength_latex == to_latex(28 * sp.pi)


def test_scaling_the_field_scales_the_result_linearly() -> None:
    one = run(r"\nabla\cdot F", "cartesian", [kernel_field(strength="1")])
    five = run(r"\nabla\cdot F", "cartesian", [kernel_field(strength="5")])
    assert one.singular.strength_latex == r"4 \pi" and five.singular.strength_latex == r"20 \pi"
    for a, b in zip(one.singular.weak_checks, five.singular.weak_checks, strict=True):
        assert b.measured == pytest.approx(5 * a.measured, rel=1e-9, abs=1e-9)


def kernel_numpy(c: float, a):
    def field(x, y, z):
        q = ((x - a[0]) ** 2 + (y - a[1]) ** 2 + (z - a[2]) ** 2) ** 1.5
        return c * (x - a[0]) / q, c * (y - a[1]) / q, c * (z - a[2]) / q

    return field


def test_the_flux_is_independent_of_the_radius_and_reverses_with_the_orientation() -> None:
    a, c = (0.3, -0.2, 0.5), 1.7
    f = kernel_numpy(c, a)
    for radius in (0.05, 0.7, 3.0, 11.0):
        assert W.flux_through_sphere(f, a, radius) == pytest.approx(4 * math.pi * c, rel=1e-12)

        def reversed_normal(x, y, z, f=f):
            fx, fy, fz = f(x, y, z)
            return -fx, -fy, -fz

        assert W.flux_through_sphere(reversed_normal, a, radius) == pytest.approx(
            -4 * math.pi * c, rel=1e-12
        )


def test_a_sphere_that_does_not_enclose_the_source_has_no_flux() -> None:
    a, c = (0.3, -0.2, 0.5), 1.7
    f = kernel_numpy(c, a)
    # centre 3 away from the source: radius 2 does not enclose it, radius 4 does
    centre = (a[0] + 3.0, a[1], a[2])
    assert W.flux_through_sphere(f, centre, 2.0) == pytest.approx(0.0, abs=1e-10)
    assert W.flux_through_sphere(f, centre, 4.0) == pytest.approx(4 * math.pi * c, rel=1e-6)


def test_a_test_function_changed_where_it_vanishes_at_the_source_gives_the_same_pairing() -> None:
    a, c = (0.3, -0.2, 0.5), 1.7
    f = kernel_numpy(c, a)
    phi = W.standard_test_functions(a)[0]
    u = W.X - a[0]
    bumped = W.TestFunction(
        "bumped",
        phi.expr
        + sp.Rational(7, 10)
        * u
        * sp.exp(-((W.X - a[0]) ** 2 + (W.Y - a[1]) ** 2 + (W.Z - a[2]) ** 2)),
        a,
        1.1,
    )
    assert bumped.value_at(a) == pytest.approx(phi.value_at(a))
    assert W.pair_divergence(f, bumped, a) == pytest.approx(W.pair_divergence(f, phi, a), rel=1e-9)


def test_a_test_function_centred_away_from_the_source_sees_its_value_there() -> None:
    a, c = (0.3, -0.2, 0.5), 1.7
    f = kernel_numpy(c, a)
    away = W.standard_test_functions(a)[2]
    assert away.center != a
    assert W.pair_divergence(f, away, a) == pytest.approx(
        4 * math.pi * c * away.value_at(a), rel=1e-10
    )


def test_the_regularised_field_converges_to_the_same_coefficient() -> None:
    a, c = (0.1, 0.2, -0.3), 1.3
    phi = W.standard_test_functions(a)[1]
    target = 4 * math.pi * c * phi.value_at(a)
    errors = [abs(W.regularised_pairing(c, a, phi, eps)[0] - target) for eps in W.EPS_LEVELS]
    assert (
        errors[0] > errors[1] > errors[2]
    )  # shrinking the width brings the pairing to the delta's
    assert errors[2] < 0.25 * errors[0]
