"""Property tests for the point-source distributional divergence.

For every bounded random source `a` and strength `C` the engine must derive `4 pi C delta^(3)(r - a)` with
every hypothesis exact and every independent pairing agreeing, and must reject the same identity with a wrong
coefficient or a wrong source. Hypothesis is derandomized in the pull-request tier (the same examples every
run); the `extended` profile searches at random and prints the failing example, which is then pinned with
`@example` here."""

import math
from fractions import Fraction

import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st
from minerva_engine import FieldDecl, VectorInput
from minerva_engine.latexprint import to_latex
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits
from minerva_engine.vector import vector_calculus
from strategies.vector_singular import Kernel, kernels, wrong_factor

L = budget_limits()
SETTINGS = settings(max_examples=6, deadline=None, derandomize=True)


def divergence(components: list[str], target: str | None = None):
    return vector_calculus(
        VectorInput(
            expression=r"\nabla\cdot F",
            fields=[FieldDecl(name="F", kind="vector", components=components)],
            action="distributional",
            target=target,
        ),
        L,
    )


def claim_text(strength: Fraction, source) -> str:
    def lit(v: Fraction) -> str:
        return f"{v.numerator}/{v.denominator}" if v.denominator != 1 else str(v.numerator)

    where = ",".join(lit(v) for v in source)
    return rf"{lit(4 * strength)}\pi\delta^{{(3)}}(\mathbf r-({where}))"


@SETTINGS
@given(kernels())
def test_the_derived_strength_and_source_are_the_ones_the_field_was_built_from(k: Kernel) -> None:
    r = divergence(k.components())
    assert r.status is Status.OK, r.message
    s = r.singular
    assert s.proof_status == "derived" and all(d.status == "established" for d in s.derivation)
    want = 4 * sp.pi * sp.Rational(k.strength.numerator, k.strength.denominator)
    assert s.strength_latex == to_latex(want)
    assert all(w.passed for w in s.weak_checks) and all(f.passed for f in s.flux_checks)
    for flux in s.flux_checks:  # the same flux through every sphere, as the derivation says
        assert flux.measured == pytest.approx(4 * math.pi * float(k.strength), rel=1e-9)


@SETTINGS
@given(kernels())
def test_the_source_the_field_is_built_from_is_the_source_in_the_claim(k: Kernel) -> None:
    ok = divergence(k.components(), claim_text(k.strength, k.source))
    assert ok.singular.claim.holds and ok.singular.claim.reason == "agrees"


@SETTINGS
@given(kernels(), wrong_factor)
def test_a_wrong_coefficient_in_the_claim_is_never_accepted(k: Kernel, factor: Fraction) -> None:
    r = divergence(k.components(), claim_text(k.strength * (1 + factor), k.source))
    assert not r.singular.claim.holds and r.singular.claim.reason == "different_strength"


@SETTINGS
@given(kernels())
def test_a_moved_source_in_the_claim_is_never_accepted(k: Kernel) -> None:
    moved = (k.source[0] + 1, k.source[1], k.source[2])
    r = divergence(k.components(), claim_text(k.strength, moved))
    assert not r.singular.claim.holds and r.singular.claim.reason == "different_source"


@SETTINGS
@given(st.data())
def test_a_one_part_in_a_million_error_in_a_pairing_is_detected(data: st.DataObject) -> None:
    from minerva_engine.vector import weak as W

    k = data.draw(kernels())
    a = tuple(float(v) for v in k.source)
    c = float(k.strength)

    def field(scale: float):
        def f(x, y, z):
            q = ((x - a[0]) ** 2 + (y - a[1]) ** 2 + (z - a[2]) ** 2) ** 1.5
            return (
                scale * c * (x - a[0]) / q,
                scale * c * (y - a[1]) / q,
                scale * c * (z - a[2]) / q,
            )

        return f

    phi = W.standard_test_functions(a)[1]
    expected = 4 * math.pi * c * phi.value_at(a)
    exact = W.pair_divergence(field(1.0), phi, a)
    off = W.pair_divergence(field(1.0 + 1e-6), phi, a)
    assert abs(exact - expected) <= W.PAIRING_TOLERANCE * max(1.0, abs(expected))
    assert abs(off - expected) > W.PAIRING_TOLERANCE * max(1.0, abs(expected))
