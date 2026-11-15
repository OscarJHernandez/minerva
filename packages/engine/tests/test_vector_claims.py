"""Vector notation is never checked by sampling numbers.

The strict parser reads `\\times` and `\\cdot` as scalar products, so `a \\times b = b \\times a`
(false for vectors) used to come out as "holds", and `(a \\cdot b) c = a (b \\cdot c)` as well."""

import pytest
from minerva_engine import Limits, check_claim, check_claims
from minerva_engine.claims import uses_vector_notation

L = Limits()


@pytest.mark.parametrize(
    "latex",
    [
        r"a \times b = b \times a",
        r"\mathbf{a} \times \mathbf{b} = \mathbf{b} \times \mathbf{a}",
        r"\vec{A} \cdot \vec{B} = \vec{B} \cdot \vec{A}",
        r"\nabla \times (\nabla \phi) = 0",
        r"\nabla^2 \phi = \nabla \cdot (\nabla \phi)",
        r"\frac{\partial \mathbf{B}}{\partial t} = 0",
        r"\hat{x} \times \hat{y} = \hat{z}",
        r"\boldsymbol{E} = -\nabla \phi",
        r"(x + y) \times (x - y) = x^2 - y^2",
    ],
)
def test_vector_notation_is_unverified_not_sampled(latex: str) -> None:
    result = check_claim(latex, {}, L)
    assert result.status == "unverified"
    assert "vector" in result.detail


@pytest.mark.parametrize(
    "latex",
    [r"3 \times 4 = 12", r"2 \times x = x + x", r"x \times 2 = 2x", r"3 \times 10^{8} = 300000000"],
)
def test_numeral_times_is_ordinary_multiplication(latex: str) -> None:
    assert not uses_vector_notation(latex)
    assert check_claim(latex, {}, L).status == "holds"


def test_declared_vector_names_are_never_sampled() -> None:
    results = check_claims([r"(a \cdot b) c = a (b \cdot c)"], {}, [], L, vectors=["a", "b"])
    assert results[0].status == "unverified"
    # the same text without the declaration is a scalar identity
    assert check_claims([r"(a \cdot b) c = a (b \cdot c)"], {}, [], L)[0].status == "holds"


def test_a_vector_statement_is_not_a_scalar_definition() -> None:
    # `E = -\nabla \phi` must not be substituted into a later scalar claim
    results = check_claims(
        [r"E = -\nabla \phi", r"E + 1 = 1 - E"],
        {},
        [],
        L,
    )
    assert results[0].status == "unverified"
    assert results[1].status != "holds"


def test_engine_entry_point_forwards_vectors(engine) -> None:
    result = engine.check_claims([r"(a \cdot b) c = a (b \cdot c)"], vectors=["a", "b"])
    assert result[0].status == "unverified"
