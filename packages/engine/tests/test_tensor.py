"""Tensor canonicalization: only the monoterm-symmetry subset SymPy supports."""

import pytest
from minerva_engine import Engine, Status, TensorInput

SPECS = {
    "S": {"rank": 2, "symmetry": "symmetric"},
    "A": {"rank": 2, "symmetry": "antisymmetric"},
    "T": {"rank": 2},
    "R": {"rank": 4, "symmetry": "riemann"},
}


def canon(engine: Engine, expr: str):
    return engine.tensor_canonicalize(TensorInput(input=expr, tensors=SPECS))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "expr",
    [
        "S(a,b)*A(-a,-b)",  # symmetric x antisymmetric contraction vanishes
        "A(a,b) + A(b,a)",
        "R(a,b,c,d) + R(b,a,c,d)",
        "R(a,b,c,d) - R(c,d,a,b)",
        "R(a,-a,b,c)",
    ],
)
def test_vanishing_expressions(engine: Engine, expr: str) -> None:
    result = canon(engine, expr)
    assert result.status is Status.OK
    assert result.result_text == "0"


def test_symmetric_reordering(engine: Engine) -> None:
    assert canon(engine, "S(b,a)").result_text == "S(a, b)"
    assert canon(engine, "A(b,a)").result_text == "-A(a, b)"


def test_nonsymmetric_tensor_is_not_wrongly_reduced(engine: Engine) -> None:
    # T has no symmetry: T(a,b)T(-b,-a) differs from T(a,b)T(-a,-b); it must not vanish or merge.
    result = canon(engine, "T(a,b)*T(-b,-a) - T(a,b)*T(-a,-b)")
    assert result.result_text != "0"


def test_dummy_index_renaming_combines_terms(engine: Engine) -> None:
    result = canon(engine, "2*S(a,b)*T(-a,-b) + 3*S(c,d)*T(-c,-d)")
    assert result.result_text is not None
    assert result.result_text.startswith("5*")


def test_bianchi_is_not_applied_and_says_so(engine: Engine) -> None:
    result = canon(engine, "R(a,b,c,d) + R(a,c,d,b) + R(a,d,b,c)")
    assert result.result_text != "0"  # true identity, but beyond monoterm symmetries
    assert "Bianchi" in result.message
    assert result.independent_verification is False


@pytest.mark.parametrize("expr", ["S(a,b)*S(a,b)", "X(a)", "S(a)", "S(a,b)**2", "__import__('os')"])
def test_invalid_tensor_input(engine: Engine, expr: str) -> None:
    assert canon(engine, expr).status is Status.PARSE_ERROR
