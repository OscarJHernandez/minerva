"""Noncommuting symbols: declared once, supported by few operations, refused by all the others."""

import itertools

import pytest
import sympy as sp
from minerva_engine import (
    Engine,
    ExpressionInput,
    LimitError,
    Status,
    UnsupportedError,
    parse_latex,
    parse_text,
)
from minerva_engine.latexprint import to_latex
from minerva_engine.ncverify import MAX_NC_DEGREE, matrix_size, nc_degree, verify_equal_nc
from minerva_engine.ops import OPERATIONS
from minerva_engine.service import NONCOMMUTATIVE_OPERATIONS

NC = {"A": ["noncommutative"], "B": ["noncommutative"], "C": ["noncommutative"]}


def read(text: str) -> sp.Expr:
    return parse_text(text, NC, allow_noncommutative=True)


# --- reading -------------------------------------------------------------------------------------
def test_order_of_factors_is_kept_by_both_parsers() -> None:
    assert read("A*B") != read("B*A")
    assert parse_latex("B A", NC, allow_noncommutative=True) != parse_latex(
        "A B", NC, allow_noncommutative=True
    )
    assert to_latex(read("B*A*B")).replace(" ", "") == "BAB"


def test_a_parser_not_told_to_accept_them_refuses() -> None:
    with pytest.raises(UnsupportedError):
        parse_text("A*B", NC)
    with pytest.raises(UnsupportedError):
        parse_latex("A B", NC)


def test_a_noncommutative_symbol_carries_no_other_assumption() -> None:
    with pytest.raises(ValueError, match="noncommutative"):
        ExpressionInput(input="A", assumptions={"A": ["noncommutative", "real"]})


# --- matrix verification --------------------------------------------------------------------------
def test_the_commutative_expansion_of_a_square_is_wrong_and_is_caught() -> None:
    right = verify_equal_nc(read("(A + B)**2"), read("A**2 + A*B + B*A + B**2"))
    wrong = verify_equal_nc(read("(A + B)**2"), read("A**2 + 2*A*B + B**2"))
    assert right.passed and right.symbolic_zero
    assert not wrong.passed and not wrong.symbolic_zero


def test_swapping_two_factors_is_caught() -> None:
    assert not verify_equal_nc(read("A*B"), read("B*A")).passed
    assert verify_equal_nc(read("A*B*C"), read("A*(B*C)")).passed
    assert not verify_equal_nc(read("A*B*C"), read("A*C*B")).passed


def test_commuting_coefficients_and_functions_of_them_are_allowed() -> None:
    x = sp.Symbol("x")
    a, b = read("A"), read("B")
    assert verify_equal_nc(
        sp.sin(x) * a * b + sp.cos(x) * b * a, sp.sin(x) * a * b + sp.cos(x) * b * a
    ).passed
    assert not verify_equal_nc(sp.sin(x) * a * b, sp.cos(x) * a * b).passed


def test_small_matrices_would_accept_a_false_identity_so_the_size_follows_the_degree() -> None:
    """The standard polynomial of degree 4 vanishes on every 2x2 matrix but not on 3x3 ones: a check
    with 2x2 matrices would call it zero."""
    symbols = sp.symbols("P Q R S", commutative=False)
    s4 = sum(
        sp.combinatorics.Permutation(list(p)).signature() * sp.Mul(*[symbols[i] for i in p])
        for p in itertools.permutations(range(4))
    )
    assert matrix_size(nc_degree(s4)) >= 3
    verdict = verify_equal_nc(s4, sp.Integer(0))
    assert not verdict.passed, "s4 is not identically zero; only 2x2 matrices cannot see it"


def test_degree_and_size() -> None:
    assert nc_degree(read("A*B**2 + C")) == 3
    assert nc_degree(read("2*A")) == 1
    assert matrix_size(2) == 3 and matrix_size(10) == 6


@pytest.mark.parametrize("text", ["sin(A)", "1/A", "A**(-1)", "A**x", "exp(A)"])
def test_functions_inverses_and_symbolic_powers_of_them_are_outside_the_subset(text: str) -> None:
    with pytest.raises(UnsupportedError):
        nc_degree(read(text))


def test_a_degree_beyond_the_limit_is_a_resource_limit_not_a_guess() -> None:
    big = read(f"(A + B)**{MAX_NC_DEGREE + 1}")
    with pytest.raises(LimitError):
        verify_equal_nc(big, big)


# --- the engine refuses them everywhere else --------------------------------------------------------
ENGINE_ONLY = sorted(set(OPERATIONS) - NONCOMMUTATIVE_OPERATIONS)


@pytest.mark.parametrize("op", ENGINE_ONLY)
def test_every_other_operation_refuses_before_running(engine: Engine, op: str) -> None:
    result, _ = engine._run(op, {"expression": {"input": "A", "assumptions": NC}})
    assert result.outcome.value == "unsupported", op


def test_public_methods_report_unsupported_not_a_wrong_answer(engine: Engine) -> None:
    expression = ExpressionInput(input="A B - B A", format="text", assumptions=NC)
    assert engine.simplify(expression).status is Status.UNSUPPORTED
    assert engine.solve("A x = B", "x", NC).status is Status.UNSUPPORTED
    assert engine.features(expression)[0] is Status.UNSUPPORTED


def test_commutative_declarations_still_work_everywhere(engine: Engine) -> None:
    result = engine.simplify(ExpressionInput(input="sin(x)**2 + cos(x)**2", format="text"))
    assert result.status is Status.OK and result.result_text == "1"
