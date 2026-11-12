"""The numeric sampler must reject identities that only hold on a small region or branch."""

import pytest
import sympy as sp
from minerva_engine.verify import numeric_detail, verify_equal

x = sp.Symbol("x")
xr = sp.Symbol("xr", real=True)
z = sp.Symbol("z")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (sp.log(sp.exp(z)), z),  # only for |Im z| < pi
        (sp.Abs(xr - 10), 10 - xr),  # only for xr <= 10
        (sp.sqrt((xr - 4) ** 2), 4 - xr),  # only for xr <= 4
        (sp.atan(sp.tan(xr / 3)), xr / 3),  # only for |xr| < 3*pi/2
        (sp.asin(sp.sin(z / 3)), z / 3),
        (xr, xr + sp.exp(xr - 40)),  # differs only at large xr
    ],
)
def test_identities_that_pass_a_small_sample_are_rejected(a: sp.Expr, b: sp.Expr) -> None:
    result = verify_equal(a, b, symbolic=False)
    assert not result.passed
    assert result.numeric_points_agreed < result.numeric_points_checked


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (sp.sin(x) ** 2 + sp.cos(x) ** 2, sp.Integer(1)),  # large Im: heavy cancellation
        (sp.exp(x) * sp.exp(z), sp.exp(x + z)),
        ((x**2 - 1) / (x - 1), x + 1),  # equal wherever both are defined
    ],
)
def test_true_identities_still_pass(a: sp.Expr, b: sp.Expr) -> None:
    result = verify_equal(a, b, symbolic=False)
    assert result.passed
    assert result.numeric_points_checked >= 12


def test_one_sided_pole_is_a_disagreement() -> None:
    n = sp.Symbol("n", integer=True)  # sampled from -6..6, so n = 1 is hit
    result = verify_equal(
        1 / (n - 1), 1 / (n - 1) + 0 * sp.tan(sp.pi * n / 2) + 1 / n, symbolic=False
    )
    assert not result.passed


def test_skipped_points_are_reported() -> None:
    checked, agreed, skipped = numeric_detail(1 / (x - x), 1 / (x - x))  # undefined on both sides
    assert (checked, agreed) == (0, 0)
    assert skipped > 0


def test_a_pole_on_one_side_only_counts_as_disagreement() -> None:
    checked, agreed, _ = numeric_detail(1 / (x - x), sp.Integer(1))
    assert checked > 0
    assert agreed == 0
