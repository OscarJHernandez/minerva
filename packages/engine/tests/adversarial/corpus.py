"""Cases that probe `verify_equal` from both sides.

`truth` is what mathematics says about the identity *as written, under the stated assumptions*:
  identity      holds for every admissible value of the symbols;
  false         fails somewhere admissible (so a verifier that accepts it is wrong: a false accept);
and `expect` is what the verifier must answer, `accept` or `reject`. A true identity that is rejected is a
false reject. Numerical agreement is evidence, not proof: `accept` means "no disagreement found", never proved.

`defect` ties a row to docs/verification-defects.md while the verifier still gets it wrong; such a row is a
strict xfail in test_adversarial.py, so fixing the defect forces the row to be promoted (defect set to None).
Never delete or loosen a row to make a suite pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sympy as sp

Truth = Literal["identity", "false"]
Expect = Literal["accept", "reject"]


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    a: sp.Expr
    b: sp.Expr
    truth: Truth
    expect: Expect
    note: str
    defect: str | None = None


def _sym(name: str, **flags: bool) -> sp.Symbol:
    return sp.Symbol(name, **flags)


x, y, z, w = (_sym(n) for n in "xyzw")  # complex
xr, yr = _sym("x", real=True), _sym("y", real=True)
xp, yp = _sym("x", positive=True), _sym("y", positive=True)
n = _sym("n", integer=True)
half = sp.Rational(1, 2)


def _id(
    domain: str, id_: str, a: sp.Expr, b: sp.Expr, note: str, defect: str | None = None
) -> Case:
    return Case(f"{domain}.{id_}", domain, a, b, "identity", "accept", note, defect)


def _no(
    domain: str, id_: str, a: sp.Expr, b: sp.Expr, note: str, defect: str | None = None
) -> Case:
    return Case(f"{domain}.{id_}", domain, a, b, "false", "reject", note, defect)


CASES: list[Case] = [
    # --- algebra: ordinary identities and their common misstatements -------------------------------------
    _id(
        "algebra",
        "binomial",
        (x + y) ** 3,
        x**3 + 3 * x**2 * y + 3 * x * y**2 + y**3,
        "expanded cube",
    ),
    _id(
        "algebra", "partial_fractions", 1 / (x * (x + 1)), 1 / x - 1 / (x + 1), "partial fractions"
    ),
    _no("algebra", "freshmans_dream", (x + y) ** 3, x**3 + y**3, "dropped cross terms"),
    _no("algebra", "sum_of_reciprocals", 1 / (x + y), 1 / x + 1 / y, "reciprocal of a sum"),
    _no(
        "algebra",
        "tiny_constant",
        xr**2 + sp.Rational(1, 10**10),
        xr**2,
        "differs by 1e-10 everywhere",
    ),
    _no(
        "algebra",
        "tiny_constant_small",
        xr + sp.Rational(1, 10**11),
        xr,
        "differs by 1e-11 everywhere",
    ),
    _no("algebra", "scaled_tiny", 10**10 * xr + 5, 10**10 * xr, "relative error 5e-11"),
    # --- singularities: removable, one-sided, and shifted poles ------------------------------------------
    _id("singularity", "removable", (x**2 - 3025) / (x - 55), x + 55, "removable at x = 55"),
    _id(
        "singularity",
        "sinc_form",
        sp.sin(x) / x,
        sp.sin(x) * (1 / x),
        "same function, written twice",
    ),
    _no("singularity", "shifted_pole", 1 / (x - 1), 1 / (x - 2), "pole moved"),
    _no("singularity", "tan_vs_sin_only", sp.tan(x), sp.sin(x), "cos dropped"),
    _id("singularity", "tan_as_ratio", sp.tan(x), sp.sin(x) / sp.cos(x), "equal wherever defined"),
    # --- branch cuts: log and sqrt ------------------------------------------------------------------------
    _no(
        "branch",
        "log_exp_complex",
        sp.log(sp.exp(z)),
        z,
        "off by 2*pi*i beyond the principal strip",
    ),
    _id("branch", "log_exp_real", sp.log(sp.exp(xr)), xr, "real argument"),
    _no(
        "branch",
        "log_exp_scaled",
        sp.log(sp.exp(z / 8)),
        z / 8,
        "strip reached only at |Im z| > 25",
    ),
    _no("branch", "asin_sin_complex", sp.asin(sp.sin(z)), z, "principal value only"),
    _no("branch", "asin_sin_scaled", sp.asin(sp.sin(z / 40)), z / 40, "needs |Re z| > 62"),
    _no("branch", "sqrt_square_complex", sp.sqrt(z**2), z, "sqrt(z^2) = +-z"),
    _id("branch", "sqrt_square_positive", sp.sqrt(xp**2), xp, "positive argument"),
    _no(
        "branch",
        "log_product_complex",
        sp.log(z * w),
        sp.log(z) + sp.log(w),
        "arguments add past pi",
    ),
    _id("branch", "log_product_positive", sp.log(xp * yp), sp.log(xp) + sp.log(yp), "positives"),
    _no(
        "branch", "sqrt_product_complex", sp.sqrt(z * w), sp.sqrt(z) * sp.sqrt(w), "branch of sqrt"
    ),
    _no(
        "branch",
        "log_reciprocal_cut",
        sp.log(1 / z),
        -sp.log(z),
        "fails only ON the negative real axis (a measure-zero set a random complex draw never hits)",
    ),
    _id("branch", "log_reciprocal_positive", sp.log(1 / xp), -sp.log(xp), "positive argument"),
    _no(
        "branch",
        "sqrt_conj_cut",
        sp.sqrt(sp.conjugate(z)),
        sp.conjugate(sp.sqrt(z)),
        "fails only on the negative real axis",
    ),
    # --- piecewise functions and domain boundaries ---------------------------------------------------------
    _no("piecewise", "abs_beyond_range", sp.Abs(xr - 60), 60 - xr, "holds only for x <= 60"),
    _id("piecewise", "abs_matches_sqrt", sp.Abs(xr), sp.sqrt(xr**2), "real x"),
    _no("piecewise", "max_collapsed", sp.Max(xr, 49), sp.Integer(49), "wrong for x > 49"),
    _id("piecewise", "max_symmetry", sp.Max(xr, yr), sp.Max(yr, xr), "commutative"),
    _no(
        "piecewise",
        "step_at_48",
        sp.Piecewise((1, xr < 48), (2, True)),
        sp.Integer(1),
        "second branch",
    ),
    _no(
        "piecewise",
        "step_near_zero",
        sp.Piecewise((1, xp > sp.Rational(1, 1000)), (2, True)),
        sp.Integer(1),
        "x <= 1/1000",
    ),
    _no("piecewise", "floor_zero", sp.floor(xp / 100), sp.Integer(0), "wrong for x >= 100"),
    _id("piecewise", "floor_of_integer", sp.floor(n), n, "integers are fixed by floor"),
    _id(
        "piecewise",
        "equal_branches",
        sp.Piecewise((x, xr < 5), (x, True)),
        x,
        "both branches agree",
    ),
    _no(
        "piecewise",
        "abs_with_spectators",
        sp.Abs(xr - 5) + yr + _sym("z", real=True),
        5 - xr + yr + _sym("z", real=True),
        "three symbols",
    ),
    _no(
        "piecewise",
        "thin_region_wide",
        sp.Piecewise((1, sp.Abs(xr - 2) < sp.Rational(1, 1000)), (2, True)),
        sp.Integer(2),
        "differs only on |x - 2| < 1/1000, found between the two roots",
    ),
    _no(
        "piecewise",
        "thin_region_narrow",
        sp.Piecewise((1, sp.Abs(xr - sp.Rational(1, 3)) < sp.Rational(1, 10**6)), (2, True)),
        sp.Integer(2),
        "differs only on |x - 1/3| < 1e-6",
    ),
    _no(
        "precision",
        "bump_at_simple_point",
        sp.exp(-(10**6) * (xr - 3) ** 2),
        sp.Integer(0),
        "peak 1 at x = 3",
    ),
    _no(
        "precision",
        "bump_at_arbitrary_point",
        sp.exp(-(10**6) * (xr - sp.Rational(1, 7)) ** 2),
        sp.Integer(0),
        "peak 1 at x = 1/7: no boundary, no simple point; the extremum probe tries the exponent's centre",
    ),
    _no(
        "precision",
        "bump_at_22_over_7",
        sp.exp(-(10**4) * (xr - sp.Rational(22, 7)) ** 2),
        sp.Integer(0),
        "peak at 22/7, scale 1e4",
    ),
    _no(
        "precision",
        "bump_at_negative_355_over_113",
        sp.exp(-(10**8) * (xr + sp.Rational(355, 113)) ** 2),
        sp.Integer(0),
        "peak at -355/113, scale 1e8",
    ),
    _no(
        "precision",
        "bump_at_symbolic_centre",
        sp.exp(-(10**6) * (xr - yr) ** 2),
        sp.Integer(0),
        "centre is the other symbol: exact for any drawn value",
    ),
    _no(
        "precision",
        "bump_two_variables",
        sp.exp(-(10**6) * ((xr - sp.Rational(1, 7)) ** 2 + (yr - sp.Rational(2, 9)) ** 2)),
        sp.Integer(0),
        "peak in two variables: the other symbol is drawn, not set to its own centre, so the probe misses (residual limit of V-014)",
        defect="V-014",
    ),
    _no(
        "precision",
        "bump_nonreal_symbol",
        sp.exp(-(10**6) * (x - sp.Rational(1, 7)) ** 2),
        sp.Integer(0),
        "complex x: centre 1/7 is admissible",
    ),
    _no(
        "precision",
        "bump_quartic_exponent",
        sp.exp(-(10**12) * (xr - sp.Rational(1, 7)) ** 4),
        sp.Integer(0),
        "degree 4 exponent: not a quadratic, so the extremum probe skips it (residual limit of V-014)",
        defect="V-014",
    ),
    _id(
        "precision",
        "bump_expanded_square",
        sp.exp(-(10**6) * (xr - sp.Rational(1, 7)) ** 2),
        sp.exp(-(10**6) * xr**2 + 2 * 10**6 * sp.Rational(1, 7) * xr - 10**6 * sp.Rational(1, 49)),
        "true twin: the same peak written expanded",
    ),
    _id(
        "precision",
        "bump_symbolic_twin",
        sp.exp(-(10**6) * (xr - yr) ** 2),
        sp.exp(-(10**6) * xr**2 + 2 * 10**6 * xr * yr - 10**6 * yr**2),
        "true twin at a symbolic centre",
    ),
    # --- integers beyond the small pool ------------------------------------------------------------------------
    _no("integer", "kronecker_at_7", sp.KroneckerDelta(n, 7), sp.Integer(0), "n = 7"),
    _no(
        "integer",
        "vanishing_product",
        sp.prod([n - k for k in range(-6, 7)]),
        sp.Integer(0),
        "n = 7",
    ),
    _no("integer", "minus_one_power", (-1) ** n, sp.Integer(1), "odd n"),
    _id("integer", "even_square", (-1) ** (2 * n), sp.Integer(1), "even exponent"),
    # --- precision and cancellation -----------------------------------------------------------------------------
    _id(
        "precision",
        "pythagoras",
        sp.sin(z) ** 2 + sp.cos(z) ** 2,
        sp.Integer(1),
        "cancels hundreds of digits at large Im",
    ),
    _id(
        "precision",
        "cosh_sinh",
        sp.cosh(z) ** 2 - sp.sinh(z) ** 2,
        sp.Integer(1),
        "heavy cancellation",
    ),
    _no("precision", "tanh_saturation", sp.tanh(12 * xr), sp.Integer(1), "1 - 2e-21 is not 1"),
    _no("precision", "sub_noise_difference", xr**12 + 1, xr**12, "constant 1 against a large term"),
    # --- conditional identities: false as written, true under the stated assumption ------------------------------
    _no("conditional", "sqrt_square_unassumed", sp.sqrt(x**2), x, "needs x >= 0 (here complex)"),
    _id(
        "conditional",
        "sqrt_square_nonneg",
        sp.sqrt(_sym("x", nonnegative=True) ** 2),
        _sym("x", nonnegative=True),
        "assumed",
    ),
    _no("conditional", "power_combine_unassumed", (z**x) ** y, z ** (x * y), "needs z > 0, y real"),
    _id(
        "conditional",
        "power_combine_assumed",
        (xp**yr) ** yr,
        xp ** (yr * yr),
        "positive base, real exponent",
    ),
    _no("conditional", "cube_root_cube", (z**3) ** sp.Rational(1, 3), z, "principal cube root"),
]

# Identities between (non)commuting matrices live in test_adversarial.py: they need explicit matrices.
