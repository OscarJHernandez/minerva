"""`check_identity`: a verdict on one stated equation, with the witness when it is not an identity.

The claim checker never calls a prose formula false unless the sides differ by a constant (it cannot tell a false
identity from an equation to solve). Here the user asked, so a disagreement is reported with the exact point; but a
relation, a matrix, a distribution or a sum is never called true or false, and an unconfirmed disagreement is shown
as nothing.
"""

import json

import pytest
from minerva_engine import Engine
from minerva_engine.identity_check import IdentityResult
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(budget_limits(max_concurrency=2))


def check(engine: Engine, latex: str, **assumptions: list[str]) -> IdentityResult:
    return engine.check_identity(latex, dict(assumptions))


# --- true identities -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "latex",
    [r"\sin^2 x + \cos^2 x = 1", r"(x+y)^2 = x^2 + 2xy + y^2", r"\cosh^2 x - \sinh^2 x = 1"],
)
def test_a_true_identity_is_proved_only_with_a_symbolic_zero(engine: Engine, latex: str) -> None:
    result = check(engine, latex)
    assert result.status is Status.OK and result.counterexample is None
    assert result.verdict in ("proved_equal", "numerically_equal")
    assert (result.verdict == "proved_equal") == bool(
        result.verification and result.verification.symbolic_zero
    )
    assert result.lhs_latex and result.rhs_latex


def test_a_cancelled_factor_is_equal_with_the_excluded_point_stated(engine: Engine) -> None:
    result = check(engine, r"\frac{x^2-1}{x-1} = x+1")
    assert result.status is Status.OK and result.verdict in ("proved_equal", "numerically_equal")
    assert result.condition and "x \\neq 1" in result.condition


# --- false identities, with a witness ------------------------------------------------------------
@pytest.mark.parametrize(
    "latex",
    [r"(x+y)^2 = x^2 + y^2", r"x + 1 = x + 2", r"\sin x = x", r"x^2 = 4x", r"|x-10| = 10-x"],
)
def test_a_false_identity_is_reported_with_a_point_where_the_sides_differ(
    engine: Engine, latex: str
) -> None:
    result = check(engine, latex)
    assert result.status is Status.OK and result.verdict == "not_identity"
    cx = result.counterexample
    assert cx is not None and cx.substitution and cx.lhs_value != cx.rhs_value
    assert "may still hold for some values" in result.message  # an equation to solve is not "false"


def test_the_witness_is_an_exact_point_that_really_separates_the_sides(engine: Engine) -> None:
    result = check(engine, r"(x+y)^2 = x^2 + y^2")
    cx = result.counterexample
    assert cx is not None
    x, y = int(cx.substitution["x"]), int(cx.substitution["y"])
    assert (x + y) ** 2 != x**2 + y**2


# --- the assumption ladder: the same equation, different declared assumptions -----------------------
def test_sqrt_of_a_square_depends_on_the_assumptions(engine: Engine) -> None:
    default = check(engine, r"\sqrt{x^2} = x")
    assert default.verdict == "conditional" and default.counterexample is not None
    assert default.condition and "positive" in default.condition
    assert check(engine, r"\sqrt{x^2} = x", x=["positive"]).verdict == "proved_equal"
    assert check(engine, r"\sqrt{x^2} = x", x=["real"]).verdict == "conditional"


def test_log_of_a_product_needs_positivity(engine: Engine) -> None:
    assert check(engine, r"\log(xy) = \log x + \log y").verdict == "conditional"
    both = check(engine, r"\log(xy) = \log x + \log y", x=["positive"], y=["positive"])
    assert both.verdict == "proved_equal"


def test_log_of_exp_fails_beyond_the_principal_strip(engine: Engine) -> None:
    result = check(engine, r"\log(\exp z) = z")
    assert result.verdict == "conditional" and result.counterexample is not None
    assert check(engine, r"\log(\exp z) = z", z=["real"]).verdict == "proved_equal"


# --- honesty: never false for what is not an identity, never a verdict from sampling what it cannot see ----
@pytest.mark.parametrize(
    "latex",
    [
        r"F = ma",  # a definition
        r"m\ddot{x} = -kx",  # an equation of motion
        r"L = T - V",
        r"\int_0^1 x\,dx = \frac{1}{2}",  # an integral keeps its own tools
        r"\operatorname{DiracDelta}(x) = 0",  # sampling cannot see a delta
    ],
)
def test_a_relation_or_unsampleable_statement_is_never_called_true_or_false(
    engine: Engine, latex: str
) -> None:
    result = check(engine, latex)
    assert result.status in (Status.UNSUPPORTED, Status.PARSE_ERROR, Status.NOT_VERIFIED)
    assert result.verdict is None and result.counterexample is None


def test_a_declared_matrix_is_not_sampled_as_numbers(engine: Engine) -> None:
    result = engine.check_identity(r"AB = BA", {"A": ["noncommutative"], "B": ["noncommutative"]})
    assert result.status is Status.UNSUPPORTED and result.verdict is None


def test_input_that_is_not_one_equation_is_a_parse_error(engine: Engine) -> None:
    for latex in ("x + 1", r"\frac{1", ""):
        assert check(engine, latex).status is Status.PARSE_ERROR, latex
    assert check(engine, "x" * 5000 + " = 1").status in (
        Status.RESOURCE_LIMIT,
        Status.PARSE_ERROR,
    )


def test_a_disagreement_that_cannot_be_confirmed_is_not_shown() -> None:
    """Cancellation, a removable singularity or a domain difference must not become a counterexample."""
    import minerva_engine.identity_check as module
    from minerva_engine.counterexample import SearchResult
    from minerva_engine.identity_check import check_identity
    from minerva_engine.models import Limits

    original = module.find_counterexample
    module.find_counterexample = lambda *a, **k: SearchResult(None, probes_tried=7)  # type: ignore[assignment]
    try:
        result = check_identity(r"x + 1 = x + 2", {}, Limits())
    finally:
        module.find_counterexample = original  # type: ignore[assignment]
    assert result.status is Status.NOT_VERIFIED and result.counterexample is None


def test_the_result_serialises_and_carries_no_hidden_fields(engine: Engine) -> None:
    result = check(engine, r"(x+y)^2 = x^2 + y^2")
    dumped = json.loads(result.model_dump_json())
    assert set(dumped["counterexample"]) == {
        "substitution",
        "lhs_value",
        "rhs_value",
        "residual",
        "probe",
    }
