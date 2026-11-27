"""The bivariate Normal (issue #229): exact results shown only if quadrature of the printed result agrees.

Wrong twins are injected into the exact algebra in process (monkeypatch); each must be caught by the
independent numerical leg and give `not_verified` with nothing shown.
"""

import pytest
import sympy as sp
from minerva_engine import Engine, MvnInput, Status
from minerva_engine.stats import mvn as M
from minerva_engine.testing import budget_limits

BASE = {"mean": ["1", "-2"], "covariance": [["4", "1"], ["1", "2"]], "variables": ["x", "y"]}
SYM = {
    "mean": ["m", "0"],
    "covariance": [["s", "r"], ["r", "s"]],
    "variables": ["x", "y"],
    "assumptions": {"m": ["real"], "s": ["positive"], "r": ["real"]},
}


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(budget_limits(max_concurrency=2))


def direct(**kw: object) -> M.MvnResult:
    """In process, so a monkeypatched algebra is the one under test."""
    return M.mvn(MvnInput(**kw), budget_limits())  # type: ignore[arg-type]


def nothing_shown(r: M.MvnResult) -> None:
    assert r.status is Status.NOT_VERIFIED and r.reason == "verification_failed", r.message
    assert not r.mean_text and not r.covariance_text and not r.density_text


# ---------------------------------------------------------------------------------- results


def test_density_of_a_numeric_model(engine) -> None:
    r = engine.multivariate_normal(MvnInput(**BASE, action="density"))
    assert r.status is Status.OK and r.evidence == "identity" and not r.premises
    assert r.mean_text == ["1", "-2"] and r.covariance_text == [["4", "1"], ["1", "2"]]
    assert {c.kind for c in r.checks} == {"normalisation", "moments"}
    assert all(c.passed for c in r.checks)
    assert "numerically checked" in r.message and "proved" not in r.message


def test_the_printed_density_is_the_textbook_one(engine) -> None:
    r = engine.multivariate_normal(MvnInput(**BASE, action="density"))
    x, y = sp.symbols("x y", real=True)
    # hand-written: Sigma^-1 = [[2, -1], [-1, 4]] / 7, det = 7
    q = (2 * (x - 1) ** 2 - 2 * (x - 1) * (y + 2) + 4 * (y + 2) ** 2) / 7
    expected = sp.exp(-q / 2) / (2 * sp.pi * sp.sqrt(7))
    shown = sp.sympify(r.density_text, locals={"x": x, "y": y})
    for point in ({x: 0, y: 0}, {x: sp.Rational(3, 2), y: -3}, {x: -2, y: 1}):
        assert abs(sp.N(shown.subs(point) - expected.subs(point), 30)) < 1e-25


def test_moments_give_the_correlation(engine) -> None:
    r = engine.multivariate_normal(MvnInput(**BASE, action="moments"))
    assert r.status is Status.OK
    assert sp.simplify(sp.sympify(r.correlation_text) - 1 / sp.sqrt(8)) == 0


def test_marginal_conditional_and_affine_against_hand_worked_answers(engine) -> None:
    marginal = engine.multivariate_normal(MvnInput(**BASE, action="marginal", keep="x"))
    assert (marginal.mean_text, marginal.covariance_text) == (["1"], [["4"]])
    cond = engine.multivariate_normal(MvnInput(**BASE, action="conditional", given={"x": "3"}))
    # mean -2 + (1/4)(3 - 1) = -3/2, variance 2 - 1/4 = 7/4
    assert (cond.mean_text, cond.covariance_text) == (["-3/2"], [["7/4"]])
    assert cond.variables == ["y"]
    image = engine.multivariate_normal(
        MvnInput(**BASE, action="affine", matrix=[["1", "0"], ["1", "1"]], offset=["0", "0"])
    )
    # [[4, 5], [5, 8]]: Var(X + Y) = 4 + 2 + 2
    assert image.mean_text == ["1", "-1"]
    assert image.covariance_text == [["4", "5"], ["5", "8"]]


def test_a_correlated_model_keeps_its_off_diagonal_covariance(engine) -> None:
    r = engine.multivariate_normal(
        MvnInput(mean=["0", "0"], covariance=[["1", "-3/5"], ["-3/5", "1"]], variables=["u", "v"], action="density")
    )  # fmt: skip
    assert r.status is Status.OK and r.covariance_text[0][1] == "-3/5"


# ------------------------------------------------------------------------- symbolic and premises


def test_undecided_positive_definiteness_is_a_premise(engine) -> None:
    r = engine.multivariate_normal(MvnInput(**SYM, action="density"))
    assert r.status is Status.OK and r.evidence == "conditional"
    assert any("det" in p for p in r.premises), r.premises  # s^2 - r^2 > 0 is not decided
    assert all(c.established is False for c in r.conditions)


def test_declared_assumptions_that_decide_it_leave_an_identity(engine) -> None:
    r = engine.multivariate_normal(
        MvnInput(mean=["m", "0"], covariance=[["s", "0"], ["0", "s"]], variables=["x", "y"],
                 assumptions={"m": ["real"], "s": ["positive"]}, action="density")
    )  # fmt: skip
    assert r.status is Status.OK and r.evidence == "identity" and not r.premises


def test_complex_symbols_are_a_premise_never_silently_real(engine) -> None:
    r = engine.multivariate_normal(
        MvnInput(mean=["a", "0"], covariance=[["1", "0"], ["0", "1"]], variables=["x", "y"], action="moments")
    )  # fmt: skip
    assert r.status is Status.OK and r.evidence == "conditional"
    assert any("real" in p for p in r.premises)


def test_symbolic_conditional_is_the_schur_complement(engine) -> None:
    r = engine.multivariate_normal(MvnInput(**SYM, action="conditional", given={"x": "1"}))
    assert r.status is Status.OK
    s, rr = sp.Symbol("s", positive=True), sp.Symbol("r", real=True)
    var = sp.sympify(r.covariance_text[0][0], locals={"s": s, "r": rr})
    assert sp.simplify(var - (s - rr**2 / s)) == 0


# ------------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("change", "status", "reason"),
    [
        ({"covariance": [["1", "2"], ["2", "1"]]}, Status.CONDITIONS_NOT_SATISFIED, "not_positive_definite"),
        ({"covariance": [["-1", "0"], ["0", "1"]]}, Status.CONDITIONS_NOT_SATISFIED, "not_positive_definite"),
        ({"covariance": [["1", "1"], ["1", "1"]]}, Status.UNSUPPORTED, "singular_covariance"),
        ({"covariance": [["0", "0"], ["0", "0"]]}, Status.UNSUPPORTED, "singular_covariance"),
        ({"covariance": [["1", "1/2"], ["0", "1"]]}, Status.PARSE_ERROR, "not_symmetric"),
        ({"mean": ["0", "0", "0"]}, Status.UNSUPPORTED, "dimension_unsupported"),
        ({"mean": ["0"]}, Status.UNSUPPORTED, "dimension_unsupported"),
        ({"covariance": [["1", "0", "0"], ["0", "1", "0"]]}, Status.PARSE_ERROR, "parse_error"),
        ({"variables": None}, Status.PARSE_ERROR, "variables_missing"),
        ({"variables": ["x", "x"]}, Status.PARSE_ERROR, "parse_error"),
    ],
)  # fmt: skip
def test_invalid_models_are_refused_before_anything_is_computed(
    engine, change, status, reason
) -> None:
    r = engine.multivariate_normal(MvnInput(**{**BASE, **change, "action": "density"}))
    assert (r.status, r.reason) == (status, reason), r.message
    assert not r.density_text and not r.mean_text


def test_a_singular_or_rank_deficient_map_is_refused(engine) -> None:
    r = engine.multivariate_normal(
        MvnInput(**BASE, action="affine", matrix=[["1", "2"], ["2", "4"]], offset=["0", "0"])
    )
    assert (r.status, r.reason) == (Status.UNSUPPORTED, "map_singular")
    r = engine.multivariate_normal(
        MvnInput(**BASE, action="affine", matrix=[["0", "0"]], offset=["1"])
    )
    assert (r.status, r.reason) == (Status.UNSUPPORTED, "map_singular")


def test_a_marginal_or_conditional_names_a_component(engine) -> None:
    for kw in ({"action": "marginal", "keep": "z"}, {"action": "conditional", "given": {"z": "1"}}):
        r = engine.multivariate_normal(MvnInput(**{**BASE, **kw}))
        assert r.reason == "index_invalid", r.message
    r = engine.multivariate_normal(
        MvnInput(**BASE, action="conditional", given={"x": "1", "y": "2"})
    )
    assert r.reason == "index_invalid"


def test_a_nearly_singular_model_is_withheld_when_quadrature_cannot_confirm_it(engine) -> None:
    r = engine.multivariate_normal(
        MvnInput(**{**BASE, "covariance": [["1", "999/1000"], ["999/1000", "1"]]}, action="density")
    )
    assert r.status in (Status.OK, Status.NOT_VERIFIED)
    if r.status is Status.NOT_VERIFIED:
        assert not r.density_text


# ------------------------------------------------------------------------------- wrong twins


def test_a_density_without_the_off_diagonal_term_is_rejected(monkeypatch) -> None:
    def wrong(m, x, y):
        dx, dy = x - m.mu[0], y - m.mu[1]
        q = (m.S[1, 1] * dx**2 + m.S[0, 0] * dy**2) / m.det
        return sp.exp(-q / 2) / (2 * sp.pi * sp.sqrt(m.det))

    monkeypatch.setattr(M, "_density_2d", wrong)
    nothing_shown(direct(**BASE, action="density"))


def test_a_density_without_its_determinant_factor_is_rejected(monkeypatch) -> None:
    right = M._density_2d
    monkeypatch.setattr(M, "_density_2d", lambda m, x, y: right(m, x, y) * sp.sqrt(m.det))
    nothing_shown(direct(**BASE, action="density"))


def test_a_density_off_by_one_in_a_million_is_rejected(monkeypatch) -> None:
    right = M._density_2d
    monkeypatch.setattr(
        M, "_density_2d", lambda m, x, y: right(m, x, y) * sp.Rational(1000001, 1000000)
    )
    nothing_shown(direct(**BASE, action="density"))


def test_a_conditional_variance_without_the_schur_complement_is_rejected(monkeypatch) -> None:
    calls = []
    right = M._one_dimensional

    def wrong(m, result, t, mean, var):
        calls.append(var)
        return right(
            m, result, t, mean, m.S[0, 0]
        )  # Sigma_11 instead of Sigma_11 - Sigma_12^2/Sigma_22

    monkeypatch.setattr(M, "_one_dimensional", wrong)
    nothing_shown(direct(**BASE, action="conditional", given={"y": "1"}))
    assert calls


def test_a_conditional_mean_off_by_one_in_a_million_is_rejected(monkeypatch) -> None:
    right = M._one_dimensional
    monkeypatch.setattr(
        M,
        "_one_dimensional",
        lambda m, result, t, mean, var: right(
            m, result, t, mean * sp.Rational(1000001, 1000000), var
        ),
    )
    nothing_shown(direct(**BASE, action="conditional", given={"y": "1"}))


def test_a_marginal_with_the_other_variance_is_rejected(monkeypatch) -> None:
    right = M._one_dimensional
    monkeypatch.setattr(
        M, "_one_dimensional", lambda m, result, t, mean, var: right(m, result, t, mean, m.S[1, 1])
    )
    nothing_shown(direct(**BASE, action="marginal", keep="x"))


def test_an_affine_covariance_without_the_transpose_is_rejected(monkeypatch) -> None:
    # A Sigma A instead of A Sigma A^T: differs for a non-symmetric A
    real_matrix = sp.Matrix
    monkeypatch.setattr(M.sp.Matrix, "T", property(lambda self: real_matrix(self)))
    nothing_shown(
        direct(**BASE, action="affine", matrix=[["1", "2"], ["0", "1"]], offset=["0", "0"])
    )


def test_the_twins_pass_when_nothing_is_patched() -> None:
    assert direct(**BASE, action="density").status is Status.OK
    assert direct(**BASE, action="conditional", given={"y": "1"}).status is Status.OK
    assert direct(**BASE, action="marginal", keep="x").status is Status.OK
    ok = direct(**BASE, action="affine", matrix=[["1", "2"], ["0", "1"]], offset=["0", "0"])
    assert ok.status is Status.OK


def test_statistical_model_points_a_multivariate_family_to_the_operation(engine) -> None:
    from minerva_engine import StatModelInput

    r = engine.statistical_model(StatModelInput(family="multivariate_normal", action="moments"))
    assert r.status is Status.UNSUPPORTED and "multivariate_normal" in r.message
