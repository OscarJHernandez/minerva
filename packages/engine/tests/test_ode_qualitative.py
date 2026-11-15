"""Equilibria, linearisation and local stability: published values, honest "undecided", and the cases
where nothing may be said."""

from __future__ import annotations

from typing import Any

import pytest
import sympy as sp
from minerva_engine import Engine, Limits, OdeAnalyzeInput, Status
from minerva_engine.ode import OdeAnalysis, analyze
from minerva_engine.ode import qualitative as qualitative_module

L = Limits(timeout_seconds=60)


def analysis(*equations: str, fmt: str = "latex", **kw: Any) -> OdeAnalysis:
    request = OdeAnalyzeInput(equations=list(equations), format=fmt, detail="qualitative", **kw)
    return analyze(request, L)


def numeric(values: list[str]) -> list[complex]:
    return sorted(
        (complex(v.replace(" ", "").replace("i", "j")) for v in values),
        key=lambda z: (z.real, z.imag),
    )


def test_the_damped_duffing_oscillator_has_one_real_equilibrium_and_it_is_a_sink() -> None:
    r = analysis("y'' + 0.2 y' + y + y^3 = 0", independent="t")
    q = r.qualitative
    assert q is not None and q.status is Status.OK
    # y^2 = -1 gives two more equilibria, but they are complex: they are not listed for real parameters
    assert len(q.equilibria) == 1 and q.complete and q.jacobian_checked
    (e,) = q.equilibria
    assert e.point_latex == ["y = 0", "y' = 0"]
    assert e.point_numeric == [0.0, 0.0]
    assert e.classification == "stable_spiral" and e.decided
    values = numeric(e.eigenvalues_numeric)
    assert values[0] == pytest.approx(complex(-0.1, -0.99498744), abs=1e-7)
    assert "asymptotically stable" in e.classification_text
    assert "Local and from the linearisation only" in q.local_note


def test_the_pendulum_centre_is_not_called_stable_and_the_list_is_not_called_complete() -> None:
    r = analysis(r"\theta'' + \sin\theta = 0", independent="t")
    q = r.qualitative
    assert q is not None
    classes = {e.point_latex[0]: e.classification for e in q.equilibria}
    assert classes == {"\\theta = 0": "centre", "\\theta = \\pi": "saddle"}
    centre = next(e for e in q.equilibria if e.classification == "centre")
    assert "does not decide" in centre.classification_text
    assert not q.complete  # sin has infinitely many zeros: these are the ones found
    assert any("there may be others" in n for n in q.notes)


def test_lorenz_equilibria_have_the_published_eigenvalues() -> None:
    r = analysis(
        "x' = s*(y-x)",
        "y' = x*(r-z)-y",
        "z' = x*y-b*z",
        fmt="text",
        independent="t",
        parameters={"s": 10.0, "r": 28.0, "b": 8 / 3},
    )
    q = r.qualitative
    assert q is not None and len(q.equilibria) == 3
    origin = next(e for e in q.equilibria if e.point_latex == ["x = 0", "y = 0", "z = 0"])
    assert sorted(v.real for v in numeric(origin.eigenvalues_numeric)) == pytest.approx(
        [-22.8277, -8 / 3, 11.8277], abs=1e-3
    )
    others = [e for e in q.equilibria if e is not origin]
    for e in others:
        values = numeric(e.eigenvalues_numeric)
        assert values[0].real == pytest.approx(-13.8546, abs=1e-3)
        assert sorted(v.real for v in values[1:]) == pytest.approx([0.09396, 0.09396], abs=1e-3)
        assert max(abs(v.imag) for v in values) == pytest.approx(10.1945, abs=1e-3)
        assert e.classification == "saddle"  # an eigenvalue of each sign


def test_the_sign_of_the_van_der_pol_damping_decides_the_origin() -> None:
    kinds = {}
    for mu in (-0.5, 0.0, 0.5):
        r = analysis(
            "y'' - mu*(1 - y^2)*y' + y = 0", fmt="text", independent="t", parameters={"mu": mu}
        )
        assert r.qualitative is not None
        kinds[mu] = [e.classification for e in r.qualitative.equilibria]
    assert kinds == {-0.5: ["stable_spiral"], 0.0: ["centre"], 0.5: ["unstable_spiral"]}


def test_symbolic_parameters_are_decided_only_where_the_assumptions_decide() -> None:
    equations = [r"\dot{x} = a x - b x y", r"\dot{y} = -c y + d x y"]
    positive = {k: ["positive"] for k in "abcd"}
    decided = analysis(*equations, assumptions=positive).qualitative
    assert decided is not None
    classes = {e.point_latex[0]: e.classification for e in decided.equilibria}
    assert classes == {"x = 0": "saddle", "x = \\frac{c}{d}": "centre"}
    assert decided.complete and all(e.decided for e in decided.equilibria)
    # the exact eigenvalues are shown, no numbers were chosen
    assert all(e.eigenvalues_numeric == [] for e in decided.equilibria)
    assert all(e.point_numeric == [] for e in decided.equilibria[1:])  # c/d, a/b stay symbolic
    undecided = analysis(*equations).qualitative
    assert undecided is not None
    origin = next(e for e in undecided.equilibria if e.point_latex[0] == "x = 0")
    assert origin.classification == "undecided" and not origin.decided
    assert "depend on parameters" in origin.classification_text
    assert any("Parameters have no values" in n for n in undecided.notes)


def test_an_eigenvalue_with_zero_real_part_leaves_the_stability_undecided() -> None:
    q = analysis("x' = -x^3", independent="t").qualitative
    assert q is not None
    (e,) = q.equilibria
    assert e.classification == "non_hyperbolic" and "does not decide" in e.classification_text


def test_a_continuum_of_equilibria_is_not_listed_as_points() -> None:
    q = analysis("x' = x*y", "y' = 0", fmt="text", independent="t").qualitative
    assert q is not None
    assert any("continuum" in n for n in q.notes)
    assert not q.complete


def test_the_direction_field_is_the_field_on_the_users_window() -> None:
    r = analysis("y'' + 0.2 y' + y + y^3 = 0", independent="t", window=[-2, 2, -1, 1])
    field = r.qualitative.direction_field if r.qualitative else None
    assert field is not None and len(field.x) == 225
    assert field.window == [-2, 2, -1, 1]
    assert min(field.x) == -2 and max(field.x) == 2 and min(field.y) == -1 and max(field.y) == 1
    for x, v, dx, dy in zip(field.x, field.y, field.dx, field.dy, strict=True):
        assert dx == pytest.approx(v, abs=1e-6)  # y' = v
        assert dy == pytest.approx(-0.2 * v - x - x**3, abs=1e-6)  # y'' by hand


def test_no_window_or_missing_values_means_no_field_and_no_guess() -> None:
    assert analysis("y'' + y = 0", independent="t").qualitative.direction_field is None  # type: ignore[union-attr]
    q = analysis("y'' + k*y = 0", independent="t", window=[-1, 1, -1, 1]).qualitative
    assert q is not None and q.direction_field is None
    assert any("value for every parameter" in n for n in q.notes)
    three = analysis(
        "x' = y", "y' = z", "z' = -x", independent="t", window=[-1, 1, -1, 1]
    ).qualitative
    assert three is not None and three.direction_field is None
    assert any("exactly two states" in n for n in three.notes)


@pytest.mark.parametrize(
    "window", [[1, 1, 0, 1], [2, 1, 0, 1], [0, 1, 1, 0], [0, 1, 0], [0, 1, 0, float("inf")]]
)
def test_a_bad_window_is_rejected(window: list[float]) -> None:
    with pytest.raises(ValueError, match="validation error"):
        OdeAnalyzeInput(equations=["y' = y"], independent="t", detail="qualitative", window=window)


@pytest.mark.parametrize(
    ("equations", "kwargs", "fragment"),
    [
        (["y'' + y = sin(t)"], {"fmt": "text"}, "autonomous"),
        (["y'''' + y = 0"], {}, "at most 3 states"),
        (["(y')^2 = y"], {}, "solved for the highest"),
    ],
)
def test_nothing_is_said_where_equilibria_are_not_defined_or_not_supported(
    equations: list[str], kwargs: dict[str, Any], fragment: str
) -> None:
    r = analysis(*equations, independent="t", **kwargs)
    assert r.status is Status.OK and r.qualitative is None
    assert fragment in " ".join(r.notes)
    methods = {m.id: m for m in r.methods}
    assert not methods["equilibrium_analysis"].applicable


def test_a_value_for_only_some_parameters_is_refused() -> None:
    q = analysis("x' = a*x + b", independent="t", parameters={"a": 1.0}).qualitative
    assert q is not None and q.status is Status.PARSE_ERROR and "missing: b" in q.message
    assert q.equilibria == []


# --- the checks reject wrong results ----------------------------------------------------------


def test_a_wrong_jacobian_shows_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    real = sp.Matrix.jacobian

    def wrong(self: sp.Matrix, *args: Any, **kwargs: Any) -> sp.Matrix:
        return real(self, *args, **kwargs) * 2  # a Jacobian that is not the field's

    monkeypatch.setattr(sp.Matrix, "jacobian", wrong)
    q = analysis("y'' + 0.2 y' + y + y^3 = 0", independent="t").qualitative
    assert q is not None and q.status is Status.NOT_VERIFIED
    assert q.equilibria == [] and "did not agree" in q.message


def test_a_claimed_equilibrium_that_is_not_one_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    real = qualitative_module.sp.solve

    def wrong(*args: Any, **kwargs: Any) -> Any:
        found = real(*args, **kwargs)
        states = args[1]
        return [*found, dict(zip(states, [sp.Integer(1), sp.Integer(1)], strict=True))]

    monkeypatch.setattr(qualitative_module.sp, "solve", wrong)
    q = analysis("y'' + 0.2 y' + y + y^3 = 0", independent="t").qualitative
    assert q is not None
    assert [e.point_latex for e in q.equilibria] == [["y = 0", "y' = 0"]]


def test_a_solve_that_runs_out_of_time_lists_no_equilibria(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*args: Any, **kwargs: Any) -> Any:
        raise qualitative_module.SolverTimeout

    monkeypatch.setattr(qualitative_module.sp, "solve", slow)
    q = analysis("y'' + y = 0", independent="t").qualitative
    assert q is not None and q.equilibria == [] and not q.complete
    assert any("time limit" in n for n in q.notes)


def test_the_diagnostics_hold_nothing_of_the_equation() -> None:
    secret = "Zqmark7919"
    r = analysis(
        f"{secret}'' + kappa_4242*{secret} = 0",
        fmt="text",
        independent="s",
        parameters={"kappa_4242": 2.0},
    )
    assert r.qualitative is not None and r.qualitative.equilibria
    dump = r.diagnostics.model_dump_json()
    for fragment in (secret, "4242", "kappa"):
        assert fragment not in dump


def test_the_engine_runs_the_qualitative_analysis_in_the_sandbox(engine: Engine) -> None:
    r = engine.ode_analyze(
        OdeAnalyzeInput(equations=["y'' + y = 0"], independent="t", detail="qualitative")
    )
    assert r.status is Status.OK and r.qualitative is not None
    assert r.qualitative.equilibria[0].classification == "centre"
    plain = engine.ode_analyze(OdeAnalyzeInput(equations=["y'' + y = 0"], independent="t"))
    assert plain.qualitative is None
