"""Killing vectors and the conserved charges of geodesics (issue #151).

The search only proposes; a vector is shown only after the Lie derivative of the metric, by central differences of
the metric and the candidate, vanishes; a charge only after `Flow.verify` on the geodesic equations. Wrong twins are
injected at the verification and must be rejected.
"""

from __future__ import annotations

import itertools
import time

import pytest
import sympy as sp
from minerva_engine.geometry import benchmark as B
from minerva_engine.geometry import killing
from minerva_engine.geometry.compute import Workspace
from minerva_engine.geometry.geometry import geometry
from minerva_engine.geometry.models import GeometryInput
from minerva_engine.geometry.setup import build_geometry
from minerva_engine.geometry.verify import Verifier
from minerva_engine.models import Status
from minerva_engine.testing import budget_limits

SCHWARZSCHILD = {
    "coordinates": ["t", "r", r"\theta", r"\phi"],
    "metric": [
        ["-(1-2M/r)", "0", "0", "0"],
        ["0", "(1-2M/r)^{-1}", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", r"r^2 \sin^2\theta"],
    ],
    "assumptions": {"M": ["positive"], "r": ["positive"]},
}
FLAT_SPHERICAL = {
    "coordinates": ["r", r"\theta", r"\phi"],
    "metric": [["1", "0", "0"], ["0", "r^2", "0"], ["0", "0", r"r^2 \sin^2\theta"]],
    "assumptions": {"r": ["positive"]},
}
BOTH = ["killing_vectors", "geodesic_constants"]


def _run(problem: dict, quantities: list[str] = BOTH):  # type: ignore[no-untyped-def, type-arg]
    return geometry(GeometryInput(**problem, quantities=quantities), budget_limits())


def _workspace(problem: dict) -> tuple[Workspace, Verifier]:  # type: ignore[type-arg]
    limits = budget_limits()
    geo = build_geometry(GeometryInput(**problem, quantities=["christoffel"]), limits, 4)
    ws = Workspace(geo, limits, time.monotonic() + limits.timeout_seconds)
    return ws, Verifier(ws)


def _sym(ws: Workspace, name: str) -> sp.Symbol:
    return {s.name: s for s in [*ws.geo.coords, *ws.geo.params, *ws.geo.velocities]}[name]


# -- results -------------------------------------------------------------------------------------
def test_schwarzschild_has_four_vectors_their_charges_and_the_total_angular_momentum() -> None:
    result = _run(SCHWARZSCHILD)
    vectors, charges = result.quantities
    assert vectors.status is Status.OK and charges.status is Status.OK
    assert vectors.killing is not None and vectors.killing.count == 4
    assert vectors.killing.bound == 10 and not vectors.killing.complete
    assert vectors.killing.largest_system <= killing.MAX_UNKNOWNS
    notes = " ".join(vectors.notes)
    assert "no further Killing vector within it" in notes
    assert "none exists" not in notes and "complete" not in notes.replace("incomplete", "")
    assert "Killing tensors (hidden quadratic symmetries) are not searched." in vectors.notes
    assert "treated as generic" in notes
    # every family above the cap is listed as not searched, never silently dropped
    assert vectors.killing.not_searched and all(
        "above the cap of 300 unknowns" in s for s in vectors.killing.not_searched
    )
    origins = [c.origin for c in charges.components]
    assert origins.count("cyclic coordinate") == 2  # d/dt and d/dphi
    assert origins.count("Killing vector") == 2
    assert origins.count("quadratic combination of Killing charges") == 1
    assert origins.count("normalisation") == 1
    labels = [c.label_latex for c in charges.components]
    assert labels == [f"Q_{{{k}}}" for k in range(1, 7)]  # never E, J_z or J^2
    quadratic = next(
        c for c in charges.components if c.origin == "quadratic combination of Killing charges"
    )
    assert r"r^{4}" in quadratic.latex


def test_schwarzschild_charges_match_the_textbook_ones_up_to_basis() -> None:
    ws, _ = _workspace(SCHWARZSCHILD)
    case = next(c for c in B.CASES if c.id == "killing_schwarzschild")
    expected = case.expect(ws)
    assert B._killing_matches(
        ws, "killing_vectors", killing.found(ws).raised, expected["killing_vectors"]
    )
    assert B._killing_matches(
        ws, "geodesic_constants", killing.charges(ws), expected["geodesic_constants"]
    )


def test_the_comparison_up_to_basis_rejects_a_wrong_basis() -> None:
    ws, _ = _workspace(SCHWARZSCHILD)
    th, ph = _sym(ws, "theta"), _sym(ws, "phi")
    found = killing.found(ws).raised
    wrong = [list(v) for v in found]
    wrong[-1] = [0, 0, sp.cos(ph), sp.sin(ph) * sp.cos(th) / sp.sin(th)]  # sign of the cot term
    assert not B._same_span(ws, found, wrong)
    assert not B._same_span(ws, found, found[:-1])
    assert B._same_span(ws, found, [found[1], found[0], found[3], found[2]])


def test_flat_space_in_spherical_coordinates_is_complete_at_six() -> None:
    result = _run(FLAT_SPHERICAL, ["killing_vectors"])
    q = result.quantities[0]
    assert q.status is Status.OK and q.killing is not None
    assert q.killing.count == 6 == q.killing.bound and q.killing.complete
    assert any("the list is complete" in n for n in q.notes)
    assert not q.killing.not_searched


def test_a_metric_without_symmetry_is_worded_within_the_family() -> None:
    problem = {
        "coordinates": ["x", "y"],
        "metric": [["1 + x^2*y", "0"], ["0", "1 + x*y^2 + y^3"]],
        "format": "text",
    }
    result = _run(problem)
    vectors, charges = result.quantities
    assert vectors.status is Status.OK and vectors.all_zero and not vectors.components
    assert any(
        "No Killing vector within the searched family; no further Killing vector within it" in n
        for n in vectors.notes
    )
    assert all("none exists" not in n for n in vectors.notes)
    assert [c.origin for c in charges.components] == ["normalisation"]


# -- wrong twins at the verification ----------------------------------------------------------------
def _rotation(ws: Workspace, sign: int = 1, factor: sp.Expr = sp.S.One) -> list[sp.Expr]:
    th, ph = _sym(ws, "theta"), _sym(ws, "phi")
    return [0, 0, sp.sin(ph), sign * factor * sp.cos(ph) * sp.cos(th) / sp.sin(th)]


def test_the_lie_derivative_leg_accepts_a_true_rotation() -> None:
    ws, verifier = _workspace(SCHWARZSCHILD)
    assert killing.lie_derivative_leg(verifier, _rotation(ws), "true").passed is True


@pytest.mark.parametrize(
    "twin",
    ["cot_sign", "factor_1e-6", "dropped_component"],
)
def test_the_lie_derivative_leg_rejects_wrong_twins(twin: str) -> None:
    ws, verifier = _workspace(SCHWARZSCHILD)
    candidate = {
        "cot_sign": _rotation(ws, sign=-1),
        "factor_1e-6": _rotation(ws, factor=1 + sp.Rational(1, 10**6)),
        "dropped_component": [*_rotation(ws)[:3], sp.S.Zero],
    }[twin]
    check = killing.lie_derivative_leg(verifier, candidate, twin)
    assert check.passed is False
    assert check.points_checked >= 4 and check.points_agreed < check.points_checked


def test_a_wrong_vector_from_the_search_withholds_both_quantities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = killing.search

    def tampered(ws: Workspace) -> killing.KillingSearch:
        out = real(ws)
        out.raised[-1] = _rotation(ws, sign=-1)
        return out

    monkeypatch.setattr(killing, "search", tampered)
    result = _run(SCHWARZSCHILD)
    for q in result.quantities:
        assert q.status is Status.NOT_VERIFIED, q.kind
        assert not q.components


def test_flow_verify_rejects_a_charge_with_sin_for_sin_squared() -> None:
    ws, _ = _workspace(SCHWARZSCHILD)
    r, th, phd = _sym(ws, "r"), _sym(ws, "theta"), _sym(ws, "phi_dot")
    flow = killing.geodesic_flow(ws.geo)
    true = killing.check_charge(
        flow, killing.Charge("Q", r**2 * sp.sin(th) ** 2 * phd, "Killing vector")
    )
    wrong = killing.check_charge(
        flow, killing.Charge("Q", r**2 * sp.sin(th) * phd, "Killing vector")
    )
    assert true.passed is True
    assert wrong.passed is False


def test_a_charge_that_fails_is_withheld_and_named(monkeypatch: pytest.MonkeyPatch) -> None:
    real = killing.candidate_charges

    def tampered(ws: Workspace, found: killing.KillingSearch) -> killing.Charges:
        out = real(ws, found)
        r, th, phd = _sym(ws, "r"), _sym(ws, "theta"), _sym(ws, "phi_dot")
        out.items[1].expr = r**2 * sp.sin(th) * phd
        return out

    monkeypatch.setattr(killing, "candidate_charges", tampered)
    charges = _run(SCHWARZSCHILD, ["geodesic_constants"]).quantities[0]
    assert charges.status is Status.OK
    assert "Q_{2}" not in [c.label_latex for c in charges.components]
    assert any("Q_{2}" in n and "is not shown" in n for n in charges.notes)


# -- refusals ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("reason", "problem"),
    [
        (
            "declared_function",
            {
                "coordinates": ["t", "r"],
                "metric": [["-f(r)", "0"], ["0", "1/f(r)"]],
                "functions": [{"name": "f", "args": ["r"]}],
            },
        ),
        (
            "non_rational",
            {"coordinates": ["x", "y"], "metric": [["1", "0"], ["0", "sqrt(x)"]], "format": "text"},
        ),
        (
            "mixed_angle",
            {
                "coordinates": ["x", "y"],
                "metric": [["1", "0"], ["0", "x^2*sin(x)^2 + 1"]],
                "format": "text",
            },
        ),
        (
            "irrational_constant",
            {
                "coordinates": ["x", "y"],
                "metric": [["1", "0"], ["0", "pi*x^2 + 1"]],
                "format": "text",
            },
        ),
    ],
)
def test_each_unsupported_metric_is_refused_with_its_reason(reason: str, problem: dict) -> None:  # type: ignore[type-arg]
    result = _run(problem)
    for q in result.quantities:
        assert q.status is Status.UNSUPPORTED, (reason, q.kind)
        assert q.message == killing.REFUSALS[reason]
        assert not q.components


def test_the_family_ladder_is_nested_and_capped() -> None:
    ws, _ = _workspace(SCHWARZSCHILD)
    variables = killing.Variables(ws.geo)
    roles = killing.classify(ws.geo, variables)
    assert roles.role == ["absent", "power", "angle", "absent"]
    assert (roles.low[1], roles.high[1]) == (-1, 2)  # 1/r in g_tt, r^2 in g_thth
    families = killing.ladder(ws.geo, roles)
    by_mode: dict[killing.Mode, list[killing.Family]] = {}
    for fam in families:
        by_mode.setdefault(fam.mode, []).append(fam)
    for fams in by_mode.values():
        for low, high in itertools.pairwise(fams):
            assert high.degree >= low.degree and high.unknowns >= low.unknowns
            for i, (lo, hi) in low.powers.items():
                assert high.powers[i][0] <= lo and high.powers[i][1] >= hi


def test_the_geodesic_lagrangian_helper_is_g_v_v_over_two() -> None:
    from minerva_engine.geometry.compute import geodesic_lagrangian

    ws, _ = _workspace(FLAT_SPHERICAL)
    r, th = _sym(ws, "r"), _sym(ws, "theta")
    rd, thd, phd = (_sym(ws, f"{n}_dot") for n in ("r", "theta", "phi"))
    expected = (rd**2 + r**2 * thd**2 + r**2 * sp.sin(th) ** 2 * phd**2) / 2
    assert sp.simplify(geodesic_lagrangian(ws.geo) - expected) == 0
