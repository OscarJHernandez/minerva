"""Electromagnetic field tensors on a metric the user gives: oracles typed by hand, refusals, mutations."""

import pytest
import sympy as sp
from minerva_engine import Limits, Status
from minerva_engine.geometry import geometry
from minerva_engine.geometry.models import GeometryInput, TensorField

L = Limits(timeout_seconds=120)
MINKOWSKI = [
    ["-1", "0", "0", "0"],
    ["0", "1", "0", "0"],
    ["0", "0", "1", "0"],
    ["0", "0", "0", "1"],
]
REAL = {"a": ["real"], "b": ["real"]}
# E = (-a t, 0, 0), B = (0, 0, b x): F_01 = -E_x = a t, F_12 = B_z = b x (signature -+++, c = 1)
F_COMPONENTS = [
    "0", "-a*t", "0", "0",
    "a*t", "0", "b*x", "0",
    "0", "-b*x", "0", "0",
    "0", "0", "0", "0",
]  # fmt: skip


def run(
    quantities: list[str],
    variance: list[str],
    components: list[str],
    metric: list[list[str]] | None = None,
    coords: list[str] | None = None,
    assumptions: dict[str, list[str]] | None = None,
):
    coords = coords or ["t", "x", "y", "z"]
    return geometry(
        GeometryInput(
            coordinates=coords,
            metric=metric or MINKOWSKI,
            format="text",
            assumptions=assumptions if assumptions is not None else REAL,
            quantities=quantities,  # type: ignore[arg-type]
            tensor=TensorField(variance=variance, components=components),  # type: ignore[arg-type]
        ),
        L,
    )


def quantity(result, kind: str):  # type: ignore[no-untyped-def]
    return next(q for q in result.quantities if q.kind == kind)


def test_the_invariant_is_twice_b_squared_minus_e_squared() -> None:
    r = run(["field_invariant"], ["down", "down"], F_COMPONENTS)
    q = quantity(r, "field_invariant")
    assert q.status is Status.OK and q.scalar_text is not None
    a, b, t, x = sp.symbols("a b t x", real=True)
    expected = 2 * (b**2 * x**2 - a**2 * t**2)  # 2 (B^2 - E^2), written by hand
    assert (
        sp.simplify(sp.sympify(q.scalar_text, locals={"a": a, "b": b, "t": t, "x": x}) - expected)
        == 0
    )
    assert any("2(B^2 - E^2)" in c for c in r.conventions)


def test_the_divergence_gives_the_sources_of_the_inhomogeneous_equations() -> None:
    q = quantity(
        run(["antisymmetric_divergence"], ["down", "down"], F_COMPONENTS),
        "antisymmetric_divergence",
    )
    assert q.status is Status.OK
    # J^x = d_t F^{tx} = a (with F^{tx} = a t) and J^y = d_x F^{xy} = b
    assert {c.label_latex: c.text for c in q.components} == {"J^{x}": "a", "J^{y}": "b"}
    assert q.verification is not None and q.verification.numeric_passed
    assert len([c for c in q.verification.checks if c.kind == "numeric"]) == 2


def test_d_of_a_potential_is_closed_and_the_dual_squares_to_minus_one_in_minkowski() -> None:
    f = quantity(
        run(["exterior_derivative"], ["down"], ["x*y", "t*z", "x**2+t", "y*z"]),
        "exterior_derivative",
    )
    assert f.status is Status.OK
    assert {c.label_latex: c.text for c in f.components}["F_{x z}"] == "-t"
    assert f.verification is not None
    assert any(c.name.startswith("dF = 0") and c.passed for c in f.verification.checks)
    dual = quantity(run(["hodge_dual"], ["down", "down"], F_COMPONENTS), "hodge_dual")
    assert dual.status is Status.OK and dual.verification is not None
    assert any("** = sign" in c.name and c.passed for c in dual.verification.checks)
    texts = {c.label_latex: c.text for c in dual.components}
    assert texts["{\\star}F_{t z}"] == "b*x" and texts["{\\star}F_{y z}"] == "a*t"


def test_a_coulomb_potential_in_spherical_coordinates_has_no_source_outside_the_charge() -> None:
    metric = [
        ["-1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "r**2", "0"],
        ["0", "0", "0", "r**2*sin(theta)**2"],
    ]
    coords = ["t", "r", "theta", "phi"]
    assumptions = {"r": ["positive"], "theta": ["positive"], "q": ["real"]}
    f = quantity(
        run(
            ["exterior_derivative"], ["down"], ["-q/r", "0", "0", "0"], metric, coords, assumptions
        ),
        "exterior_derivative",
    )
    assert f.status is Status.OK
    assert f.components[0].text == "-q/r**2"  # F_tr = d_t A_r - d_r A_t
    full = run(
        ["antisymmetric_divergence"],
        ["down", "down"],
        ["0", "q/r**2", "0", "0", "-q/r**2", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"],
        metric,
        coords,
        assumptions,
    )
    j = quantity(full, "antisymmetric_divergence")
    assert j.status is Status.OK and j.all_zero


def test_a_tensor_that_is_not_antisymmetric_is_refused() -> None:
    comps = ["0", "1", "0", "0", "1", *["0"] * 11]  # F_01 = F_10
    for kind in ("field_invariant", "antisymmetric_divergence", "hodge_dual"):
        q = quantity(run([kind], ["down", "down"], comps), kind)
        assert q.status is Status.PARSE_ERROR and "antisymmetric" in q.message, kind


def test_contravariant_components_and_the_wrong_rank_are_refused() -> None:
    assert (
        quantity(run(["field_invariant"], ["up", "up"], F_COMPONENTS), "field_invariant").status
        is Status.PARSE_ERROR
    )
    assert (
        quantity(
            run(["exterior_derivative"], ["down", "down", "down"], ["0"] * 64),
            "exterior_derivative",
        ).status
        is Status.PARSE_ERROR
    )


def test_the_hodge_dual_needs_four_dimensions_and_a_decided_sign() -> None:
    q = quantity(
        run(
            ["hodge_dual"],
            ["down", "down"],
            ["0", "1", "-1", "0"],
            [["1", "0"], ["0", "1"]],
            ["x", "y"],
            {},
        ),
        "hodge_dual",
    )
    assert q.status is Status.UNSUPPORTED and "four dimensions" in q.message
    undecided = [
        ["-1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "s"],
    ]
    q = quantity(run(["hodge_dual"], ["down", "down"], F_COMPONENTS, undecided), "hodge_dual")
    assert q.status is Status.UNSUPPORTED and "sign of det g" in q.message


def test_a_wrong_candidate_is_not_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    from minerva_engine.geometry import compute, field

    real = field.field_invariant

    def wrong(ws: compute.Workspace) -> sp.Expr:
        value = real(ws)
        ws._memo["field_invariant"] = value * (1 + sp.Rational(1, 10**6))  # a 1e-6 error
        return ws._memo["field_invariant"]  # type: ignore[no-any-return]

    monkeypatch.setattr(field, "field_invariant", wrong)
    r = run(["field_invariant"], ["down", "down"], F_COMPONENTS)
    q = quantity(r, "field_invariant")
    assert q.status is Status.NOT_VERIFIED and q.scalar_text is None


def test_the_divergence_formula_with_density_agrees_on_a_curved_metric() -> None:
    metric = [
        ["-(1 - 2/r)", "0", "0", "0"],
        ["0", "1/(1 - 2/r)", "0", "0"],
        ["0", "0", "r**2", "0"],
        ["0", "0", "0", "r**2*sin(theta)**2"],
    ]
    comps = ["0", "q/r**2", "0", "0", "-q/r**2", "0", "0", "0", *["0"] * 8]
    q = quantity(
        run(
            ["antisymmetric_divergence"],
            ["down", "down"],
            comps,
            metric,
            ["t", "r", "theta", "phi"],
            {"r": ["positive"], "theta": ["positive"], "q": ["real"]},
        ),
        "antisymmetric_divergence",
    )
    assert q.status is Status.OK and q.all_zero, q.message
