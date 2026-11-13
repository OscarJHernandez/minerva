r"""Differential geometry: analytical reference cases, honesty cases, wrong candidates, coordinate changes.

Reference values come from standard results (flat space, the round sphere of radius a, Minkowski space,
the Schwarzschild solution, FLRW with a general scale factor), not from the code under test.
"""

import json
import time

import pytest
import sympy as sp
from minerva_engine import Limits, Status
from minerva_engine.geometry import GeometryInput, geometry
from minerva_engine.geometry.compute import Workspace
from minerva_engine.geometry.setup import build_geometry
from minerva_engine.geometry.tensors import Tensor
from minerva_engine.geometry.verify import Verifier

LIMITS = Limits(timeout_seconds=60)
SPHERE = [["a^2", "0"], ["0", r"a^2 \sin^2\theta"]]
SCHWARZSCHILD = [
    ["-(1-2M/r)", "0", "0", "0"],
    ["0", "(1-2M/r)^{-1}", "0", "0"],
    ["0", "0", "r^2", "0"],
    ["0", "0", "0", r"r^2 \sin^2\theta"],
]
ALL_SCALARS = ["ricci_scalar", "kretschmann", "ricci_squared"]


def run(coordinates, metric, quantities, **extra):  # type: ignore[no-untyped-def]
    data = GeometryInput(coordinates=coordinates, metric=metric, quantities=quantities, **extra)
    return geometry(data, LIMITS)


def workspace(coordinates, metric, **extra):  # type: ignore[no-untyped-def]
    data = GeometryInput(
        coordinates=coordinates, metric=metric, quantities=["christoffel"], **extra
    )
    geo = build_geometry(data, LIMITS, 6)
    return Workspace(geo, LIMITS, time.monotonic() + 600)


def quantity(result, kind):  # type: ignore[no-untyped-def]
    (q,) = [q for q in result.quantities if q.kind == kind]
    return q


def comps(q):  # type: ignore[no-untyped-def]
    return {tuple(c.indices): c.latex for c in q.components}


def sym(ws, text: str) -> sp.Expr:  # type: ignore[no-untyped-def]
    """Parse an expected value in the symbols of a workspace."""
    names = {s.name: s for s in [*ws.geo.coords, *ws.geo.params]}
    return sp.sympify(text, locals=names)  # test-only parsing of trusted literals


# --- reference cases -------------------------------------------------------------------------
def test_euclidean_space_in_cartesian_coordinates_is_flat() -> None:
    eye = [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]]
    r = run(["x", "y", "z"], eye, ["christoffel", "riemann", "ricci", "einstein", *ALL_SCALARS])
    assert r.status is Status.OK
    for kind in ("christoffel", "riemann", "ricci", "einstein"):
        assert quantity(r, kind).all_zero and quantity(r, kind).components == []
    for kind in ALL_SCALARS:
        assert quantity(r, kind).scalar_text == "0"
    assert r.signature == "(+, +, +)"
    assert r.restrictions_latex == [] and r.restrictions_complete


def test_euclidean_plane_in_polar_coordinates() -> None:
    r = run(
        ["r", "theta"],
        [["1", "0"], ["0", "r^2"]],
        ["inverse_metric", "determinant", "christoffel", "riemann", "ricci_scalar", "geodesics"],
        format="text",
        assumptions={"r": ["positive"]},
    )
    assert r.status is Status.OK
    gamma = comps(quantity(r, "christoffel"))
    assert gamma == {(0, 1, 1): "- r", (1, 0, 1): r"\frac{1}{r}"}
    assert quantity(r, "determinant").scalar_text == "r**2"
    assert quantity(r, "riemann").all_zero and quantity(r, "ricci_scalar").scalar_text == "0"
    geo = {c.indices[0]: c.text for c in quantity(r, "geodesics").components}
    assert geo == {0: "r_ddot = r*theta_dot**2", 1: "theta_ddot = -2*r_dot*theta_dot/r"}


def test_two_sphere_curvature_and_christoffel_symbols() -> None:
    r = run(
        ["theta", "phi"],
        SPHERE,
        ["christoffel", "riemann", "ricci", "einstein", *ALL_SCALARS],
        assumptions={"a": ["positive"]},
    )
    assert r.status is Status.OK
    assert comps(quantity(r, "christoffel")) == {
        (0, 1, 1): r"- \sin{\left(\theta \right)} \cos{\left(\theta \right)}",
        (1, 0, 1): r"\cot{\left(\theta \right)}",
    }
    assert comps(quantity(r, "ricci")) == {
        (0, 0): "1",
        (1, 1): r"\sin^{2}{\left(\theta \right)}",
    }
    assert quantity(r, "ricci_scalar").scalar_text == "2/a**2"
    assert quantity(r, "kretschmann").scalar_text == "4/a**4"
    assert quantity(r, "ricci_squared").scalar_text == "2/a**4"
    assert quantity(r, "einstein").all_zero  # the Einstein tensor vanishes identically in 2D
    assert r.restrictions_latex == [r"\sin{\left(\theta \right)} \neq 0"]
    assert r.signature is None  # a^2 > 0 is known but sin^2(theta) >= 0 is not strictly positive


def test_minkowski_space_is_flat_with_lorentzian_signature() -> None:
    eta = [
        ["-1", "0", "0", "0"],
        ["0", "1", "0", "0"],
        ["0", "0", "1", "0"],
        ["0", "0", "0", "1"],
    ]
    r = run(["t", "x", "y", "z"], eta, ["christoffel", "riemann", "einstein", "geodesics"])
    assert r.status is Status.OK
    assert r.signature == "(-, +, +, +)"
    assert quantity(r, "riemann").all_zero and quantity(r, "einstein").all_zero
    assert quantity(r, "geodesics").all_zero


def test_schwarzschild_is_ricci_flat_with_kretschmann_48_m_squared_over_r_to_the_sixth() -> None:
    r = run(
        ["t", "r", "theta", "phi"],
        SCHWARZSCHILD,
        ["christoffel", "riemann", "ricci", "ricci_scalar", "einstein", "kretschmann"],
        assumptions={"M": ["positive"], "r": ["positive"]},
    )
    assert r.status is Status.OK
    assert quantity(r, "ricci").all_zero and quantity(r, "einstein").all_zero
    assert quantity(r, "ricci_scalar").scalar_text == "0"
    assert quantity(r, "kretschmann").scalar_text == "48*M**2/r**6"
    gamma = comps(quantity(r, "christoffel"))
    assert gamma[(1, 2, 2)] == "2 M - r"  # Gamma^r_{theta theta} = -(r - 2M)
    assert gamma[(2, 1, 2)] == r"\frac{1}{r}"
    assert gamma[(0, 0, 1)] == r"\frac{M}{r \left(- 2 M + r\right)}"
    assert r.signature is None  # the sign of 1 - 2M/r is not decided by the assumptions
    assert "- 2 M + r \\neq 0" in r.restrictions_latex  # the horizon is a coordinate singularity


def test_schwarzschild_geodesics_match_the_known_equations() -> None:
    r = run(
        ["t", "r", "theta", "phi"],
        SCHWARZSCHILD,
        ["geodesics"],
        assumptions={"M": ["positive"], "r": ["positive"]},
    )
    ws = workspace(
        ["t", "r", "theta", "phi"],
        SCHWARZSCHILD,
        assumptions={"M": ["positive"], "r": ["positive"]},
    )
    acc = ws.accelerations()
    rr, th = sym(ws, "r"), sym(ws, "theta")
    M = sym(ws, "M")
    vt, vr, vth, vph = ws.geo.velocities
    # t'' = -2M/(r(r-2M)) t' r' ; theta'' = sin theta cos theta phi'^2 - 2 r' theta'/r
    assert sp.simplify(acc[0] + 2 * M * vt * vr / (rr * (rr - 2 * M))) == 0
    assert sp.simplify(acc[2] - (sp.sin(th) * sp.cos(th) * vph**2 - 2 * vr * vth / rr)) == 0
    assert quantity(r, "geodesics").status is Status.OK


def test_static_spherical_metric_with_an_unspecified_function() -> None:
    metric = [
        ["-f(r)", "0", "0", "0"],
        ["0", "1/f(r)", "0", "0"],
        ["0", "0", "r^2", "0"],
        ["0", "0", "0", r"r^2 \sin^2\theta"],
    ]
    kw = {"assumptions": {"r": ["positive"]}, "functions": [{"name": "f", "args": ["r"]}]}
    r = run(["t", "r", "theta", "phi"], metric, ["christoffel", "ricci_scalar", "einstein"], **kw)
    assert r.status is Status.OK
    ws = workspace(["t", "r", "theta", "phi"], metric, **kw)
    f = sp.Function("f", real=True)(ws.geo.coords[1])
    rr = ws.geo.coords[1]
    expected = -sp.diff(f, rr, 2) - 4 * sp.diff(f, rr) / rr - 2 * (f - 1) / rr**2
    assert sp.simplify(ws.ricci_scalar() - expected) == 0
    # Schwarzschild-like f = 1 - 2M/r makes the Ricci scalar vanish
    M = sp.Symbol("M")
    assert sp.simplify(expected.subs(f, 1 - 2 * M / rr).doit()) == 0


def test_flrw_with_a_general_scale_factor() -> None:
    metric = [
        ["-1", "0", "0", "0"],
        ["0", "a(t)^2", "0", "0"],
        ["0", "0", "a(t)^2 r^2", "0"],
        ["0", "0", "0", r"a(t)^2 r^2 \sin^2\theta"],
    ]
    kw = {"assumptions": {"r": ["positive"]}, "functions": [{"name": "a", "args": ["t"]}]}
    r = run(["t", "r", "theta", "phi"], metric, ["ricci_scalar", "einstein"], **kw)
    assert r.status is Status.OK
    ws = workspace(["t", "r", "theta", "phi"], metric, **kw)
    t = ws.geo.coords[0]
    a = sp.Function("a", real=True)(t)
    # flat FLRW: R = 6 (a a'' + a'^2) / a^2, G_tt = 3 (a'/a)^2, G_rr = -(2 a a'' + a'^2)
    assert sp.simplify(ws.ricci_scalar() - 6 * (a * a.diff(t, 2) + a.diff(t) ** 2) / a**2) == 0
    ein = ws.einstein()
    assert sp.simplify(ein.get((0, 0)) - 3 * (a.diff(t) / a) ** 2) == 0
    assert sp.simplify(ein.get((1, 1)) + 2 * a * a.diff(t, 2) + a.diff(t) ** 2) == 0


def test_a_correct_einstein_tensor_is_not_rejected_for_numerical_noise() -> None:
    # Found by the property tests: the contracted Bianchi residual (terms of size ~1e2 cancelling to ~0)
    # used to be compared with zero on an absolute scale of 1 and failed at one sample point.
    grid = [
        ["x - y^2/2 + z + 3", "0", "0"],
        ["0", "x - z + 3", "0"],
        ["0", "0", "-2*x + y^2/2 + 3"],
    ]
    r = run(["x", "y", "z"], grid, ["einstein"], format="text")
    q = quantity(r, "einstein")
    assert q.status is Status.OK and q.verification is not None
    bianchi = next(c for c in q.verification.checks if "Bianchi" in c.name)
    assert bianchi.passed and bianchi.points_agreed == bianchi.points_checked >= 4


def test_conventions_are_stated_in_every_result() -> None:
    r = run(["x"], [["1"]], ["christoffel"])
    text = " ".join(r.conventions)
    for needle in ("Christoffel", "Riemann", "Ricci", "Einstein", "Levi-Civita", "2/a^2"):
        assert needle in text
    assert any("complex" in line for line in r.assumptions_used)


# --- covariant derivatives and index operations -----------------------------------------------
POLAR = (["r", "theta"], [["1", "0"], ["0", "r^2"]])
POLAR_KW = {"format": "text", "assumptions": {"r": ["positive"]}}


def tensor_run(kind, variance, components, **extra):  # type: ignore[no-untyped-def]
    return run(
        *POLAR,
        [kind],
        tensor={"variance": variance, "components": components},
        **POLAR_KW,
        **extra,
    )


def test_covariant_derivative_of_a_vector_uses_the_connection() -> None:
    # V = d/dr: (nabla_theta V)^theta = Gamma^theta_{theta r} V^r = 1/r
    r = tensor_run("covariant_derivative", ["up"], ["1", "0"])
    q = quantity(r, "covariant_derivative")
    assert q.status is Status.OK and q.variance == ["down", "up"]
    assert {tuple(c.indices): c.text for c in q.components} == {(1, 1): "1/r"}


def test_a_constant_cartesian_vector_field_has_zero_covariant_derivative() -> None:
    # d/dx in polar coordinates: V^r = cos(theta), V^theta = -sin(theta)/r
    r = tensor_run("covariant_derivative", ["up"], ["cos(theta)", "-sin(theta)/r"])
    assert quantity(r, "covariant_derivative").status is Status.OK
    assert quantity(r, "covariant_derivative").all_zero


def test_covariant_derivative_of_a_covector_has_the_opposite_sign() -> None:
    # omega = dr: (nabla_theta omega)_theta = -Gamma^r_{theta theta} omega_r = r
    r = tensor_run("covariant_derivative", ["down"], ["1", "0"])
    q = quantity(r, "covariant_derivative")
    assert {tuple(c.indices): c.text for c in q.components} == {(1, 1): "r"}


def test_the_metric_is_covariantly_constant() -> None:
    r = tensor_run("covariant_derivative", ["down", "down"], ["1", "0", "0", "r^2"])
    assert quantity(r, "covariant_derivative").status is Status.OK
    assert quantity(r, "covariant_derivative").all_zero


def test_lowering_and_raising_an_index() -> None:
    r = tensor_run("lower_index", ["up"], ["1", "1"])
    assert {tuple(c.indices): c.text for c in quantity(r, "lower_index").components} == {
        (0,): "1",
        (1,): "r**2",
    }
    r = tensor_run("raise_index", ["down"], ["1", "r^2"])
    assert {tuple(c.indices): c.text for c in quantity(r, "raise_index").components} == {
        (0,): "1",
        (1,): "1",
    }


def test_rank_two_index_operation_acts_on_the_chosen_index() -> None:
    r = tensor_run("lower_index", ["up", "down"], ["0", "1", "1", "0"], tensor_index=0)
    q = quantity(r, "lower_index")
    assert q.variance == ["down", "down"]
    assert {tuple(c.indices): c.text for c in q.components} == {(0, 1): "1", (1, 0): "r**2"}


# --- honesty: invalid and unsupported input ------------------------------------------------------
@pytest.mark.parametrize(
    ("metric", "needle"),
    [
        ([["1", "0"]], "by 2 grid"),  # not square
        ([["1", "x"], ["0", "1"]], "symmetric"),  # asymmetric
    ],
)
def test_invalid_metrics_are_parse_errors(metric: list[list[str]], needle: str) -> None:
    r = run(["x", "y"], metric, ["christoffel"], format="text")
    assert r.status is Status.PARSE_ERROR and needle in r.message
    assert r.quantities == []


def test_a_degenerate_metric_is_unsupported_not_inverted() -> None:
    r = run(["x", "y"], [["1", "0"], ["0", "0"]], ["christoffel"], format="text")
    assert r.status is Status.UNSUPPORTED and "degenerate" in r.message
    r = run(["x", "y"], [["1", "x"], ["x", "x^2"]], ["christoffel"], format="text")
    assert r.status is Status.UNSUPPORTED  # det = x^2 - x^2 = 0 only after simplification


def test_metric_singular_only_somewhere_is_reported_as_a_restriction_not_refused() -> None:
    r = run(["x", "y"], [["x", "0"], ["0", "1"]], ["christoffel"], format="text")
    assert r.status is Status.OK
    assert r.restrictions_latex == ["x \\neq 0"]


def test_indefinite_and_unknown_signs_are_never_assumed() -> None:
    r = run(["x", "y"], [["x", "0"], ["0", "1"]], ["determinant"], format="text")
    assert r.signature is None and "not determined" in r.signature_note
    r = run(["x", "y"], [["1", "x"], ["x", "3"]], ["determinant"], format="text")
    assert r.signature is None and "not diagonal" in r.signature_note


def test_an_undeclared_function_is_refused_not_read_as_a_product() -> None:
    r = run(["r"], [["f(r)"]], ["christoffel"])
    assert r.status is Status.PARSE_ERROR and "declare" in r.message


def test_a_function_may_only_depend_on_coordinates() -> None:
    r = run(["r"], [["f(r)"]], ["christoffel"], functions=[{"name": "f", "args": ["s"]}])
    assert r.status is Status.PARSE_ERROR


def test_a_delta_or_step_in_a_metric_is_refused_as_unsupported_input() -> None:
    # distributions belong to the `distribution` operation; pointwise checks cannot handle them
    r = run(["x"], [[r"1 + \operatorname{Heaviside}(x)"]], ["christoffel"])
    assert r.status is Status.UNSUPPORTED and r.quantities == []


def test_dimension_above_the_cap_is_unsupported() -> None:
    names = ["x1", "x2", "x3", "x4", "x5"]
    eye = [["1" if i == j else "0" for j in range(5)] for i in range(5)]
    r = run(names, eye, ["christoffel"])
    assert r.status is Status.UNSUPPORTED and "Dimension 5" in r.message
    assert (
        geometry(
            GeometryInput(coordinates=names, metric=eye, quantities=["determinant"]),
            LIMITS,
            max_dimension=5,
        ).status
        is Status.OK
    )


def test_tensor_quantities_need_a_tensor_and_valid_variance() -> None:
    r = run(*POLAR, ["lower_index"], **POLAR_KW)
    assert quantity(r, "lower_index").status is Status.PARSE_ERROR
    r = tensor_run("lower_index", ["down"], ["1", "0"])
    q = quantity(r, "lower_index")
    assert q.status is Status.PARSE_ERROR and "already down" in q.message
    r = tensor_run("raise_index", ["up"], ["1", "0"])
    assert quantity(r, "raise_index").status is Status.PARSE_ERROR
    r = tensor_run("lower_index", ["up"], ["1", "0"], tensor_index=1)
    assert quantity(r, "lower_index").status is Status.PARSE_ERROR  # no second index


def test_a_tensor_with_the_wrong_number_of_components_is_rejected() -> None:
    r = tensor_run("covariant_derivative", ["up", "up"], ["1", "0", "0"])
    assert r.status is Status.PARSE_ERROR and "4 components" in r.message


def test_names_reserved_for_velocities_are_rejected() -> None:
    r = run(["x"], [["x_dot"]], ["christoffel"], format="text")
    assert r.status is Status.PARSE_ERROR and "reserved" in r.message


def test_an_exhausted_time_budget_is_a_resource_limit_that_names_the_stage() -> None:
    data = GeometryInput(
        coordinates=["t", "r", "theta", "phi"], metric=SCHWARZSCHILD, quantities=["riemann"]
    )
    r = geometry(data, Limits(timeout_seconds=0.0001))
    assert r.status is Status.RESOURCE_LIMIT and "stage" in r.message


def test_the_engine_runs_geometry_in_the_sandbox(engine) -> None:  # type: ignore[no-untyped-def]
    data = GeometryInput(
        coordinates=["r", "theta"],
        metric=[["1", "0"], ["0", "r^2"]],
        format="text",
        assumptions={"r": ["positive"]},
        quantities=["christoffel", "ricci_scalar"],
    )
    r = engine.geometry(data)
    assert r.status is Status.OK and r.duration_ms > 0
    assert quantity(r, "ricci_scalar").scalar_text == "0"


# --- deliberately wrong candidates must be rejected -----------------------------------------------
def verified(ws: Workspace, kind: str):  # type: ignore[no-untyped-def]
    return getattr(Verifier(ws), kind)()


def rejected_by_disagreement(v) -> bool:  # type: ignore[no-untyped-def]
    """Not shown because valid sample points disagreed, not merely because points were missing."""
    return not v.numeric_passed and any(
        c.kind == "numeric" and c.points_checked >= 4 and c.points_agreed < c.points_checked
        for c in v.checks
    )


def sphere_workspace() -> Workspace:
    return workspace(["theta", "phi"], SPHERE, assumptions={"a": ["positive"]})


def poison(ws: Workspace, key: str, value: object) -> None:
    ws._memo[key] = value


def test_a_christoffel_symbol_with_a_flipped_sign_is_rejected() -> None:
    ws = sphere_workspace()
    gamma = ws.christoffel()
    wrong = Tensor(2, gamma.variance, dict(gamma.data))
    wrong.data[(1, 0, 1)] = -wrong.data[(1, 0, 1)]
    wrong.data[(1, 1, 0)] = wrong.data[(1, 0, 1)]
    poison(ws, "christoffel", wrong)
    v = verified(ws, "christoffel")
    assert rejected_by_disagreement(v)


def test_a_one_part_in_a_million_error_is_rejected() -> None:
    ws = sphere_workspace()
    gamma = ws.christoffel()
    wrong = Tensor(2, gamma.variance, dict(gamma.data))
    for key in ((0, 1, 1), (1, 0, 1), (1, 1, 0)):
        wrong.data[key] = wrong.data[key] * (1 + sp.Rational(1, 1_000_000))
    poison(ws, "christoffel", wrong)
    assert rejected_by_disagreement(verified(ws, "christoffel"))


def test_the_opposite_riemann_sign_convention_is_rejected() -> None:
    ws = sphere_workspace()
    riem = ws.riemann()
    poison(ws, "riemann", riem.map(lambda e: -e))
    v = verified(ws, "riemann")
    assert rejected_by_disagreement(v)
    curvature = next(c for c in v.checks if c.name == "agrees with the numerical curvature")
    assert curvature.passed is False


def test_a_ricci_tensor_off_by_a_factor_two_is_rejected() -> None:
    ws = sphere_workspace()
    ws.ricci()
    poison(ws, "ricci", ws.ricci().map(lambda e: 2 * e))
    assert rejected_by_disagreement(verified(ws, "ricci"))


def test_an_einstein_tensor_with_the_wrong_trace_term_is_rejected() -> None:
    # a curved metric (exponentially expanding space), so the trace term matters
    ws = workspace(
        ["t", "x", "y", "z"],
        [
            ["-1", "0", "0", "0"],
            ["0", "e^{2t}", "0", "0"],
            ["0", "0", "e^{2t}", "0"],
            ["0", "0", "0", "e^{2t}"],
        ],
    )
    right = ws.einstein()
    wrong = Tensor(4, right.variance)
    for idx in right.indices():
        wrong.set(idx, right.get(idx) + ws.geo.g[idx] * ws.ricci_scalar() / 2)  # +R g/2 extra
    assert wrong.data != right.data
    poison(ws, "einstein", wrong)
    assert rejected_by_disagreement(verified(ws, "einstein"))


def test_a_geodesic_equation_with_a_missing_factor_of_two_is_rejected() -> None:
    ws = workspace(
        ["r", "theta"], [["1", "0"], ["0", "r^2"]], format="text", assumptions={"r": ["positive"]}
    )
    acc = ws.accelerations()
    poison(ws, "geodesics", [acc[0], acc[1] / 2])
    assert rejected_by_disagreement(verified(ws, "geodesics"))


def test_a_wrong_inverse_and_a_wrong_determinant_are_rejected() -> None:
    ws = sphere_workspace()
    inv = ws.inverse()
    poison(ws, "inverse_metric", sp.Matrix(2, 2, lambda i, j: inv[i, j] * 2))
    assert rejected_by_disagreement(verified(ws, "inverse_metric"))
    ws = sphere_workspace()
    poison(ws, "determinant", ws.determinant() + 1)
    assert rejected_by_disagreement(verified(ws, "determinant"))


def test_correct_results_report_every_numeric_leg_passed() -> None:
    ws = sphere_workspace()
    for kind in ("christoffel", "riemann", "ricci", "ricci_scalar", "einstein", "geodesics"):
        v = verified(ws, kind)
        assert v.numeric_passed, kind
        assert all(
            c.points_agreed == c.points_checked >= 4 for c in v.checks if c.kind == "numeric"
        )


# --- coordinate changes ----------------------------------------------------------------------------
def test_christoffel_symbols_obey_the_inhomogeneous_transformation_law() -> None:
    cart = workspace(["x", "y"], [["1", "0"], ["0", "1"]], format="text")
    polar = workspace(*POLAR, **POLAR_KW)
    r, th = polar.geo.coords
    x, y = r * sp.cos(th), r * sp.sin(th)  # Cartesian coordinates as functions of the polar ones
    jac = sp.Matrix([[sp.diff(c, q) for q in (r, th)] for c in (x, y)])  # dx^a/dx'^m
    inv = jac.inv()  # dx'^r/dx^a
    q = (r, th)
    gamma = polar.christoffel()
    assert all(v == 0 for v in cart.christoffel().data.values())
    for rho in range(2):
        for m in range(2):
            for n in range(2):
                # Gamma'^rho_{mn} = dx'^rho/dx^a d^2 x^a/dx'^m dx'^n (the Cartesian Gamma vanishes)
                law = sum(inv[rho, a] * sp.diff((x, y)[a], q[m], q[n]) for a in range(2))
                assert sp.simplify(gamma.get((rho, m, n)) - law) == 0


def test_the_ricci_tensor_and_scalar_transform_as_a_tensor_and_a_scalar() -> None:
    # the sphere in (theta, phi) and in (u = cos theta, phi)
    old = sphere_workspace()
    new = workspace(
        ["u", "phi"], [["a^2/(1-u^2)", "0"], ["0", "a^2 (1-u^2)"]], assumptions={"a": ["positive"]}
    )
    u = new.geo.coords[0]
    jac = sp.diff(sp.acos(u), u)  # dtheta/du (phi is unchanged)
    point = {u: sp.Rational(3, 10), new.geo.params[0]: 2}
    theta0 = sp.acos(sp.Rational(3, 10))
    old_point = {old.geo.coords[0]: theta0, old.geo.params[0]: 2}
    # R'_{uu} = (dtheta/du)^2 R_{theta theta}, R'_{phi phi} = R_{phi phi}
    lhs_uu = new.ricci().get((0, 0)).subs(point)
    rhs_uu = (jac**2 * old.ricci().get((0, 0))).subs(u, sp.Rational(3, 10)).subs(old_point)
    assert sp.N(lhs_uu - rhs_uu, 30) == 0
    lhs_pp = new.ricci().get((1, 1)).subs(point)
    assert sp.N(lhs_pp - old.ricci().get((1, 1)).subs(old_point), 30) == 0
    # the Ricci scalar is invariant
    assert sp.simplify(new.ricci_scalar() - old.ricci_scalar()) == 0


# --- diagnostics ---------------------------------------------------------------------------------------
def test_diagnostics_name_each_stage_and_carry_no_expressions() -> None:
    r = run(["theta", "phi"], SPHERE, ["christoffel", "riemann"], assumptions={"a": ["positive"]})
    stages = [d.stage for d in r.diagnostics]
    for needed in (
        "setup",
        "determinant",
        "inverse_metric",
        "christoffel",
        "christoffel:verify",
        "riemann",
        "riemann:verify",
    ):
        assert needed in stages
    assert all(d.status == "ok" for d in r.diagnostics)
    assert any(d.ops for d in r.diagnostics if d.stage == "riemann")
    blob = json.dumps([d.model_dump() for d in r.diagnostics])
    assert "theta" not in blob and "sin" not in blob and "a^2" not in blob
    assert r.peak_memory_mb is not None and r.peak_memory_mb > 0


def test_a_failing_stage_is_visible_in_the_diagnostics() -> None:
    r = run(["x", "y"], [["1", "0"], ["0", "0"]], ["christoffel"], format="text")
    assert any(d.stage == "determinant" and d.status == "ok" for d in r.diagnostics)
    assert any(d.stage == "inverse_metric" and d.status == "failed" for d in r.diagnostics)


def test_results_are_deterministic() -> None:
    first = run(["theta", "phi"], SPHERE, ["christoffel", "ricci"], assumptions={"a": ["positive"]})
    second = run(
        ["theta", "phi"], SPHERE, ["christoffel", "ricci"], assumptions={"a": ["positive"]}
    )
    assert [q.model_dump() for q in first.quantities] == [q.model_dump() for q in second.quantities]
