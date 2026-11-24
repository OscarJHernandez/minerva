"""Adaptive discovery of conserved quantities (`discovery.py`, `flow.py`, `independence.py`).

The expected invariants below are written out here, by hand, and compared with what the engine found;
they are never given to the engine.
"""

import pytest
import sympy as sp
from minerva_engine.discovery import DiscoveryReport, Pins, discover_invariants
from minerva_engine.flow import lagrangian_flow
from minerva_engine.latexparse import parse_latex
from minerva_engine.models import Limits, Status
from minerva_engine.testing import budget_limits

LIMITS = Limits(timeout_seconds=120)
KEPLER = r"\frac{1}{2}m(\dot{r}^2+r^2\dot{\theta}^2)+\frac{k}{r}"
KEPLER_H = r"\frac{p_r^2}{2m}+\frac{p_\theta^2}{2 m r^2}-\frac{k}{r}"
KEPLER_CARTESIAN = r"\frac{1}{2}m(\dot{x}^2+\dot{y}^2)+\frac{k}{\sqrt{x^2+y^2}}"
OSCILLATOR = r"\frac{1}{2}m(\dot{x}^2+\dot{y}^2)-\frac{1}{2}k(x^2+y^2)"
OSCILLATOR_3D = r"\frac{1}{2}(\dot{x}^2+\dot{y}^2+\dot{z}^2)-\frac{1}{2}(x^2+y^2+z^2)"
HENON_HEILES = r"\frac{1}{2}(\dot{x}^2+\dot{y}^2)-\frac{1}{2}(x^2+y^2)-x^2 y+\frac{1}{3}y^3"


def run(
    function: str,
    coordinates: list[str],
    *,
    system: str = "lagrangian",
    momenta: list[str] | None = None,
    angles: list[str] | None = None,
    budget: str = "standard",
    pins: Pins | None = None,
) -> DiscoveryReport:
    return discover_invariants(
        system, function, coordinates, momenta, angles or [], budget, pins, "t", {}, LIMITS
    )


def parsed(report: DiscoveryReport, kind: str = "new") -> list[sp.Expr]:
    out = []
    for c in report.constants:
        if c.kind == kind:
            out.append(parse_latex(c.expression_latex.split("=", 1)[1], {}, LIMITS))
    return out


def symbols(expressions: list[sp.Expr]) -> dict[str, sp.Symbol]:
    return {s.name: s for e in expressions for s in e.free_symbols}


PARAMETERS = {"m", "k", "F", "g", "l"}


def in_span(basis: list[sp.Expr], target: sp.Expr) -> bool:
    """Is `target` a linear combination of `basis`? The coefficients may depend on the parameters."""
    cos_s, sin_s = sp.Symbol("COS"), sp.Symbol("SIN")

    def prep(e: sp.Expr) -> sp.Expr:
        e = e.subs({sp.cos(s): cos_s for s in e.free_symbols}).subs(
            {sp.sin(s): sin_s for s in e.free_symbols}
        )
        return sp.expand(sp.rem(sp.expand(e), sin_s**2 + cos_s**2 - 1, sin_s))

    basis = [prep(b) for b in basis]
    target = prep(target)
    unknown = sp.symbols(f"a0:{len(basis)}")
    residual = sp.numer(
        sp.together(target - sum(a * b for a, b in zip(unknown, basis, strict=True)))
    )
    residual = sp.expand(residual)
    variables = [s for s in residual.free_symbols if s not in unknown and s.name not in PARAMETERS]
    eqs = sp.Poly(residual, *variables).coeffs() if variables else [residual]
    return bool(sp.linsolve(eqs, *unknown))


# --- Kepler: the critical regression ----------------------------------------------------------------
def test_kepler_polar_discovers_both_runge_lenz_components_without_hints() -> None:
    report = run(KEPLER, ["r", "theta"])  # no angles, no power bound, no ansatz
    assert report.status is Status.OK
    new = parsed(report)
    assert len(new) == 2
    assert all(c.verification and c.verification.symbolic_zero for c in report.new_constants)
    s = symbols([*new, *parsed(report, "known")])
    r, th, rd, thd, m, k = (s[n] for n in ("r", "theta", "r_dot", "theta_dot", "m", "k"))
    ax = m * r**3 * thd**2 * sp.cos(th) + m * r**2 * rd * thd * sp.sin(th) - k * sp.cos(th)
    ay = m * r**3 * thd**2 * sp.sin(th) - m * r**2 * rd * thd * sp.cos(th) - k * sp.sin(th)
    assert in_span(new, ax) and in_span(new, ay)
    # The Noether constants are recognised as known, not counted as discoveries.
    known = {c.source for c in report.constants if c.kind == "known"}
    assert known == {"cyclic", "energy"}


def test_kepler_independence_is_three_with_the_relation_to_energy_and_angular_momentum() -> None:
    report = run(KEPLER, ["r", "theta"])
    ind = report.independence
    assert ind is not None
    assert (ind.rank_lower, ind.rank_upper, ind.rank_proved) == (3, 3, True)
    assert ind.max_possible == 3
    assert len(ind.relations) == 1  # |A|^2 in terms of h and p_theta, found by an exact nullspace
    assert [c.independence for c in report.constants] == [
        "independent",
        "independent",
        "independent",
        "dependent",
    ]
    assert "C_{1}" in ind.relations[0].latex and "C_{2}" in ind.relations[0].latex


def test_kepler_a_restricted_ansatz_finds_neither_and_expansion_reaches_the_sufficient_one() -> (
    None
):
    small = run(KEPLER, ["r", "theta"], pins=Pins(degree=2, power_bound=2, harmonics=1))
    assert small.status is Status.OK
    assert small.new_constants == []
    assert len(small.tiers) == 1
    enough = run(KEPLER, ["r", "theta"], pins=Pins(degree=2, power_bound=3, harmonics=1))
    assert len(enough.new_constants) == 2
    adaptive = run(KEPLER, ["r", "theta"])
    first = {c.tier for c in adaptive.new_constants}
    assert first == {3}  # the 4th family of the ladder: degree 2, powers of r up to 3
    assert [t.code for t in adaptive.tiers[:4]] == ["d1p1h1t0", "d2p1h1t0", "d2p2h1t0", "d2p3h1t0"]


def test_kepler_hamiltonian_form_gives_the_same_structure() -> None:
    report = run(KEPLER_H, ["r", "theta"], system="hamiltonian", momenta=["p_r", "p_theta"])
    assert report.status is Status.OK and report.system == "hamiltonian"
    assert len(report.new_constants) == 2
    ind = report.independence
    assert ind is not None and ind.rank_lower == 3 and ind.rank_proved and len(ind.relations) == 1
    s = symbols(parsed(report))
    r, th, pr, pth, m, k = (s[n] for n in ("r", "theta", "p_r", "p_theta", "m", "k"))
    ax = pth**2 * sp.sin(th) / r - pr * pth * sp.cos(th) - k * m * sp.sin(th)
    ay = -(pth**2) * sp.cos(th) / r - pr * pth * sp.sin(th) + k * m * sp.cos(th)
    assert in_span(parsed(report), ax) and in_span(parsed(report), ay)


def test_discovery_is_deterministic() -> None:
    a = run(OSCILLATOR, ["x", "y"])
    b = run(OSCILLATOR, ["x", "y"])
    assert [t.code for t in a.tiers] == [t.code for t in b.tiers]
    assert [c.expression_latex for c in a.constants] == [c.expression_latex for c in b.constants]
    assert [e.name for e in a.events] == [e.name for e in b.events]


# --- other systems -----------------------------------------------------------------------------------
def test_isotropic_oscillator_finds_the_quadratic_integrals() -> None:
    report = run(OSCILLATOR, ["x", "y"])
    s = symbols(parsed(report))
    x, y, xd, yd, m, k = (s[n] for n in ("x", "y", "x_dot", "y_dot", "m", "k"))
    new = parsed(report)
    assert len(new) == 2
    assert in_span(new, k * x * y + m * xd * yd) and in_span(new, k * x**2 + m * xd**2)
    ind = report.independence
    assert ind is not None and ind.rank_lower == 3 and ind.rank_proved
    assert sum(c.independence == "dependent" for c in report.constants) == 1


def test_three_dimensional_oscillator_has_five_independent_integrals() -> None:
    report = run(OSCILLATOR_3D, ["x", "y", "z"])
    ind = report.independence
    assert ind is not None
    assert (ind.rank_lower, ind.rank_proved, ind.max_possible) == (5, True, 5)
    assert len(report.constants) >= 8
    # every constant after the first five adds nothing, and the report says so with proof
    assert [c.independence for c in report.constants[5:]] == ["dependent"] * (
        len(report.constants) - 5
    )


def test_hamiltonian_oscillator_matches_the_lagrangian_one() -> None:
    report = run(r"\frac{p_x^2+p_y^2}{2m}+\frac{k}{2}(x^2+y^2)", ["x", "y"], system="hamiltonian")
    s = symbols(parsed(report))
    x, y, px, py, m, k = (s[n] for n in ("x", "y", "p_x", "p_y", "m", "k"))
    assert in_span(parsed(report), k * m * x * y + px * py)
    assert in_span(parsed(report), k * m * x**2 + px**2)


def test_free_particle_has_only_dependent_known_constants_and_the_boost_needs_time() -> None:
    plain = run(r"\frac{1}{2}m\dot{x}^2", ["x"])
    assert plain.new_constants == []
    assert plain.cause == "ansatz_exhausted"
    assert plain.independence is not None and plain.independence.rank_lower == 1
    assert [c.independence for c in plain.constants] == ["independent", "dependent"]
    thorough = run(r"\frac{1}{2}m\dot{x}^2", ["x"], budget="thorough")
    new = parsed(thorough)
    assert len(new) == 1
    s = symbols(new)
    assert in_span(new, s["t"] * s["x_dot"] - s["x"])  # x - v t: conserved, explicit in t
    assert thorough.independence is not None and thorough.independence.rank_lower == 2


def test_time_dependent_system_found_with_time_powers() -> None:
    report = run(r"\frac{1}{2}m\dot{x}^2+F t x", ["x"])
    new = parsed(report)
    assert len(new) == 1
    s = symbols(new)
    assert in_span(new, 2 * s["m"] * s["x_dot"] - s["F"] * s["t"] ** 2)
    assert not any(c.source == "energy" for c in report.constants)


def test_central_force_without_a_hidden_constant_reports_none_in_family() -> None:
    report = run(r"\frac{1}{2}m(\dot{r}^2+r^2\dot{\theta}^2)-\frac{k}{r^2}", ["r", "theta"])
    assert report.status is Status.OK
    assert report.new_constants == []
    assert report.independence is not None and report.independence.rank_lower == 2
    assert "none-in-family" in " ".join(report.features)
    assert report.cause in ("resource_limit", "ansatz_exhausted")
    assert "proves nothing" in " ".join(report.limitations)


def test_henon_heiles_negative_benchmark() -> None:
    report = run(HENON_HEILES, ["x", "y"])
    assert report.status is Status.OK
    assert report.new_constants == []
    assert [c.source for c in report.constants] == ["energy"]
    assert report.stop_reason in ("unknown_cap", "ladder_exhausted")
    assert report.unexplored or report.stop_reason == "ladder_exhausted"


def test_pendulum_angle_is_detected_without_declaration() -> None:
    report = run(r"\frac{1}{2}m l^2\dot{\theta}^2+m g l\cos(\theta)", ["theta"])
    assert report.status is Status.OK
    assert report.angles == ["theta"]
    assert report.new_constants == []  # one degree of freedom: only the energy


# --- limits stated, not hidden ---------------------------------------------------------------------
def test_cartesian_kepler_is_unsupported_not_guessed() -> None:
    report = run(KEPLER_CARTESIAN, ["x", "y"])
    assert report.status is Status.UNSUPPORTED
    assert report.cause == "unsupported_function"
    assert report.constants == []


def test_quick_budget_stops_at_the_proved_maximum() -> None:
    report = run(OSCILLATOR, ["x", "y"], budget="quick")
    assert report.stop_reason == "maximal_independent_set"
    assert (
        len(report.new_constants) == 1
    )  # the larger families that would add the dependent ones were skipped
    assert report.unexplored
    assert report.independence is not None and report.independence.rank_lower == 3


def test_resource_limit_when_nothing_fits() -> None:
    report = run(OSCILLATOR_3D, ["x", "y", "z"], budget="quick", pins=Pins(degree=2, power_bound=4))
    # pinned family of 500+ unknowns is refused by the cap of the quick budget: nothing is claimed
    assert report.status in (Status.OK, Status.RESOURCE_LIMIT)
    if report.status is Status.OK:
        assert report.tiers


def test_bad_arguments_are_refused() -> None:
    assert run(KEPLER, ["r", "theta"], angles=["phi"]).status is Status.PARSE_ERROR
    assert run(KEPLER, ["a", "b", "c", "d", "e"]).status is Status.UNSUPPORTED  # above the cap of 4
    assert run(KEPLER, ["r", "theta"], budget="huge").status is Status.UNSUPPORTED
    missing = run(r"\frac{p_x^2}{2m}+x^2", ["x"], system="hamiltonian", momenta=["q"])
    assert missing.status is Status.UNSUPPORTED and "momentum" in missing.message
    singular = run(r"p_x+x^2", ["x"], system="hamiltonian")
    assert singular.status is Status.UNSUPPORTED


# --- verification is the gate -----------------------------------------------------------------------
def test_a_wrong_candidate_fails_the_verification_used_by_the_search() -> None:
    flow = lagrangian_flow(KEPLER, ["r", "theta"], "t", {}, LIMITS)
    r, th, rd, thd = flow.q[0], flow.q[1], flow.v[0], flow.v[1]
    m = next(p for p in flow.params if p.name == "m")
    wrong = m * r**3 * thd**2 * sp.cos(th) + r**2 * rd * thd * sp.sin(th)
    ok, _ = flow.verify(wrong)
    assert not ok


def test_events_carry_counters_only_and_cover_the_stages() -> None:
    report = run(KEPLER, ["r", "theta"])
    names = {e.name for e in report.events}
    for stage in (
        "started",
        "family_generated",
        "determining_equations",
        "nullspace_computed",
        "candidate_found",
        "verification_completed",
        "independence_checked",
        "family_expanded",
        "completed",
    ):
        assert f"invariant_search.{stage}" in names
    completed = report.events[-1]
    assert completed.name == "invariant_search.completed"
    assert completed.data["new"] == 2 and completed.data["rejected"] == 0
    for e in report.events:
        for value in e.data.values():
            assert isinstance(value, int | float | bool | str)
            if isinstance(value, str):  # fixed labels only: no expression can appear here
                assert len(value) < 40 and not any(ch in value for ch in "\\^*+=(")
    # the nullspace stage reports matrix shape, rank and nullity
    stage = next(e for e in report.events if e.name.endswith("nullspace_computed"))
    assert {"rows", "cols", "rank", "nullity"} <= set(stage.data)


def test_constants_are_named_with_a_subscript_like_the_relation_between_them() -> None:
    # The model writes C_1 = ...; the tool used to print C1 = ..., so a verified constant was marked
    # "unchecked" over spelling alone. The id stays C1 (it is a reference), the LaTeX is C_{1}.
    report = run(KEPLER, ["r", "theta"])
    assert [c.id for c in report.constants[:2]] == ["C1", "C2"]
    assert [c.expression_latex.split(" = ")[0] for c in report.constants] == [
        "C_{1}",
        "C_{2}",
        "C_{3}",
        "C_{4}",
    ]


def test_text_syntax_is_read_by_the_strict_text_parser() -> None:
    latex = run(KEPLER, ["r", "theta"], angles=["theta"])
    text = discover_invariants(
        "lagrangian",
        "1/2*m*(r_dot^2 + r^2*theta_dot^2) + k/r",
        ["r", "theta"],
        None,
        ["theta"],
        "standard",
        None,
        "t",
        {},
        LIMITS,
        "text",
    )
    assert text.status is Status.OK and latex.status is Status.OK
    assert len(text.constants) == len(latex.constants)
    hamiltonian = discover_invariants(
        "hamiltonian",
        "p_r^2/(2*m) + p_theta^2/(2*m*r^2) - k/r",
        ["r", "theta"],
        None,
        [],
        "standard",
        None,
        "t",
        {},
        LIMITS,
        "text",
    )
    assert hamiltonian.status is Status.OK and hamiltonian.constants


def test_more_than_four_coordinates_is_a_typed_refusal_that_names_the_limit() -> None:
    from minerva_engine.notation.models import InputCode

    report = run(
        r"\frac{1}{2}(\dot{a}^2+\dot{b}^2+\dot{c}^2+\dot{d}^2+\dot{e}^2)", ["a", "b", "c", "d", "e"]
    )
    assert report.status is Status.UNSUPPORTED
    assert report.diagnostic is not None
    assert report.diagnostic.code is InputCode.TOO_MANY_COORDINATES
    assert report.diagnostic.field == "coordinates" and not report.diagnostic.recoverable
    assert "at most 4 coordinates" in report.message and "5 were given" in report.message
    assert "not available" in report.message
    assert not report.constants  # nothing was searched and nothing was dropped


# -- four coordinates (issue #152) -------------------------------------------------------------------------
# The isotropic oscillator in four Cartesian coordinates. Its quadratic constants are written here by hand,
# from outside Minerva (the U(4) symmetry of the isotropic oscillator; Fradkin, Am. J. Phys. 33 (1965) 207):
# the four energies (x_i^2 + v_i^2)/2, the six rotations x_i v_j - x_j v_i and the six x_i x_j + v_i v_j.
OSCILLATOR_4D = "(a_dot^2+b_dot^2+c_dot^2+d_dot^2)/2 - (a^2+b^2+c^2+d^2)/2"


def _oscillator_4d_reference() -> tuple[list[sp.Expr], list[sp.Expr], list[sp.Expr]]:
    x = sp.symbols("a b c d")
    v = sp.symbols("a_dot b_dot c_dot d_dot")  # the names the LaTeX parser gives the velocities
    energies = [(x[i] ** 2 + v[i] ** 2) / 2 for i in range(4)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    rotations = [x[i] * v[j] - x[j] * v[i] for i, j in pairs]
    products = [x[i] * x[j] + v[i] * v[j] for i, j in pairs]
    return energies, rotations, products


def run_4d(budget: str) -> DiscoveryReport:
    return discover_invariants(
        "lagrangian",
        OSCILLATOR_4D,
        ["a", "b", "c", "d"],
        None,
        [],
        budget,
        None,
        "t",
        {},
        budget_limits(),
        "text",
    )


def _rename(found: list[sp.Expr], reference: list[sp.Expr]) -> list[sp.Expr]:
    by_name = {s.name: s for e in reference for s in e.free_symbols}
    return [e.xreplace({s: by_name.get(s.name, s) for s in e.free_symbols}) for e in found]


def test_four_coordinates_are_searched_and_every_constant_lies_in_the_reference_span() -> None:
    report = run_4d("quick")
    assert report.status is Status.OK
    energies, rotations, products = _oscillator_4d_reference()
    reference = energies + rotations + products
    known = _rename(parsed(report, "known"), reference)
    assert len(known) == 7  # the energy and the six rotations (Noether), each verified on the flow
    assert all(c.verification and c.verification.passed for c in report.constants)
    for found in known:
        assert in_span(reference, found)
    for rotation in rotations:
        assert in_span(known, rotation)
    assert in_span(known, sum(energies, sp.Integer(0)))
    # The family searched at four coordinates is stated, and the larger ones are listed as not searched.
    assert [t.code for t in report.tiers] == ["d1p1h1t0"] and report.tiers[0].unknowns == 80
    assert any("needs 240 unknown coefficients (limit 150)" in u for u in report.unexplored)
    # One relation between the six rotations (the Pfaffian) makes the rank 6, proved.
    assert report.independence is not None and report.independence.rank_lower == 6


@pytest.mark.extended
def test_four_coordinates_at_the_standard_budget_find_the_products_and_not_the_single_energies() -> (
    None
):
    report = run_4d("standard")
    assert report.status is Status.OK
    energies, rotations, products = _oscillator_4d_reference()
    reference = energies + rotations + products
    found = _rename(parsed(report, "known") + parsed(report, "new"), reference)
    assert all(c.verification and c.verification.symbolic_zero for c in report.new_constants)
    for constant in found:
        assert in_span(reference, constant)
    for product in products:
        assert in_span(found, product)
    # The single energies need the power bound 2 (1215 unknowns): outside the searched families, and said so.
    assert not in_span(found, energies[0])
    assert any("power bound to 2 needs 1215" in u for u in report.unexplored)
    assert report.independence is not None and report.independence.rank_lower == 7


def test_four_coordinates_as_a_hamiltonian_are_searched_and_five_are_refused() -> None:
    hamiltonian = "(p_a^2+p_b^2+p_c^2+p_d^2)/2 + (a^2+b^2+c^2+d^2)/2"
    args = ([], "quick", None, "t", {}, budget_limits(), "text")
    four = discover_invariants("hamiltonian", hamiltonian, ["a", "b", "c", "d"], None, *args)
    assert four.status is Status.OK and len(four.constants) == 7
    assert all(c.verification and c.verification.passed for c in four.constants)
    five = discover_invariants(
        "hamiltonian", hamiltonian + " + p_e^2/2", ["a", "b", "c", "d", "e"], None, *args
    )
    assert five.status is Status.UNSUPPORTED and five.diagnostic is not None
    assert five.diagnostic.code.value == "too_many_coordinates"


def test_a_four_coordinate_system_whose_smallest_family_is_too_large_is_a_resource_limit() -> None:
    """Must still fail: every coordinate in a denominator gives 5 * 3^4 = 405 unknowns in the smallest
    family, above the standard cap of 350. Nothing is searched and the family and its size are the reason."""
    report = discover_invariants(
        "lagrangian",
        "(a_dot^2+b_dot^2+c_dot^2+d_dot^2)/2 - 1/a - 1/b - 1/c - 1/d",
        ["a", "b", "c", "d"],
        None,
        [],
        "standard",
        None,
        "t",
        {},
        budget_limits(),
        "text",
    )
    assert report.status is Status.RESOURCE_LIMIT and report.cause == "resource_limit"
    assert not report.tiers and not report.new_constants
    assert "the smallest family (" in report.message
    assert "405 unknown coefficients (limit 350)" in report.message
    assert report.unexplored and "405" in report.unexplored[0]


# -- tan, cot, sec and csc of an angle (issue #191) --------------------------------------------------------
# Kepler in spherical coordinates: `derive` writes tan(theta) in the phi equation. The search reads it as
# sin/cos; the equations every candidate is verified on are the original ones.
KEPLER_SPHERICAL = "(r_dot^2 + r^2*theta_dot^2 + r^2*sin(theta)^2*phi_dot^2)/2 + k/r"


def test_reciprocal_trig_functions_are_written_through_cos_and_sin() -> None:
    from minerva_engine.discovery import _reciprocal_trig

    x = sp.Symbol("x")
    assert _reciprocal_trig(sp.tan(x)) == sp.sin(x) / sp.cos(x)
    assert _reciprocal_trig(sp.cot(x)) == sp.cos(x) / sp.sin(x)
    assert _reciprocal_trig(sp.sec(x)) == 1 / sp.cos(x)
    assert _reciprocal_trig(sp.csc(x)) == 1 / sp.sin(x)
    assert _reciprocal_trig(sp.cosh(x) + sp.sin(x)) == sp.cosh(x) + sp.sin(
        x
    )  # closed: nothing else


def test_kepler_in_spherical_coordinates_is_searched_and_its_constants_pass_the_flow_check() -> (
    None
):
    # One pinned family (the first of the ladder, 108 unknowns) under the thorough budget's deadline: the
    # quick budget's 8 s deadline could end the tier on a loaded runner before it started (`outcome
    # 'deadline'`), which made the test depend on the machine's speed, not on the mathematics.
    report = discover_invariants(
        "lagrangian",
        KEPLER_SPHERICAL,
        ["r", "theta", "phi"],
        None,
        [],
        "thorough",
        Pins(degree=1, power_bound=1, harmonics=1, time_degree=0),
        "t",
        {"k": ["positive"]},
        budget_limits(),
        "text",
    )
    assert report.status is Status.OK  # was `unsupported` before issue #191
    assert [t.code for t in report.tiers] == ["d1p1h1t0"] and report.tiers[0].unknowns == 108
    assert all(t.outcome == "searched" for t in report.tiers)
    assert "theta" in report.angles
    flow = lagrangian_flow(
        KEPLER_SPHERICAL, ["r", "theta", "phi"], "t", {"k": ["positive"]}, budget_limits(), "text"
    )
    by_name = {s.name: s for s in [*flow.q, *flow.v, *flow.params]}
    assert report.constants
    for constant in report.constants:
        expr = parse_latex(constant.expression_latex.split("=", 1)[1], {}, LIMITS)
        expr = expr.xreplace({s: by_name.get(s.name, s) for s in expr.free_symbols})
        ok, ver = flow.verify(expr)
        assert ok and ver is not None and ver.passed


def test_a_wrong_sign_in_the_tan_rewrite_gives_a_quantity_that_fails_the_flow_check() -> None:
    """Wrong twin: with tan(theta) read as -sin/cos, the phi equation would conserve r^2 phi_dot / sin^2
    instead of r^2 sin^2 phi_dot. The flow check, on the original equations, must reject it."""
    flow = lagrangian_flow(
        KEPLER_SPHERICAL, ["r", "theta", "phi"], "t", {"k": ["positive"]}, budget_limits(), "text"
    )
    r, theta, _ = flow.q
    _, _, phi_dot = flow.v
    true = r**2 * sp.sin(theta) ** 2 * phi_dot
    twin = r**2 * phi_dot / sp.sin(theta) ** 2
    # The twin is exactly what the sign-flipped rewrite of the same equations conserves.
    flipped = [
        e.xreplace({a: -sp.sin(a.args[0]) / sp.cos(a.args[0]) for a in e.atoms(sp.tan)})
        for e in flow.v_rate
    ]
    drift = sp.diff(twin, theta) * flow.q_rate[1] + sp.diff(twin, r) * flow.q_rate[0]
    drift += sp.diff(twin, phi_dot) * flipped[2]
    assert sp.simplify(drift) == 0
    assert flow.verify(true)[0]
    ok, _ = flow.verify(twin)
    assert not ok


SCHWARZSCHILD = (
    "(-(1-2*M/r)*t_dot^2 + r_dot^2/(1-2*M/r) + r^2*theta_dot^2 + r^2*sin(theta)^2*phi_dot^2)/2"
)
# x is coupled to the velocity of the cyclic z (p_z = z_dot + x), w is a free cyclic coordinate. With p_z held
# fixed the reduced system is two unit oscillators about (p_z, 0), whose angular momentum and mixed product are
# odd in p_z: a wrong sign in the back-substitution changes them.
GYROSCOPIC = "(x_dot^2 + y_dot^2 + z_dot^2 + w_dot^2)/2 + x*z_dot - y^2/2"


def text_run(
    function: str,
    coordinates: list[str],
    budget: str,
    *,
    time: str = "t",
    pins: Pins | None = None,
    **assumed: list[str],
) -> DiscoveryReport:
    return discover_invariants(
        "lagrangian",
        function,
        coordinates,
        None,
        [],
        budget,
        pins,
        time,
        dict(assumed),
        budget_limits(),
        "text",
    )


def _proportional(found: sp.Expr, expected: sp.Expr) -> bool:
    ratio = sp.simplify(found / expected)
    return bool(ratio.is_number and ratio != 0)


def test_schwarzschild_in_four_coordinates_is_searched_after_reducing_t_and_phi() -> None:
    """Issue #152 with #153: t and phi are cyclic, so p_t and p_phi are held fixed and the Routhian of (r, theta)
    is searched (the whole system's smallest family, 405 unknowns, is above the standard cap of 350). Expected,
    by hand (Wald §6.3; Carroll §5.4): E = (1 - 2M/r) t_dot, L_z = r^2 sin^2(theta) phi_dot and the energy
    function, which for this L (homogeneous of degree 2 in the velocities) is L itself. J^2 = r^4 theta_dot^2 +
    L_z^2 / sin^2(theta) is conserved too, but in the reduced variables it needs sin(theta) in a denominator, so
    it is outside every family searched and must not be reported.

    One family of the reduced system is pinned (degree 2, powers of r to 2, first harmonic: 90 unknowns) so
    that the test does not depend on runner speed: the whole standard ladder takes about 10.5 s on a workstation,
    a third of its deadline, and a tier the deadline cuts would fail the "every family searched" assertion."""
    report = text_run(
        SCHWARZSCHILD,
        ["t", "r", "theta", "phi"],
        "standard",
        time="s",
        pins=Pins(degree=2, power_bound=2, harmonics=1, time_degree=0),
        M=["positive"],
    )
    assert report.status is Status.OK, report.message
    M, r, theta = sp.symbols("M r theta")
    t_dot, r_dot, theta_dot, phi_dot = sp.symbols("t_dot r_dot theta_dot phi_dot")
    f = 1 - 2 * M / r
    energy = f * t_dot
    l_z = r**2 * sp.sin(theta) ** 2 * phi_dot
    lagrangian = (-f * t_dot**2 + r_dot**2 / f + r**2 * theta_dot**2 + l_z * phi_dot) / 2
    j_squared = r**4 * theta_dot**2 + l_z**2 / sp.sin(theta) ** 2
    found = [parsed_constant(c.expression_latex) for c in report.constants]
    assert len(found) == 3
    for expected in (energy, l_z, lagrangian):
        assert sum(_proportional(c, expected) for c in found) == 1
    assert not any(_proportional(c, j_squared) for c in found)

    # Each one verified on the whole four-coordinate system, and so is J^2: its absence is the family's.
    flow = lagrangian_flow(
        SCHWARZSCHILD, ["t", "r", "theta", "phi"], "s", {"M": ["positive"]}, budget_limits(), "text"
    )
    for constant in report.constants:
        assert constant.verification is not None and constant.verification.passed
    for expected in (energy, l_z, lagrangian, j_squared):
        assert flow.verify(_on(flow, expected))[0]

    reduction = report.reduction
    assert reduction is not None
    assert reduction.cyclic == ["t", "phi"] and reduction.remaining == ["r", "theta"]
    assert [h.symbol_latex for h in reduction.held_fixed] == ["p_{t}", r"p_{\phi}"]
    assert reduction.verification is not None and reduction.verification.passed
    assert any(r"\sin" in x for x in reduction.restrictions_latex)
    assert any("M" in x for x in reduction.restrictions_latex)
    assert reduction.rejected_on_full_system == 0
    assert [t.code for t in report.tiers] == ["d2p2h1t0"] and report.tiers[0].unknowns == 90
    assert all(t.outcome == "searched" for t in report.tiers)
    assert all("theta" in t.description and " t," not in t.description for t in report.tiers)
    notes = " ".join(report.notes)
    assert "families of the reduced system" in notes
    assert "verified again on the equations of motion of the whole system" in notes
    assert "cos or sin of an angle in a denominator" in notes
    assert report.independence is not None
    assert report.independence.rank_proved and report.independence.rank_lower == 3
    assert report.independence.max_possible == 7  # the whole system's 2n - 1, not the reduced one's


def parsed_constant(latex: str) -> sp.Expr:
    expr = parse_latex(latex.split("=", 1)[1])
    return expr.xreplace({s: sp.Symbol(s.name) for s in expr.free_symbols})


def _on(flow: object, expr: sp.Expr) -> sp.Expr:
    """`expr` (plain symbols) in the symbols of `flow` (which carry the declared assumptions)."""
    by_name = {s.name: s for s in [*flow.q, *flow.v, *flow.params]}  # type: ignore[attr-defined]
    return expr.xreplace({s: by_name.get(s.name, s) for s in expr.free_symbols})


def test_reduced_constants_odd_in_a_held_momentum_are_mapped_back_and_verified() -> None:
    """Reduced by z and w (p_z = z_dot + x, p_w = w_dot), the search finds constants of (x, y) that contain p_z;
    with p_z replaced by z_dot + x each is a constant of the whole system (written by hand here: the reduced
    oscillators have centre (p_z, 0))."""
    report = text_run(GYROSCOPIC, ["x", "y", "z", "w"], "quick")
    assert report.status is Status.OK, report.message
    assert report.reduction is not None and report.reduction.cyclic == ["z", "w"]
    x, y, z_dot, x_dot, y_dot = sp.symbols("x y z_dot x_dot y_dot")
    p_z = z_dot + x
    angular = (x - p_z) * y_dot - y * x_dot  # = -(z_dot y_dot + y x_dot)
    mixed = x_dot * y_dot + (x - p_z) * y  # = x_dot y_dot - z_dot y
    new = [parsed_constant(c.expression_latex) for c in report.new_constants]
    assert len(new) == 2
    for expected in (angular, mixed):
        assert sum(_proportional(c, sp.expand(expected)) for c in new) == 1
    assert all(c.verification and c.verification.symbolic_zero for c in report.new_constants)
    assert report.independence is not None and report.independence.rank_lower == 5
    assert "of the reduced system was reached" in report.message


def test_a_wrong_sign_in_the_back_substitution_fails_on_the_whole_system_and_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrong twin of the lift: p_z -> -(z_dot + x). The reduced search finds the same candidates, but mapped
    back with the wrong sign they are not constant on the whole system, so the check drops and counts them."""
    from minerva_engine import discovery

    honest = discovery._held_values
    monkeypatch.setattr(
        discovery,
        "_held_values",
        lambda d, index, momenta: {p: -v for p, v in honest(d, index, momenta).items()},
    )
    report = text_run(GYROSCOPIC, ["x", "y", "z", "w"], "quick")
    assert report.status is Status.OK
    assert report.reduction is not None and report.reduction.rejected_on_full_system >= 2
    assert not report.new_constants
    assert report.cause == "verification_failed"
    assert sum(t.rejected for t in report.tiers) >= 2
    assert any("failed the check on the whole system" in n for n in report.notes)
    assert [e for e in report.events if e.name.endswith("lift_verified")]
    assert not any(e.data.get("passed") for e in report.events if e.name.endswith("lift_verified"))

    # The twin itself, written by hand: the reduced angular momentum with p_z -> -(z_dot + x).
    flow = lagrangian_flow(GYROSCOPIC, ["x", "y", "z", "w"], "t", {}, budget_limits(), "text")
    x, y, z_dot, x_dot, y_dot = sp.symbols("x y z_dot x_dot y_dot")
    honest_lift = (x - (z_dot + x)) * y_dot - y * x_dot
    wrong_lift = (x + (z_dot + x)) * y_dot - y * x_dot
    assert flow.verify(_on(flow, honest_lift))[0]
    assert not flow.verify(_on(flow, wrong_lift))[0]


def test_a_reduction_that_fails_its_own_check_searches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong Routhian (one term with the wrong sign) injected at the solve: `routh.reduce` refuses it as
    not verified, and the search does not run on it or on the whole system."""
    from minerva_engine import routh

    honest = routh.solve_reduction

    def wrong(*args: object, **kwargs: object) -> routh.Reduction:
        reduction = honest(*args, **kwargs)  # type: ignore[arg-type]
        x = sp.Symbol("x")
        return routh.Reduction(
            routhian=reduction.routhian + 2 * sp.Symbol("p_z") * x,
            velocities=reduction.velocities,
            determinant=reduction.determinant,
        )

    monkeypatch.setattr(routh, "solve_reduction", wrong)
    report = text_run(GYROSCOPIC, ["x", "y", "z", "w"], "quick")
    assert report.status is Status.NOT_VERIFIED
    assert report.stop_reason == "reduction_not_verified" and report.cause == "verification_failed"
    assert not report.tiers and not report.constants


def test_four_coordinates_without_a_cyclic_one_are_still_a_resource_limit() -> None:
    """Must still fail: the reduction needs a cyclic coordinate, and this system has none (every coordinate is
    in a denominator: 5 * 3^4 = 405 unknowns in the smallest family, above the standard cap of 350)."""
    report = text_run(
        "(a_dot^2+b_dot^2+c_dot^2+d_dot^2)/2 - 1/a - 1/b - 1/c - 1/d",
        ["a", "b", "c", "d"],
        "standard",
    )
    assert report.status is Status.RESOURCE_LIMIT and report.reduction is None
    assert "405 unknown coefficients (limit 350)" in report.message


def test_a_cyclic_coordinate_the_reduction_refuses_leaves_a_resource_limit() -> None:
    """Must still fail: w is cyclic but p_w = w_dot^3 is not linear in w_dot, so `routh.reduce` refuses
    (`nonlinear_momentum`) and the whole system is searched as before: 5 * 3^4 = 405 unknowns > 350. The note
    says why nothing was reduced; no coordinate is dropped to make it fit."""
    report = text_run(
        "(a_dot^2+b_dot^2+c_dot^2)/2 + w_dot^4/4 - 1/a - 1/b - 1/c",
        ["a", "b", "c", "w"],
        "standard",
    )
    assert report.status is Status.RESOURCE_LIMIT and report.reduction is None
    assert "405 unknown coefficients (limit 350)" in report.message
    assert any("were not reduced" in n and "linear" in n for n in report.notes)
