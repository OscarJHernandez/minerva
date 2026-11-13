"""Adaptive discovery of conserved quantities (`discovery.py`, `flow.py`, `independence.py`).

The expected invariants below are written out here, by hand, and compared with what the engine found;
they are never given to the engine.
"""

import sympy as sp
from minerva_engine.discovery import DiscoveryReport, Pins, discover_invariants
from minerva_engine.flow import lagrangian_flow
from minerva_engine.latexparse import parse_latex
from minerva_engine.models import Limits, Status

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
    assert run(KEPLER, ["a", "b", "c", "d"]).status is Status.UNSUPPORTED
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
