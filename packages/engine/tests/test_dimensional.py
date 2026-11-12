"""Dimensional analysis: units, Pi groups, missing quantities, and equation unit checks."""

from fractions import Fraction

import pytest
import sympy as sp
from minerva_engine import (
    ConstantCandidate,
    Engine,
    Limits,
    MinervaMathError,
    ParseError,
    QuantitySpec,
    Status,
    pi_groups,
)
from minerva_engine.dimensions import (
    BASE_UNITS,
    ZERO,
    DimensionError,
    UnsupportedDimensionConstruct,
    check_equation_dimensions,
    dim_text,
    dimension_of,
    read_kind,
    read_unit,
)
from minerva_engine.pi_groups import analyze_dimensions, check_units

L = Limits()
CONSTANTS = [
    ConstantCandidate(id="const.c", latex="c", unit="m/s"),
    ConstantCandidate(id="const.G", latex="G", unit="m^3/(kg s^2)"),
    ConstantCandidate(id="const.h", latex="h", unit="J s"),
    ConstantCandidate(id="const.k_b", latex="k_B", unit="J/K"),
]


def q(name: str, unit: str | None = None, kind: str | None = None) -> QuantitySpec:
    return QuantitySpec(name=name, unit=unit, kind=kind)


def vec(**exps: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(exps.get(u, 0)) for u in BASE_UNITS)


# --- reading units ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kg", vec(kg=1)),
        ("kg*m/s**2", vec(kg=1, m=1, s=-2)),
        ("N", vec(kg=1, m=1, s=-2)),
        ("J s", vec(kg=1, m=2, s=-1)),
        ("m^3/(kg s^2)", vec(m=3, kg=-1, s=-2)),
        ("Pa s", vec(kg=1, m=-1, s=-1)),
        ("m**(1/2)", (Fraction(0), Fraction(1, 2)) + (Fraction(0),) * 5),
        ("rad", ZERO),
        ("1", ZERO),
        ("ohm", vec(kg=1, m=2, s=-3, A=-2)),
        ("T", vec(kg=1, s=-2, A=-1)),
    ],
)
def test_read_unit(text: str, expected: tuple[Fraction, ...]) -> None:
    assert read_unit(text) == expected


@pytest.mark.parametrize(
    "text", ["cm", "eV", "foo", "m + s", "2 m", "m^pi", "sin(m)", "", "m" * 300, "__import__('os')"]
)
def test_read_unit_rejects(text: str) -> None:
    with pytest.raises(MinervaMathError):
        read_unit(text)


def test_non_coherent_unit_is_explained() -> None:
    with pytest.raises(ParseError, match="coherent SI"):
        read_unit("cm")


def test_read_kind_and_aliases() -> None:
    assert read_kind("Dynamic Viscosity") == read_unit("Pa s")
    assert read_kind("speed") == read_unit("m/s")
    assert read_kind("surface-tension") == read_unit("N/m")
    with pytest.raises(ParseError, match="unknown quantity kind"):
        read_kind("vibes")


def test_every_kind_and_derived_unit_is_readable() -> None:
    from minerva_engine.dimensions import DERIVED_UNITS, QUANTITY_KINDS

    for name in DERIVED_UNITS:
        read_unit(name)
    for kind in QUANTITY_KINDS:
        read_kind(kind)


def test_dimension_of_rules() -> None:
    x, t = sp.symbols("x t")
    units = {"x": read_unit("m"), "t": read_unit("s")}
    assert dim_text(dimension_of(x / t, units)) == "m s^-1"
    assert dimension_of(sp.exp(x / x), units) == ZERO
    with pytest.raises(DimensionError) as sum_err:
        dimension_of(x + t, units)
    assert sum_err.value.kind == "sum" and len(sum_err.value.parts) == 2
    with pytest.raises(DimensionError) as fn_err:
        dimension_of(sp.sin(x), units)
    assert fn_err.value.kind == "function"
    with pytest.raises(UnsupportedDimensionConstruct):
        dimension_of(sp.Derivative(sp.Function("f")(x), x), units)


def test_handbook_equation_check_is_unchanged() -> None:
    units = {"F": "kg*m/s**2", "m": "kg", "a": "m/s**2"}
    check_equation_dimensions("F", "m*a", units)
    with pytest.raises(DimensionError):
        check_equation_dimensions("F", "m*a**2", units)
    with pytest.raises(ParseError):
        check_equation_dimensions("F", "m*b", units)


# --- Pi groups -------------------------------------------------------------------------------
def test_pendulum_is_determined_up_to_a_constant() -> None:
    r = analyze_dimensions(
        [
            q("T", kind="period"),
            q("L", kind="length"),
            q("g", kind="acceleration"),
            q("m", kind="mass"),
        ],
        "T",
        [],
        CONSTANTS,
        L,
    )
    assert r.status is Status.OK and r.closure == "determined" and r.rank == 3
    assert [g.latex for g in r.groups] == [r"\frac{T \sqrt{g}}{\sqrt{L}}"]
    assert r.groups[0].exponents == {"T": "1", "L": "-1/2", "g": "1/2"}
    assert r.relation_latex == r"T = C\,\sqrt{\frac{L}{g}}"
    assert r.cannot_enter == ["m"]  # a mass has nothing to cancel against
    assert r.missing is None and all(g.dimensionless_verified for g in r.groups)


def test_pipe_flow_with_requested_repeating_quantities() -> None:
    r = analyze_dimensions(
        [
            q("dp", "Pa"),
            q("rho", kind="density"),
            q("v", "m/s"),
            q("D", "m"),
            q("mu", kind="dynamic viscosity"),
            q("L", "m"),
        ],
        "dp",
        ["rho", "v", "D"],
        [],
        L,
    )
    assert r.closure == "up_to_function" and r.repeating == ["rho", "v", "D"]
    assert [g.latex for g in r.groups] == [
        r"\frac{\mathrm{dp}}{\rho v^{2}}",
        r"\frac{\mu}{\rho v D}",  # inverse Reynolds number
        r"\frac{L}{D}",
    ]
    assert r.groups[0].contains_target and not r.groups[1].contains_target
    assert r.relation_latex == r"\mathrm{dp} = \rho v^{2}\,f\left(\Pi_{2},\Pi_{3}\right)"


def test_groups_without_a_target() -> None:
    r = analyze_dimensions([q("x", "m"), q("theta", "rad"), q("t", "s")], None, [], [], L)
    assert r.closure is None and r.relation_latex is None
    assert [g.latex for g in r.groups] == [r"\theta"]  # an angle is its own group
    assert r.rank == 2


def test_no_group_exists_for_independent_dimensions() -> None:
    r = analyze_dimensions([q("x", "m"), q("t", "s"), q("M", "kg")], None, [], [], L)
    assert r.status is Status.OK and r.groups == [] and "no dimensionless group" in r.message


def test_every_group_is_dimensionless_on_a_larger_problem() -> None:
    specs = [
        q("F", "N"), q("rho", "kg/m^3"), q("v", "m/s"), q("A", "m^2"), q("mu", "Pa s"),
        q("g", "m/s^2"), q("sigma", "N/m"), q("c", "m/s"),
    ]  # fmt: skip
    r = analyze_dimensions(specs, "F", [], [], L)
    assert r.status is Status.OK and len(r.groups) == len(specs) - r.rank
    units = {s.name: read_unit(s.unit or "") for s in specs}
    for g in r.groups:
        total = [Fraction(0)] * len(BASE_UNITS)
        for name, exp in g.exponents.items():
            total = [a + Fraction(exp) * b for a, b in zip(total, units[name], strict=True)]
        assert all(x == 0 for x in total)


# --- missing quantities ----------------------------------------------------------------------
def test_energy_from_mass_is_missing_a_velocity_and_c_closes_it() -> None:
    r = analyze_dimensions([q("m", "kg"), q("E", "J")], "E", [], CONSTANTS, L)
    assert r.closure == "missing" and r.groups == [] and r.relation_latex is None
    assert r.missing and r.missing.verified
    assert r.missing.unbalanced == ["m", "s"] and r.missing.missing_dimension_text == "m s^-1"
    assert [s.ids for s in r.missing.suggestions] == [["const.c"]]
    assert r.missing.suggestions[0].relation_latex == r"E = C\,m c^{2}"
    assert "dimensional:missing-quantity" in r.features


def test_length_from_mass_needs_two_constants() -> None:
    r = analyze_dimensions([q("M", "kg"), q("r", "m")], "r", [], CONSTANTS, L)
    assert r.missing is not None and r.missing.missing_dimension_text == "m"
    ids = sorted(tuple(s.ids) for s in r.missing.suggestions)
    assert ids == [("const.G", "const.h"), ("const.c", "const.G"), ("const.c", "const.h")]
    by_ids = {tuple(s.ids): s.relation_latex for s in r.missing.suggestions}
    assert by_ids[("const.c", "const.G")] == r"r = C\,\frac{M G}{c^{2}}"  # Schwarzschild scale
    assert any("pairs" in t for t in r.tested)
    assert any("three or more" in t for t in r.tested)


def test_a_single_constant_is_preferred_and_pairs_are_skipped() -> None:
    r = analyze_dimensions([q("M", "kg"), q("t", "s"), q("r", "m")], "r", [], CONSTANTS, L)
    assert r.missing is not None
    assert {tuple(s.ids) for s in r.missing.suggestions} == {
        ("const.c",),
        ("const.G",),
        ("const.h",),
    }
    assert not any("pairs" in t for t in r.tested)


def test_missing_without_any_constant_that_helps() -> None:
    r = analyze_dimensions([q("m", "kg"), q("I", "A")], "I", [], CONSTANTS[:1], L)
    assert r.missing is not None and r.missing.suggestions == []
    assert r.missing.missing_dimension_text == "A"


def test_unreadable_candidate_unit_is_skipped_and_reported() -> None:
    bad = ConstantCandidate(id="const.bad", latex="b", unit="parsec")
    r = analyze_dimensions([q("m", "kg"), q("E", "J")], "E", [], [bad, *CONSTANTS], L)
    assert r.missing is not None and [s.ids for s in r.missing.suggestions] == [["const.c"]]
    assert any("const.bad" in t for t in r.tested)


def test_target_alone_is_missing_with_its_own_dimension() -> None:
    r = analyze_dimensions([q("E", "J")], "E", [], [], L)
    assert r.closure == "missing" and r.missing is not None
    assert r.missing.missing_dimension_text == "kg m^2 s^-2"


def test_dimensionless_target_is_its_own_group() -> None:
    r = analyze_dimensions([q("eta", "1"), q("x", "m")], "eta", [], [], L)
    assert r.closure == "determined" and r.relation_latex == r"\eta = C"


def test_a_quantity_called_c_does_not_clash_with_the_constant() -> None:
    r = analyze_dimensions(
        [q("C", kind="capacitance"), q("R", kind="resistance"), q("tau", "s")], "tau", [], [], L
    )
    assert r.closure == "determined" and r.relation_latex == r"\tau = C_{0}\,C R"


# --- honest failure --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("specs", "target", "repeating"),
    [
        ([q("x", "m"), q("x", "s")], None, []),  # duplicate names
        ([q("x", "m")], "y", []),  # target not listed
        ([q("x", "m"), q("t", "s")], "x", ["x"]),  # target as repeating
        ([q("x", "m")], None, ["y"]),  # unknown repeating
        ([q("x", "m"), q("t", "s")], None, ["t", "t"]),
        ([q("x", "cm")], None, []),  # non-coherent unit
        ([q("x", "m + s")], None, []),  # sum inside a unit
        ([q("x", kind="vibes")], None, []),
    ],
)
def test_bad_input_is_a_parse_error(specs, target, repeating) -> None:  # type: ignore[no-untyped-def]
    r = analyze_dimensions(specs, target, repeating, [], L)
    assert r.status is Status.PARSE_ERROR and r.groups == [] and r.message


def test_too_many_quantities_is_unsupported() -> None:
    specs = [q(f"x{i}", "m") for i in range(17)]
    assert analyze_dimensions(specs, None, [], [], L).status is Status.UNSUPPORTED


def test_spec_needs_exactly_one_of_unit_or_kind() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        QuantitySpec(name="x")
    with pytest.raises(ValueError, match="exactly one"):
        QuantitySpec(name="x", unit="m", kind="length")
    with pytest.raises(ValueError, match="invalid quantity name"):
        QuantitySpec(name="x y", unit="m")


def test_dependent_repeating_quantities_are_reported() -> None:
    r = analyze_dimensions(
        [q("a", "m"), q("b", "m"), q("c", "s"), q("T", "s")], "T", ["a", "b", "c"], [], L
    )
    assert r.status is Status.OK and r.repeating == ["a", "c"]
    assert any("depend on earlier" in n and "b" in n for n in r.notes)
    assert any(g.latex == r"\frac{b}{a}" for g in r.groups)


def test_a_group_that_is_not_dimensionless_is_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    real = pi_groups._nullspace

    def corrupted(matrix: sp.Matrix) -> list[sp.Matrix]:
        vectors = real(matrix)
        vectors[0] = vectors[0] * 2 + sp.Matrix([0] * (len(vectors[0]) - 1) + [0])
        vectors[0][0] = vectors[0][0] + 1  # break the balance
        return vectors

    monkeypatch.setattr(pi_groups, "_nullspace", corrupted)
    r = analyze_dimensions([q("T", "s"), q("L", "m"), q("g", "m/s^2")], "T", [], [], L)
    assert r.status is Status.NOT_VERIFIED and r.groups == [] and r.relation_latex is None


# --- unit check of an equation -----------------------------------------------------------------
def test_consistent_equation() -> None:
    r = check_units(r"F = m a", [q("F", "N"), q("m", "kg"), q("a", "m/s^2")], L)
    assert r.status is Status.OK and r.consistent is True and len(r.sides) == 2
    assert r.sides[0].dimension_text == "kg m s^-2"


def test_missing_factor_is_reported() -> None:
    r = check_units(r"E = m c", [q("E", "J"), q("m", "kg"), q("c", "m/s")], L.model_copy())
    assert r.status is Status.PARSE_ERROR  # E is a reserved name in the LaTeX parser
    r = check_units(r"W = m c", [q("W", "J"), q("m", "kg"), q("c", "m/s")], L)
    assert r.consistent is False and r.missing_factor_text == "m s^-1"
    assert "factor of dimension m s^-1" in r.message


def test_sum_of_unlike_terms_is_reported() -> None:
    r = check_units(r"x = x_0 + v", [q("x", "m"), q("x_0", "m"), q("v", "m/s")], L)
    assert r.status is Status.OK and r.consistent is False
    assert {t.dimension_text for t in r.problem_terms} == {"m", "m s^-1"}


def test_function_argument_must_be_dimensionless() -> None:
    r = check_units(r"y = \sin(x)", [q("y", "1"), q("x", "m")], L)
    assert r.consistent is False and "sin" in (r.problem or "")
    ok = check_units(r"y = \sin(k x)", [q("y", "1"), q("x", "m"), q("k", "1/m")], L)
    assert ok.consistent is True


def test_equation_check_errors() -> None:
    assert check_units("x", [q("x", "m")], L).status is Status.PARSE_ERROR  # no equals sign
    assert check_units("x = y", [q("x", "m")], L).status is Status.PARSE_ERROR  # no unit for y
    assert check_units("x = 1", [q("x", "cm")], L).status is Status.PARSE_ERROR


def test_chained_equation_compares_every_part() -> None:
    r = check_units(
        r"p = F / A = \rho g h",
        [
            q("p", "Pa"),
            q("F", "N"),
            q("A", "m^2"),
            q("rho", "kg/m^3"),
            q("g", "m/s^2"),
            q("h", "m"),
        ],
        L,
    )
    assert r.consistent is True and len(r.sides) == 3


# --- through the sandbox -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine()


def test_engine_runs_both_operations_in_the_sandbox(engine: Engine) -> None:
    r = engine.dimensional_analysis([q("m", "kg"), q("E", "J")], target="E", candidates=CONSTANTS)
    assert r.status is Status.OK and r.missing is not None and r.duration_ms >= 0
    assert r.missing.suggestions[0].ids == ["const.c"]
    u = engine.check_units(r"F = m a", [q("F", "N"), q("m", "kg"), q("a", "m/s^2")])
    assert u.status is Status.OK and u.consistent is True


def test_hostile_input_is_stopped(engine: Engine) -> None:
    huge = engine.dimensional_analysis([q("x", "m**100000")])
    assert huge.status in {Status.PARSE_ERROR, Status.RESOURCE_LIMIT}
    with pytest.raises(ValueError, match="at most 200"):
        q("x", "(" * 150 + "m" * 100 + ")" * 150)
    deep = engine.dimensional_analysis([q("x", "(" * 90 + "m" + ")" * 90)])
    assert deep.status in {Status.OK, Status.PARSE_ERROR, Status.RESOURCE_LIMIT}
    assert engine.check_units("x = " + "(" * 500 + "x" + ")" * 500, [q("x", "m")]).status in {
        Status.PARSE_ERROR,
        Status.RESOURCE_LIMIT,
    }
    eval_attempt = engine.dimensional_analysis([q("x", "__import__('os').system('true')")])
    assert eval_attempt.status is Status.PARSE_ERROR
