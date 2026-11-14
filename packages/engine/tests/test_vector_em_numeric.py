"""Electrostatics with numbers the user gives: every result is two methods that must agree."""

import pytest
from minerva_engine import Limits, Status, VectorInput
from minerva_engine.vector import (
    BoundaryProblem,
    CheckProblem,
    FaceCondition,
    FieldProblem,
    MultipoleProblem,
    PointCharge,
    Region,
    vector_calculus,
)

L = Limits(timeout_seconds=120)
UNIT = Region(bounds=[("0", "1")] * 3)


def run(problem, constants: list[str] | None = None):  # type: ignore[no-untyped-def]
    return vector_calculus(
        VectorInput(
            expression="",
            format="text",
            action="em_compute",
            em=problem,
            constants=constants or [],
        ),
        L,
    )


def test_a_dipole_field_agrees_with_the_gradient_of_the_potential() -> None:
    r = run(
        FieldProblem(
            kind="field",
            charges=[
                PointCharge(q="q", position=("0", "0", "0")),
                PointCharge(q="-q", position=("0", "0", "d")),
            ],
            point=("1", "2", "3"),
        ),
        ["q", "d"],
    )
    assert r.status is Status.OK and r.em is not None
    assert r.em.agreement < 1e-8 and not r.em.approximate


def test_the_field_at_a_point_charge_is_refused() -> None:
    r = run(
        FieldProblem(
            kind="field",
            charges=[PointCharge(q="1", position=("0", "0", "0"))],
            point=("0", "0", "0"),
        )
    )
    assert r.status is Status.UNSUPPORTED and r.diagnostics.reason == "field_point_in_source"


def test_a_density_field_outside_the_source_matches_the_point_charge_far_away() -> None:
    r = run(
        FieldProblem(kind="field", charges=[], density="1", box=UNIT, point=("30", "0.5", "0.5"))
    )
    assert r.status is Status.OK and r.em is not None and r.em.approximate
    ez = r.em.items[-1].numeric or ""
    assert ez.startswith("(") and abs(float(ez.strip("()").split(",")[0]) - 1 / 29.5**2) < 1e-4


def test_a_field_point_inside_the_density_is_refused() -> None:
    r = run(FieldProblem(kind="field", charges=[], density="1", box=UNIT, point=("0.5",) * 3))
    assert r.status is Status.UNSUPPORTED and r.diagnostics.reason == "field_point_in_source"


def test_multipole_moments_and_the_measured_decay() -> None:
    r = run(
        MultipoleProblem(
            kind="multipole",
            charges=[
                PointCharge(q="1", position=("0", "0", "1")),
                PointCharge(q="-1", position=("0", "0", "-1")),
                PointCharge(q="1", position=("1", "0", "0")),
            ],
        )
    )
    assert r.status is Status.OK and r.em is not None
    assert r.em.items[0].latex == "1"
    assert "\\left(1, 0, 2\\right)" in (r.em.items[1].latex or "")
    assert "measured" in r.em.checks[0] and "never a bound" in r.em.checks[0]


def test_laplace_series_agrees_with_the_relaxation_and_is_an_approximation() -> None:
    r = run(BoundaryProblem(kind="boundary_value", a="1", b="1", top="sin(pi*x)"))
    assert r.status is Status.OK and r.em is not None and r.em.approximate
    centre = next(i for i in r.em.items if i.label.startswith("phi(0.5, 0.5)"))
    # phi = sin(pi x) sinh(pi y) / sinh(pi) at the centre
    assert float(centre.numeric or "nan") == pytest.approx(0.19926840, abs=1e-6)


def test_poisson_with_a_source() -> None:
    r = run(BoundaryProblem(kind="boundary_value", a="1", b="1", source="1"))
    assert r.status is Status.OK and r.em is not None
    centre = next(i for i in r.em.items if i.label.startswith("phi(0.5, 0.5)"))
    assert float(centre.numeric or "nan") == pytest.approx(0.0736713, abs=2e-5)


def test_poisson_with_nonzero_boundary_is_refused() -> None:
    r = run(BoundaryProblem(kind="boundary_value", a="1", b="1", top="1", source="1"))
    assert r.status is Status.UNSUPPORTED


def test_a_series_that_disagrees_with_the_relaxation_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from minerva_engine.vector import em_numeric

    real = em_numeric._sor
    monkeypatch.setattr(
        em_numeric,
        "_sor",
        lambda *a, **k: [[v * 1.05 for v in row] for row in real(*a, **k)],
    )
    r = run(BoundaryProblem(kind="boundary_value", a="1", b="1", top="sin(pi*x)"))
    assert r.status is Status.NOT_VERIFIED and r.diagnostics.reason == "solver_disagreement"
    assert r.em is None


def test_a_candidate_solution_is_checked_and_a_wrong_one_is_named() -> None:
    good = run(
        CheckProblem(
            kind="check_solution",
            potential="sin(pi*x)*sinh(pi*y)",
            box=UNIT,
            conditions=[FaceCondition(face="y=lo", value="0")],
        )
    )
    assert good.em is not None and good.em.holds is True
    wrong = run(
        CheckProblem(
            kind="check_solution",
            potential="x**2",
            box=UNIT,
            conditions=[FaceCondition(face="y=lo", value="0")],
        )
    )
    assert wrong.em is not None and wrong.em.holds is False
    assert "does NOT" in wrong.message
    source_ok = run(CheckProblem(kind="check_solution", potential="x**2", source="-2", box=UNIT))
    assert source_ok.em is not None and source_ok.em.holds is True


def test_undeclared_constants_are_refused() -> None:
    r = run(
        FieldProblem(
            kind="field",
            charges=[PointCharge(q="Q", position=("0", "0", "0"))],
            point=("1", "0", "0"),
        )
    )
    assert r.status is Status.PARSE_ERROR
