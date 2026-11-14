"""The ODE reference problems: their judges must catch a wrong answer, a lost answer and an answer where a
refusal was due. A benchmark that cannot fail proves nothing."""

from __future__ import annotations

from typing import Any

import pytest
from minerva_engine import Engine, Limits, Status
from minerva_engine.ode import benchmark
from minerva_engine.ode.models import OdeNumericReport, OdeSolution
from minerva_engine.reference_benchmark import PROBLEMS

ODE = {p.id: p for p in benchmark.ODE_PROBLEMS}


def test_the_corpus_covers_every_kind_of_problem_in_the_brief() -> None:
    assert {p.category for p in benchmark.ODE_PROBLEMS} == {
        "ode-analytic",
        "ode-numeric",
        "ode-stiff",
        "ode-coupled",
        "ode-boundary",
        "ode-approximation",
        "ode-qualitative",
        "ode-refusal",
    }
    assert all(p.source for p in benchmark.ODE_PROBLEMS)
    assert len({p.id for p in PROBLEMS}) == len(PROBLEMS)
    assert set(ODE) <= {p.id for p in PROBLEMS}  # they are part of the reference report CI checks
    refusals = [p for p in benchmark.ODE_PROBLEMS if p.expected == "refused"]
    assert len(refusals) >= 5  # failure behaviour is tested as hard as success


class Doctored(Engine):
    """The real engine, with one result changed on its way to the judge."""

    def __init__(self, **changes: Any) -> None:
        super().__init__(Limits(timeout_seconds=60, max_concurrency=2))
        self.changes = changes

    def ode_solve_numerical(self, request: Any) -> OdeNumericReport:
        report = super().ode_solve_numerical(request)
        change = self.changes.get("numeric")
        return change(report) if change else report

    def ode_solve_symbolic(self, request: Any) -> OdeSolution:
        report = super().ode_solve_symbolic(request)
        change = self.changes.get("symbolic")
        return change(report) if change else report


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=60, max_concurrency=2))


def scaled(factor: float):  # type: ignore[no-untyped-def]
    def change(report: OdeNumericReport) -> OdeNumericReport:
        for s in report.series:
            s.values = [v * factor for v in s.values]
        return report

    return change


def test_the_real_engine_passes_a_sample_of_each_kind(engine: Engine) -> None:
    for pid in (
        "ode-linear-second-order",
        "ode-harmonic-oscillator",
        "ode-bvp-linear",
        "ode-series-geometric",
        "ode-blow-up-refused",
    ):
        outcome, detail = ODE[pid].judge(engine)
        assert outcome in ("correct", "refused_as_expected"), (pid, detail)


@pytest.mark.parametrize("pid", ["ode-harmonic-oscillator", "ode-logistic", "ode-robertson-stiff"])
def test_a_numerical_answer_that_is_off_by_a_thousandth_is_called_wrong(pid: str) -> None:
    outcome, _ = ODE[pid].judge(Doctored(numeric=scaled(1.001)))
    assert outcome == "wrong"


def test_a_boundary_answer_that_is_off_is_called_wrong() -> None:
    assert ODE["ode-bvp-linear"].judge(Doctored(numeric=scaled(1.01)))[0] == "wrong"


def test_a_closed_form_that_is_not_the_solution_is_called_wrong() -> None:
    def wrong(report: OdeSolution) -> OdeSolution:
        for item in report.solutions:
            item.rhs_text = ["2*" + item.rhs_text[0]]
        return report

    assert ODE["ode-linear-second-order"].judge(Doctored(symbolic=wrong))[0] == "wrong"
    assert ODE["ode-series-geometric"].judge(Doctored(symbolic=wrong))[0] == "wrong"


def test_a_lost_answer_is_not_a_pass() -> None:
    def lost(report: OdeNumericReport) -> OdeNumericReport:
        report.status = Status.UNSUPPORTED
        return report

    assert (
        ODE["ode-harmonic-oscillator"].judge(Doctored(numeric=lost))[0] == "unsupported_regression"
    )


def test_a_result_where_a_refusal_was_due_is_called_wrong(engine: Engine) -> None:
    def invented(report: OdeNumericReport) -> OdeNumericReport:
        report.status = Status.OK
        return report

    for pid in (
        "ode-blow-up-refused",
        "ode-numbers-not-given-refused",
        "ode-bvp-no-solution-refused",
    ):
        assert ODE[pid].judge(Doctored(numeric=invented))[0] == "wrong", pid

    def invented_series(report: OdeSolution) -> OdeSolution:
        report.status = Status.OK
        return report

    for pid in ("ode-series-singular-point-refused", "ode-delay-refused"):
        assert ODE[pid].judge(Doctored(symbolic=invented_series))[0] == "wrong", pid
