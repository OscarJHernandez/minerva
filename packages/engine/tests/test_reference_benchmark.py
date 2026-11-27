"""The reference problems (answers established outside Minerva) stay as expected, and no judge can pass a wrong one.

The first half checks the engine against the oracles. The second half checks the oracles against the engine:
each judge is also run on an engine whose results are deliberately wrong (every value shifted by 1/1000, or every
refusal turned into an `ok`), and must not call that correct.
"""

import json
from pathlib import Path
from typing import Any

import mpmath as mp
import pytest
from minerva_engine import Engine, Limits, Status
from minerva_engine.reference import algebra
from minerva_engine.reference.core import EVIDENCE, ORACLES, TIERS
from minerva_engine.reference_benchmark import (
    GOOD,
    PROBLEMS,
    _verdict,
    deterministic,
    run_problems,
)

ROOT = Path(__file__).resolve().parents[3]
PR_PROBLEMS = [p for p in PROBLEMS if p.tier == "pr"]


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(Limits(timeout_seconds=60, max_concurrency=2))


@pytest.fixture(scope="module")
def rows(engine: Engine) -> list[dict[str, object]]:
    return run_problems(engine, tier="pr")


def test_every_problem_comes_out_as_expected(rows: list[dict[str, object]]) -> None:
    bad = [
        (r["id"], r["outcome"], r["detail"])
        for r in rows
        if r["outcome"] not in (*GOOD, "pending")  # a pending problem is a known gap, never a pass
    ]
    assert not bad


def test_outcomes_match_the_committed_report(rows: list[dict[str, object]]) -> None:
    committed = json.loads((ROOT / "docs/benchmarks/reference.json").read_text())["problems"]
    pr_committed = [r for r in committed if r["tier"] == "pr"]
    assert deterministic(rows) == pr_committed, (
        "run: python -m minerva_engine.reference_benchmark --tier extended"
    )
    report = (ROOT / "docs/reference-benchmark.md").read_text()
    assert all(f"`{r['id']}`" in report for r in rows)


@pytest.mark.parametrize("ambient_dps", [15, 20, 25])
def test_a_judge_does_not_depend_on_the_precision_a_previous_test_left_behind(
    engine: Engine, ambient_dps: int
) -> None:
    # Regression (nightly 2026-11-20 onward): the Pythagorean oracle's 1e-25 threshold failed at 20 digits.
    problem = next(p for p in PROBLEMS if p.id == "simplify-pythagorean-identity")
    before = mp.mp.dps
    mp.mp.dps = ambient_dps
    try:
        outcome, detail = problem.judge(engine)
        assert mp.mp.dps == ambient_dps  # the judge restores what it found
    finally:
        mp.mp.dps = before
    assert outcome == "correct", detail


def test_the_corpus_covers_the_requested_domains_with_both_kinds_of_problem() -> None:
    categories = {p.category for p in PROBLEMS}
    assert categories >= {
        "matrix",
        "distribution",
        "substitution",
        "mechanics",
        "calculus",
        "series",
        "algebra",
        "statmech",
        "variational",
        "ode",
    }
    assert all(p.source for p in PROBLEMS) and len({p.id for p in PROBLEMS}) == len(PROBLEMS)
    for category in categories:
        kinds = {p.expected for p in PROBLEMS if p.category == category}
        if category not in ("substitution", "mechanics"):
            assert kinds == {"correct", "refused"}, category
    refused = [p for p in PR_PROBLEMS if p.expected == "refused"]
    assert len(refused) >= 10  # cases that must be reported unsupported are a quarter of the corpus


def test_every_problem_declares_an_independent_oracle_and_the_kind_of_evidence() -> None:
    for p in PROBLEMS:
        assert p.oracle in ORACLES and p.evidence in EVIDENCE and p.tier in TIERS, p.id
        assert (p.evidence == "refusal") == (p.expected == "refused"), p.id
    # numerical agreement is never recorded as a proof, and a proof is recorded only where a symbolic zero exists
    proofs = {p.id for p in PROBLEMS if p.evidence == "symbolic_proof"}
    assert proofs, "at least one result is backed by a symbolic zero test"
    assert not any(p.oracle == "minerva" for p in PROBLEMS)


def test_a_wrong_answer_a_wrong_result_and_a_lost_answer_are_each_caught() -> None:
    assert _verdict("correct", True, False, "d")[0] == "wrong"  # disagrees with the oracle
    assert _verdict("refused", True, None, "d")[0] == "wrong"  # an answer where a refusal was due
    assert _verdict("correct", False, None, "d")[0] == "unsupported_regression"
    assert _verdict("correct", True, True, "d")[0] == "correct"
    assert _verdict("refused", False, None, "d")[0] == "refused_as_expected"


# --- the oracles against a wrong engine -----------------------------------------------------------------------
def _corrupt(result: Any) -> Any:
    """The same result with its value shifted by 1/1000 (or a constant made to drift)."""
    updates: dict[str, Any] = {}
    numeric = getattr(result, "numeric", None)
    if isinstance(numeric, str):
        updates["numeric"] = str(mp.mpmathify(numeric) * mp.mpf("1.001"))
    text = getattr(result, "result_text", None)
    if isinstance(text, str):
        # not a constant: an antiderivative shifted by a constant is still an antiderivative
        updates["result_text"] = f"({text}) + x/1000"
    latex = getattr(result, "result_latex", None)
    if isinstance(latex, str) and "begin" not in latex:
        updates["result_latex"] = f"{latex} + \\frac{{1}}{{1000}}"
    entries = getattr(result, "result_entries", None)
    if entries:
        first = [list(row) for row in entries]
        first[0][0] = f"({first[0][0]}) + 1/1000"
        updates["result_entries"] = first
    quantities = getattr(result, "quantities", None)
    if quantities and hasattr(quantities[0], "scalar_text"):
        # geometry: the invariant of a field tensor
        updates["quantities"] = [
            q.model_copy(update={"scalar_text": f"({q.scalar_text})*1.001"}) if q.scalar_text else q
            for q in quantities
        ]
    elif quantities:  # statistical mechanics: every returned formula is scaled by 1.001
        updates["quantities"] = [
            q.model_copy(update={"text": f"({q.text})*1.001"}) for q in quantities
        ]
    # vector calculus: component formulas, derived equations, integral values and field samples are shifted
    components = getattr(result, "components_text", None)
    if components:
        updates["components_text"] = [f"({c}) + 1/1000" for c in components]
    equations = getattr(result, "equations_latex", None)
    if equations:
        updates["equations_latex"] = [f"{e} + \\frac{{1}}{{1000}}" for e in equations]
    integral = getattr(result, "integral", None)
    if integral is not None and integral.values:
        updates["integral"] = integral.model_copy(
            update={
                "values": [
                    v.model_copy(
                        update={
                            "text": f"({v.text})*1.001",
                            "numeric": str(mp.mpmathify(v.numeric) * mp.mpf("1.001"))
                            if v.numeric
                            else v.numeric,
                        }
                    )
                    for v in integral.values
                ]
            }
        )
    em = getattr(result, "em", None)
    if em is not None and em.items:
        updates["em"] = em.model_copy(
            update={
                "items": [
                    i.model_copy(
                        update={
                            "numeric": str(mp.mpmathify(i.numeric) * mp.mpf("1.001"))
                            if i.numeric
                            else i.numeric
                        }
                    )
                    for i in em.items
                ]
            }
        )
    solution = getattr(result, "solution", None)
    if solution is not None and getattr(solution, "paths", None):  # variational: stationary actions
        paths = [
            path.model_copy(update={"action": str(mp.mpmathify(path.action) * mp.mpf("1.001"))})
            for path in solution.paths
        ]
        updates["solution"] = solution.model_copy(update={"paths": paths})
    verification = getattr(result, "verification", None)
    if verification is not None and getattr(verification, "perturbations", None):
        perturbed = [
            r.model_copy(update={"analytic": str(mp.mpmathify(r.analytic) * mp.mpf("1.001"))})
            for r in verification.perturbations
        ]
        updates["verification"] = verification.model_copy(update={"perturbations": perturbed})
    # ODE results: closed forms and series are scaled by 1.001, numeric series likewise, and each equilibrium's
    # eigenvalues are scaled and its class flipped. The frequency of a perturbation solution is scaled too.
    ode_solutions = getattr(result, "solutions", None)
    if ode_solutions:
        scaled = []
        for item in ode_solutions:
            change: dict[str, Any] = {}
            if item.rhs_text:
                change["rhs_text"] = [f"({t})*1.001" for t in item.rhs_text]
            if item.kind == "perturbation" and len(item.equations_latex) > 1:
                lhs, _, rhs = item.equations_latex[1].partition("=")
                change["equations_latex"] = [
                    item.equations_latex[0],
                    f"{lhs}= ({rhs})\\cdot 1.001",
                    *item.equations_latex[2:],
                ]
            scaled.append(item.model_copy(update=change))
        updates["solutions"] = scaled
    ode_series = getattr(result, "series", None)
    if ode_series:
        updates["series"] = [
            s.model_copy(update={"values": [v * 1.001 for v in s.values]}) for s in ode_series
        ]
    qualitative = getattr(result, "qualitative", None)
    if qualitative is not None and qualitative.equilibria:
        flip = {"centre": "saddle", "saddle": "centre"}

        def _eigen(text: str) -> str:
            try:
                z = complex(text.replace(" ", "").replace("i", "j")) * 1.001
            except ValueError:
                return text
            return f"{z.real}{z.imag:+}i"

        shifted = [
            eq.model_copy(
                update={
                    "eigenvalues_numeric": [_eigen(v) for v in eq.eigenvalues_numeric],
                    "classification": flip.get(eq.classification, "non_hyperbolic"),
                }
            )
            for eq in qualitative.equilibria
        ]
        updates["qualitative"] = qualitative.model_copy(update={"equilibria": shifted})
    modes = getattr(result, "modes", None)
    if modes:  # normal modes of an expansion: every squared frequency off by 1/1000
        updates["modes"] = [
            m.model_copy(
                update={
                    "omega_squared_latex": (
                        rf"\omega^2 = ({m.omega_squared_latex.split('=', 1)[1]})\cdot 1.001"
                    )
                }
            )
            for m in modes
        ]
    if getattr(result, "mean_text", None):  # the bivariate Normal: shift every mean entry
        updates["mean_text"] = [f"{t} + 1/1000" for t in result.mean_text]
    if getattr(result, "covariance_text", None):
        updates["covariance_text"] = [
            [f"{t} + 1/1000" for t in row] for row in result.covariance_text
        ]
    if getattr(result, "frequency_latex", None):
        updates["frequency_latex"] = "\\omega = 2"
    if getattr(result, "restrictions_latex", None):
        updates["restrictions_latex"] = []
    if getattr(result, "constants", None):
        updates["constants"] = [
            c.model_copy(update={"expression_latex": f"{c.expression_latex} + q_1"})
            for c in result.constants
        ]
    if getattr(result, "conserved", None):
        updates["conserved"] = [
            c.model_copy(update={"expression_latex": f"{c.expression_latex} + x"})
            for c in result.conserved
        ]
    return result.model_copy(update=updates)


def _force_ok(result: Any) -> Any:
    """The same result presented as an accepted one: a refusal turned into an answer."""
    updates: dict[str, Any] = {"status": Status.OK}
    if hasattr(result, "result_text"):
        updates["result_text"] = "0"
    quantities = getattr(result, "quantities", None)
    if quantities and hasattr(
        quantities[0], "scalar_text"
    ):  # geometry: each quantity carries a status
        updates["quantities"] = [q.model_copy(update={"status": Status.OK}) for q in quantities]
    if hasattr(result, "conserved"):
        from minerva_engine.lagrangian import ConservedQuantity

        updates["conserved"] = [
            ConservedQuantity(
                name="E", kind="energy", expression_latex="E = 1", reason="forced", verified=True
            )
        ]
    return result.model_copy(update=updates)


class Mutant:
    """An engine whose results are wrong on purpose."""

    def __init__(self, engine: Engine, mutate: Any) -> None:
        self._engine = engine
        self._mutate = mutate

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._engine, name)
        if not callable(attribute):
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            result = attribute(*args, **kwargs)
            return self._mutate(result) if hasattr(result, "model_copy") else result

        return call


@pytest.fixture
def wrong_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """`solve` is called in-process by the algebra judges: make it return shifted roots, and accept anything."""
    original = algebra.solve_polynomial

    def solve(*args: Any, **kwargs: Any) -> Any:
        report = original(*args, **kwargs)
        shifted = [
            r.model_copy(update={"numeric": str(float(r.numeric) + 0.001)}) if r.real else r
            for r in report.roots
        ]
        return report.model_copy(update={"roots": shifted})

    monkeypatch.setattr(algebra, "solve_polynomial", solve)


@pytest.mark.parametrize(
    "problem",
    [p for p in PR_PROBLEMS if p.expected == "correct" and p.pending is None],
    ids=lambda p: p.id,
)
def test_no_judge_accepts_a_wrong_result(problem: Any, engine: Engine, wrong_roots: None) -> None:
    outcome, detail = problem.judge(Mutant(engine, _corrupt))
    assert outcome not in GOOD, (
        f"{problem.id}: a result shifted by 1/1000 was called {outcome} ({detail})"
    )


@pytest.mark.parametrize(
    "problem", [p for p in PR_PROBLEMS if p.expected == "refused"], ids=lambda p: p.id
)
def test_no_judge_accepts_an_answer_where_a_refusal_was_due(problem: Any, engine: Engine) -> None:
    outcome, detail = problem.judge(Mutant(engine, _force_ok))
    assert outcome == "wrong", (
        f"{problem.id}: an answer where a refusal was due was called {outcome} ({detail})"
    )
