"""Identity-finder benchmark: hand-written expression -> expected / forbidden identities.

The cases are written by a person, independently of any generator, so that adding handbook content
or changing retrieval can be measured instead of assumed to help. A case passes when every `expect`
entry is APPLICABLE and no `forbid` entry is APPLICABLE. `pending` cases describe identities the
handbook is known not to have yet: they are reported but never fail a run.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import yaml
from minerva_engine import Applicability, Engine, ExpressionInput, Status, parse_text
from pydantic import BaseModel, Field

from .identities import chain_candidates, find_identities, transform_expression
from .kb import KnowledgeBase
from .loader import DATA_DIR

CASES_FILE = "benchmark/identity_cases.yaml"
TRANSFORM_CASES_FILE = "benchmark/transform_cases.yaml"


class Case(BaseModel):
    id: str
    input: str
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    expect: list[str] = Field(default_factory=list)
    forbid: list[str] = Field(default_factory=list)
    pending: bool = False


class CaseResult(BaseModel):
    case: Case
    status: str
    applicable: list[str]
    missing: list[str]  # expected but not applicable
    forbidden_shown: list[str]  # forbidden but applicable
    candidates: int
    milliseconds: int

    @property
    def passed(self) -> bool:
        return self.status == Status.OK.value and not self.missing and not self.forbidden_shown


class Report(BaseModel):
    results: list[CaseResult]

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if not r.case.pending]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.scored if not r.passed]

    def summary(self) -> dict[str, float]:
        scored = self.scored
        expected = sum(len(r.case.expect) for r in scored)
        found = sum(len(r.case.expect) - len(r.missing) for r in scored)
        forbidden = sum(len(r.forbidden_shown) for r in scored)
        times = [r.milliseconds for r in self.results] or [0]
        pending = [r for r in self.results if r.case.pending]
        return {
            "cases": len(scored),
            "cases_passed": sum(r.passed for r in scored),
            "expected_found": found,
            "expected_total": expected,
            "forbidden_shown": forbidden,
            "mean_applicable": statistics.fmean(len(r.applicable) for r in self.results)
            if self.results
            else 0.0,
            "pending_cases": len(pending),
            "pending_now_satisfied": sum(r.passed for r in pending),
            "pending_forbidden_shown": sum(len(r.forbidden_shown) for r in pending),
            "pending_expected_missing": sum(len(r.missing) for r in pending),
            "latency_median_ms": statistics.median(times),
            "latency_max_ms": max(times),
        }


def load_cases(data_dir: Path = DATA_DIR) -> list[Case]:
    with (data_dir / CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{CASES_FILE} must contain a list of cases")
    cases = [Case(**item) for item in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate benchmark case ids")
    return cases


def run_case(kb: KnowledgeBase, engine: Engine, case: Case) -> CaseResult:
    started = time.monotonic()
    response = find_identities(
        kb, engine, ExpressionInput(input=case.input, assumptions=case.assumptions)
    )
    elapsed = int((time.monotonic() - started) * 1000)
    applicable = [
        m.entry.id
        for m in response.matches
        if m.application.applicability is Applicability.APPLICABLE
    ]
    return CaseResult(
        case=case,
        status=response.status.value,
        applicable=applicable,
        missing=[e for e in case.expect if e not in applicable],
        forbidden_shown=[e for e in case.forbid if e in applicable],
        candidates=response.candidates_considered,
        milliseconds=elapsed,
    )


def run_benchmark(kb: KnowledgeBase, engine: Engine, cases: list[Case] | None = None) -> Report:
    return Report(results=[run_case(kb, engine, c) for c in (cases or load_cases())])


def format_report(report: Report) -> str:
    lines = []
    for r in report.results:
        tag = "pend" if r.case.pending else ("ok  " if r.passed else "FAIL")
        detail = []
        if r.missing:
            detail.append(f"missing {r.missing}")
        if r.forbidden_shown:
            detail.append(f"forbidden shown {r.forbidden_shown}")
        if r.status != Status.OK.value:
            detail.append(f"status {r.status}")
        lines.append(
            f"{tag} {r.case.id:<34} applicable={len(r.applicable):<2} "
            f"cand={r.candidates:<3} {r.milliseconds:>5}ms {'; '.join(detail)}"
        )
    s = report.summary()
    lines += [
        "",
        f"cases passed:            {int(s['cases_passed'])}/{int(s['cases'])}",
        f"expected found:          {int(s['expected_found'])}/{int(s['expected_total'])}",
        f"forbidden shown:         {int(s['forbidden_shown'])}",
        f"mean applicable matches: {s['mean_applicable']:.2f}",
        f"pending cases:           {int(s['pending_cases'])} "
        f"({int(s['pending_now_satisfied'])} already satisfied)",
        f"pending: forbidden shown {int(s['pending_forbidden_shown'])}, "
        f"expected still missing {int(s['pending_expected_missing'])}",
        f"latency median / max:    {s['latency_median_ms']:.0f} / {s['latency_max_ms']:.0f} ms",
    ]
    return "\n".join(lines)


# --- retrieval quality ------------------------------------------------------------------------
class RetrievalReport(BaseModel):
    """Where the expected identities rank among the candidates retrieval proposes.

    Retrieval never decides correctness, but a rule that is not proposed can never be applied, and
    a late rank is wasted engine time. Uses the same cases as `run_benchmark` (expected identities
    only, pending cases left out).
    """

    ranks: dict[
        str, int | None
    ]  # "case_id/entry_id" -> 1-based rank among candidates, None if absent
    pool_sizes: list[int]
    chain_pool_sizes: list[int]
    chain_pool_missing: list[str]  # expected entries the chain search would not even be offered

    def summary(self) -> dict[str, float]:
        ranks = list(self.ranks.values())
        total = len(ranks) or 1

        def at(k: int) -> float:
            return sum(1 for r in ranks if r is not None and r <= k) / total

        return {
            "expected_total": len(ranks),
            "in_pool": sum(1 for r in ranks if r is not None) / total,
            "recall_at_1": at(1),
            "recall_at_3": at(3),
            "recall_at_5": at(5),
            "recall_at_10": at(10),
            "mrr": sum(1 / r for r in ranks if r is not None) / total,
            "mean_pool": statistics.fmean(self.pool_sizes) if self.pool_sizes else 0.0,
            "mean_chain_pool": statistics.fmean(self.chain_pool_sizes)
            if self.chain_pool_sizes
            else 0.0,
            "chain_pool_missing": len(self.chain_pool_missing),
        }


def run_retrieval(
    kb: KnowledgeBase, engine: Engine, cases: list[Case] | None = None
) -> RetrievalReport:
    ranks: dict[str, int | None] = {}
    pools: list[int] = []
    chain_pools: list[int] = []
    chain_missing: list[str] = []
    for case in cases or load_cases():
        if case.pending or not case.expect:
            continue
        status, _, features, _ = engine.features(
            ExpressionInput(input=case.input, assumptions=case.assumptions)
        )
        if status is not Status.OK:
            continue
        ordered = [e.id for e in kb.candidate_rules(features, limit=200)]
        pools.append(len(ordered))
        in_chain = {e.id for e in chain_candidates(kb, features, [], "", 60)}
        chain_pools.append(len(in_chain))
        for entry_id in case.expect:
            ranks[f"{case.id}/{entry_id}"] = (
                ordered.index(entry_id) + 1 if entry_id in ordered else None
            )
            if entry_id not in in_chain:
                chain_missing.append(f"{case.id}/{entry_id}")
    return RetrievalReport(
        ranks=ranks,
        pool_sizes=pools,
        chain_pool_sizes=chain_pools,
        chain_pool_missing=chain_missing,
    )


def format_retrieval(report: RetrievalReport) -> str:
    s = report.summary()
    return "\n".join(
        [
            f"expected identities:     {int(s['expected_total'])}",
            f"in the candidate pool:   {s['in_pool']:.0%}",
            f"recall@1 / @3 / @5 / @10: {s['recall_at_1']:.0%} / {s['recall_at_3']:.0%} / "
            f"{s['recall_at_5']:.0%} / {s['recall_at_10']:.0%}",
            f"mean reciprocal rank:    {s['mrr']:.3f}",
            f"mean pool size:          {s['mean_pool']:.1f} (chain search: {s['mean_chain_pool']:.1f})",
            f"missing from chain pool: {int(s['chain_pool_missing'])}",
        ]
    )


# --- multi-step transformations -------------------------------------------------------------------
class TransformCase(BaseModel):
    """An input, a goal and what a verified chain must (or must not) look like.

    `results` are accepted result forms in text syntax, compared structurally with the engine's
    result (a different but equal form is a failure: it measures the ranking, not correctness,
    which the engine's verification already guarantees). `expect_rules` must all appear among the
    chain's steps; `forbid_rules` must appear in no chain.
    """

    id: str
    input: str
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    goal: str = "simplify"
    target: list[str] = Field(default_factory=list)
    status: str = "ok"
    results: list[str] = Field(default_factory=list)
    expect_rules: list[str] = Field(default_factory=list)
    forbid_rules: list[str] = Field(default_factory=list)
    pending: bool = False


class TransformCaseResult(BaseModel):
    case: TransformCase
    status: str
    result_text: str | None
    rules: list[str]
    result_ok: bool
    missing_rules: list[str]
    forbidden_used: list[str]
    baseline_ok: bool | None  # SymPy simplify alone, simplify-goal cases only
    steps: int
    milliseconds: int

    @property
    def passed(self) -> bool:
        return (
            self.status == self.case.status
            and (self.case.status != "ok" or self.result_ok)
            and not self.missing_rules
            and not self.forbidden_used
        )


class TransformReport(BaseModel):
    results: list[TransformCaseResult]

    @property
    def scored(self) -> list[TransformCaseResult]:
        return [r for r in self.results if not r.case.pending]

    @property
    def failures(self) -> list[TransformCaseResult]:
        return [r for r in self.scored if not r.passed]

    def summary(self) -> dict[str, float]:
        scored = self.scored
        chain = [r for r in scored if r.case.status == "ok" and r.case.goal == "simplify"]
        times = [r.milliseconds for r in self.results] or [0]
        return {
            "cases": len(scored),
            "cases_passed": sum(r.passed for r in scored),
            "ok_cases": sum(1 for r in scored if r.case.status == "ok"),
            "ok_passed": sum(r.passed for r in scored if r.case.status == "ok"),
            "negative_cases": sum(1 for r in scored if r.case.status != "ok"),
            "negative_passed": sum(r.passed for r in scored if r.case.status != "ok"),
            "simplify_cases": len(chain),
            "chain_matches": sum(r.passed for r in chain),
            "sympy_only_matches": sum(1 for r in chain if r.baseline_ok),
            "cited_identities": sum(len(set(r.rules)) for r in scored),
            "pending_cases": len(self.results) - len(scored),
            "pending_now_satisfied": sum(r.passed for r in self.results if r.case.pending),
            "latency_median_ms": statistics.median(times),
            "latency_max_ms": max(times),
        }


def load_transform_cases(data_dir: Path = DATA_DIR) -> list[TransformCase]:
    with (data_dir / TRANSFORM_CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{TRANSFORM_CASES_FILE} must contain a list of cases")
    cases = [TransformCase(**item) for item in raw]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate transform case ids")
    return cases


def _same_expression(a: str, b: str) -> bool:
    try:
        return bool(parse_text(a) == parse_text(b))
    except Exception:  # an unreadable form simply does not match
        return False


def _matches_any(result_text: str | None, accepted: list[str]) -> bool:
    return result_text is not None and any(_same_expression(result_text, r) for r in accepted)


def run_transform_case(
    kb: KnowledgeBase, engine: Engine, case: TransformCase
) -> TransformCaseResult:
    started = time.monotonic()
    expression = ExpressionInput(input=case.input, assumptions=case.assumptions)
    response = transform_expression(kb, engine, expression, case.goal, case.target)
    elapsed = int((time.monotonic() - started) * 1000)
    chains = [response.steps, *(c.steps for c in response.alternatives)]
    used = [s.name for steps in chains for s in steps if s.kind == "rule"]
    shown = [s.name for s in response.steps if s.kind == "rule"]
    baseline: bool | None = None
    if case.goal == "simplify" and case.status == "ok":
        plain = engine.simplify(expression)
        baseline = plain.status is Status.OK and _matches_any(plain.result_text, case.results)
    return TransformCaseResult(
        case=case,
        status=response.status.value,
        result_text=response.result_text,
        rules=shown,
        result_ok=response.status is Status.OK and _matches_any(response.result_text, case.results),
        missing_rules=[r for r in case.expect_rules if r not in shown],
        forbidden_used=[r for r in case.forbid_rules if r in used],
        baseline_ok=baseline,
        steps=len(response.steps),
        milliseconds=elapsed,
    )


def run_transform_benchmark(
    kb: KnowledgeBase, engine: Engine, cases: list[TransformCase] | None = None
) -> TransformReport:
    return TransformReport(
        results=[run_transform_case(kb, engine, c) for c in (cases or load_transform_cases())]
    )


def format_transform_report(report: TransformReport) -> str:
    lines = []
    for r in report.results:
        tag = "pend" if r.case.pending else ("ok  " if r.passed else "FAIL")
        detail = []
        if r.status != r.case.status:
            detail.append(f"status {r.status}, wanted {r.case.status}")
        if r.case.status == "ok" and not r.result_ok:
            detail.append(f"result {r.result_text}")
        if r.missing_rules:
            detail.append(f"missing {r.missing_rules}")
        if r.forbidden_used:
            detail.append(f"forbidden used {r.forbidden_used}")
        lines.append(
            f"{tag} {r.case.id:<34} goal={r.case.goal:<8} steps={r.steps} "
            f"{r.milliseconds:>5}ms {'; '.join(detail)}"
        )
    s = report.summary()
    lines += [
        "",
        f"cases passed:             {int(s['cases_passed'])}/{int(s['cases'])} "
        f"(chains {int(s['ok_passed'])}/{int(s['ok_cases'])}, "
        f"negative {int(s['negative_passed'])}/{int(s['negative_cases'])})",
        f"simplify goal, expected form: chain search {int(s['chain_matches'])}/"
        f"{int(s['simplify_cases'])}, SymPy simplify alone {int(s['sympy_only_matches'])}/"
        f"{int(s['simplify_cases'])}",
        f"identities cited:         {int(s['cited_identities'])}",
        f"pending cases:            {int(s['pending_cases'])} "
        f"({int(s['pending_now_satisfied'])} already satisfied)",
        f"latency median / max:     {s['latency_median_ms']:.0f} / {s['latency_max_ms']:.0f} ms",
    ]
    return "\n".join(lines)
