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
from minerva_engine import Applicability, Engine, ExpressionInput, Status
from pydantic import BaseModel, Field

from .identities import find_identities
from .kb import KnowledgeBase
from .loader import DATA_DIR

CASES_FILE = "benchmark/identity_cases.yaml"


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
