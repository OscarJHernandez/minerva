"""Distribution-calculus benchmark: problems from mathematical physics with analytic answers.

The oracle is a hand-written result (never produced by the engine). A run measures, over the whole
case file: correctness on supported cases, how many cases are supported versus honestly refused,
retrieval of the handbook identities, how often accepted results carry passed verification, latency
(median and p95), the reasons for refusals, and whether wrong rewritings are rejected. Results are
machine-readable (``to_json``) and two runs can be compared (``compare``), so a change shows up as
a per-case difference instead of an impression.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import sympy as sp
import yaml
from minerva_engine import (
    DistributionInput,
    Engine,
    ExpressionInput,
    MinervaMathError,
    Status,
    parse_text,
)
from minerva_engine.distributions.models import Endpoint
from minerva_engine.distributions.normal_form import Refusal
from minerva_engine.distributions.pairing import verify_pairing
from pydantic import BaseModel, Field

from .kb import KnowledgeBase, SearchQuery
from .loader import DATA_DIR

CASES_FILE = "benchmark/distribution_cases.yaml"
FORMAT = 1
RETRIEVAL_LIMIT = 20

Kind = Literal["supported", "unsupported", "invalid_transform"]


class DistCase(BaseModel):
    id: str
    kind: Kind
    note: str = ""
    action: Literal["simplify", "integrate", "differentiate"]
    input: str
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    real_variable: bool = True  # False: leave the variable without the `real` declaration
    lower: str | None = None
    upper: str | None = None
    endpoint: Endpoint | None = None
    # supported
    result: str | None = None
    forbid_results: list[str] = Field(default_factory=list)
    entries: list[str] = Field(default_factory=list)
    retrieval: list[str] = Field(default_factory=list)
    # unsupported
    status: str | None = None
    reason: str | None = None
    # invalid_transform
    candidate: str | None = None


class DistCaseResult(BaseModel):
    id: str
    kind: Kind
    passed: bool
    status: str
    reason: str | None = None
    result_text: str | None = None
    cited: list[str] = Field(default_factory=list)
    verified: bool | None = None  # supported cases that returned ok: pairing and conditions passed
    problems: list[str] = Field(default_factory=list)
    retrieval_ranks: dict[str, int | None] = Field(default_factory=dict)
    milliseconds: float = 0.0


class DistReport(BaseModel):
    results: list[DistCaseResult]

    def summary(self) -> dict[str, Any]:
        def of(kind: str) -> list[DistCaseResult]:
            return [r for r in self.results if r.kind == kind]

        supported, unsupported, invalid = (
            of("supported"),
            of("unsupported"),
            of("invalid_transform"),
        )
        answered = [r for r in supported if r.status == Status.OK.value]
        times = sorted(r.milliseconds for r in self.results) or [0.0]
        ranks = [rank for r in self.results for rank in r.retrieval_ranks.values()]
        hits = [rank for rank in ranks if rank is not None]
        return {
            "cases": len(self.results),
            "cases_passed": sum(r.passed for r in self.results),
            "supported_cases": len(supported),
            "supported_correct": sum(r.passed for r in supported),
            "supported_answered": len(answered),
            "unsupported_cases": len(unsupported),
            "unsupported_refused_correctly": sum(r.passed for r in unsupported),
            "coverage": round(len(supported) / max(1, len(supported) + len(unsupported)), 4),
            "verification_success": round(
                sum(1 for r in answered if r.verified) / max(1, len(answered)), 4
            ),
            "invalid_cases": len(invalid),
            "invalid_rejected": sum(r.passed for r in invalid),
            "retrieval_expected": len(ranks),
            "retrieval_recall_at_1": _rate(hits, ranks, 1),
            "retrieval_recall_at_3": _rate(hits, ranks, 3),
            "retrieval_recall_at_5": _rate(hits, ranks, 5),
            "retrieval_mrr": round(sum(1 / h for h in hits) / max(1, len(ranks)), 4),
            "latency_median_ms": round(statistics.median(times), 1),
            "latency_p95_ms": round(_p95(times), 1),
            "refusal_reasons": dict(
                sorted(Counter(r.reason for r in self.results if r.reason).items())
            ),
            "failed_transformations": [r.id for r in self.results if not r.passed],
        }

    def to_json(self, label: str = "") -> dict[str, Any]:
        return {
            "format": FORMAT,
            "label": label,
            "summary": self.summary(),
            "cases": [r.model_dump(mode="json") for r in self.results],
        }

    @property
    def failures(self) -> list[DistCaseResult]:
        return [r for r in self.results if not r.passed]


def _rate(hits: list[int], ranks: list[int | None], k: int) -> float:
    return round(sum(1 for h in hits if h <= k) / max(1, len(ranks)), 4)


def _p95(sorted_times: list[float]) -> float:
    index = max(0, min(len(sorted_times) - 1, -(-95 * len(sorted_times) // 100) - 1))
    return sorted_times[index]


def load_distribution_cases(data_dir: Path = DATA_DIR) -> list[DistCase]:
    with (data_dir / CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{CASES_FILE} must contain a list of cases")
    cases = [DistCase(**item) for item in raw]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate distribution case ids")
    return cases


def _declared(case: DistCase) -> dict[str, list[str]]:
    declared = dict(case.assumptions)
    if case.real_variable:
        declared["x"] = sorted({*declared.get("x", []), "real"})
    return declared


def _equal(a: str, b: str, assumptions: dict[str, list[str]]) -> bool:
    try:
        return bool(
            sp.simplify(sp.expand(parse_text(a, assumptions) - parse_text(b, assumptions))) == 0
        )
    except MinervaMathError:
        return False


def _run_engine_case(case: DistCase, engine: Engine) -> DistCaseResult:
    assumptions = _declared(case)
    started = time.monotonic()
    result = engine.distribution(
        DistributionInput(
            expression=ExpressionInput(input=case.input, format="text", assumptions=assumptions),
            action=case.action,
            variable="x",
            lower=case.lower,
            upper=case.upper,
            endpoint=case.endpoint,
        )
    )
    ms = (time.monotonic() - started) * 1000
    cited = sorted({s.entry_id for s in result.steps})
    problems: list[str] = []
    verified = None
    if case.kind == "supported":
        if result.status is not Status.OK or result.result_text is None:
            problems.append(f"expected ok, got {result.status.value} ({result.reason})")
        else:
            verified = bool(
                result.verification
                and result.verification.passed
                and result.verification.conditions_established
            )
            if not verified:
                problems.append("verification did not pass")
            if case.result is not None and not _equal(result.result_text, case.result, assumptions):
                problems.append(f"result {result.result_text} != {case.result}")
            for wrong in case.forbid_results:
                if _equal(result.result_text, wrong, assumptions):
                    problems.append(f"returned the forbidden result {wrong}")
            problems.extend(f"did not cite {e}" for e in case.entries if e not in cited)
    else:  # unsupported
        if result.status.value != case.status:
            problems.append(f"expected {case.status}, got {result.status.value}")
        if result.reason != case.reason:
            problems.append(f"expected reason {case.reason}, got {result.reason}")
        if result.result_text is not None:
            problems.append("a result was shown")
    return DistCaseResult(
        id=case.id,
        kind=case.kind,
        passed=not problems,
        status=result.status.value,
        reason=result.reason,
        result_text=result.result_text,
        cited=cited,
        verified=verified,
        problems=problems,
        milliseconds=round(ms, 1),
    )


def _bound(text: str | None) -> sp.Expr:
    if text is None:
        raise ValueError("an integrate case needs both limits")
    return parse_text(text)


def _run_invalid_case(case: DistCase) -> DistCaseResult:
    """The wrong rewriting must be rejected by the test-function pairing."""
    assumptions = _declared(case)
    started = time.monotonic()
    original = parse_text(case.input, assumptions)
    candidate = parse_text(case.candidate or "0", assumptions)
    x = next((s for s in original.free_symbols if s.name == "x"), sp.Symbol("x", real=True))
    bounds = (_bound(case.lower), _bound(case.upper)) if case.action == "integrate" else None
    try:
        passed = verify_pairing(
            original,
            candidate,
            x,
            action=case.action,
            bounds=bounds,
            endpoint=case.endpoint,
            deadline=time.monotonic() + 60,
        ).passed
    except Refusal:
        passed = False
    return DistCaseResult(
        id=case.id,
        kind=case.kind,
        passed=not passed,
        status="rejected" if not passed else "accepted",
        problems=["the invalid rewriting was accepted"] if passed else [],
        milliseconds=round((time.monotonic() - started) * 1000, 1),
    )


def _retrieval_ranks(case: DistCase, kb: KnowledgeBase, engine: Engine) -> dict[str, int | None]:
    if not case.retrieval:
        return {}
    status, _, features, _ = engine.features(
        ExpressionInput(input=case.input, format="text", assumptions=_declared(case))
    )
    if status is not Status.OK:
        return {entry_id: None for entry_id in case.retrieval}
    ordered = [
        e.id
        for e in kb.search(
            SearchQuery(features=[*features, f"distribution:{case.action}"], limit=RETRIEVAL_LIMIT)
        )
    ]
    return {
        entry_id: (ordered.index(entry_id) + 1 if entry_id in ordered else None)
        for entry_id in case.retrieval
    }


def run_distribution_benchmark(
    kb: KnowledgeBase, engine: Engine, cases: list[DistCase] | None = None
) -> DistReport:
    results: list[DistCaseResult] = []
    for case in cases or load_distribution_cases():
        if case.kind == "invalid_transform":
            result = _run_invalid_case(case)
        else:
            result = _run_engine_case(case, engine)
            result.retrieval_ranks = _retrieval_ranks(case, kb, engine)
        results.append(result)
    return DistReport(results=results)


def format_distribution_report(report: DistReport) -> str:
    s = report.summary()
    lines = [
        f"cases passed:            {s['cases_passed']}/{s['cases']}",
        f"supported, correct:      {s['supported_correct']}/{s['supported_cases']} "
        f"(coverage {s['coverage']:.0%} of supported + refused cases)",
        f"refused as designed:     {s['unsupported_refused_correctly']}/{s['unsupported_cases']}",
        f"invalid rewrites caught: {s['invalid_rejected']}/{s['invalid_cases']}",
        f"verification success:    {s['verification_success']:.0%} of accepted results",
        f"retrieval recall@1/3/5:  {s['retrieval_recall_at_1']:.0%} / "
        f"{s['retrieval_recall_at_3']:.0%} / {s['retrieval_recall_at_5']:.0%} "
        f"(MRR {s['retrieval_mrr']:.3f}, {s['retrieval_expected']} expected entries)",
        f"latency median / p95:    {s['latency_median_ms']:.0f} ms / {s['latency_p95_ms']:.0f} ms",
        f"refusal reasons:         {s['refusal_reasons']}",
    ]
    for r in report.failures:
        lines.append(f"FAIL {r.id}: {'; '.join(r.problems)}")
    return "\n".join(lines)


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> tuple[str, bool]:
    """A concise comparison of two ``to_json`` documents; the flag is True if anything regressed."""
    if baseline.get("format") != current.get("format"):
        return "benchmark formats differ; regenerate the baseline", True
    before = {c["id"]: c for c in baseline["cases"]}
    after = {c["id"]: c for c in current["cases"]}
    regressions: list[str] = []
    improvements: list[str] = []
    changes: list[str] = []
    for case_id, new in after.items():
        old = before.get(case_id)
        if old is None:
            changes.append(f"new case {case_id}: {'pass' if new['passed'] else 'FAIL'}")
        elif old["passed"] and not new["passed"]:
            regressions.append(f"REGRESSION {case_id}: {'; '.join(new['problems'])}")
        elif not old["passed"] and new["passed"]:
            improvements.append(f"fixed {case_id}")
        elif old["status"] != new["status"] or old["result_text"] != new["result_text"]:
            changes.append(
                f"changed {case_id}: {old['status']} {old['result_text']} -> "
                f"{new['status']} {new['result_text']}"
            )
    changes.extend(f"removed case {case_id}" for case_id in before if case_id not in after)
    lines = ["metric                      baseline   current"]
    for key, new_value in current["summary"].items():
        old_value = baseline["summary"].get(key)
        if isinstance(new_value, int | float) and isinstance(old_value, int | float):
            marker = "" if old_value == new_value else "  <-"
            lines.append(f"{key:<26} {old_value!s:>9} {new_value!s:>9}{marker}")
    lines.extend(regressions + improvements + changes)
    if len(lines) == 1 + sum(
        1 for k, v in current["summary"].items() if isinstance(v, int | float)
    ):
        lines.append("no per-case differences")
    return "\n".join(lines), bool(regressions)
