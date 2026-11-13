"""Benchmark for controlled symbolic transformations: substitute, collect, isolate, rewrite one part.

The cases (`data/benchmark/step_cases.yaml`) are worked by hand and separate from implementation tests.
Two adapters run the same cases:

- `existing`: the nearest operation Minerva had before these transformations (simplify, expand or factor
  goals, the polynomial solver, applying a handbook identity to the whole expression). This is the
  recorded baseline: it shows what the old functionality could not do, honestly, including the cases
  where it happens to give the requested answer.
- `steps`: the new operations.

A case passes when the status is the expected one and, for a result, the expected form was produced
(`match: structural` is the requested form, `equivalent` is the same function by a numeric check). That
check scores the output against the hand-worked answer; it is never how the engine decides a result.
Latency and effort are measured only by explicit request (`--repeat`): CI asserts correctness, never time.
"""

from __future__ import annotations

import ast
import contextlib
import json
import operator
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import sympy as sp
import yaml
from minerva_engine import (
    Applicability,
    Engine,
    ExpressionInput,
    MinervaMathError,
    parse_latex,
    parse_text,
)
from minerva_engine.verify import verify_equal
from pydantic import BaseModel, Field, model_validator

from .identities import simplify_expression, transform_expression
from .kb import KnowledgeBase
from .loader import DATA_DIR

STEP_CASES_FILE = "benchmark/step_cases.yaml"
BASELINE_FILE = "benchmark/step_baseline.json"

Operation = Literal["substitute", "collect", "isolate", "rewrite"]


class StepCase(BaseModel):
    id: str
    operation: Operation
    domain: Literal["algebra", "physics", "applied", "operators"] = "algebra"
    tags: list[str] = Field(default_factory=list)
    input: str
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    relations: list[str] = Field(default_factory=list)
    order: str | None = None
    variable: str | None = None
    by: list[str] = Field(default_factory=list)
    part: str | None = None
    occurrence: int = 1
    action: str | None = None
    rule_id: str | None = None
    reverse: bool = False
    status: str = "ok"
    results: list[str] = Field(default_factory=list)
    solutions: list[str] | None = None
    match: Literal["equivalent", "structural"] = "equivalent"
    conditions: list[str] = Field(default_factory=list)
    note: str = ""
    pending: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> StepCase:
        if self.operation == "isolate":
            if self.variable is None:
                raise ValueError(f"{self.id}: isolate needs a variable")
            if self.status == "ok" and self.solutions is None:
                raise ValueError(
                    f"{self.id}: an ok isolate case lists its solutions (possibly none)"
                )
        else:
            if self.solutions is not None:
                raise ValueError(f"{self.id}: only isolate cases have solutions")
            if self.status == "ok" and not self.results:
                raise ValueError(f"{self.id}: an ok case lists its accepted results")
        if self.operation == "substitute" and not self.relations:
            raise ValueError(f"{self.id}: substitute needs relations")
        if self.operation == "collect" and not self.by:
            raise ValueError(f"{self.id}: collect needs targets")
        if self.operation == "rewrite" and (self.part is None or self.action is None):
            raise ValueError(f"{self.id}: rewrite needs a part and an action")
        if (self.action == "rule") != (self.rule_id is not None):
            raise ValueError(f"{self.id}: rule_id goes with action: rule")
        return self


@dataclass
class StepOutcome:
    """What an adapter produced for one case. Effort fields are zero for the baseline."""

    status: str
    results: list[str] = field(default_factory=list)
    message: str = ""
    steps: int = 0
    candidates: int = 0
    rejected: int = 0
    verify_ms: int = 0
    conditions: list[str] = field(default_factory=list)


class StepCaseResult(BaseModel):
    case: StepCase
    status: str
    results: list[str]
    matched: bool
    passed: bool
    milliseconds: list[int]
    steps: int = 0
    candidates: int = 0
    rejected: int = 0
    verify_ms: int = 0
    message: str = ""

    @property
    def refusal(self) -> bool:
        return self.case.status != "ok"


Adapter = Callable[[KnowledgeBase, Engine, StepCase], StepOutcome]


def load_step_cases(data_dir: Path = DATA_DIR) -> list[StepCase]:
    with (data_dir / STEP_CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{STEP_CASES_FILE} must contain a list of cases")
    cases = [StepCase(**item) for item in raw]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate step case ids")
    return cases


# --- scoring ---------------------------------------------------------------------------------------
def _nc_names(assumptions: dict[str, list[str]]) -> set[str]:
    return {k for k, v in assumptions.items() if "noncommutative" in v}


_NC_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def _read_noncommutative(text: str, nc: set[str]) -> sp.Expr:
    """Read a polynomial in some noncommutative symbols (sums, products, differences, small integer
    powers, numeric division) by walking the syntax tree: nothing is evaluated, and factor order is
    kept, which a commutative parse would lose. Scoring only; the engine has its own parser."""

    def walk(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return sp.Rational(str(node.value))
        if isinstance(node, ast.Name):
            return sp.Symbol(node.id, commutative=node.id not in nc)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            if type(node.op) in _NC_BINARY:
                return _NC_BINARY[type(node.op)](walk(node.left), walk(node.right))
            if isinstance(node.op, ast.Div):
                return walk(node.left) / walk(node.right)
            if isinstance(node.op, ast.Pow):
                exponent = walk(node.right)
                if exponent.is_Integer and 0 <= exponent <= 8:
                    return walk(node.left) ** exponent
        raise ValueError("outside the noncommutative polynomial subset")

    return walk(ast.parse(text, mode="eval"))


def _forms_equal(a: str, b: str, case: StepCase) -> bool:
    """Same tree (`structural`) or same function (`equivalent`) as the hand-worked answer."""
    nc = _nc_names(case.assumptions)
    if nc:
        try:
            x, y = _read_noncommutative(a, nc), _read_noncommutative(b, nc)
        except (ValueError, SyntaxError):
            return False
        if x == y:
            return True
        return case.match == "equivalent" and bool(sp.expand(x - y) == 0)
    try:
        x = parse_text(a, case.assumptions)
        y = parse_text(b, case.assumptions)
    except MinervaMathError:
        return False
    if x == y:
        return True
    if case.match == "structural":
        return False
    return bool(verify_equal(x, y, symbolic=False).passed)


def _same_set(got: list[str], expected: list[str], case: StepCase) -> bool:
    """Every expected solution matched by a distinct returned one, and nothing extra."""
    if len(got) != len(expected):
        return False
    unused = list(got)
    for want in expected:
        hit = next((g for g in unused if _forms_equal(g, want, case)), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


def score(case: StepCase, outcome: StepOutcome) -> tuple[bool, bool]:
    """(matched the expected result, passed). A refusal case has nothing to match."""
    if outcome.status != case.status:
        return False, False
    if case.status != "ok":
        return True, True
    if case.operation == "isolate":
        ok = _same_set(outcome.results, case.solutions or [], case)
    else:
        ok = any(_forms_equal(r, want, case) for r in outcome.results for want in case.results)
    return ok, ok


# --- the baseline: nearest existing operation -------------------------------------------------------
_RULE_STATUS = {
    Applicability.APPLICABLE: "ok",
    Applicability.CONDITIONS_NOT_SATISFIED: "conditions_not_satisfied",
    Applicability.NOT_VERIFIED: "not_verified",
    Applicability.NO_MATCH: "unchanged",
}


def run_existing(kb: KnowledgeBase, engine: Engine, case: StepCase) -> StepOutcome:
    """What Minerva could do before: no relations, no part selection, no collect, no isolate for anything
    but polynomials. Each case is given to the closest operation as it stood, and whatever it returns is
    scored with the same rules as the new operations."""
    try:
        expression = ExpressionInput(input=case.input, assumptions=case.assumptions)
    except ValueError:  # `noncommutative` is not a declaration the old operations know
        return StepOutcome("unsupported", message="noncommutative symbols are not supported")

    if case.operation == "isolate":
        report = engine.solve(case.input, case.variable, case.assumptions)
        roots: list[str] = []
        for factor in report.factors:
            for latex in factor.exact_roots_latex:
                with contextlib.suppress(MinervaMathError):
                    roots.append(str(parse_latex(latex, case.assumptions)))
        return StepOutcome(report.status.value, roots, report.message)

    if case.operation == "rewrite" and case.action == "rule":
        entry = kb.get(case.rule_id or "")
        if entry is None or entry.rule is None:
            return StepOutcome("unsupported", message="unknown rule")
        applied = engine.apply_rules(expression, [entry.rule]).applications
        if not applied:
            return StepOutcome("unsupported")
        first = applied[0]
        text = [first.rewritten_text] if first.rewritten_text else []
        return StepOutcome(_RULE_STATUS[first.applicability], text, first.message)

    # collect, and rewrite with a builtin: only the whole-expression goals existed. Take every goal's
    # answer, so the baseline is as generous as it can be.
    goals = ["simplify", "expand", "factor"]
    if case.operation == "rewrite" and case.action in {"expand", "factor", "simplify"}:
        goals = [case.action]
    elif case.operation == "rewrite":
        return StepOutcome("unsupported", message=f"no whole-expression {case.action}")
    texts: list[str] = []
    first_status = "unchanged"
    for index, goal in enumerate(goals):
        if goal == "simplify":
            response = simplify_expression(kb, engine, expression)
        else:
            response = transform_expression(kb, engine, expression, goal)
        if index == 0:
            first_status = response.status.value
        if response.status.value == "ok" and response.result_text:
            texts.append(response.result_text)
    return StepOutcome("ok" if texts else first_status, texts)


ADAPTERS: dict[str, Adapter] = {"existing": run_existing}


# --- running and reporting --------------------------------------------------------------------------
def run_step_case(
    kb: KnowledgeBase, engine: Engine, case: StepCase, adapter: Adapter, repeat: int = 1
) -> StepCaseResult:
    timings: list[int] = []
    outcome = StepOutcome("internal_error")
    for _ in range(max(1, repeat)):
        started = time.monotonic()
        outcome = adapter(kb, engine, case)
        timings.append(int((time.monotonic() - started) * 1000))
    matched, passed = score(case, outcome)
    return StepCaseResult(
        case=case,
        status=outcome.status,
        results=outcome.results,
        matched=matched,
        passed=passed,
        milliseconds=timings,
        steps=outcome.steps,
        candidates=outcome.candidates,
        rejected=outcome.rejected,
        verify_ms=outcome.verify_ms,
        message=outcome.message,
    )


class StepReport(BaseModel):
    adapter: str
    results: list[StepCaseResult]

    @property
    def scored(self) -> list[StepCaseResult]:
        return [r for r in self.results if not r.case.pending]

    @property
    def failures(self) -> list[StepCaseResult]:
        return [r for r in self.scored if not r.passed]

    def solved_ids(self) -> dict[str, bool]:
        return {r.case.id: r.passed for r in self.results}

    def summary(self) -> dict[str, float]:
        scored = self.scored
        positive = [r for r in scored if not r.refusal]
        refusals = [r for r in scored if r.refusal]
        times = [t for r in self.results for t in r.milliseconds] or [0]
        ordered = sorted(times)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        return {
            "cases": len(scored),
            "passed": sum(r.passed for r in scored),
            "positive_cases": len(positive),
            "positive_passed": sum(r.passed for r in positive),
            "refusal_cases": len(refusals),
            "refusal_passed": sum(r.passed for r in refusals),
            "wrong_status_on_ok": sum(1 for r in positive if r.status != "ok"),
            "wrong_result": sum(1 for r in positive if r.status == "ok" and not r.matched),
            "pending_cases": len(self.results) - len(scored),
            "pending_now_satisfied": sum(r.passed for r in self.results if r.case.pending),
            "resource_limit": sum(1 for r in self.results if r.status == "resource_limit"),
            "latency_median_ms": statistics.median(times),
            "latency_p95_ms": p95,
            "latency_max_ms": max(times),
            "candidates": sum(r.candidates for r in self.results),
            "rejected": sum(r.rejected for r in self.results),
            "verify_ms": sum(r.verify_ms for r in self.results),
        }

    def by_operation(self) -> dict[str, tuple[int, int]]:
        groups: dict[str, list[StepCaseResult]] = defaultdict(list)
        for r in self.scored:
            groups[r.case.operation].append(r)
        return {k: (sum(r.passed for r in v), len(v)) for k, v in sorted(groups.items())}


def run_step_benchmark(
    kb: KnowledgeBase,
    engine: Engine,
    adapter_name: str = "existing",
    cases: list[StepCase] | None = None,
    repeat: int = 1,
) -> StepReport:
    adapter = ADAPTERS[adapter_name]
    return StepReport(
        adapter=adapter_name,
        results=[
            run_step_case(kb, engine, c, adapter, repeat) for c in (cases or load_step_cases())
        ],
    )


def format_step_report(report: StepReport, *, timing: bool = False) -> str:
    lines = []
    for r in report.results:
        tag = "pend" if r.case.pending else ("ok  " if r.passed else "FAIL")
        detail = []
        if r.status != r.case.status:
            detail.append(f"status {r.status}, wanted {r.case.status}")
        elif r.case.status == "ok" and not r.matched:
            detail.append(f"got {r.results}")
        suffix = f" {max(r.milliseconds)}ms" if timing else ""
        lines.append(f"{tag} {r.case.id:<44} {r.case.operation:<10}{suffix} {'; '.join(detail)}")
    s = report.summary()
    lines += [
        "",
        f"adapter:                  {report.adapter}",
        f"cases passed:             {int(s['passed'])}/{int(s['cases'])} "
        f"(results {int(s['positive_passed'])}/{int(s['positive_cases'])}, "
        f"refusals {int(s['refusal_passed'])}/{int(s['refusal_cases'])})",
        "by operation:             "
        + ", ".join(f"{k} {a}/{b}" for k, (a, b) in report.by_operation().items()),
        f"result cases: wrong status {int(s['wrong_status_on_ok'])}, "
        f"wrong result {int(s['wrong_result'])}",
        f"pending cases:            {int(s['pending_cases'])} "
        f"({int(s['pending_now_satisfied'])} already satisfied)",
        f"resource limits hit:      {int(s['resource_limit'])}",
    ]
    if timing:
        lines += [
            f"latency median/p95/max:   {s['latency_median_ms']:.0f} / {s['latency_p95_ms']:.0f} / "
            f"{s['latency_max_ms']:.0f} ms",
            f"search effort:            {int(s['candidates'])} candidates, {int(s['rejected'])} rejected, "
            f"{int(s['verify_ms'])} ms verifying",
        ]
    return "\n".join(lines)


# --- the recorded baseline ----------------------------------------------------------------------------
def load_baseline(data_dir: Path = DATA_DIR) -> dict[str, bool]:
    with (data_dir / BASELINE_FILE).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {str(k): bool(v) for k, v in raw["passed"].items()}


def baseline_json(report: StepReport) -> str:
    """Per-case pass/fail of the existing functionality: no timings, so the file is reproducible."""
    body = {
        "adapter": report.adapter,
        "note": "Generated by `python -m minerva_knowledge step-benchmark --write-baseline` before the "
        "transformation operations existed. Pass/fail only; see docs/transformations-benchmark.md.",
        "passed": dict(sorted(report.solved_ids().items())),
    }
    return json.dumps(body, indent=2) + "\n"
