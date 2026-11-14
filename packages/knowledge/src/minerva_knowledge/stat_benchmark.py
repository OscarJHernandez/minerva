"""Statistical derivation benchmark: hand-derived reference answers, adversarial cases, a baseline.

The cases (`data/benchmark/stat_cases.yaml`) were written before the operations they exercise and
are never derived from an engine run. Three kinds:

* ``supported``: the operation must answer ``ok`` with passed verification, and each named output must
  agree with the reference at concrete data (`stat_oracle`, which shares no code with the engine);
* ``unsupported``: the operation must refuse with the stated status (and reason) and show nothing;
* ``invalid_candidate``: asked to check a wrong answer to the request, the operation must reject it,
  while a correct ``control`` answer to the same request is accepted (so "always reject" fails).

A request the engine cannot express yet (no input model, no such field, no such method) is reported as
``not_expressible``. ``pending`` cases are known gaps: reported, never asserted, and a pending case
that passes is stale and fails the suite, as in the matrix benchmark.

Timings are measurements on one machine and are never asserted. Peak memory is not measured per case:
every operation runs in a sandboxed child process whose address space is capped by the engine's
limits, and that cap is what the benchmark relies on.
"""

from __future__ import annotations

import copy
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import minerva_engine
import yaml
from minerva_engine import Engine
from pydantic import BaseModel, Field, ValidationError

from .loader import DATA_DIR
from .stat_oracle import OracleError, environments, equal_texts

CASES_FILE = "benchmark/stat_cases.yaml"
FORMAT = 1
NOT_EXPRESSIBLE = "not_expressible"
REJECTED = {"not_verified", "conditions_not_satisfied"}
UNKNOWN_OPTION = {"extra_forbidden", "literal_error", "enum"}
GROUPS = ("sums", "expectation", "families", "mle", "fisher", "posterior", "matrix")

# operation -> (engine input model, Engine method)
OPERATIONS: dict[str, tuple[str, str]] = {
    "sums": ("SumsInput", "sums"),
    "expectation": ("ExpectationInput", "expectation"),
    "statistical_model": ("StatModelInput", "statistical_model"),
    "matrix_calculus": ("MatrixCalculusInput", "matrix_calculus"),
}

# The five derivations the work must complete (the "minimum successful outcome").
MINIMUM_OUTCOME = (
    "mle-gaussian-mean-and-variance",
    "mle-bernoulli",
    "posterior-beta-binomial",
    "fisher-normal-mean-and-sd",
    "matrix-gaussian-gradient-covariance-symmetric",
)

Kind = Literal["supported", "unsupported", "invalid_candidate"]


class Inflate(BaseModel):
    """Make the input enormous: ``unit`` repeated ``count`` times is appended at ``path``."""

    unit: str
    count: int
    path: str | None = None  # a dotted path into `request`; None: the expression text


class Expect(BaseModel):
    status: str = "ok"
    reason: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    matrix: list[list[str]] | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    evidence: Literal["conditional", "identity"] | None = None
    optimum: Literal["global", "local", "stationary_only"] | None = None
    has_conditions: bool | None = None

    def all_outputs(self) -> dict[str, str]:
        return flatten({**self.outputs, **({"matrix": self.matrix} if self.matrix else {})})


class StatCase(BaseModel):
    id: str
    group: str
    kind: Kind
    operation: str
    source: str
    input: str | None = None
    format: Literal["text", "latex"] = "text"
    assumptions: dict[str, list[str]] = Field(default_factory=dict)
    matrices: list[dict[str, Any]] = Field(default_factory=list)
    vars: dict[str, str] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    inflate: Inflate | None = None
    expect: Expect = Field(default_factory=Expect)
    forbid: dict[str, list[str]] = Field(default_factory=dict)
    candidate: dict[str, Any] | None = None
    control: dict[str, Any] | None = None
    pending: bool = False
    note: str = ""

    @property
    def independent(self) -> list[list[str]]:
        return [list(g) for g in self.request.get("independent", [])]


def flatten(entries: dict[str, Any]) -> dict[str, str]:
    """``matrix: [[a, b], [c, d]]`` becomes ``matrix[0][0]`` ... keys; everything else is text."""
    out: dict[str, str] = {}
    for key, value in entries.items():
        if key == "matrix" and isinstance(value, list):
            for i, row in enumerate(value):
                for j, text in enumerate(row):
                    out[f"matrix[{i}][{j}]"] = str(text)
        else:
            out[key] = str(value)
    return out


def load_cases(data_dir: Path = DATA_DIR) -> list[StatCase]:
    with (data_dir / CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{CASES_FILE} must contain a list of cases")
    cases = [StatCase(**item) for item in raw]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("duplicate statistics case ids")
    for case in cases:
        if case.operation not in OPERATIONS:
            raise ValueError(f"{case.id}: unknown operation {case.operation}")
        if case.group not in GROUPS:
            raise ValueError(f"{case.id}: unknown group {case.group}")
        if case.kind == "supported" and not (case.expect.outputs or case.expect.matrix):
            raise ValueError(f"{case.id}: a supported case needs outputs")
        if case.kind == "invalid_candidate" and not (case.candidate and case.control):
            raise ValueError(f"{case.id}: an invalid_candidate case needs candidate and control")
        if not case.source.strip():
            raise ValueError(f"{case.id}: every case cites a source locator")
    return cases


# --------------------------------------------------------------------------- requests


class _NotExpressible(Exception):
    pass


def _set_path(body: dict[str, Any], path: str, text: str) -> None:
    node = body
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    node[last] = text


def request_body(case: StatCase, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    body = copy.deepcopy(case.request)
    if case.input is not None:
        text = case.input
        if case.inflate and case.inflate.path is None:
            text += case.inflate.unit * case.inflate.count
        body["expression"] = {"input": text, "format": case.format, "assumptions": case.assumptions}
    else:
        body["assumptions"] = case.assumptions
    if case.inflate and case.inflate.path is not None:
        node = body
        *parents, last = case.inflate.path.split(".")
        for key in parents:
            node = node[key]
        _set_path(body, case.inflate.path, str(node[last]) + case.inflate.unit * case.inflate.count)
    if case.matrices:
        body["matrices"] = copy.deepcopy(case.matrices)
    if candidate is not None:
        body["candidate"] = flatten(candidate)
    return body


def _dropped(sent: Any, kept: Any, where: str = "") -> str | None:
    """The first key of `sent` that the model silently discarded (it does not know the field)."""
    if isinstance(sent, dict) and isinstance(kept, dict):
        for key, value in sent.items():
            if key not in kept:
                return f"{where}{key}"
            found = _dropped(value, kept[key], f"{where}{key}.")
            if found:
                return found
    elif isinstance(sent, list) and isinstance(kept, list) and len(sent) == len(kept):
        for item_sent, item_kept in zip(sent, kept, strict=True):
            found = _dropped(item_sent, item_kept, where)
            if found:
                return found
    return None


def build_request(case: StatCase, candidate: dict[str, Any] | None = None) -> Any:
    """The engine input model for a case. Raises _NotExpressible if the engine cannot express it."""
    model_name, _ = OPERATIONS[case.operation]
    model = getattr(minerva_engine, model_name, None)
    if model is None:
        raise _NotExpressible(model_name)
    body = request_body(case, candidate)
    try:
        request = model.model_validate(body)
    except ValidationError as exc:
        unknown = [e for e in exc.errors() if e["type"] in UNKNOWN_OPTION]
        if unknown:
            where = ".".join(str(part) for part in unknown[0]["loc"])
            raise _NotExpressible(f"{where}: an option the model does not have") from exc
        raise
    missing = _dropped(body, request.model_dump(mode="json"))
    if missing:
        raise _NotExpressible(missing)
    return request


# --------------------------------------------------------------------------- outcomes


class Outcome(BaseModel):
    status: str
    reason: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)
    labels: dict[str, str] = Field(default_factory=dict)
    evidence: str | None = None
    verified: bool | None = None
    optimums: list[str] = Field(default_factory=list)
    has_conditions: bool | None = None
    result_chars: int = 0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _outputs(result: Any) -> tuple[dict[str, str], dict[str, str], list[str], bool | None]:
    outputs: dict[str, str] = {}
    labels: dict[str, str] = {}
    optimums: list[str] = []
    conditions: bool | None = None
    text = getattr(result, "result_text", None)
    if isinstance(text, str):
        outputs["result"] = text
    for estimate in getattr(result, "estimates", None) or []:
        outputs[f"estimate:{estimate.parameter}"] = estimate.text
        optimums.append(_text(getattr(estimate, "optimum", None)) or "")
        admissible = getattr(estimate, "admissible_if", None)
        if admissible:
            conditions = True
        elif conditions is None:
            conditions = False
    matrix = getattr(result, "matrix_text", None)
    if matrix:
        outputs.update(flatten({"matrix": matrix}))
    posterior = getattr(result, "posterior", None)
    if posterior is not None:
        labels["posterior:family"] = _text(posterior.family) or ""
        outputs.update({f"posterior:{k}": v for k, v in posterior.parameters.items()})
    for name, value in (getattr(result, "moments", None) or {}).items():
        outputs[f"moment:{name}"] = value
    derivative = getattr(result, "derivative", None)
    if derivative is not None:
        for key in ("constraint", "convention"):
            value = _text(getattr(derivative, key, None))
            if value:
                labels[key] = value
        for alt in getattr(derivative, "alternatives", None) or []:
            convention = (_text(alt.convention) or "").removeprefix("symmetric_")
            outputs[f"alternative:{convention}"] = alt.text
    return outputs, labels, optimums, conditions


def _verified(result: Any) -> bool | None:
    verification = getattr(result, "verification", None)
    if verification is not None:
        return bool(getattr(verification, "passed", False))
    checks = getattr(result, "checks", None)
    if checks:
        return all(c.passed for c in checks)
    return None


def outcome_of(result: Any) -> Outcome:
    outputs, labels, optimums, conditions = _outputs(result)
    if conditions is None and getattr(result, "conditions", None):
        conditions = True
    return Outcome(
        status=_text(result.status) or "",
        reason=_text(getattr(result, "reason", None)),
        outputs=outputs,
        labels=labels,
        evidence=_text(getattr(result, "evidence", None)),
        verified=_verified(result),
        optimums=optimums,
        has_conditions=conditions,
        result_chars=sum(len(v) for v in outputs.values()),
    )


def execute(case: StatCase, engine: Engine, candidate: dict[str, Any] | None = None) -> Outcome:
    _, method = OPERATIONS[case.operation]
    runner = getattr(engine, method, None)
    if runner is None:
        raise _NotExpressible(f"Engine.{method}")
    return outcome_of(runner(build_request(case, candidate)))


# --------------------------------------------------------------------------- judging


def judge_supported(case: StatCase, outcome: Outcome, envs: list[Any]) -> list[str]:
    expect = case.expect
    problems: list[str] = []
    if outcome.status != expect.status:
        return [f"expected {expect.status}, got {outcome.status} ({outcome.reason})"]
    for key, expected in expect.all_outputs().items():
        actual = outcome.outputs.get(key)
        if actual is None:
            problems.append(f"no output {key}")
            continue
        ok, why = equal_texts(actual, expected, envs)
        if not ok:
            problems.append(f"{key}: {why}")
    for key, expected in expect.labels.items():
        if outcome.labels.get(key) != expected:
            problems.append(f"{key} is {outcome.labels.get(key)!r}, expected {expected!r}")
    for key, wrongs in case.forbid.items():
        actual = outcome.outputs.get(key)
        for wrong in wrongs:
            if actual is not None and equal_texts(actual, wrong, envs)[0]:
                problems.append(f"{key} returned the forbidden result {wrong}")
    if outcome.verified is not True:
        problems.append("verification did not pass")
    if expect.evidence and outcome.evidence != expect.evidence:
        problems.append(f"evidence {outcome.evidence!r}, expected {expect.evidence!r}")
    if expect.optimum and (
        not outcome.optimums or any(o != expect.optimum for o in outcome.optimums)
    ):
        problems.append(f"optimum {outcome.optimums}, expected {expect.optimum}")
    if expect.has_conditions is not None and outcome.has_conditions != expect.has_conditions:
        problems.append("the conditions under which the result holds are not reported")
    return problems


def judge_unsupported(case: StatCase, outcome: Outcome) -> list[str]:
    expect = case.expect
    problems: list[str] = []
    if outcome.status != expect.status:
        problems.append(f"expected {expect.status}, got {outcome.status} ({outcome.reason})")
    if expect.reason is not None and outcome.reason != expect.reason:
        problems.append(f"expected reason {expect.reason}, got {outcome.reason}")
    if outcome.outputs and expect.status != "unchanged":
        problems.append("a result was shown")
    return problems


def run_case(case: StatCase, engine: Engine) -> CaseResult:
    started = time.monotonic()
    try:
        envs = environments(case.vars, case.matrices, case.independent)
    except OracleError as exc:
        return CaseResult.of(case, NOT_EXPRESSIBLE, [f"the oracle cannot draw data: {exc}"], 0.0)
    try:
        if case.kind == "invalid_candidate":
            assert case.candidate is not None and case.control is not None  # noqa: S101
            wrong = execute(case, engine, case.candidate)
            right = execute(case, engine, case.control)
            problems: list[str] = []
            if wrong.status not in REJECTED:
                problems.append(f"the wrong candidate was answered {wrong.status}")
            if wrong.outputs:
                problems.append("a result was shown for the wrong candidate")
            if right.status != "ok" or right.verified is not True:
                problems.append(f"the correct control was answered {right.status} ({right.reason})")
            outcome = wrong
        else:
            outcome = execute(case, engine)
            problems = (
                judge_supported(case, outcome, envs)
                if case.kind == "supported"
                else judge_unsupported(case, outcome)
            )
    except _NotExpressible as exc:
        return CaseResult.of(case, NOT_EXPRESSIBLE, [f"not expressible: {exc}"], 0.0)
    milliseconds = (time.monotonic() - started) * 1000
    return CaseResult.of(case, outcome.status, problems, milliseconds, outcome)


class CaseResult(BaseModel):
    id: str
    group: str
    kind: Kind
    pending: bool
    passed: bool
    status: str
    reason: str | None = None
    verified: bool | None = None
    problems: list[str] = Field(default_factory=list)
    result_chars: int = 0
    milliseconds: float = 0.0

    @classmethod
    def of(
        cls,
        case: StatCase,
        status: str,
        problems: list[str],
        milliseconds: float,
        outcome: Outcome | None = None,
    ) -> CaseResult:
        return cls(
            id=case.id,
            group=case.group,
            kind=case.kind,
            pending=case.pending,
            passed=status != NOT_EXPRESSIBLE and not problems,
            status=status,
            reason=outcome.reason if outcome else None,
            verified=outcome.verified if outcome else None,
            problems=problems,
            result_chars=outcome.result_chars if outcome else 0,
            milliseconds=round(milliseconds, 1),
        )


def _p95(sorted_times: list[float]) -> float:
    index = max(0, min(len(sorted_times) - 1, -(-95 * len(sorted_times) // 100) - 1))
    return sorted_times[index]


class StatReport(BaseModel):
    results: list[CaseResult]

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if not r.pending]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.scored if not r.passed]

    @property
    def stale_pending(self) -> list[CaseResult]:
        return [r for r in self.results if r.pending and r.passed]

    def summary(self) -> dict[str, Any]:
        def of(kind: str) -> list[CaseResult]:
            return [r for r in self.results if r.kind == kind]

        supported, refused, invalid = of("supported"), of("unsupported"), of("invalid_candidate")
        ran = [r for r in self.results if r.status != NOT_EXPRESSIBLE]
        answered = [r for r in supported if r.status == "ok"]
        times = sorted(r.milliseconds for r in ran) or [0.0]
        by_id = {r.id: r for r in self.results}
        groups = {
            g: {
                "cases": sum(r.group == g for r in self.results),
                "passing": sum(r.group == g and r.passed for r in self.results),
            }
            for g in GROUPS
        }
        return {
            "cases": len(self.results),
            "pending": sum(r.pending for r in self.results),
            "not_expressible": sum(r.status == NOT_EXPRESSIBLE for r in self.results),
            "scored_cases": len(self.scored),
            "scored_passing": sum(r.passed for r in self.scored),
            "supported_cases": len(supported),
            "supported_passing": sum(r.passed for r in supported),
            "unsupported_cases": len(refused),
            "refused_correctly": sum(r.passed for r in refused),
            "invalid_cases": len(invalid),
            "invalid_rejected": sum(r.passed for r in invalid),
            "coverage": round(sum(r.passed for r in supported) / max(1, len(supported)), 4),
            "verification_success": round(
                sum(1 for r in answered if r.verified) / max(1, len(answered)), 4
            ),
            "minimum_outcome_passing": sum(
                1 for case_id in MINIMUM_OUTCOME if case_id in by_id and by_id[case_id].passed
            ),
            "minimum_outcome_total": len(MINIMUM_OUTCOME),
            "latency_median_ms": round(statistics.median(times), 1),
            "latency_p95_ms": round(_p95(times), 1),
            "result_chars_max": max((r.result_chars for r in ran), default=0),
            "refusal_reasons": dict(sorted(Counter(r.reason for r in ran if r.reason).items())),
            "groups": groups,
            "failed": [r.id for r in self.failures],
        }

    def to_json(self, label: str = "") -> dict[str, Any]:
        return {
            "format": FORMAT,
            "label": label,
            "summary": self.summary(),
            "cases": [r.model_dump(mode="json") for r in self.results],
        }


def run_stat_benchmark(
    engine: Engine, cases: list[StatCase] | None = None, *, group: str | None = None
) -> StatReport:
    selected = [c for c in (cases or load_cases()) if group in (None, c.group)]
    return StatReport(results=[run_case(c, engine) for c in selected])


def format_report(report: StatReport) -> str:
    s = report.summary()
    lines = [
        f"cases:                   {s['cases']} ({s['pending']} pending, "
        f"{s['not_expressible']} not expressible by the engine yet)",
        f"scored cases passing:    {s['scored_passing']}/{s['scored_cases']}",
        f"supported, passing:      {s['supported_passing']}/{s['supported_cases']}"
        f" (verification success {s['verification_success']:.0%} of answers)",
        f"refused as designed:     {s['refused_correctly']}/{s['unsupported_cases']}",
        f"wrong answers rejected:  {s['invalid_rejected']}/{s['invalid_cases']}",
        f"minimum outcome:         {s['minimum_outcome_passing']}/{s['minimum_outcome_total']}"
        " derivations complete",
        f"latency median / p95:    {s['latency_median_ms']:.0f} ms / {s['latency_p95_ms']:.0f} ms",
    ]
    lines.append(
        "by group:                "
        + ", ".join(f"{g} {v['passing']}/{v['cases']}" for g, v in s["groups"].items())
    )
    for r in report.failures:
        lines.append(f"FAIL {r.id}: {'; '.join(r.problems)}")
    for r in report.stale_pending:
        lines.append(f"STALE {r.id}: passes now, remove `pending`")
    return "\n".join(lines)


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> tuple[str, bool]:
    """A concise comparison of two ``to_json`` documents; the flag is True if anything regressed."""
    if baseline.get("format") != current.get("format"):
        return "benchmark formats differ; regenerate the baseline", True
    before = {c["id"]: c for c in baseline["cases"]}
    after = {c["id"]: c for c in current["cases"]}
    regressions: list[str] = []
    notes: list[str] = []
    for case_id, new in after.items():
        old = before.get(case_id)
        if old is None:
            notes.append(f"new case {case_id}: {'pass' if new['passed'] else 'not yet'}")
        elif old["passed"] and not new["passed"]:
            regressions.append(f"REGRESSION {case_id}: {'; '.join(new['problems'])}")
        elif not old["passed"] and new["passed"]:
            notes.append(f"now passes {case_id}")
        elif old["status"] != new["status"]:
            notes.append(f"changed {case_id}: {old['status']} -> {new['status']}")
    notes.extend(f"removed case {case_id}" for case_id in before if case_id not in after)
    lines = ["metric                      baseline   current"]
    for key, value in current["summary"].items():
        old_value = baseline["summary"].get(key)
        if isinstance(value, int | float) and isinstance(old_value, int | float):
            marker = "" if old_value == value else "  <-"
            lines.append(f"{key:<26} {old_value!s:>9} {value!s:>9}{marker}")
    lines.extend(regressions + notes)
    return "\n".join(lines), bool(regressions)
