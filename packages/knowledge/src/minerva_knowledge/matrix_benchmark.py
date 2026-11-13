"""Matrix calculus benchmark: hand-written cases -> expected status, method and result kind.

The cases (`data/benchmark/matrix_cases.yaml`) are written by a person, independently of the code
under test. A case passes when the status, and (where given) the method, result kind and `evaluated`
flag, match. `pending` cases are known gaps: reported, never asserted. A request the engine cannot
express yet (a field or action that does not exist) is reported as status `not_expressible`, so the
same case file can be run against an older commit for a "before" measurement.

Timings are measurements on one machine and are never asserted anywhere.
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import yaml
from minerva_engine import Engine, ExpressionInput
from minerva_engine.matrix import MatrixCalculusInput
from pydantic import BaseModel, Field, ValidationError

from .loader import DATA_DIR

CASES_FILE = "benchmark/matrix_cases.yaml"
GROUPS = ("existing", "frechet", "duhamel", "evolution", "unsupported", "limits")
NOT_EXPRESSIBLE = "not_expressible"


class MatrixCase(BaseModel):
    id: str
    group: str
    input: str
    action: str = "simplify"
    matrices: list[dict[str, Any]]
    commuting: list[list[str]] = Field(default_factory=list)
    invertible: list[str] = Field(default_factory=list)
    self_commuting: list[str] = Field(default_factory=list)
    variable: str | None = None
    lower: str | None = None
    upper: str | None = None
    pending: bool = False
    expect_status: str
    expect_method: str | None = None
    expect_kind: str | None = None
    expect_evaluated: bool | None = None

    def request(self) -> MatrixCalculusInput:
        """Build the engine request. Raises ValidationError if the engine cannot express it."""
        body: dict[str, Any] = {
            "expression": ExpressionInput(input=self.input, format="latex"),
            "matrices": self.matrices,
            "commuting": [tuple(p) for p in self.commuting],
            "invertible": self.invertible,
            "action": self.action,
            "variable": self.variable,
            "lower": self.lower,
            "upper": self.upper,
        }
        if self.self_commuting:
            body["self_commuting"] = self.self_commuting
        # A field the engine does not know is ignored by pydantic; detect that explicitly.
        if any("depends_on" in m for m in self.matrices) or self.self_commuting:
            fields = MatrixCalculusInput.model_fields
            if self.self_commuting and "self_commuting" not in fields:
                raise _NotExpressible("self_commuting")
            from minerva_engine.matrix import MatrixDecl

            if (
                any("depends_on" in m for m in self.matrices)
                and "depends_on" not in MatrixDecl.model_fields
            ):
                raise _NotExpressible("depends_on")
        return MatrixCalculusInput(**body)


class _NotExpressible(Exception):
    pass


class CaseResult(BaseModel):
    case: MatrixCase
    status: str
    method: str | None
    kind: str | None
    evaluated: bool | None
    verified: bool | None  # None: no verification ran
    exact_leg: bool | None  # symbolic / series leg: None = not run
    residual: float | None
    diagnostics: dict[str, Any] | None = None
    milliseconds: list[float]  # one entry per repeat

    @property
    def median_ms(self) -> float:
        return statistics.median(self.milliseconds) if self.milliseconds else 0.0

    @property
    def passed(self) -> bool:
        c = self.case
        if self.status != c.expect_status:
            return False
        if c.expect_method is not None and self.method != c.expect_method:
            return False
        if c.expect_kind is not None and self.kind != c.expect_kind:
            return False
        return c.expect_evaluated is None or self.evaluated == c.expect_evaluated


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


class MatrixReport(BaseModel):
    results: list[CaseResult]

    @property
    def scored(self) -> list[CaseResult]:
        return [r for r in self.results if not r.case.pending]

    @property
    def failures(self) -> list[CaseResult]:
        return [r for r in self.scored if not r.passed]

    @property
    def stale_pending(self) -> list[CaseResult]:
        return [r for r in self.results if r.case.pending and r.passed]

    def by_group(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for g in GROUPS:
            rs = [r for r in self.results if r.case.group == g]
            if not rs:
                continue
            out[g] = {
                "cases": len(rs),
                "passing": sum(r.passed for r in rs),
                "pending": sum(r.case.pending for r in rs),
            }
        return out

    def summary(self) -> dict[str, float]:
        results = self.results
        samples = [ms for r in results for ms in r.milliseconds]
        ok = [r for r in results if r.status == "ok"]
        attempted = [r for r in results if r.verified is not None]
        with_method = [r for r in results if r.case.expect_method is not None]
        residuals = [r.residual for r in results if r.residual is not None]
        return {
            "cases": len(results),
            "passing": sum(r.passed for r in results),
            "scored": len(self.scored),
            "scored_passing": sum(r.passed for r in self.scored),
            "pending": sum(r.case.pending for r in results),
            "status_ok": len(ok),
            "verification_attempted": len(attempted),
            "verification_passed": sum(
                bool(r.verified) and r.exact_leg is not False for r in attempted
            ),
            "method_cases": len(with_method),
            "method_correct": sum(r.method == r.case.expect_method for r in with_method),
            "resource_limit": sum(r.status == "resource_limit" for r in results),
            "not_expressible": sum(r.status == NOT_EXPRESSIBLE for r in results),
            "latency_median_ms": statistics.median(samples) if samples else 0.0,
            "latency_p95_ms": percentile(samples, 95),
            "latency_max_ms": max(samples) if samples else 0.0,
            "max_residual": max(residuals) if residuals else float("nan"),
        }


def load_cases(data_dir: Path = DATA_DIR) -> list[MatrixCase]:
    with (data_dir / CASES_FILE).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{CASES_FILE} must contain a list of cases")
    cases = [MatrixCase(**item) for item in raw]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate matrix benchmark case ids")
    unknown = {c.group for c in cases} - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown matrix benchmark groups: {sorted(unknown)}")
    return cases


def _residual(result: Any) -> float | None:
    diagnostics = getattr(result, "diagnostics", None)
    value = getattr(diagnostics, "max_residual", None)
    return float(value) if value is not None else None


def run_case(engine: Engine, case: MatrixCase, repeat: int = 1) -> CaseResult:
    try:
        request = case.request()
    except (ValidationError, _NotExpressible):
        return CaseResult(
            case=case,
            status=NOT_EXPRESSIBLE,
            method=None,
            kind=None,
            evaluated=None,
            verified=None,
            exact_leg=None,
            residual=None,
            milliseconds=[],
        )
    started = time.monotonic()
    result = engine.matrix_calculus(request)
    times = [(time.monotonic() - started) * 1000]
    for _ in range(max(1, repeat) - 1):
        started = time.monotonic()
        engine.matrix_calculus(request)
        times.append((time.monotonic() - started) * 1000)
    v = result.verification
    return CaseResult(
        case=case,
        status=result.status.value,
        method=result.method or None,
        kind=result.result_kind,
        evaluated=result.evaluated if result.status.value == "ok" else None,
        verified=v.passed if v is not None else None,
        exact_leg=v.symbolic_zero if v is not None else None,
        residual=_residual(result),
        diagnostics=result.diagnostics.model_dump() if result.diagnostics else None,
        milliseconds=times,
    )


def run_matrix_benchmark(
    engine: Engine, cases: list[MatrixCase] | None = None, repeat: int = 1
) -> MatrixReport:
    chosen = cases if cases is not None else load_cases()
    if chosen:  # warm-up: the first call in a process pays for imports and the fork
        run_case(engine, chosen[0])
    return MatrixReport(results=[run_case(engine, c, repeat) for c in chosen])


def _trace_line(r: CaseResult) -> str:
    d = r.diagnostics
    if not d:
        return "      (no diagnostics)"
    stages = " ".join(f"{s['name']}={s['ms']:.0f}ms" for s in d["stages"])
    tried = ",".join(f"{m['id']}:{m['outcome']}" for m in d["methods_tried"])
    residual = "-" if d["max_residual"] is None else f"{d['max_residual']:.1e}"
    return (
        f"      tried[{tried}] dims={d['dimensions']} commutation={d['commutation']} "
        f"digits={d['digits']} tol={d['tolerance']} residual={residual} "
        f"points={d['points_agreed']}/{d['points_checked']} fallbacks={d['fallbacks']} "
        f"failure={d['failure_reason']} {stages}"
    )


def convergence_line() -> str:
    """Observed order of the RK4 cross-check on a smooth generator (Airy equation)."""
    import sympy as sp
    from minerva_engine.matrix.evolution import rk4_observed_order

    t = sp.Symbol("t")
    errors, order = rk4_observed_order(sp.Matrix([[0, 1], [-t, 0]]), t, 0.0, 1.5)
    return (
        f"RK4 cross-check, Airy generator on [0, 1.5]: errors {errors[0]:.1e}, {errors[1]:.1e}, "
        f"{errors[2]:.1e} at N, 2N, 4N steps; observed order {order:.2f} (expected 4)"
    )


def format_report(report: MatrixReport, trace: bool = False) -> str:
    lines = []
    for r in report.results:
        tag = "pend" if r.case.pending else ("ok  " if r.passed else "FAIL")
        if r.case.pending and r.passed:
            tag = "STALE"
        lines.append(
            f"{tag:<5} {r.case.group:<11} {r.case.id:<44} {r.status:<24} "
            f"{(r.method or '-'):<24} {(r.kind or '-'):<14} {r.median_ms:>7.0f}ms"
        )
        if trace:
            lines.append(_trace_line(r))
    s = report.summary()
    lines += [
        "",
        f"cases passing:            {int(s['passing'])}/{int(s['cases'])} "
        f"(scored {int(s['scored_passing'])}/{int(s['scored'])}, pending {int(s['pending'])})",
        f"verification passed:      {int(s['verification_passed'])}/{int(s['verification_attempted'])}"
        " (cases where a verification ran)",
        f"method selected:          {int(s['method_correct'])}/{int(s['method_cases'])}",
        f"resource limits:          {int(s['resource_limit'])}",
        f"not expressible:          {int(s['not_expressible'])}",
        f"latency median/p95/max:   {s['latency_median_ms']:.0f} / {s['latency_p95_ms']:.0f} / "
        f"{s['latency_max_ms']:.0f} ms",
    ]
    if not math.isnan(s["max_residual"]):
        lines.append(f"max residual:             {s['max_residual']:.2e}")
    return "\n".join(lines)


SECTION_PREFIX = "## Run: "
HEADER = """# Matrix calculus benchmark

`uv run python -m minerva_knowledge matrix-benchmark [--repeat N] [--section LABEL] [--out FILE]` runs
hand-written cases (`packages/knowledge/src/minerva_knowledge/data/benchmark/matrix_cases.yaml`)
through the real engine and sandbox. Groups: `existing` (behaviour that shipped before this work),
`frechet`, `duhamel`, `evolution`, `unsupported` (gaps that must stay explicit) and `limits`.

A case passes when status, and where given method, result kind and the `evaluated` flag, match.
`pending` cases are known gaps: reported, never asserted. A pending case that already passes fails the
test suite until `pending` is removed. A request the engine cannot express (for example a field that
does not exist yet) is reported as `not_expressible`, so the final case file can be run against an
older commit for the "before" run. Each `## Run:` section below is one run of the same case file.

Limits of this evidence: timings are medians over the repeats on one machine (the first call in a
process is excluded, forking the sandbox is included), are noisy and are never asserted; the cases are
small hand-written problems, not a sample of real use; agreement with the expected status and method
shows the engine behaves as designed, not that every result is right: correctness is established by the
checks inside the engine and by the regression tests.
"""


def format_markdown(
    report: MatrixReport, label: str, repeat: int, extra: list[str] | None = None
) -> str:
    s = report.summary()
    lines = [
        f"{SECTION_PREFIX}{label}",
        "",
        f"{int(s['cases'])} cases, {repeat} repeat(s) each.",
        "",
        "| Measure | Value |",
        "| --- | --- |",
        f"| Cases passing (all, incl. pending) | {int(s['passing'])} / {int(s['cases'])} |",
        f"| Scored cases passing | {int(s['scored_passing'])} / {int(s['scored'])} |",
        f"| Pending (known gaps) | {int(s['pending'])} |",
        f"| Results with status ok | {int(s['status_ok'])} |",
        f"| Verification passed (where one ran) | {int(s['verification_passed'])} / "
        f"{int(s['verification_attempted'])} |",
        f"| Method selected as expected | {int(s['method_correct'])} / {int(s['method_cases'])} |",
        f"| Resource-limit results | {int(s['resource_limit'])} |",
        f"| Not expressible | {int(s['not_expressible'])} |",
        f"| Latency median / p95 / max (ms) | {s['latency_median_ms']:.0f} / "
        f"{s['latency_p95_ms']:.0f} / {s['latency_max_ms']:.0f} |",
    ]
    if not math.isnan(s["max_residual"]):
        lines.append(f"| Max verification residual | {s['max_residual']:.2e} |")
    lines += ["", "| Group | Cases | Passing | Pending |", "| --- | --- | --- | --- |"]
    for g, v in report.by_group().items():
        lines.append(f"| {g} | {int(v['cases'])} | {int(v['passing'])} | {int(v['pending'])} |")
    lines += [
        "",
        "| Case | Group | Status | Method | Kind | Pass | ms |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        mark = "pending" if r.case.pending and not r.passed else ("yes" if r.passed else "NO")
        lines.append(
            f"| {r.case.id} | {r.case.group} | {r.status} | {r.method or '-'} | {r.kind or '-'} "
            f"| {mark} | {r.median_ms:.0f} |"
        )
    if extra:
        lines += ["", *extra]
    return "\n".join(lines) + "\n"


def write_section(path: Path, label: str, section: str) -> None:
    """Insert or replace one `## Run:` section, keeping the others in order."""
    sections: list[tuple[str, str]] = []
    if path.exists():
        parts = path.read_text(encoding="utf-8").split("\n" + SECTION_PREFIX)
        for part in parts[1:]:
            head, _, body = part.partition("\n")
            sections.append((head.strip(), SECTION_PREFIX + head + "\n" + body))
    replaced = False
    out: list[tuple[str, str]] = []
    for name, text in sections:
        if name == label:
            out.append((label, section))
            replaced = True
        else:
            out.append((name, text))
    if not replaced:
        out.append((label, section))
    body = "\n".join(text.rstrip("\n") + "\n" for _, text in out)
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
