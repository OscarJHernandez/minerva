"""One table across saved reports: which model, with or without thinking, answers best within what a chat tolerates.

The rule is fixed before the runs and written in docs/reliability/model-comparison.md; this module only applies it:

1. a configuration with a false verified result, or not fully on the GPU (its latency would measure the spill), is
   not eligible (it is still listed, with the reason);
2. among the eligible, a higher pass rate counts only when the gap is larger than the run-to-run noise of both
   (`max(1, 2 x the larger std)` passes per repeat, the rule of `--compare`);
3. a tie goes to the lower p95.

Speed is reported twice and never decides alone: the best configuration **at any speed**, and the best **within
the chat latency ceiling**. A chat may be slow when it is accurate (the maintainer's call, 2026-11-19), so a slow
configuration stays in the screen and in the result; the ceiling only says which ones also answer quickly.

Every row names the model, its digest, whether thinking was requested and the reply budget, so a number cannot be
quoted without them. The host's address is never in a report.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from .cases import ASK, NO_TOOL

# Whole turn on the host that serves the deployed model, about twice the 2026-11-19 baseline (7.0 s, 20.2 s).
MEDIAN_CEILING_SECONDS = 15.0
P95_CEILING_SECONDS = 40.0
# Fraction of the model that must sit on the GPU for its latency to count as the model's.
MIN_GPU_SHARE = 0.99


_MODEL_TAG = re.compile(r"^[a-z][a-z0-9.\-]*:[a-z0-9][a-z0-9._\-]*$")


def _is_model_tag(name: str) -> bool:
    """`family:size` (`gemma4:26b`, `qwen3.5:35b`) names a model; an operator's alias (`minerva`) does not."""
    return bool(_MODEL_TAG.match(name))


@dataclass
class Row:
    source: int
    model: str
    weights: str
    digest: str | None
    thinking: bool | None
    reply_budget: str
    gpu_share: float | None
    runs: int
    passed: int
    repeats: list[int]
    std: float
    tool_selection: float | None
    valid_arguments: float | None
    completion: float | None
    false_verified: int
    no_tool_runs: int
    cut_off: int
    median_s: float | None
    p95_s: float | None
    tokens_out: int | None
    reasons: list[str] = field(default_factory=list)  # keeps a row out of every result
    slow: list[str] = field(
        default_factory=list
    )  # over the chat latency ceiling: only out of the quick result
    base_model: str | None = None  # what the host says a derived build is made from

    @property
    def eligible(self) -> bool:
        """Eligible at any speed."""
        return not self.reasons

    @property
    def quick(self) -> bool:
        """Eligible and within the chat latency ceiling."""
        return self.eligible and not self.slow

    @property
    def rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0

    @property
    def mean_passes(self) -> float:
        return statistics.fmean(self.repeats) if self.repeats else float(self.passed)

    @property
    def name(self) -> str:
        think = {True: "thinking on", False: "thinking off", None: "thinking not recorded"}[
            self.thinking
        ]
        return f"{self.display} ({think})"

    @property
    def display(self) -> str:
        """The model by what it is, not by the name a host serves it under (an alias can be re-pointed, and a
        name like `minerva` says nothing about the weights)."""
        if _is_model_tag(self.model) or (
            self.base_model and self.base_model.removesuffix(":latest") == self.model
        ):
            return (
                self.model
            )  # `qwen3.5:35b` is the model; a parent the host reports is a detail, not its name
        if self.base_model and self.base_model != self.model:
            return f"{self.base_model} (served as {self.model})"
        return self.model


def _weights(identity: dict[str, Any] | None) -> tuple[str, str | None, float | None]:
    if not identity:
        return ("weights not recorded", None, None)
    if identity.get("hosted"):
        return ("hosted by the provider (weights not pinned)", None, None)
    parts = [
        f"based on {identity['base_model']}" if identity.get("base_model") else None,
        identity.get("parameter_size"),
        identity.get("quantization"),
    ]
    total, gpu = identity.get("loaded_bytes"), identity.get("loaded_gpu_bytes")
    share = (gpu / total) if total and gpu is not None else None
    return (
        ", ".join(str(p) for p in parts if p) or "weights not recorded",
        identity.get("digest"),
        share,
    )


def _budget(params: dict[str, Any], identity: dict[str, Any] | None) -> str:
    if params.get("num_predict"):
        return str(params["num_predict"])
    host = ((identity or {}).get("host_defaults") or {}).get("num_predict")
    return f"host default {host}" if host else "host default (not reported)"


def rows_of(docs: list[dict[str, Any]]) -> list[Row]:
    rows: list[Row] = []
    for index, doc in enumerate(docs):
        meta = doc.get("meta") or {}
        params = meta.get("params") or {}
        for model, entry in doc["models"].items():
            summary = entry.get("summary")
            if not summary:
                continue  # a model that did not run is not a row
            results = [r for r in entry["results"] if not r.get("informational")]
            identity = entry.get("identity")
            weights, digest, share = _weights(identity)
            latency = summary.get("latency_seconds") or {}
            row = Row(
                source=index + 1,
                model=model,
                base_model=(identity or {}).get("base_model") or None,
                weights=weights,
                digest=digest,
                thinking=params.get("think", meta.get("think")),
                reply_budget=_budget(params, identity),
                gpu_share=share,
                runs=summary["runs"],
                passed=summary["passed"],
                repeats=list(summary.get("pass_counts_per_repeat") or []),
                std=float(summary.get("pass_count_stdev") or 0.0),
                tool_selection=summary.get("tool_selection_accuracy"),
                valid_arguments=summary.get("valid_arguments_rate"),
                completion=summary.get("completion_rate"),
                false_verified=int(summary.get("false_verified") or 0),
                no_tool_runs=sum(
                    1
                    for r in results
                    if not r.get("calls") and r.get("expected_tool") not in (NO_TOOL, ASK)
                ),
                cut_off=sum(1 for r in results if r.get("stop_reason") == "output_limit"),
                median_s=latency.get("median"),
                p95_s=latency.get("p95"),
                tokens_out=summary.get("tokens_out"),
            )
            row.reasons, row.slow = _reasons(row)
            rows.append(row)
    return rows


def _reasons(row: Row) -> tuple[list[str], list[str]]:
    """(what keeps a row out of every result, what only keeps it out of the quick result)."""
    out: list[str] = []
    slow: list[str] = []
    if row.false_verified:
        out.append(f"{row.false_verified} false verified result(s)")
    if row.gpu_share is None:
        out.append("GPU share not recorded, so its latency cannot be attributed to the model")
    elif row.gpu_share < MIN_GPU_SHARE:
        out.append(f"only {100 * row.gpu_share:.0f}% on the GPU: its latency measures the spill")
    if row.median_s is None or row.p95_s is None:
        out.append("latency not recorded")
    else:
        if row.median_s > MEDIAN_CEILING_SECONDS:
            slow.append(
                f"median {row.median_s:.1f} s is over the {MEDIAN_CEILING_SECONDS:.0f} s ceiling"
            )
        if row.p95_s > P95_CEILING_SECONDS:
            slow.append(f"p95 {row.p95_s:.1f} s is over the {P95_CEILING_SECONDS:.0f} s ceiling")
    return out, slow


def noise_limit(a: Row, b: Row) -> float:
    """Passes per repeat below which a difference between two rows is not shown as real."""
    return max(1.0, 2 * max(a.std, b.std))


def within_noise(a: Row, b: Row) -> bool | None:
    """None when either side has fewer than two repeats: the noise cannot be measured."""
    if len(a.repeats) < 2 or len(b.repeats) < 2:
        return None
    return abs(a.mean_passes - b.mean_passes) <= noise_limit(a, b)


def screen(rows: list[Row], keep: int = 3) -> list[Row]:
    """The stage-1 cut: every eligible row within noise of the `keep`-th best, not a fixed number.

    Rows that are not eligible are dropped first. With fewer than `keep` eligible rows all of them go on.
    """
    eligible = sorted((r for r in rows if r.eligible), key=lambda r: -r.mean_passes)
    if len(eligible) <= keep:
        return eligible
    third = eligible[keep - 1]
    return [
        r
        for r in eligible
        if r.mean_passes >= third.mean_passes or within_noise(r, third) is not False
    ]


def decide(rows: list[Row], *, quick: bool = False) -> tuple[Row | None, list[Row]]:
    """(the best eligible row, the eligible rows that cannot be told apart from it).

    Ties inside the noise go to the lower p95, so the first element is the answer under the written rule.
    `quick=True` considers only rows within the chat latency ceiling; the default is any speed.
    """
    eligible = [r for r in rows if (r.quick if quick else r.eligible)]
    if not eligible:
        return None, []
    top = max(eligible, key=lambda r: r.mean_passes)
    tied = [r for r in eligible if r is top or within_noise(r, top) is not False]
    best = min(tied, key=lambda r: r.p95_s if r.p95_s is not None else float("inf"))
    return best, [r for r in tied if r is not best]


def _p(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def matrix_report(docs: list[dict[str, Any]]) -> str:
    rows = rows_of(docs)
    if not rows:
        return "No model with results in these reports.\n"
    case_sets = sorted({str((d.get("meta") or {}).get("cases_digest")) for d in docs})
    seeds = sorted({str((d.get("meta") or {}).get("seed")) for d in docs})
    lines = [
        "# Model comparison",
        "",
        f"{len(rows)} configuration(s) from {len(docs)} report(s). Case sets {', '.join(case_sets)}"
        f" (a row is comparable with another only on the same case set); seeds {', '.join(seeds)}. "
        "Decision rule: no false verified result and fully on the GPU; then pass rate where the gap exceeds the "
        "run-to-run noise; ties go to the lower p95. Speed is reported apart: the best at any speed, and the best "
        f"within the chat ceiling (median <= {MEDIAN_CEILING_SECONDS:.0f} s, p95 <= {P95_CEILING_SECONDS:.0f} s per turn "
        "on this host).",
        "",
        "## Configurations",
        "",
        "| Report | Model | Weights as the host reports them | Digest | Thinking | Reply budget | On GPU | "
        "Runs | Passed | Per repeat | Tool selection | Valid arguments | Completion | False verified | "
        "No tool called | Cut off | Median s | p95 s | Tokens out | Eligible | Within chat ceiling |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        think = {True: "on", False: "off", None: "not recorded"}[r.thinking]
        gpu = "not recorded" if r.gpu_share is None else f"{100 * r.gpu_share:.0f}%"
        verdict = "yes" if r.eligible else "no: " + "; ".join(r.reasons)
        quick = "yes" if r.quick else ("no: " + "; ".join(r.slow) if r.eligible else "n/a")
        lines.append(
            f"| {r.source} | {r.display} | {r.weights} | {r.digest or 'not recorded'} | {think} | "
            f"{r.reply_budget} | {gpu} | {r.runs} | {r.passed} ({_p(r.rate)}) | {r.repeats} | "
            f"{_p(r.tool_selection)} | {_p(r.valid_arguments)} | {_p(r.completion)} | {r.false_verified} | "
            f"{r.no_tool_runs} | {r.cut_off} | {_s(r.median_s)} | {_s(r.p95_s)} | "
            f"{'n/a' if r.tokens_out is None else r.tokens_out} | {verdict} | {quick} |"
        )
    lines += ["", "## Result under the rule", ""]
    if len(case_sets) > 1:
        lines.append(
            "Not decided: the reports ran different case sets, and a pass rate on one set says nothing about "
            "another. Run every configuration on the same cases."
        )
        return "\n".join(lines) + "\n"
    for title, only_quick in (("At any speed", False), ("Within the chat ceiling", True)):
        best, tied = decide(rows, quick=only_quick)
        if best is None:
            lines.append(
                f"- **{title}**: no configuration qualifies; the reasons are in the table above."
            )
            continue
        text = (
            f"- **{title}**: {best.name}, digest {best.digest or 'not recorded'}, reply budget {best.reply_budget}, "
            f"median {_s(best.median_s)} s, p95 {_s(best.p95_s)} s."
        )
        if tied:
            text += (
                " Not distinguishable from it within the run-to-run noise: "
                + ", ".join(
                    f"{t.name} (digest {t.digest or 'not recorded'}, median {_s(t.median_s)} s)"
                    for t in tied
                )
                + "."
            )
        lines.append(text)
    eligible = [r for r in rows if r.eligible]
    if eligible:
        top = max(eligible, key=lambda r: r.mean_passes)
        lines.append(
            f"- Highest pass rate: {top.name}, digest {top.digest or 'not recorded'}, {top.passed}/{top.runs} "
            f"({_p(top.rate)}), median {_s(top.median_s)} s. The result above names the fastest of the "
            "configurations that cannot be told apart from the most accurate, so a slower one that is no worse "
            "never loses to a faster one that is merely tied."
        )
    unmeasured = [r.name for r in rows if r.eligible and len(r.repeats) < 2]
    if unmeasured:
        lines.append(
            "- Noise could not be measured (fewer than two repeats) for: "
            + ", ".join(unmeasured)
            + "."
        )
    kept = screen(rows)
    lines += [
        "",
        "Stage-1 cut (every eligible configuration within noise of third place, at any speed): "
        + (", ".join(r.name for r in kept) or "none")
        + ".",
        "",
    ]
    return "\n".join(lines) + "\n"
