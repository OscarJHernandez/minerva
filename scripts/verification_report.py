"""One report of how well Minerva's verification detects wrong mathematics.

It reads artefacts that already exist and adds nothing that could be tuned to flatter the result:

  docs/benchmarks/reference.json       independent reference problems (outcomes recorded by the reference benchmark)
  packages/engine/tests/adversarial/   false identities and true identities, run through `verify_equal` here
  docs/benchmarks/geometry.json        geometry corpus with closed-form references
  docs/benchmarks/distributions.json   Dirac delta corpus (supported, unsupported, invalid transformations)
  the agent evaluation suites          coverage only (cases, turns, failure classes); model results are optional

For every source and domain it counts what matters for a verifier: correct answers, correct refusals,
**false accepts** (a refusal or rejection was due and an answer was given), **incorrect results accepted**
(an answer that disagrees with the oracle), **false rejects** (a correct answer was not given), pending problems
(known gaps, never counted as correct), known open defects, and counterexamples found by targeted probes. It
keeps the kinds of evidence apart: a symbolic proof, numerical agreement, a structural check, a conditional
result. Numerical agreement is never called a proof.

  python scripts/verification_report.py            # write docs/benchmarks/verification-summary.json + docs/verification-report.md
  python scripts/verification_report.py --check    # fail if the deterministic part differs from the committed file
  python scripts/verification_report.py --compare docs/benchmarks/verification-summary-baseline.json
  python scripts/verification_report.py --probes off --reference-rev f0015b0 --json-out OUT.json
        # the "before": the numerical verifier without targeted probes, the reference report of an earlier commit

Timings and memory are reported in a separate `measurements` section that `--check` ignores.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / "packages/engine/tests")
)  # the adversarial corpus lives with the tests

SUMMARY = ROOT / "docs/benchmarks/verification-summary.json"
REPORT = ROOT / "docs/verification-report.md"
FORMAT = 1
COLUMNS = (
    "problems",
    "correct",
    "refused_as_expected",
    "false_accepts",
    "incorrect_accepted",
    "false_rejects",
    "pending",
    "known_open",
    "counterexamples",
)
EVIDENCE_KINDS = ("symbolic_proof", "numeric_agreement", "structural", "conditional", "refusal")


def _row(source: str, domain: str, **counts: int) -> dict[str, Any]:
    row: dict[str, Any] = {"source": source, "domain": domain, **dict.fromkeys(COLUMNS, 0)}
    row.update(counts)
    return row


# --- sources -----------------------------------------------------------------------------------------------
def reference_rows(problems: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Reference problems: the oracle is independent of Minerva; a refusal problem expects no answer."""
    by_domain: dict[str, dict[str, Any]] = {}
    evidence: Counter[str] = Counter()
    for p in problems:
        row = by_domain.setdefault(p["category"], _row("reference", p["category"]))
        row["problems"] += 1
        outcome, expected = p["outcome"], p["expected"]
        if outcome == "correct":
            row["correct"] += 1
        elif outcome == "refused_as_expected":
            row["refused_as_expected"] += 1
        elif outcome == "wrong":
            row["false_accepts" if expected == "refused" else "incorrect_accepted"] += 1
        elif outcome == "unsupported_regression":
            row["false_rejects"] += 1
        elif outcome in ("pending", "pending_now_passes"):
            row["pending"] += 1
        evidence[
            p.get("evidence", "unrecorded")
            if outcome in ("correct", "refused_as_expected")
            else "none"
        ] += 1
    return sorted(by_domain.values(), key=lambda r: r["domain"]), evidence


def adversarial_rows(probe: bool) -> tuple[list[dict[str, Any]], Counter[str], list[float]]:
    by_domain: dict[str, dict[str, Any]] = {}
    by_probe: Counter[str] = Counter()
    timings: list[float] = []
    for case, outcome, verification in _timed(probe, timings):
        row = by_domain.setdefault(case.domain, _row("adversarial", case.domain))
        row["problems"] += 1
        if case.defect and outcome != "correct":
            row["known_open"] += 1  # a documented limit (strict xfail), not a new false accept
        elif outcome == "correct":
            row["correct" if case.expect == "accept" else "refused_as_expected"] += 1
        elif outcome == "false_accept":
            row["false_accepts"] += 1
        else:
            row["false_rejects"] += 1
        if case.expect == "reject" and verification.probe_kind is not None:
            row["counterexamples"] += 1
            by_probe[verification.probe_kind] += 1
    return sorted(by_domain.values(), key=lambda r: r["domain"]), by_probe, timings


def _timed(probe: bool, timings: list[float]) -> list[Any]:
    """Every corpus case through the verifier, timing each (the timings go to the non-gated section)."""
    from adversarial.corpus import CASES
    from adversarial.run import outcome_of, verify_case

    out = []
    for case in CASES:
        start = time.monotonic()
        verification = verify_case(case, probe=probe)
        timings.append(time.monotonic() - start)
        out.append((case, outcome_of(case, verification), verification))
    return out


def geometry_row(data: dict[str, Any]) -> dict[str, Any]:
    cover = data["deterministic"]["coverage"]
    return _row(
        "geometry",
        "differential geometry",
        problems=cover["corpus"] + cover["refusal_cases"],
        correct=cover["computed_and_correct"],
        refused_as_expected=cover["refusal_cases"],
    )


def distribution_row(data: dict[str, Any]) -> dict[str, Any]:
    s = data["summary"]
    return _row(
        "distribution benchmark",
        "dirac delta",
        problems=s["cases"],
        correct=s["supported_correct"],
        refused_as_expected=s["unsupported_refused_correctly"] + s["invalid_rejected"],
        false_accepts=(s["unsupported_cases"] - s["unsupported_refused_correctly"])
        + (s["invalid_cases"] - s["invalid_rejected"]),
        incorrect_accepted=s["supported_answered"] - s["supported_correct"],
        false_rejects=s["supported_cases"] - s["supported_answered"],
    )


def agent_suite() -> dict[str, Any]:
    """Coverage of the agent evaluation: what is asked, and which failures can be told apart."""
    from minerva_api.agent.evaluation.cases import load_suite
    from minerva_api.agent.evaluation.score import FailureClass

    core, reliability = load_suite("core"), load_suite("reliability")
    return {
        "core_cases": len(core),
        "reliability_cases": len(reliability),
        "reliability_turns": sum(len(c.turns()) for c in reliability),
        "failure_classes": sorted(f.value for f in FailureClass),
    }


# --- assembling ----------------------------------------------------------------------------------------------
def _reference_problems(rev: str | None) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]]
    if rev is None:
        problems = json.loads((ROOT / "docs/benchmarks/reference.json").read_text())["problems"]
        return problems
    shown = subprocess.run(  # noqa: S603 - fixed argv; the revision is a command-line argument of a dev tool
        ["git", "show", f"{rev}:docs/benchmarks/reference.json"],  # noqa: S607
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    problems = json.loads(shown.stdout)["problems"]
    return problems


def build(probe: bool = True, reference_rev: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    ref_rows, evidence = reference_rows(_reference_problems(reference_rev))
    adv_rows, by_probe, timings = adversarial_rows(probe)
    rows = [
        *ref_rows,
        *adv_rows,
        geometry_row(json.loads((ROOT / "docs/benchmarks/geometry.json").read_text())),
        distribution_row(json.loads((ROOT / "docs/benchmarks/distributions.json").read_text())),
    ]
    totals = {k: sum(r[k] for r in rows) for k in COLUMNS}
    deterministic: dict[str, Any] = {
        "verifier": "targeted probes on" if probe else "targeted probes off",
        "reference_report": reference_rev or "current",
        "rows": rows,
        "totals": totals,
        "evidence_of_accepted_reference_results": {
            k: evidence.get(k, 0) for k in (*EVIDENCE_KINDS, "unrecorded")
        },
        "counterexamples_by_probe": dict(sorted(by_probe.items())),
        "agent_suite": agent_suite(),
    }
    return {
        "format": FORMAT,
        "deterministic": deterministic,
        "measurements": {
            "adversarial_total_s": round(sum(timings), 2),
            "adversarial_median_ms": round(statistics.median(timings) * 1000, 1) if timings else 0,
            "adversarial_max_ms": round(max(timings) * 1000, 1) if timings else 0,
            "report_wall_s": round(time.monotonic() - started, 2),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1),
        },
    }


# --- rendering ------------------------------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]], totals: dict[str, int]) -> list[str]:
    head = "| Source | Domain | " + " | ".join(c.replace("_", " ") for c in COLUMNS) + " |"
    lines = [head, "| --- | --- | " + " | ".join("---:" for _ in COLUMNS) + " |"]
    for r in rows:
        lines.append(
            f"| {r['source']} | {r['domain']} | " + " | ".join(str(r[c]) for c in COLUMNS) + " |"
        )
    lines.append("| **total** | | " + " | ".join(f"**{totals[c]}**" for c in COLUMNS) + " |")
    return lines


def markdown(doc: dict[str, Any]) -> str:
    d, m = doc["deterministic"], doc["measurements"]
    t = d["totals"]
    lines = [
        "# Verification report",
        "",
        "Generated by `python scripts/verification_report.py`; see its docstring for the sources and the commands that "
        "reproduce and compare it. The numbers below are deterministic; timings are in the last section and are not gated.",
        "",
        f"Verifier: **{d['verifier']}**. Reference report: {d['reference_report']}.",
        "",
        f"Across {t['problems']} problems: {t['correct']} correct answers, {t['refused_as_expected']} correct refusals "
        f"or rejections, **{t['false_accepts']} false accepts**, **{t['incorrect_accepted']} incorrect results accepted**, "
        f"{t['false_rejects']} false rejects, {t['pending']} pending (known gaps, not counted as correct), "
        f"{t['known_open']} known open limits, {t['counterexamples']} counterexamples found by targeted probes.",
        "",
        "A *false accept* is an answer or acceptance where a refusal or rejection was due; an *incorrect result accepted* "
        "is an answer that disagrees with its independent oracle; a *false reject* is a correct answer that was not given. "
        "A *known open limit* is a documented defect pinned by a strict expected failure "
        "(`docs/verification-defects.md`).",
        "",
        *_table(d["rows"], t),
        "",
        "## Kinds of evidence",
        "",
        "What backs a result the engine accepted (reference problems; the engine reports it, the report never upgrades it). "
        "Only a symbolic zero test is a proof of an identity; numerical agreement is evidence, not proof.",
        "",
        "| symbolic proof | numerical agreement | structural check | conditional result | refusal | unrecorded |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            str(d["evidence_of_accepted_reference_results"][k])
            for k in (*EVIDENCE_KINDS, "unrecorded")
        )
        + " |",
        "",
        "Counterexamples found by probe family: "
        + (", ".join(f"{k} {v}" for k, v in d["counterexamples_by_probe"].items()) or "none")
        + ".",
        "",
        "## Agent evaluation",
        "",
        f"{d['agent_suite']['core_cases']} core cases and {d['agent_suite']['reliability_cases']} reliability cases "
        f"({d['agent_suite']['reliability_turns']} live turns). Failure classes a run can report: "
        + ", ".join(f"`{c}`" for c in d["agent_suite"]["failure_classes"])
        + ". Model results come from `scripts/eval_models.py` (see `docs/model-evaluation.md`); they depend on a model "
        "host and are not part of this report.",
        "",
        "## Measurements (not gated)",
        "",
        f"Adversarial corpus: {m['adversarial_total_s']} s in total, median {m['adversarial_median_ms']} ms and "
        f"maximum {m['adversarial_max_ms']} ms per case. Report wall time {m['report_wall_s']} s; peak memory "
        f"{m['peak_rss_mb']} MB.",
    ]
    return "\n".join(lines) + "\n"


def compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    a, b = old["deterministic"], new["deterministic"]
    lines = [
        f"before: {a['verifier']}, reference report {a['reference_report']}",
        f"after:  {b['verifier']}, reference report {b['reference_report']}",
        "",
        f"{'metric':<22}{'before':>8}{'after':>8}{'change':>8}",
    ]
    for key in COLUMNS:
        x, y = a["totals"][key], b["totals"][key]
        lines.append(f"{key.replace('_', ' '):<22}{x:>8}{y:>8}{y - x:>+8}")
    index = {(r["source"], r["domain"]): r for r in a["rows"]}
    changed = []
    for r in b["rows"]:
        before = index.get((r["source"], r["domain"]))
        if before is None:
            changed.append(f"  new: {r['source']} / {r['domain']} ({r['problems']} problems)")
        elif any(
            before[c] != r[c] for c in ("false_accepts", "false_rejects", "incorrect_accepted")
        ):
            changed.append(
                f"  {r['source']} / {r['domain']}: false accepts {before['false_accepts']} -> {r['false_accepts']}, "
                f"false rejects {before['false_rejects']} -> {r['false_rejects']}"
            )
    return "\n".join(lines + (["", "domains that changed:", *changed] if changed else [])) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if the deterministic part differs"
    )
    parser.add_argument(
        "--compare", metavar="OLD.json", help="print the change from a saved report"
    )
    parser.add_argument(
        "--probes", choices=("on", "off"), default="on", help="off: the verifier before probes"
    )
    parser.add_argument(
        "--reference-rev", help="read docs/benchmarks/reference.json as of this git revision"
    )
    parser.add_argument(
        "--json-out", help="write the machine-readable report here instead of the default"
    )
    args = parser.parse_args(argv)

    doc = build(probe=args.probes == "on", reference_rev=args.reference_rev)
    if args.compare:
        old = json.loads(Path(args.compare).read_text())
        print(compare(old, doc))
        return 0
    if args.check:
        committed = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else None
        if committed is None or committed["deterministic"] != doc["deterministic"]:
            print(
                "the verification summary differs from docs/benchmarks/verification-summary.json; "
                "run: python scripts/verification_report.py",
                file=sys.stderr,
            )
            return 1
        return 0
    target = Path(args.json_out) if args.json_out else SUMMARY
    target.write_text(json.dumps(doc, indent=2) + "\n")
    if not args.json_out:
        REPORT.write_text(markdown(doc))
    print(markdown(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
