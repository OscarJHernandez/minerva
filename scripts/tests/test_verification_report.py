"""The verification report: its committed files are consistent, and the baseline really is the verifier before probes.

The full recomputation (`python scripts/verification_report.py --check`, about ten seconds) runs as its own CI step;
this file checks the committed summary and baseline without recomputing, and the comparison logic.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "verification_report", ROOT / "scripts" / "verification_report.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CURRENT = json.loads((ROOT / "docs/benchmarks/verification-summary.json").read_text())
BASELINE = json.loads((ROOT / "docs/benchmarks/verification-summary-baseline.json").read_text())


def test_totals_are_the_sums_of_the_rows_and_every_row_adds_up() -> None:
    for doc in (CURRENT, BASELINE):
        d = doc["deterministic"]
        for column in MODULE.COLUMNS:
            assert d["totals"][column] == sum(r[column] for r in d["rows"]), column
        for r in d["rows"]:
            counted = (
                r["correct"]
                + r["refused_as_expected"]
                + r["false_accepts"]
                + r["incorrect_accepted"]
                + r["false_rejects"]
                + r["pending"]
                + r["known_open"]
            )
            assert counted == r["problems"], r  # every problem lands in exactly one outcome


def test_the_current_verifier_has_no_false_accept_and_no_incorrect_accepted_result() -> None:
    totals = CURRENT["deterministic"]["totals"]
    assert totals["false_accepts"] == 0 and totals["incorrect_accepted"] == 0
    assert totals["false_rejects"] == 0  # tightening must not reject true identities


def test_the_baseline_is_the_verifier_without_probes_and_the_original_reference_report() -> None:
    before, after = BASELINE["deterministic"], CURRENT["deterministic"]
    assert before["verifier"] == "targeted probes off" and after["verifier"] == "targeted probes on"
    assert before["reference_report"] != "current"
    # The point of the comparison: the probes closed false accepts, and the corpus grew.
    assert before["totals"]["false_accepts"] > after["totals"]["false_accepts"]
    assert before["totals"]["counterexamples"] == 0 < after["totals"]["counterexamples"]
    assert before["totals"]["problems"] < after["totals"]["problems"]


def test_numerical_agreement_is_never_counted_as_a_proof() -> None:
    evidence = CURRENT["deterministic"]["evidence_of_accepted_reference_results"]
    assert set(evidence) == {*MODULE.EVIDENCE_KINDS, "unrecorded"}
    # symbolic proofs are a minority of accepted results: most results are checked numerically or structurally
    assert evidence["symbolic_proof"] < evidence["numeric_agreement"] + evidence["structural"]


def test_compare_reports_the_change_per_metric_and_per_domain() -> None:
    text = MODULE.compare(BASELINE, CURRENT)
    assert "false accepts" in text and "before" in text and "after" in text
    assert "adversarial / piecewise: false accepts 8 -> 0" in text
