"""The probe suite (`--suite probe`, cases_probe.yaml) is frozen.

It was written while the 2026-11-18 baseline run was still going, before any trace or report of that run was
opened, so it is not shaped by the failures a later experiment tries to fix. Editing a prompt would turn it into
a suite tuned to those failures: these tests make that a deliberate, visible act.
"""

from __future__ import annotations

import hashlib
import json

from minerva_api.agent.evaluation.cases import cases_digest, load_suite

# The digest of the prompts alone (each first-turn prompt and its follow-up prompts, in file order). It must not
# change when expectations are tightened or an `accept:` block is added; it changes only if a question changes.
PROMPTS_DIGEST = "bd8f394f3c86"
# The digest of the whole cases (prompts, expectations, references). A change here with the prompts digest
# unchanged means an expectation changed: say so in docs/reliability/experiment-log.md.
CASES_DIGEST = "6c095266ac4c"

CATEGORIES_COVERED = {
    "calculus",
    "false_claim",
    "symbolic",
    "ambiguous",
    "multistep",
    "unsupported",
    "retrieval",
    "followup",
    "physics",
    "tensor_matrix",
}


def _prompts_digest() -> str:
    cases = load_suite("probe")
    blob = "\n".join(json.dumps([c.prompt, [f.prompt for f in c.followups]]) for c in cases)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def test_the_probe_prompts_are_frozen() -> None:
    assert _prompts_digest() == PROMPTS_DIGEST


def test_the_probe_cases_are_pinned() -> None:
    assert cases_digest(load_suite("probe")) == CASES_DIGEST


def test_the_probe_suite_is_not_part_of_all_or_heldout() -> None:
    probe = {c.id for c in load_suite("probe")}
    assert probe
    assert not probe & {c.id for c in load_suite("all")}
    assert not probe & {c.id for c in load_suite("heldout")}


def test_every_probe_case_names_an_independent_source() -> None:
    for case in load_suite("probe"):
        assert case.id.startswith("probe-")
        assert case.source.strip(), case.id
        for follow in case.followups:
            assert follow.prompt.strip()


def test_the_probe_suite_covers_the_failure_classes_it_was_written_for() -> None:
    cases = load_suite("probe")
    assert len(cases) >= 35
    assert {c.category for c in cases} >= CATEGORIES_COVERED
    ids = {c.id for c in cases}
    # one case each for the situations the reliability work names
    assert {
        "probe-false-exp-sum",  # false identity
        "probe-true-square-difference",  # its true twin
        "probe-log-power-needs-sign",  # missing assumption
        "probe-cancel-then-differentiate",  # restriction across turns
        "probe-laurent-expansion",  # unsupported, not impossible
        "probe-premise-is-not-proof",  # a premise is not a proof
        "probe-text-derivative",  # plain-text notation
        "probe-explore-derivative-then-integral",  # multi-step
    } <= ids
    # at least a few cases accept an alternative valid tool
    assert sum(1 for c in cases if c.also_ok) >= 8
    # and at least a few are multi-turn
    assert sum(1 for c in cases if c.followups) >= 6
