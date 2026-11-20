"""The second held-out suite (`--suite heldout2`, cases_heldout2.yaml) is frozen and untouched.

Its 35 cases were written by an agent that was given only the capabilities document, the case schema and the tool
definitions: no trace, no report and no failure table of any run (2026-11-20). They exist to check that a gain on the
cases the work reads traces from is not a gain from reading them. So: the prompts are pinned, the suite is not part
of `all`, and its traces are not opened or tuned against until the close-out measurement. Editing a prompt, or
fixing a failure it shows, turns it into one more tuned suite; these tests make that a deliberate, visible act.

Known weaknesses, kept as written and not tightened (tightening would change difficulty after the freeze): a few
`reply_matches` patterns are so permissive that a numeric reply satisfies them (for example `flat|zero|vanish|0`),
so those cases are decided by the tool choice and the card, not by the wording.
"""

from __future__ import annotations

import hashlib
import json

from minerva_api.agent.evaluation.cases import cases_digest, load_suite

PROMPTS_DIGEST = "bf78f392fd85"
CASES_DIGEST = "a5242089b868"


def _prompts_digest() -> str:
    cases = load_suite("heldout2")
    blob = "\n".join(json.dumps([c.prompt, [f.prompt for f in c.followups]]) for c in cases)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def test_the_heldout2_prompts_are_frozen() -> None:
    assert _prompts_digest() == PROMPTS_DIGEST


def test_the_heldout2_cases_are_pinned() -> None:
    assert cases_digest(load_suite("heldout2")) == CASES_DIGEST


def test_heldout2_is_not_part_of_any_other_suite() -> None:
    held = {c.id for c in load_suite("heldout2")}
    assert len(held) >= 30
    for other in ("all", "heldout", "probe", "explore"):
        assert not held & {c.id for c in load_suite(other)}, other


def test_every_heldout2_case_has_an_independent_source_and_a_checkable_reference() -> None:
    for case in load_suite("heldout2"):
        assert case.id.startswith("h2-")
        assert case.source.strip(), case.id
        if case.expected_tool not in ("none", "ask") and not case.informational:
            assert case.reference is not None, (
                f"{case.id}: a case that expects a tool needs a reference call"
            )


def test_heldout2_covers_the_tools_by_use() -> None:
    expected = {c.expected_tool for c in load_suite("heldout2")}
    for tool in (
        "statistical_mechanics",
        "vector_calculus",
        "integrate",
        "matrix_calculus",
        "statistical_model",
        "analyze_lagrangian",
        "geometry",
        "ode_solve_symbolic",
        "check_identity",
        "ask",
        "none",
    ):
        assert tool in expected or any(
            tool in {a for a in (c.acceptable_tools or [])} for c in load_suite("heldout2")
        ), tool
