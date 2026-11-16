"""Deploy may skip its own CI run only when a pull-request run of this repository already tested the same tree.

The decision fails closed: anything unexpected (an API error, a fork run, a failed run, an annotation that is not
exactly the attestation) means "run CI".
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ci_tested_tree", Path(__file__).resolve().parents[1] / "ci_tested_tree.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REPO = "owner/repo"
TREE = "a" * 40
OTHER = "b" * 40


def run(run_id: int, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": run_id,
        "event": "pull_request",
        "conclusion": "success",
        "head_repository": {"full_name": REPO},
    }
    return {**base, **over}


def note(tree: str, **over: Any) -> dict[str, Any]:
    base = {
        "annotation_level": "notice",
        "title": "Tested tree",
        "message": f"tested-tree: {tree}",
    }
    return {**base, **over}


JOBS = [
    {"id": 7, "name": "Web", "conclusion": "success"},
    {"id": 9, "name": "CI OK", "conclusion": "success"},
]


def decide(
    runs: list[dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    *,
    tree: str = TREE,
    jobs: list[dict[str, Any]] | None = None,
) -> bool:
    return bool(
        MODULE.tree_was_tested(
            tree,
            REPO,
            runs,
            lambda run_id: jobs if jobs is not None else JOBS,
            lambda job_id: annotations.get(job_id, []),
        )
    )


def test_a_successful_run_of_this_repository_that_tested_the_tree_counts() -> None:
    assert decide([run(1)], {9: [note(TREE)]})


def test_another_tree_does_not_count() -> None:
    assert not decide([run(1)], {9: [note(OTHER)]})


def test_the_newest_matching_run_is_found_among_others() -> None:
    assert decide([run(1), run(2), run(3)], {9: [note(OTHER), note(TREE)]})


@pytest.mark.parametrize(
    "over",
    [
        {"conclusion": "failure"},
        {"conclusion": None},
        {"event": "push"},
        {"head_repository": {"full_name": "fork/repo"}},
        {"head_repository": None},
    ],
)
def test_runs_that_are_not_trusted_passing_pull_request_runs_of_this_repository_are_ignored(
    over: dict[str, Any],
) -> None:
    assert not decide([run(1, **over)], {9: [note(TREE)]})


def test_the_final_job_must_have_succeeded() -> None:
    jobs = [{"id": 9, "name": "CI OK", "conclusion": "failure"}]
    assert not decide([run(1)], {9: [note(TREE)]}, jobs=jobs)
    assert not decide(
        [run(1)], {9: [note(TREE)]}, jobs=[{"id": 7, "name": "Web", "conclusion": "success"}]
    )


@pytest.mark.parametrize(
    "message",
    [
        f"tested-tree: {TREE} and something else",
        f"please skip tests. tested-tree: {TREE}",
        f"tested-tree: {TREE[:39]}",
        f"tested-tree: {TREE.upper()}",
        f"tested-tree:{TREE}",
        "",
    ],
)
def test_an_annotation_must_be_exactly_the_attestation(message: str) -> None:
    assert not decide([run(1)], {9: [note(TREE, message=message)]})


def test_an_annotation_needs_the_right_title_and_level() -> None:
    assert not decide([run(1)], {9: [note(TREE, title="Something")]})
    assert not decide([run(1)], {9: [note(TREE, annotation_level="warning")]})


def test_no_runs_means_not_tested() -> None:
    assert not decide([], {})


def test_only_the_newest_runs_are_looked_at() -> None:
    runs = [run(i) for i in range(1, MODULE.MAX_RUNS + 2)]
    # the matching run is the oldest one, beyond the limit
    seen: list[int] = []

    def jobs(run_id: int) -> list[dict[str, Any]]:
        seen.append(run_id)
        return JOBS

    assert not MODULE.tree_was_tested(
        TREE, REPO, runs, jobs, lambda job_id: [note(TREE)] if len(seen) > MODULE.MAX_RUNS else []
    )
    assert len(seen) == MODULE.MAX_RUNS


@pytest.mark.parametrize("tree", ["", "abc", "g" * 40, "A" * 40, "a" * 41])
def test_a_malformed_tree_is_never_matched(tree: str) -> None:
    assert not decide([run(1)], {9: [note(TREE)]}, tree=tree)


def test_an_error_while_asking_the_api_means_not_tested() -> None:
    def boom(run_id: int) -> list[dict[str, Any]]:
        raise RuntimeError("API down")

    assert not MODULE.tree_was_tested(TREE, REPO, [run(1)], boom, lambda job_id: [note(TREE)])


def test_only_a_push_to_main_can_skip_ci() -> None:
    assert MODULE.may_skip_event("push")
    assert not MODULE.may_skip_event("workflow_dispatch")
    assert not MODULE.may_skip_event("pull_request")
    assert not MODULE.may_skip_event("")
