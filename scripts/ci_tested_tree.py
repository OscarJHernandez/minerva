"""Has a pull-request run already tested this exact tree? (used by the Deploy workflow)

A merge to main whose tree is byte-for-byte the tree a passing pull-request run tested does not need the same
fifteen minutes of CI again. The pull-request run records what it tested as a notice annotation on its final job
(`.github/workflows/ci.yml`, step "Record which tree this run tested"):

    title   Tested tree
    message tested-tree: <40 hex digits, the git tree of the merge commit that CI checked out>

This script looks for that annotation on recent successful pull-request runs of THIS repository (never a fork)
and prints `tested=true` or `tested=false` for `$GITHUB_OUTPUT`. It fails closed: an API error, a malformed tree,
a run that did not pass, or an annotation that is not exactly the attestation all give `false`, and Deploy then
runs the full CI as before. It only ever reads; the token needs `actions: read` and `checks: read`.

Trust: the attestation is written by workflow code on a branch of this repository, so whoever can push such a
branch could forge one. That person can already run code on the self-hosted runners, and the tree that is skipped
is the tree of a commit that has been merged into main; a forged attestation can skip tests, never add code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from typing import Any

MAX_RUNS = 20
FINAL_JOB = "CI OK"
TITLE = "Tested tree"
TREE = re.compile(r"[0-9a-f]{40}")
ATTESTATION = re.compile(r"tested-tree: ([0-9a-f]{40})")


def may_skip_event(event: str) -> bool:
    """Only a push to main may skip CI; a manual dispatch always runs it."""
    return event == "push"


def attested_trees(annotations: list[dict[str, Any]]) -> set[str]:
    """The trees that `annotations` attest, each only if the annotation is exactly the attestation."""
    trees: set[str] = set()
    for note in annotations:
        if note.get("annotation_level") != "notice" or note.get("title") != TITLE:
            continue
        match = ATTESTATION.fullmatch(str(note.get("message") or ""))
        if match:
            trees.add(match.group(1))
    return trees


def trusted_runs(runs: list[dict[str, Any]], repo: str) -> list[dict[str, Any]]:
    """Passing pull-request runs whose head branch lives in `repo` itself (a fork's run never counts)."""
    return [
        r
        for r in runs
        if r.get("event") == "pull_request"
        and r.get("conclusion") == "success"
        and (r.get("head_repository") or {}).get("full_name") == repo
    ]


def tree_was_tested(
    tree: str,
    repo: str,
    runs: list[dict[str, Any]],
    jobs_of: Callable[[int], list[dict[str, Any]]],
    annotations_of: Callable[[int], list[dict[str, Any]]],
) -> bool:
    if not TREE.fullmatch(tree):
        return False
    try:
        for run in trusted_runs(runs, repo)[:MAX_RUNS]:
            for job in jobs_of(int(run["id"])):
                passed = job.get("name") == FINAL_JOB and job.get("conclusion") == "success"
                if passed and tree in attested_trees(annotations_of(int(job["id"]))):
                    return True
    except Exception as exc:
        print(f"could not decide ({type(exc).__name__}); CI will run", file=sys.stderr)
        return False
    return False


def _gh(path: str) -> Any:
    out = subprocess.run(  # noqa: S603 - fixed program, repository-controlled path
        ["gh", "api", path],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    return json.loads(out)


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    tree = os.environ.get("TREE", "")
    tested = False
    if may_skip_event(os.environ.get("GITHUB_EVENT_NAME", "")):
        try:
            runs = _gh(
                f"repos/{repo}/actions/workflows/ci.yml/runs?event=pull_request&status=success&per_page=50"
            )["workflow_runs"]
        except Exception as exc:
            print(f"could not list runs ({type(exc).__name__}); CI will run", file=sys.stderr)
            runs = []
        tested = tree_was_tested(
            tree,
            repo,
            runs,
            lambda run_id: _gh(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")["jobs"],
            lambda job_id: _gh(f"repos/{repo}/check-runs/{job_id}/annotations?per_page=100"),
        )
    line = f"tested={'true' if tested else 'false'}"
    print(line)
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
