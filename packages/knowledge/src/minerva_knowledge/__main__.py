"""Command line: `python -m minerva_knowledge verify` checks every entry."""

from __future__ import annotations

import sys

from .checks import run_check
from .loader import load_entries


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "verify"
    if command != "verify":
        print("usage: python -m minerva_knowledge verify", file=sys.stderr)
        return 2
    entries = load_entries()
    failures = 0
    stated = 0
    for entry in entries:
        report = run_check(entry)
        mark = "ok  " if report.passed else "FAIL"
        if entry.verification == "stated":
            stated += 1
        print(f"{mark} {entry.id:<40} {entry.verification:<11} {report.detail}")
        failures += 0 if report.passed else 1
    print(f"\n{len(entries)} entries, {failures} failed, {stated} stated (unchecked)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
