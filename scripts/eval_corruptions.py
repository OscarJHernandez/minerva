#!/usr/bin/env python3
"""Corruption test: every known model mistake applied to every reference call of the benchmark.

    uv run python scripts/eval_corruptions.py                       # PR tier: report to stdout
    uv run python scripts/eval_corruptions.py --json-out docs/benchmarks/corruption.json \
        --out docs/benchmarks/corruption.md
    uv run python scripts/eval_corruptions.py --check                # compare with the stored report
    uv run python scripts/eval_corruptions.py --full                 # run every corrupted call (slow)

Exit status is 1 when a corrupted call produced a different verified result or an untyped failure, or when
`--check` finds the stored report out of date. See `minerva_api.agent.evaluation.corruption`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minerva_api.agent.evaluation import corruption
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

DEFAULT = Path("docs/benchmarks/corruption.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--full", action="store_true", help="run every corrupted call through the engine"
    )
    parser.add_argument(
        "--without-cleaning",
        action="store_true",
        help="skip prepare_arguments: what the cleaning layer adds, on a sample (every call is run)",
    )
    parser.add_argument("--per-pair", type=int, default=1, help="calls run per operator and tool")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--out", type=Path, help="Markdown report")
    parser.add_argument("--check", nargs="?", const=DEFAULT, type=Path, metavar="STORED")
    args = parser.parse_args(argv)

    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    report = corruption.run_corruptions(
        ctx, full=args.full, clean=not args.without_cleaning, per_pair=args.per_pair
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out:
        args.out.write_text(corruption.markdown(report))
    if not args.json_out and not args.out and not args.check:
        sys.stdout.write(corruption.markdown(report))

    status = 0
    if any(report["totals"][d] for d in corruption.DEFECTS):
        print("DEFECT: see the report", file=sys.stderr)
        status = 1
    if args.check is not None:
        stored = json.loads(args.check.read_text())
        if corruption.deterministic_view(stored) != corruption.deterministic_view(report):
            print(f"{args.check} is out of date: regenerate it", file=sys.stderr)
            status = 1
        else:
            print("corruption report is up to date")
    return status


if __name__ == "__main__":
    sys.exit(main())
