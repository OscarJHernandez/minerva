#!/usr/bin/env python3
"""Offline orchestration suite: the real agent loop and tools with a replayed model (no network).

    uv run python scripts/eval_offline.py --json-out docs/benchmarks/orchestration-offline.json \
        --out docs/benchmarks/orchestration-offline.md
    uv run python scripts/eval_offline.py --check            # deterministic fields against the stored report
    uv run python scripts/eval_offline.py --compare docs/benchmarks/orchestration-offline-baseline.json

`--check` fails on any change of a deterministic field (calls, estimated tokens, marks, correctness);
timings are written to the report and never gated. `--compare` lists what changed against an older report,
flagging the changes for the worse as REGRESSION. See `minerva_api.agent.evaluation.offline`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minerva_api.agent.evaluation import offline
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine, Limits
from minerva_knowledge import KnowledgeBase

DEFAULT = Path("docs/benchmarks/orchestration-offline.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--ids", help="comma-separated trajectory ids (default: all)")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--out", type=Path, help="Markdown report")
    parser.add_argument("--check", nargs="?", const=DEFAULT, type=Path, metavar="STORED")
    parser.add_argument("--compare", type=Path, metavar="OLD.json")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for t in offline.TRAJECTORIES:
            print(f"{t.id:42} {t.category:10} {t.description}")
        return 0

    ctx = ToolContext(
        engine=Engine(Limits(timeout_seconds=120, max_concurrency=2)), kb=KnowledgeBase.load()
    )
    ids = args.ids.split(",") if args.ids else None
    report = offline.run_suite(ctx, ids)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out:
        args.out.write_text(offline.markdown(report))
    if not (args.json_out or args.out or args.check or args.compare):
        sys.stdout.write(offline.markdown(report))
    status = 0
    if args.compare:
        lines = offline.compare(json.loads(args.compare.read_text()), report)
        print("\n".join(lines) or "no differences in the comparable trajectories")
        status = 1 if any(line.startswith("REGRESSION") for line in lines) else 0
    if args.check:
        stored = json.loads(args.check.read_text())
        mine, theirs = offline.deterministic_view(report), offline.deterministic_view(stored)
        if ids:  # a subset: compare those trajectories only
            theirs["results"] = {k: v for k, v in theirs["results"].items() if k in mine["results"]}
            theirs["summary"] = mine["summary"]
        if mine != theirs:
            for key in sorted(set(mine["results"]) | set(theirs["results"])):
                if mine["results"].get(key) != theirs["results"].get(key):
                    print(
                        f"changed: {key}\n  stored: {theirs['results'].get(key)}\n  now:    {mine['results'].get(key)}"
                    )
            print("offline orchestration report differs: rerun with --json-out and review the diff")
            return 1
        print("offline orchestration report is up to date")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
