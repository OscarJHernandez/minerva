#!/usr/bin/env python3
"""Which diagnostic codes do real tool failures carry? Count them from the API's JSON log.

    uv run python scripts/diagnostic_codes.py api.log [more.log ...]
    kubectl logs deploy/minerva-api | uv run python scripts/diagnostic_codes.py -
    uv run python scripts/diagnostic_codes.py api.log --json --top 10 --since 2026-11-01

Reads the `tool.failed` and `tool.declined` lines (docs/observability.md, "Tool outcomes in the log") and
prints the most frequent (tool, reason_code, error_category) combinations, most frequent first. The most
common code in real use says which normalisation or schema change to add next. It reads only the closed
fields the log already carries (never an expression, a symbol or a message), writes none, and ignores any
line that is not a JSON object.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from typing import Any

EVENTS = ("tool.failed", "tool.declined")
FIELDS = ("event", "tool", "reason_code", "error_category", "outcome_kind")


def records(lines: Iterable[str], since: str | None = None) -> Iterable[dict[str, Any]]:
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("event") not in EVENTS:
            continue
        stamp = record.get("timestamp")
        if since and isinstance(stamp, str) and stamp < since:
            continue
        yield record


def count(lines: Iterable[str], since: str | None = None) -> Counter[tuple[str, ...]]:
    """Counts of (event, tool, reason_code, error_category); a missing field is `-`."""
    counts: Counter[tuple[str, ...]] = Counter()
    for record in records(lines, since):
        counts[tuple(str(record.get(f) or "-") for f in FIELDS[:4])] += 1
    return counts


def render(counts: Counter[tuple[str, ...]], top: int) -> str:
    rows = counts.most_common(top)
    if not rows:
        return "no tool failures or declines in the log\n"
    out = [
        "count  event          tool                        reason_code                    category"
    ]
    for (event, tool, code, category), n in rows:
        out.append(f"{n:5d}  {event:<14} {tool:<27} {code:<30} {category}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("logs", nargs="+", help="log files, or - for standard input")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--since", help="ISO timestamp prefix: ignore earlier lines")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    counts: Counter[tuple[str, ...]] = Counter()
    for name in args.logs:
        if name == "-":
            counts += count(sys.stdin, args.since)
        else:
            with open(name, encoding="utf-8", errors="replace") as handle:
                counts += count(handle, args.since)
    if args.json:
        rows = [
            dict(zip(FIELDS[:4], key, strict=True), count=n)
            for key, n in counts.most_common(args.top)
        ]
        sys.stdout.write(json.dumps(rows, indent=2) + "\n")
    else:
        sys.stdout.write(render(counts, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
