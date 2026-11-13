"""Command line: `python -m minerva_knowledge verify` checks every entry."""

from __future__ import annotations

import sys
from dataclasses import replace

from .checks import run_check
from .loader import load_entries


def _benchmark() -> int:
    from minerva_engine import Engine, Limits

    from .benchmark import (
        format_report,
        format_retrieval,
        format_transform_report,
        run_benchmark,
        run_retrieval,
        run_transform_benchmark,
    )
    from .kb import KnowledgeBase

    kb = KnowledgeBase.load()
    engine = Engine(Limits(timeout_seconds=30, max_concurrency=2))
    report = run_benchmark(kb, engine)
    print(format_report(report))
    print("\n== retrieval quality ==")
    print(format_retrieval(run_retrieval(kb, engine)))
    print("\n== multi-step transformations ==")
    chains = run_transform_benchmark(kb, engine)
    print(format_transform_report(chains))
    return 1 if report.failures or chains.failures else 0


def _step_benchmark(argv: list[str]) -> int:
    """Controlled-transformation cases. `--adapter existing|steps`, `--repeat N` and `--timing` for the
    local latency run, `--write-baseline` to record the existing functionality's pass/fail."""
    from minerva_engine import Engine, Limits

    from .kb import KnowledgeBase
    from .loader import DATA_DIR
    from .step_benchmark import (
        ADAPTERS,
        BASELINE_FILE,
        baseline_json,
        format_step_report,
        run_step_benchmark,
    )

    def option(name: str, default: str) -> str:
        return (
            argv[argv.index(name) + 1]
            if name in argv and argv.index(name) + 1 < len(argv)
            else default
        )

    adapter = option("--adapter", "steps" if "steps" in ADAPTERS else "existing")
    if adapter not in ADAPTERS:
        print(f"unknown adapter {adapter!r}; choose from {sorted(ADAPTERS)}", file=sys.stderr)
        return 2
    repeat = int(option("--repeat", "1"))
    timing = "--timing" in argv or repeat > 1
    kb = KnowledgeBase.load()
    engine = Engine(Limits(timeout_seconds=30, max_concurrency=2))
    report = run_step_benchmark(kb, engine, adapter, repeat=repeat)
    print(format_step_report(report, timing=timing))
    if "--write-baseline" in argv:
        if adapter != "existing":
            print(
                "the baseline records the existing functionality: use --adapter existing",
                file=sys.stderr,
            )
            return 2
        (DATA_DIR / BASELINE_FILE).write_text(baseline_json(report), encoding="utf-8")
        print(f"wrote {BASELINE_FILE}")
        return 0
    return 1 if report.failures and adapter == "steps" else 0


def _generate(check: bool) -> int:
    from minerva_engine import Engine, Limits

    from . import generate

    generated = generate.build(
        generate.SEEDS, Engine(Limits(timeout_seconds=30, max_concurrency=2))
    )
    for rejection in generated.rejected:
        print(f"REJECTED {rejection.seed_id}: {rejection.reason}", file=sys.stderr)
    if generated.rejected:
        print(f"{len(generated.rejected)} seed(s) rejected; nothing written.", file=sys.stderr)
        return 1
    if check:
        stale = generate.stale_files(generated)
        for name in stale:
            print(f"STALE {name}: run `python -m minerva_knowledge generate`", file=sys.stderr)
        print(f"{generated.count} generated entries, {len(stale)} stale file(s)")
        return 1 if stale else 0
    for path in generate.write(generated):
        print(f"wrote {path.name}")
    print(f"{generated.count} generated entries")
    return 0


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "verify"
    if command == "benchmark":
        return _benchmark()
    if command == "step-benchmark":
        return _step_benchmark(argv[1:])
    if command == "generate":
        return _generate(check="--check" in argv[1:])
    if command != "verify":
        print(
            "usage: python -m minerva_knowledge verify | benchmark | "
            "step-benchmark [--adapter existing|steps] [--repeat N] [--timing] [--write-baseline] | "
            "generate [--check]",
            file=sys.stderr,
        )
        return 2
    entries = load_entries()
    failures = 0
    stated = 0
    for entry in entries:
        report = run_check(entry)
        # `symbolic` means a symbolic zero was found, not only numeric agreement.
        if entry.verification == "symbolic" and report.passed and not report.symbolic_proof:
            report = replace(
                report,
                passed=False,
                detail="declared symbolic but no symbolic zero: " + report.detail,
            )
        mark = "ok  " if report.passed else "FAIL"
        if entry.verification == "stated":
            stated += 1
        print(f"{mark} {entry.id:<40} {entry.verification:<11} {report.detail}")
        failures += 0 if report.passed else 1
    print(f"\n{len(entries)} entries, {failures} failed, {stated} stated (unchecked)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
