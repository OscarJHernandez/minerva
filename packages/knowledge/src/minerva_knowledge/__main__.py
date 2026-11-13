"""Command line: `python -m minerva_knowledge verify` checks every entry."""

from __future__ import annotations

import sys
from dataclasses import replace

from .checks import run_check
from .loader import load_entries


def _flag(args: list[str], name: str) -> str | None:
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return None


def _distribution_benchmark(args: list[str]) -> int:
    import json
    from pathlib import Path

    from minerva_engine import Engine, Limits

    from .distribution_benchmark import (
        compare,
        format_distribution_report,
        run_distribution_benchmark,
    )
    from .kb import KnowledgeBase

    report = run_distribution_benchmark(
        KnowledgeBase.load(), Engine(Limits(timeout_seconds=30, max_concurrency=2))
    )
    print(format_distribution_report(report))
    document = report.to_json(label=_flag(args, "--label") or "")
    out = _flag(args, "--json")
    if out:
        Path(out).write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {out}")
    regressed = False
    baseline = _flag(args, "--compare")
    if baseline:
        text, regressed = compare(json.loads(Path(baseline).read_text(encoding="utf-8")), document)
        print("\n== comparison with " + baseline + " ==\n" + text)
    return 1 if report.failures or regressed else 0


def _benchmark(args: list[str] | None = None) -> int:
    from minerva_engine import Engine, Limits

    if args and _flag(args, "--only") == "distributions":
        return _distribution_benchmark(args)

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
    print("\n== distribution calculus ==")
    distributions = _distribution_benchmark(args or [])
    return 1 if report.failures or chains.failures or distributions else 0


def _matrix_benchmark(args: list[str]) -> int:
    from pathlib import Path

    from minerva_engine import Engine, Limits

    from .matrix_benchmark import (
        convergence_line,
        format_markdown,
        format_report,
        run_matrix_benchmark,
        write_section,
    )

    def option(name: str, default: str) -> str:
        return args[args.index(name) + 1] if name in args else default

    repeat = int(option("--repeat", "1"))
    engine = Engine(Limits(timeout_seconds=20, max_concurrency=2))
    report = run_matrix_benchmark(engine, repeat=repeat)
    print(format_report(report, trace="--trace" in args))
    extra: list[str] = []
    if "--convergence" in args:
        extra = [convergence_line()]
        print(extra[0])
    if "--out" in args:
        label = option("--section", "current")
        write_section(
            Path(option("--out", "")), label, format_markdown(report, label, repeat, extra)
        )
        print(f"\nwrote section '{label}' to {option('--out', '')}")
    return 1 if report.failures or report.stale_pending else 0


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
        return _benchmark(argv[1:])
    if command == "matrix-benchmark":
        return _matrix_benchmark(argv[1:])
    if command == "generate":
        return _generate(check="--check" in argv[1:])
    if command != "verify":
        print(
            "usage: python -m minerva_knowledge verify | benchmark [--only distributions] "
            "[--json PATH] [--compare BASELINE.json] [--label NAME] | matrix-benchmark "
            "[--repeat N] [--trace] [--convergence] [--section LABEL] [--out FILE] | "
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
