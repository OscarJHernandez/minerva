"""Benchmarks for the discovery of conserved quantities on generated Hamiltonians.

    uv run python scripts/bench_invariants.py check              # fixtures regenerate byte for byte
    uv run python scripts/bench_invariants.py run                # adaptive search on the fixed-seed cases
    uv run python scripts/bench_invariants.py compare [--out docs/invariant-benchmark.md]
                                                                 # adaptive vs fixed-parameter search
    uv run python scripts/bench_invariants.py sweep --seeds 40   # larger randomized evaluation (offline)
    uv run python scripts/bench_invariants.py promote osc2_res:17:2:1   # failing case -> fixed regression

The reference side lives in `packages/engine/tests/bench` and shares no code with the discovery
implementation; Minerva is given the Hamiltonian and the variable names only. Everything runs through the
real `Engine` (sandboxed child process), so runtimes and peak memory are those of production. Fixed seeds
run in CI (`test_discovery_generated.py`); `sweep` and `compare` are for offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "engine" / "tests"))

import sympy as sp  # noqa: E402
from bench import cases, evaluate, legendre  # noqa: E402
from bench.systems import BUILDERS  # noqa: E402
from minerva_engine import Engine, Limits  # noqa: E402
from minerva_engine.latexparse import parse_latex  # noqa: E402

FIXED_CONFIGS = [(2, 1, 1), (2, 2, 1), (2, 3, 1), (2, 4, 1)]  # (degree, power bound, harmonics)
DEFAULT_FIXED = (2, 3, 1)  # the defaults of the original search


def environment() -> dict[str, str]:
    """What a stored result needs to be reproduced: engine commit, versions, generator version."""
    import platform
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "engine_commit": commit,
        "python": platform.python_version(),
        "sympy": sp.__version__,
        "generator_version": str(cases.GENERATOR_VERSION),
        "budgets": "quick 8s/150, standard 30s/350, thorough 90s/600 (seconds / unknown coefficients)",
    }


def engine() -> Engine:
    return Engine(Limits(timeout_seconds=120, max_concurrency=1, memory_mb=2048))


def run_adaptive(eng: Engine, case: cases.Case, budget: str) -> tuple[object, evaluate.Evaluation]:
    inp = case.engine_input()
    report = eng.discover_invariants(
        inp.hamiltonian_latex, inp.coordinates, "hamiltonian", inp.momenta, [], budget
    )
    return report, evaluate.evaluate(case, report)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


# ---------------------------------------------------------------------------------------------------------
def cmd_check(_: argparse.Namespace) -> int:
    stored = cases.fixed_path().read_text()
    fresh = (
        json.dumps([cases.to_json(c) for c in cases.build_fixed()], indent=1, sort_keys=True) + "\n"
    )
    if stored != fresh:
        print("fixtures differ from what the generator produces: run `generate`", file=sys.stderr)
        return 1
    known = {c.id for c in cases.load_fixed()}
    unknown = set(cases.load_pending()) - known
    if unknown:
        print(f"pending.json names unknown cases: {sorted(unknown)}", file=sys.stderr)
        return 1
    print(f"{len(known)} cases regenerate exactly (generator version {cases.GENERATOR_VERSION})")
    return 0


def cmd_generate(_: argparse.Namespace) -> int:
    print(f"wrote {cases.write_fixed()} cases to {cases.fixed_path()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    eng = engine()
    pending = cases.load_pending()
    bad = 0
    for case in cases.load_fixed():
        if args.ids and not any(i in case.id for i in args.ids):
            continue
        _, ev = run_adaptive(eng, case, args.budget)
        misses = [
            f"{r.name}:{r.recovered if r.recovered != 'no' else r.failure}"
            for r in ev.refs
            if r.recovered != "linear" and not case.broken.get(r.name)
        ]
        expected = [f"{k}:{v.removeprefix('no:')}" for k, v in pending.get(case.id, {}).items()]
        flag = (
            "" if sorted(misses) == sorted(expected) and not ev.false_positives else "  <-- CHANGED"
        )
        bad += bool(flag)
        print(
            f"{case.id:30s} {ev.status:11s} new={ev.new_constants} fp={len(ev.false_positives)} "
            f"rank={ev.rank_lower}/{ev.reference_rank} tiers={ev.tiers} unk={ev.unknowns:5d} "
            f"first={ev.first_discovery_ms}ms total={ev.runtime_ms}ms mem={ev.peak_mb}MB "
            f"misses={misses}{flag}"
        )
    return 1 if bad else 0


# --- adaptive vs fixed --------------------------------------------------------------------------------------
def baseline(eng: Engine, case: cases.Case) -> dict[str, object] | None:
    """The original fixed-parameter search on the verified Legendre-transformed Lagrangian."""
    lf = legendre.to_lagrangian(case.H, case.q, case.p)
    if lf is None:
        return None
    variables = [*case.q, *lf.velocities]
    coords = [s.name for s in case.q]
    sub = dict(zip(case.p, lf.momenta_of_velocity, strict=True))
    refs = [r.expr.subs(sub, simultaneous=True) for r in case.refs]
    limits = Limits(timeout_seconds=60)
    noether: list[sp.Expr] = []
    lag = eng.analyze_lagrangian(lf.latex, coords)
    for c in lag.conserved:
        noether.append(parse_latex(c.expression_latex.split("=", 1)[1], {}, limits))
    out: dict[str, object] = {}
    for config in FIXED_CONFIGS:
        degree, power, harmonics = config
        started = time.monotonic()
        report = eng.search_invariants(lf.latex, coords, [], degree, power, harmonics)
        elapsed = time.monotonic() - started
        found = [parse_latex(f.expression_latex.split("=", 1)[1], {}, limits) for f in report.found]
        space = evaluate.algebra([*noether, *found])
        recovered = [
            evaluate.in_span(space, r, variables) if report.status.value == "ok" else False
            for r in refs
        ]
        out[str(config)] = {
            "status": report.status.value,
            "recovered": recovered,
            "unknowns": report.unknowns,
            "seconds": elapsed,
        }
    return out


def cmd_compare(args: argparse.Namespace) -> int:
    eng = engine()
    rows = []
    for case in cases.load_fixed():
        if case.kind != "discovery":
            continue
        _, ev = run_adaptive(eng, case, "standard")
        base = baseline(eng, case)
        rows.append((case, ev, base))
        print(
            case.id,
            "adaptive",
            f"{ev.coverage:.0%}",
            "baseline" if base else "n/a (no Lagrangian)",
            flush=True,
        )

    env = environment()
    lines = ["# Invariant discovery benchmark: adaptive search against fixed-parameter search", ""]
    lines += [
        f"Reproduce with `uv run python scripts/bench_invariants.py compare`. Engine commit `{env['engine_commit']}`, "
        f"Python {env['python']}, SymPy {env['sympy']}, generator version {env['generator_version']}. "
        f"Budgets: {env['budgets']}. Seeds, recipes (exact matrices) and parameters of every case are stored in "
        "`packages/engine/tests/bench/fixtures/fixed_seed.json`; timings are from one machine and vary.",
        "",
    ]
    lines += [
        "Generated by `scripts/bench_invariants.py compare` on the fixed-seed cases (exact canonical maps of",
        "curated and constructed integrable systems; see `packages/engine/tests/bench`). Minerva is given",
        "the Hamiltonian only. The baseline is the original `search_invariants` (Lagrangian, one ansatz chosen",
        "by the caller) run on the verified Legendre transform of each Hamiltonian that is quadratic in the",
        "momenta, once per configuration below; the adaptive search is `discover_invariants` with the",
        "`standard` budget. Both are capped at 350 unknown coefficients per family. A reference invariant",
        "counts as recovered when it lies in the exact span of the reported constants, 1 and their pairwise",
        "products (the Noether constants are included for both). Runtimes include sandbox process start.",
        "",
    ]
    comparable = [(c, ev, b) for c, ev, b in rows if b is not None]
    total_refs = sum(len(c.refs) for c, _, _ in comparable)

    def count(label: str) -> int:
        return sum(sum(b[label]["recovered"]) for _, _, b in comparable)  # type: ignore[index, arg-type]

    adaptive_ref = sum(sum(r.recovered == "linear" for r in ev.refs) for _, ev, _ in comparable)
    lines += [
        "## References recovered",
        "",
        "| Method | Recovered | Cases fully recovered | Median s | p95 s |",
        "| --- | --- | --- | --- | --- |",
    ]
    adaptive_t = [ev.runtime_ms / 1000 for _, ev, _ in comparable]
    full = sum(all(r.recovered == "linear" for r in ev.refs) for _, ev, _ in comparable)
    lines.append(
        f"| adaptive (standard) | {adaptive_ref} / {total_refs} | {full} / {len(comparable)} | "
        f"{statistics.median(adaptive_t):.2f} | {percentile(adaptive_t, 0.95):.2f} |"
    )
    for config in FIXED_CONFIGS:
        label = str(config)
        times = [b[label]["seconds"] for _, _, b in comparable]  # type: ignore[index]
        full_f = sum(all(b[label]["recovered"]) for _, _, b in comparable)  # type: ignore[index, arg-type]
        lines.append(
            f"| fixed degree {config[0]}, powers {config[1]}, harmonics {config[2]} | {count(label)} / {total_refs} | "
            f"{full_f} / {len(comparable)} | {statistics.median(times):.2f} | {percentile(times, 0.95):.2f} |"
        )
    best = sum(
        max(sum(b[str(cfg)]["recovered"]) for cfg in FIXED_CONFIGS)  # type: ignore[index, arg-type]
        for _, _, b in comparable
    )
    lines.append(
        f"| best fixed configuration per case (needs the answer in advance) | {best} / {total_refs} | | | |"
    )
    lines += [
        "",
        f"{len(comparable)} of {len([r for r in rows])} cases have a Lagrangian form; the rest are "
        "not quadratic in the momenta (or are algebraic) and the old search cannot take them.",
        "",
    ]

    lines += [
        "## Per case",
        "",
        "| Case | Refs | Adaptive | Fixed "
        + " | Fixed ".join(str(c) for c in FIXED_CONFIGS)
        + " | Families | Unknowns (sum) | First found (ms) | Peak MB |",
        "| --- | --- | --- | "
        + " | ".join("---" for _ in FIXED_CONFIGS)
        + " | --- | --- | --- | --- |",
    ]
    wins = ties = losses = 0
    for case, ev, base in rows:
        n = len(case.refs)
        a = sum(r.recovered == "linear" for r in ev.refs)
        if base is None:
            fixed_cells = ["n/a"] * len(FIXED_CONFIGS)
        else:
            fixed_cells = [f"{sum(base[str(c)]['recovered'])}/{n}" for c in FIXED_CONFIGS]  # type: ignore[index, arg-type]
            default = sum(base[str(DEFAULT_FIXED)]["recovered"])  # type: ignore[index, arg-type]
            wins += a > default
            ties += a == default
            losses += a < default
        lines.append(
            f"| {case.id} | {n} | {a}/{n} | "
            + " | ".join(fixed_cells)
            + f" | {ev.tiers} | {ev.unknowns} | {ev.first_discovery_ms} | {ev.peak_mb} |"
        )
    lines += [
        "",
        f"Against the original defaults {DEFAULT_FIXED}: adaptive recovers more on {wins} cases, "
        f"the same on {ties}, fewer on {losses}.",
        "",
    ]

    classes = Counter(r.failure for _, ev, _ in rows for r in ev.refs if r.failure)
    lines += ["## Misses by cause (adaptive)", ""]
    lines += [f"- `{k}`: {v}" for k, v in sorted(classes.items())] or ["- none"]
    fps = sum(len(ev.false_positives) for _, ev, _ in rows)
    triv = sum(ev.trivial_reported for _, ev, _ in rows)
    lines += [
        "",
        f"False conservation claims (adaptive): {fps}. Trivial combinations reported as new: {triv}.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


# --- randomized sweep ---------------------------------------------------------------------------------------
def cmd_sweep(args: argparse.Namespace) -> int:
    eng = engine()
    bases = args.bases or [b for b in BUILDERS if b not in ("osc3_iso", "kepler_cartesian")]
    results = []
    failures: list[tuple[str, int, int, bool]] = []
    classes: Counter[str] = Counter()
    for seed in range(args.start, args.start + args.seeds):
        for base in bases:
            rng = random.Random(f"sweep:{base}:{seed}")  # noqa: S311 - deterministic, not cryptographic
            steps = rng.randint(0, args.max_steps)
            mixing = rng.random() < 0.7
            try:
                case = cases.build_case(base, seed, steps, mixing=mixing)
            except RuntimeError:
                continue  # no map of acceptable complexity for this seed
            _, ev = run_adaptive(eng, case, args.budget)
            results.append((case, ev))
            for r in ev.refs:
                if r.failure:
                    classes[r.failure] += 1
            suspicious = (
                ev.false_positives
                or ev.trivial_reported
                or any(
                    r.failure in ("solver_failure", "verification_failure", "unclassified")
                    for r in ev.refs
                )
            )
            if suspicious:
                failures.append((base, seed, steps, mixing))
            print(
                f"{case.id:32s} {ev.status:11s} cov={ev.linear_coverage:4.0%} fp={len(ev.false_positives)} "
                f"triv={ev.trivial_reported} {ev.runtime_ms}ms"
                + ("  <-- suspicious" if suspicious else ""),
                flush=True,
            )
    refs = sum(len(ev.refs) for _, ev in results)
    linear = sum(r.recovered == "linear" for _, ev in results for r in ev.refs)
    in_family = [r for _, ev in results for r in ev.refs if r.dominated]
    found = sum(r.recovered == "linear" for r in in_family)
    times = [ev.runtime_ms / 1000 for _, ev in results]
    summary = {
        "environment": environment(),
        "budget": args.budget,
        "start_seed": args.start,
        "seeds": args.seeds,
        "cases": len(results),
        "references": refs,
        "recovered_linear": linear,
        "in_searched_family": len(in_family),
        "in_searched_family_recovered": found,
        "false_positives": sum(len(ev.false_positives) for _, ev in results),
        "trivial_reported": sum(ev.trivial_reported for _, ev in results),
        "misses_by_cause": dict(classes),
        "median_s": statistics.median(times) if times else 0,
        "p95_s": percentile(times, 0.95),
        "suspicious": [list(f) for f in failures],
    }
    print(json.dumps(summary, indent=1))
    if args.promote:
        for base, seed, steps, mixing in failures:
            promote_spec(base, seed, steps, mixing)
    return 0


def promote_spec(base: str, seed: int, steps: int, mixing: bool) -> None:
    """Add a failing case to the permanent fixed-seed set, marked pending with its observed misses."""
    path = cases.promoted_path()
    specs = json.loads(path.read_text()) if path.exists() else []
    spec = [base, seed, steps, mixing, []]
    if spec not in specs:
        specs.append(spec)
        path.write_text(json.dumps(specs, indent=1) + "\n")
    cases.write_fixed()
    eng = engine()
    case = next(
        c
        for c in cases.load_fixed()
        if c.id == cases.build_case(base, seed, steps, mixing=mixing).id
    )
    _, ev = run_adaptive(eng, case, "standard")
    pending = cases.load_pending()
    misses = {
        r.name: (r.recovered if r.recovered != "no" else f"no:{r.failure}")
        for r in ev.refs
        if r.recovered != "linear"
    }
    if misses:
        pending[case.id] = misses
        cases.pending_path().write_text(json.dumps(pending, indent=1, sort_keys=True) + "\n")
    print(f"promoted {case.id}: {misses}")


def cmd_promote(args: argparse.Namespace) -> int:
    base, seed, steps, mixing = args.spec.split(":")
    promote_spec(base, int(seed), int(steps), bool(int(mixing)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("generate").set_defaults(func=cmd_generate)
    run = sub.add_parser("run")
    run.add_argument("ids", nargs="*")
    run.add_argument("--budget", default="standard")
    run.set_defaults(func=cmd_run)
    compare = sub.add_parser("compare")
    compare.add_argument("--out")
    compare.set_defaults(func=cmd_compare)
    sweep = sub.add_parser("sweep")
    sweep.add_argument("--seeds", type=int, default=20)
    sweep.add_argument("--start", type=int, default=100)
    sweep.add_argument("--max-steps", type=int, default=3)
    sweep.add_argument("--bases", nargs="*")
    sweep.add_argument("--budget", default="standard")
    sweep.add_argument(
        "--promote", action="store_true", help="add suspicious cases to the fixed set"
    )
    sweep.set_defaults(func=cmd_sweep)
    promote = sub.add_parser("promote")
    promote.add_argument("spec", help="base:seed:steps:mixing (mixing 0 or 1)")
    promote.set_defaults(func=cmd_promote)
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
