"""Generate the site's benchmark record from the committed reports in docs/benchmarks.

Writes apps/site/src/data/benchmarks.json, which the landing site's /benchmarks page renders. Nothing in the output
is typed by hand except the one-sentence description of what each suite checks: every count, rate and time is read
from a committed report, and a report that is missing or has a shape this script does not know stops the build.
The model host's address is not in any report and is never written here (a test checks it).

Two kinds of record:

* **Engine suites** (deterministic): problems whose answers were fixed before the code was written, from outside
  Minerva where possible. Each problem ends in one of four ways, and the four add up to the total: the engine
  computed the right answer (`correct`), refused for a stated reason (`refused`), rejected a wrong answer to the same
  request (`rejected`), or is a known gap (`pending`).
* **Chat reliability** (live, model-dependent): the agent run through the real loop on a named model. Every row names
  the model as its host reported it (base model, size, quantisation, digest), whether thinking was requested, the
  reply limit and the case set, so no number can be read without them. Targets are shown next to the measurement and
  are met or not met, never adjusted.

Usage:
  uv run python scripts/benchmarks_site.py            # regenerate the file
  uv run python scripts/benchmarks_site.py --check    # fail if the file on disk differs (the tests run this)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs" / "benchmarks"
OUT = ROOT / "apps" / "site" / "src" / "data" / "benchmarks.json"


def load(name: str) -> dict[str, Any]:
    path = BENCH / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing report: docs/benchmarks/{name}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{name}.json: not a report this script can read")
    return data


def outcome(
    total: int, correct: int, refused: int = 0, rejected: int = 0, pending: int = 0
) -> dict[str, int]:
    if correct + refused + rejected + pending != total:
        raise SystemExit(
            f"counts do not add up: {correct} + {refused} + {rejected} + {pending} != {total}"
        )
    return {
        "total": total,
        "correct": correct,
        "refused": refused,
        "rejected": rejected,
        "pending": pending,
    }


# ----------------------------------------------------------------------------------------------- engine suites
def statistics() -> dict[str, int]:
    kinds = Counter(
        (c["kind"], bool(c["passed"]), bool(c.get("pending"))) for c in load("statistics")["cases"]
    )
    bad = [k for k in kinds if not k[1] and not k[2]]
    if bad:
        raise SystemExit("statistics.json holds a failing case; fix it before publishing")
    return outcome(
        sum(kinds.values()),
        correct=sum(n for (k, _, _), n in kinds.items() if k == "supported"),
        refused=sum(n for (k, _, _), n in kinds.items() if k == "unsupported"),
        rejected=sum(n for (k, _, _), n in kinds.items() if k == "invalid_candidate"),
    )


def distributions() -> dict[str, int]:
    cases = load("distributions")["cases"]
    if not all(c["passed"] for c in cases):
        raise SystemExit("distributions.json holds a failing case; fix it before publishing")
    kinds = Counter(c["kind"] for c in cases)
    return outcome(
        len(cases),
        correct=kinds["supported"],
        refused=kinds["unsupported"],
        rejected=kinds["invalid_transform"],
    )


def geometry() -> dict[str, int]:
    cover = load("geometry")["deterministic"]["coverage"]
    if cover["computed_and_correct"] != cover["corpus"]:
        raise SystemExit("geometry.json: a computed case is not correct")
    if cover["refusals_as_expected"] != cover["refusal_cases"]:
        raise SystemExit("geometry.json: a refusal was lost")
    return outcome(
        cover["corpus"] + cover["refusal_cases"],
        correct=cover["computed_and_correct"],
        refused=cover["refusals_as_expected"],
    )


def vector() -> dict[str, int]:
    cover = load("vector")["deterministic"]["coverage"]
    if cover["correct"] != cover["corpus"]:
        raise SystemExit("vector.json: a case is not correct")
    return outcome(
        cover["corpus"], correct=cover["shown_results"], refused=cover["refused_as_expected"]
    )


def passed_list(name: str) -> Callable[[], dict[str, int]]:
    def build() -> dict[str, int]:
        det = load(name)["deterministic"]
        if det["passed"] != det["total"]:
            raise SystemExit(f"{name}.json: a case fails")
        return outcome(det["total"], correct=det["passed"])

    return build


def problems(name: str, key: str = "problems") -> Callable[[], dict[str, int]]:
    def build() -> dict[str, int]:
        counts = Counter(p["outcome"] for p in load(name)[key])
        known = {"correct", "refused_as_expected", "rejected_as_expected", "pending"}
        if set(counts) - known:
            raise SystemExit(f"{name}.json: unknown outcome {sorted(set(counts) - known)}")
        return outcome(
            sum(counts.values()),
            correct=counts["correct"],
            refused=counts["refused_as_expected"],
            rejected=counts["rejected_as_expected"],
            pending=counts["pending"],
        )

    return build


# id, title, what it checks (the one hand-written sentence), builder, source files, how to run it, doc
SUITES: list[tuple[str, str, str, Callable[[], dict[str, int]], list[str], str, str]] = [
    (
        "statistics",
        "Statistical derivations",
        "Sums and products, expectations, moments of probability families, maximum-likelihood estimators, "
        "Fisher information, Bayesian posteriors and matrix gradients, against hand-written references.",
        statistics,
        ["statistics.json"],
        "uv run python -m minerva_knowledge stat-benchmark",
        "docs/statistics-benchmark.md",
    ),
    (
        "vector",
        "Vector calculus",
        "Gradient, divergence, curl and Laplacian in Cartesian, cylindrical and spherical coordinates, "
        "identities, derivations, integral theorems and point sources, with textbook references and refusals.",
        vector,
        ["vector.json"],
        "uv run python -m minerva_engine.vector.benchmark --check",
        "docs/vector-benchmark.md",
    ),
    (
        "geometry",
        "Differential geometry",
        "Christoffel symbols, curvature and geodesics of explicit metrics, compared with closed forms and "
        "with an independent numerical recomputation, plus malformed metrics that must be refused.",
        geometry,
        ["geometry.json"],
        "uv run python -m minerva_engine.geometry.benchmark --check",
        "docs/geometry-benchmark.md",
    ),
    (
        "reference",
        "Problems answered from outside Minerva",
        "Differential equations, distributions, calculus and matrices whose expected answers come from "
        "other tools (mpmath, quadrature, plain arithmetic), written by hand and never regenerated from the code.",
        problems("reference"),
        ["reference.json"],
        "uv run python -m minerva_engine.reference_benchmark --check",
        "docs/reference-benchmark.md",
    ),
    (
        "statmech",
        "Statistical mechanics",
        "Partition functions and the quantities derived from them against closed forms, with refusals that "
        "must carry a named reason (a divergent sum, a substituted ensemble).",
        problems("statmech"),
        ["statmech.json"],
        "uv run python -m minerva_engine.statmech.benchmark --check",
        "docs/statmech-benchmark.md",
    ),
    (
        "telescoping",
        "Parameterised integrals",
        "Integrals with a parameter, where a certificate is verified exactly, forged certificates must be "
        "rejected and problems beyond the searched class must be declined.",
        problems("creative_telescoping"),
        ["creative_telescoping.json"],
        "uv run python -m minerva_engine.creative_telescoping.benchmark --check",
        "docs/creative-telescoping-benchmark.md",
    ),
    (
        "integration",
        "Integration strategies",
        "Named techniques (symmetry, Frullani, Schwinger and others) checked against quadrature, with "
        "techniques whose conditions are not established reported as blocked.",
        problems("integration_strategies", "cases"),
        ["integration_strategies.json"],
        "uv run python -m minerva_engine.integration_strategy.benchmark --check",
        "docs/integration-strategies-benchmark.md",
    ),
    (
        "distributions",
        "Dirac delta and Heaviside",
        "Distributions as distributions: roots, scaling and jumps, with a test-function pairing as separate "
        "evidence, and wrong transformations that must be rejected.",
        distributions,
        ["distributions.json"],
        "uv run python -m minerva_knowledge benchmark --only distributions",
        "docs/distribution-benchmark.md",
    ),
    (
        "variational",
        "Variational calculus",
        "Euler-Lagrange equations, boundary terms and stationary paths of explicit actions, against closed forms.",
        passed_list("variational"),
        ["variational.json"],
        "uv run python -m minerva_engine.variational.benchmark --check",
        "docs/variational-benchmark.md",
    ),
    (
        "approximation",
        "Approximations",
        "Taylor, Pade, Chebyshev, Fourier, asymptotic and Euler-Maclaurin results, with answers from outside "
        "Minerva (Bessel and Hurwitz functions, OEIS, Stirling).",
        passed_list("approximation"),
        ["approximation.json"],
        "uv run python -m minerva_engine.approx.benchmark --check",
        "docs/approximation-benchmark.md",
    ),
]


def verification() -> dict[str, Any]:
    det = load("verification-summary")["deterministic"]
    totals = det["totals"]
    if totals["false_accepts"] or totals["false_rejects"] or totals["incorrect_accepted"]:
        raise SystemExit(
            "verification-summary.json records a wrong verdict; fix it before publishing"
        )
    return {
        "problems": totals["problems"],
        "correct": totals["correct"],
        "refused": totals["refused_as_expected"],
        "false_accepts": totals["false_accepts"],
        "false_rejects": totals["false_rejects"],
        "counterexamples_found": totals["counterexamples"],
        "pending": totals["pending"],
        "known_open": totals["known_open"],
        "evidence": det["evidence_of_accepted_reference_results"],
        "source": "verification-summary.json",
        "doc": "docs/verification-testing.md",
        "command": "uv run python scripts/verification_report.py --check",
    }


# --------------------------------------------------------------------------------------------- chat reliability
TARGETS = [
    ("pass_rate", "Pass rate", 0.85, "min"),
    ("tool_selection", "Tool selection", 0.95, "min"),
    ("valid_arguments", "Valid arguments", 0.99, "min"),
    ("completion", "Completion", 0.95, "min"),
    ("false_verified", "Results wrongly shown as verified", 0, "max"),
]
CHAT_REPORTS = [
    ("all", "baseline-2026-11-19-all", "Core and reliability cases"),
    ("probe", "baseline-2026-11-19-probe", "Probe suite (written before the traces were read)"),
    ("heldout", "baseline-2026-11-19-heldout", "Held-out workflows in other words"),
]


def chat_row(key: str, name: str, title: str) -> dict[str, Any]:
    doc = load(name)
    meta = doc["meta"]
    model, entry = next(iter(doc["models"].items()))
    summary, identity = entry["summary"], entry.get("identity") or {}
    if not identity or identity.get("hosted"):
        raise SystemExit(
            f"{name}.json does not record the model's identity; it cannot be published"
        )
    params = meta.get("params") or {}
    latency = summary["latency_seconds"]
    measured = {
        "pass_rate": summary["pass_rate"],
        "tool_selection": summary["tool_selection_accuracy"],
        "valid_arguments": summary["valid_arguments_rate"],
        "completion": summary["completion_rate"],
        "false_verified": summary["false_verified"],
    }
    return {
        "id": key,
        "title": title,
        "report": f"{name}.json",
        "date": meta["date"],
        "model": {
            "alias": model,
            "base_model": identity.get("base_model"),
            "family": identity.get("family"),
            "parameter_size": identity.get("parameter_size"),
            "quantization": identity.get("quantization"),
            "digest": identity.get("digest"),
            "thinking_requested": bool(params.get("think", meta.get("think"))),
            "reply_limit": params.get("num_predict")
            or (identity.get("host_defaults") or {}).get("num_predict"),
            "reply_limit_is_host_default": not params.get("num_predict"),
        },
        "run": {
            "runs": summary["runs"],
            "cases": summary["cases"],
            "repeats": meta["repeats"],
            "seed": meta["seed"],
            "compute_limit_seconds": meta["compute_timeout"],
            "cases_digest": meta["cases_digest"],
            "code": str(meta["git_sha"])[:10],
        },
        "passed": summary["passed"],
        "per_repeat": summary["pass_counts_per_repeat"],
        "measured": measured,
        "median_seconds": round(latency["median"], 1),
        "p95_seconds": round(latency["p95"], 1),
    }


def chat() -> dict[str, Any]:
    rows = [chat_row(*r) for r in CHAT_REPORTS]
    full = next(r for r in rows if r["id"] == "all")
    targets = []
    for key, label, target, kind in TARGETS:
        value = full["measured"][key]
        met = value >= target if kind == "min" else value <= target
        targets.append({"id": key, "label": label, "target": target, "measured": value, "met": met})
    return {"rows": rows, "targets": targets}


# ----------------------------------------------------------------------------------------------- model comparison
MODEL_DIR = "model-comparison-2026-11-20"
# (full-suite report, held-out report, what the configuration is in the comparison: a label, not a number)
STAGE2 = [
    (
        "stage2-all-qwen3.5-35b-thinking-on",
        "stage2-heldout-qwen3.5-35b-thinking-on",
        "the model the chat was switched to on 2026-11-19",
    ),
    (
        "stage2-all-gemma4-26b-deployed-build-thinking-off",
        "stage2-heldout-gemma4-26b-deployed-build-thinking-off",
        "the reference: the build deployed until 2026-11-19",
    ),
    (
        "stage2-all-glm-4.7-flash-thinking-off",
        "stage2-heldout-glm-4.7-flash-thinking-off",
        "the other finalist of the screen",
    ),
]
_TAG = re.compile(r"^[a-z][a-z0-9.\-]*:[a-z0-9][a-z0-9._\-]*$")


def model_name(alias: str, identity: dict[str, Any]) -> str:
    """The model by what it is: a tag (`qwen3.5:35b`) is its own name, an operator's alias (`minerva`) is described by
    the base model the host reports. Never the alias alone, and never an internal parent name over a real tag."""
    base = identity.get("base_model")
    if _TAG.match(alias) or (base and str(base).removesuffix(":latest") == alias):
        return alias
    return f"{base} (served as {alias})" if base and base != alias else alias


def stage1_screen() -> list[dict[str, Any]]:
    """One row per model and digest from the probe-suite reports: passes with thinking off and on."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((BENCH / MODEL_DIR).glob("stage1-probe-*.json")):
        doc = load(f"{MODEL_DIR}/{path.stem}")
        alias, entry = next(iter(doc["models"].items()))
        summary, identity = entry["summary"], entry.get("identity") or {}
        if not identity or identity.get("hosted") or not summary:
            raise SystemExit(
                f"{path.name} does not record the model's identity; it cannot be published"
            )
        name = model_name(alias, identity)
        row = rows.setdefault(
            (name, str(identity.get("digest"))),
            {
                "model": name,
                "digest": identity.get("digest"),
                "parameter_size": identity.get("parameter_size"),
                "off": None,
                "on": None,
            },
        )
        key = (
            "on"
            if (doc["meta"].get("params") or {}).get("think", doc["meta"].get("think"))
            else "off"
        )
        row[key] = {
            "passed": summary["passed"],
            "runs": summary["runs"],
            "median_seconds": round(summary["latency_seconds"]["median"], 1),
            "false_verified": summary["false_verified"],
        }
    return sorted(
        rows.values(),
        key=lambda r: -max((r[k]["passed"] for k in ("on", "off") if r[k]), default=0),
    )


def model_comparison() -> dict[str, Any]:
    final = []
    for full, held, label in STAGE2:
        row = chat_row("all", f"{MODEL_DIR}/{full}", "Full suite")
        out = chat_row("heldout", f"{MODEL_DIR}/{held}", "Held-out suite")
        identity = next(iter(load(f"{MODEL_DIR}/{full}")["models"].values()))["identity"]
        row["model"]["name"] = model_name(row["model"]["alias"], identity)
        row["role"] = label
        row["held_out"] = {"passed": out["passed"], "runs": out["run"]["runs"]}
        fv = row["measured"]["false_verified"]
        row["eligible"] = fv == 0
        row["not_eligible_reason"] = (
            None
            if fv == 0
            else f"{fv} false-verified result(s) in this run (see the note below the table)"
        )
        final.append(row)
    return {
        "directory": f"docs/benchmarks/{MODEL_DIR}",
        "doc": "docs/reliability/model-comparison.md",
        "final": final,
        "screen": stage1_screen(),
    }


def build() -> dict[str, Any]:
    suites = []
    for sid, title, what, builder, files, command, doc in SUITES:
        if not (ROOT / doc).exists():
            raise SystemExit(f"{sid}: {doc} does not exist; the page would link to nothing")
        suites.append(
            {
                "id": sid,
                "title": title,
                "what": what,
                **builder(),
                "sources": files,
                "command": command,
                "doc": doc,
            }
        )
    return {
        "suites": suites,
        "verification": verification(),
        "chat": chat(),
        "models": model_comparison(),
    }


def render() -> str:
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        # compared as data: a formatter may lay the file out differently without changing a number
        if not OUT.exists() or json.loads(OUT.read_text(encoding="utf-8")) != json.loads(text):
            print(
                "apps/site/src/data/benchmarks.json is out of date: "
                "run `uv run python scripts/benchmarks_site.py`",
                file=sys.stderr,
            )
            return 1
        print("benchmarks.json: ok")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
