"""The model comparison applies a rule fixed in advance: it never ranks what it was not allowed to compare."""

from typing import Any

from minerva_api.agent.evaluation.matrix import (
    decide,
    matrix_report,
    rows_of,
    screen,
)


def entry(
    passes: list[int],
    *,
    median: float = 6.0,
    p95: float = 20.0,
    gpu: float = 1.0,
    false_verified: int = 0,
    digest: str = "aaaaaaaaaaaa",
    cut: int = 0,
) -> dict[str, Any]:
    runs = 100 * len(passes)
    results = [
        {
            "informational": False,
            "calls": 0 if i < 3 else 1,
            "expected_tool": "simplify",
            "stop_reason": "output_limit" if i < cut else "completed",
        }
        for i in range(10)
    ]
    mean = sum(passes) / len(passes)
    std = (sum((p - mean) ** 2 for p in passes) / len(passes)) ** 0.5
    return {
        "identity": {
            "hosted": False,
            "digest": digest,
            "base_model": "base",
            "parameter_size": "25B",
            "quantization": "Q4_K_M",
            "loaded_bytes": 1000,
            "loaded_gpu_bytes": int(1000 * gpu),
            "host_defaults": {"num_predict": 1200},
        },
        "results": results,
        "summary": {
            "runs": runs,
            "passed": sum(passes),
            "pass_counts_per_repeat": passes,
            "pass_count_stdev": std,
            "tool_selection_accuracy": 0.9,
            "valid_arguments_rate": 0.95,
            "completion_rate": 0.95,
            "false_verified": false_verified,
            "latency_seconds": {"median": median, "p95": p95, "max": p95},
            "tokens_out": 10,
        },
    }


def doc(
    models: dict[str, dict[str, Any]], *, think: bool = False, cases: str = "c1"
) -> dict[str, Any]:
    return {
        "schema": 1,
        "meta": {"think": think, "params": {"think": think}, "cases_digest": cases, "seed": 7},
        "models": models,
    }


def test_every_row_names_model_digest_thinking_and_budget() -> None:
    text = matrix_report([doc({"minerva": entry([70, 71, 69])}, think=True)])
    row = next(line for line in text.splitlines() if line.startswith("| 1 | minerva"))
    assert "aaaaaaaaaaaa" in row and "| on |" in row and "host default 1200" in row


def test_a_false_verified_result_is_never_eligible_however_well_it_scores() -> None:
    rows = rows_of([doc({"a": entry([90, 90, 90], false_verified=1), "b": entry([60, 61, 59])})])
    best, _ = decide(rows)
    assert best is not None and best.model == "b"
    assert "false verified" in " ".join(rows[0].reasons)


def test_over_the_latency_ceiling_or_off_the_gpu_is_not_eligible() -> None:
    rows = rows_of(
        [
            doc(
                {
                    "slow": entry([90, 90, 90], p95=60.0),
                    "spill": entry([90, 90, 90], gpu=0.6),
                    "ok": entry([70, 70, 70]),
                }
            )
        ]
    )
    best, _ = decide(rows)
    assert best is not None and best.model == "ok"
    assert [r.eligible for r in rows] == [False, False, True]


def test_a_gap_inside_the_noise_is_a_tie_and_the_lower_p95_wins() -> None:
    rows = rows_of([doc({"a": entry([70, 72, 68], p95=30.0), "b": entry([69, 71, 67], p95=18.0)})])
    best, tied = decide(rows)
    assert best is not None and best.model == "b" and [t.model for t in tied] == ["a"]


def test_a_gap_beyond_the_noise_is_not_a_tie() -> None:
    rows = rows_of([doc({"a": entry([80, 80, 80], p95=30.0), "b": entry([60, 60, 60], p95=10.0)})])
    best, tied = decide(rows)
    assert best is not None and best.model == "a" and tied == []


def test_the_screen_keeps_everything_within_noise_of_third_place() -> None:
    models = {
        "a": entry([90, 90, 90]),
        "b": entry([80, 80, 80]),
        "c": entry([70, 71, 69]),
        "d": entry([69, 70, 68]),  # within noise of third
        "e": entry([40, 40, 40]),
        "f": entry([95, 95, 95], false_verified=2),  # best score, dropped first
    }
    kept = {r.model for r in screen(rows_of([doc(models)]))}
    assert kept == {"a", "b", "c", "d"}


def test_different_case_sets_are_not_ranked() -> None:
    text = matrix_report(
        [doc({"a": entry([70, 70, 70])}, cases="c1"), doc({"b": entry([60, 60, 60])}, cases="c2")]
    )
    assert "Not decided" in text and "Best eligible" not in text


def test_an_unrecorded_gpu_share_is_said_not_assumed() -> None:
    e = entry([70, 70, 70])
    e["identity"].pop("loaded_bytes")
    rows = rows_of([doc({"a": e})])
    assert not rows[0].eligible and "GPU share not recorded" in rows[0].reasons[0]


def test_cut_off_and_no_tool_runs_are_counted() -> None:
    rows = rows_of([doc({"a": entry([70, 70, 70], cut=2)})])
    assert rows[0].cut_off == 2 and rows[0].no_tool_runs == 3
