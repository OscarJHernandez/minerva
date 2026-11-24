"""Row numbers for `variation`, `statistics` and `expansion`, and the whole-result number of `sums`, `expectation`
and `distribution` (issue #133, parts 2 and 3, PR 2 of 3).

A reader lists only formulas the card really displays, only for a card that was accepted, and only when the row
itself was checked: an unverified expansion or mode, a result that holds given a premise, a numerical value,
an unevaluated representation and the user's own input get no number. A number is a name, never a verification:
`@eN` of a row substitutes its LaTeX exactly and the result says it came from one component of an earlier result.

The keys and the row texts are pinned against `packages/ui/fixtures/equation-rows.json`, which the browser test
renders (`equation-rows.test.tsx`): the keys the server numbers and the formulas the panels mark cannot drift.
The fixture is real engine output; `MINERVA_REGEN_ROW_FIXTURE=1 pytest -k regenerate` adds the entries that are
missing (it never rewrites an existing one).
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from minerva_api.agent.card_equations import (
    _GREEK,
    LIST_NOT_YET_NUMBERED,
    LIST_READERS,
    NESTED_RESULT_CARDS,
    SINGLE_RESULT,
    canonical_modes,
    expansion,
    listed_equations,
    name_latex,
    result_formula,
    statistics,
    variation,
)
from minerva_api.agent.equations import (
    MAX_EQUATION_CHARS,
    EquationTable,
    build_table,
    number_stored,
)
from minerva_api.agent.refs import COMPONENT_SUFFIX, NO_ID_SUFFIX, ResultStore
from minerva_api.agent.replay import call, say
from minerva_api.agent.tools import ToolContext, execute
from minerva_engine import Engine
from minerva_engine.notation.models import InputCode
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

from .test_equation_rows_contract import FIXTURE, card_from, number
from .test_total_derivative_handles import Row, card_of, play

NEW = ("variation", "variation_second", "statistics_moments", "statistics_estimate",
       "statistics_expressions", "statistics_fisher", "expansion", "expansion_two_modes")  # fmt: skip
CARD_TYPE = {
    "variation": "variation",
    "variation_second": "variation",
    "statistics_moments": "statistics",
    "statistics_estimate": "statistics",
    "statistics_expressions": "statistics",
    "statistics_fisher": "statistics",
    "expansion": "expansion",
    "expansion_two_modes": "expansion",
}
NORMAL = {
    "family": "normal",
    "parametrization": "variance",
    "parameters": {"v": "v"},
    "observation": "x",
    "estimate": ["mu"],
    "assumptions": {"x": ["real"], "v": ["positive"], "n": ["integer", "positive"]},
}
POTENTIAL = {
    "integrand": r"\frac{m}{2}\dot{x}^2-V(x)",
    "functions": ["x"],
    "given_functions": [{"name": "V", "args": ["x"]}],
}
REQUESTS: dict[str, tuple[str, dict[str, Any]]] = {
    "variation": ("variation", {"action": "first", **POTENTIAL}),
    "variation_second": ("variation", {"action": "second", **POTENTIAL}),
    "statistics_moments": (
        "statistical_model",
        {
            "family": "gamma",
            "parameters": {"alpha": "alpha", "beta": "beta"},
            "action": "moments",
            "assumptions": {"alpha": ["positive"], "beta": ["positive"]},
        },
    ),
    "statistics_estimate": ("statistical_model", {**NORMAL, "action": "mle"}),
    "statistics_expressions": ("statistical_model", {**NORMAL, "action": "score"}),
    "statistics_fisher": (
        "statistical_model",
        {**NORMAL, "estimate": ["mu", "v"], "action": "fisher_information"},
    ),
    "expansion": (
        "expand_motion",
        {
            "lagrangian": r"\frac{1}{2}m\dot{x}^2-\frac{1}{2}kx^2",
            "coordinates": ["x"],
            "time": "t",
            "assumptions": {"m": ["positive"], "k": ["positive"]},
            "order": 3,
            "equilibrium": {},
        },
    ),
    "expansion_two_modes": (
        "expand_motion",
        {
            "lagrangian": r"\frac12 (\dot{x}^2 + \dot{y}^2) - \frac12 (3 x^2 + 3 y^2 - 2 x y)",
            "coordinates": ["x", "y"],
            "time": "t",
            "assumptions": {},
            "order": 2,
            "equilibrium": {},
        },
    ),
}
EXPECTED_KEYS = {
    "variation": ["el:x", "first", "stationarity:0", "natural:lower:x:0", "natural:upper:x:0"],
    "variation_second": ["second"],
    "statistics_moments": ["moment:mean", "moment:variance", "moment:second"],
    "statistics_estimate": ["estimate:mu"],
    "statistics_expressions": ["expr:loglik", "expr:score:mu"],
    "statistics_fisher": ["fisher"],
    "expansion": ["equil:x", "eq:x", "char", "mode:0:omega2", "mode:0:omega", "mode:0:vector"],
    "expansion_two_modes": [
        "equil:x",
        "equil:y",
        "eq:x",
        "eq:y",
        "char",
        "mode:0:omega2",
        "mode:0:omega",
        "mode:0:vector",
        "mode:1:omega2",
        "mode:1:omega",
        "mode:1:vector",
    ],
}
COMMENT = (
    "Real engine output of the cards below and the row keys the server numbers, with the LaTeX of each numbered "
    "row. The Python tests (test_equation_rows_contract.py, test_equation_rows_variation_stats_expansion.py) "
    "check the keys and the rows against the readers; the vitest test checks that the panels put each number on "
    "exactly that formula. The handbook `methods` of a card are left out. Regenerate with the helper in the "
    "second test file (MINERVA_REGEN_ROW_FIXTURE=1), which adds missing entries and never rewrites one."
)


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def test_regenerate_the_missing_fixture_entries(ctx: ToolContext) -> None:
    """A helper, not a check: it runs only with `MINERVA_REGEN_ROW_FIXTURE=1` and only adds what is missing."""
    if os.environ.get("MINERVA_REGEN_ROW_FIXTURE") != "1":
        pytest.skip("set MINERVA_REGEN_ROW_FIXTURE=1 to add the missing fixture entries")
    data = json.loads(FIXTURE.read_text())
    data["comment"] = COMMENT
    for name, (tool, args) in REQUESTS.items():
        if name in data:
            continue
        out = execute(tool, args, ctx)
        assert out.status == "ok" and out.card.type == CARD_TYPE[name], name
        card = {k: v for k, v in out.card.data.items() if k != "methods"}
        rows = dict(listed_equations(out.card.type, card))
        data[name] = {"data": card, "keys": list(rows), "rows": rows}
    FIXTURE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


# --- the keys and rows the browser marks ------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEW)
def test_the_fixture_keys_and_rows_are_the_readers_and_the_ones_written_by_hand(
    fixture: dict[str, Any], name: str
) -> None:
    entry = fixture[name]
    rows = listed_equations(CARD_TYPE[name], entry["data"])
    assert [k for k, _ in rows] == entry["keys"] == EXPECTED_KEYS[name]
    assert dict(rows) == entry["rows"]


@pytest.mark.parametrize("name", NEW)
def test_a_live_engine_card_is_read_like_the_fixture(
    ctx: ToolContext, fixture: dict[str, Any], name: str
) -> None:
    tool, args = REQUESTS[name]
    out = execute(tool, args, ctx)
    assert out.status == "ok" and out.card.type == CARD_TYPE[name]
    assert [k for k, _ in listed_equations(out.card.type, out.card.data)] == fixture[name]["keys"]


def test_the_three_types_are_list_readers_and_the_remaining_ones_are_still_listed() -> None:
    for kind in ("variation", "statistics", "expansion"):
        assert kind in LIST_READERS and kind not in LIST_NOT_YET_NUMBERED
    assert not LIST_NOT_YET_NUMBERED  # the matrix rows and the vector rows came with #166


# --- variation ----------------------------------------------------------------------------------------------
def test_variation_numbers_the_euler_lagrange_equation_the_first_variation_and_the_natural_conditions(
    fixture: dict[str, Any],
) -> None:
    data = fixture["variation"]["data"]
    card = card_from("variation", data)
    table = number(card)
    assert set(card.numbers) == set(EXPECTED_KEYS["variation"])
    assert card.number is None
    assert all(table.items[n].part_of == "variation" for n in card.numbers.values())
    report = data["report"]
    texts = {latex for _, latex in variation(data)}
    # not numbered: the functional as read, the integrand of the variation, the integrations by parts, the
    # boundary form and the endpoint terms
    not_numbered = {
        report["analysis"]["functional_latex"],
        report["first"]["delta_integrand_latex"],
        report["first"]["boundary_form_latex"],
        report["first"]["endpoint_terms_latex"],
        *(s["integrand_latex"] for s in report["first"]["stages"]),
    }
    assert not texts & not_numbered
    # the natural-condition key is the React key of the table row
    boundary = report["first"]["boundary"][0]
    assert f"natural:{boundary['end']}:{boundary['function']}:{boundary['order']}" in card.numbers


def test_the_second_variation_is_numbered_only_when_its_own_check_passed(
    fixture: dict[str, Any],
) -> None:
    data = fixture["variation_second"]["data"]
    assert [k for k, _ in variation(data)] == ["second"]
    failed = copy.deepcopy(data)
    failed["report"]["second"]["checks"]["deciding"] = "inconclusive"
    assert variation(failed) == []


def test_a_null_lagrangian_has_no_equation_rows_but_keeps_its_natural_conditions(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["variation"]["data"])
    first = data["report"]["first"]
    first["null_lagrangian"] = True
    first["equations"][0]["equation_latex"] = "0 = 0"
    first["stationarity_latex"] = []
    keys = [k for k, _ in variation(data)]
    assert "el:x" not in keys and not any(k.startswith("stationarity") for k in keys)
    assert keys == ["first", "natural:lower:x:0", "natural:upper:x:0"]


def test_a_variation_that_was_not_accepted_or_not_verified_lists_nothing(
    fixture: dict[str, Any],
) -> None:
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"):
        data = copy.deepcopy(fixture["variation"]["data"])
        data["report"]["status"] = status
        assert variation(data) == [], status
    data = copy.deepcopy(fixture["variation"]["data"])
    data["report"]["first"]["verification"]["passed"] = False
    assert (
        variation(data) == []
    )  # the first-variation rows need the check; the card has no second here
    both = copy.deepcopy(fixture["variation"]["data"])
    both["report"]["second"] = copy.deepcopy(
        fixture["variation_second"]["data"]["report"]["second"]
    )
    both["report"]["first"]["verification"]["passed"] = False
    assert [k for k, _ in variation(both)] == [
        "second"
    ]  # its own check decides the second variation


def test_a_variation_row_over_500_characters_is_not_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["variation"]["data"])
    data["report"]["first"]["equations"][0]["equation_latex"] = "x + " * 200 + "1 = 0"
    assert len(data["report"]["first"]["equations"][0]["equation_latex"]) > MAX_EQUATION_CHARS
    card = card_from("variation", data)
    number(card)
    assert "el:x" not in card.numbers and "first" in card.numbers


# --- statistics ---------------------------------------------------------------------------------------------
def test_an_estimate_row_is_the_formula_the_panel_draws(fixture: dict[str, Any]) -> None:
    data = fixture["statistics_estimate"]["data"]
    assert dict(statistics(data)) == {"estimate:mu": r"\hat{\mu} = x"}


def test_the_greek_table_is_the_one_of_the_panel() -> None:
    source = Path(FIXTURE.parents[3], "apps/web/src/components/statistics-panel.tsx").read_text()
    block = re.search(r"const GREEK = new Set\(\[(.*?)\]\)", source, re.S)
    assert block is not None
    assert set(re.findall(r'"([a-z]+)"', block.group(1))) == set(_GREEK)
    assert name_latex("lam") == r"\lambda" and name_latex("mu") == r"\mu" and name_latex("p") == "p"
    assert 'if (name === "lam") return "\\\\lambda"' in source


def test_moments_and_expressions_are_numbered_by_name(fixture: dict[str, Any]) -> None:
    moments = fixture["statistics_moments"]["data"]["result"]["moments_latex"]
    assert dict(statistics(fixture["statistics_moments"]["data"])) == {
        f"moment:{k}": v for k, v in moments.items()
    }
    expressions = fixture["statistics_expressions"]["data"]["result"]["expressions_latex"]
    assert dict(statistics(fixture["statistics_expressions"]["data"])) == {
        f"expr:{k}": v for k, v in expressions.items()
    }


def test_a_result_that_holds_given_a_premise_or_an_assumption_lists_nothing(
    fixture: dict[str, Any],
) -> None:
    base = fixture["statistics_moments"]["data"]
    for edit in (
        {"evidence": "conditional"},
        {"premises": ["the observations are independent and identically distributed"]},
        {"conditions": [{"latex": "p > 0", "established": False, "how": "assumed"}]},
        {"status": "not_verified"},
        {"status": "unsupported"},
    ):
        data = copy.deepcopy(base)
        data["result"].update(edit)
        assert statistics(data) == [], edit


def test_a_live_independence_statement_makes_the_estimate_conditional_and_unnumbered(
    ctx: ToolContext,
) -> None:
    out = execute(
        "statistical_model",
        {**NORMAL, "action": "mle", "sample_size": "n", "iid": True},
        ctx,
    )
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "conditional"
    assert statistics(out.card.data) == []


def test_an_estimate_that_exists_only_under_an_unestablished_condition_is_left_out(
    fixture: dict[str, Any],
) -> None:
    data = copy.deepcopy(fixture["statistics_estimate"]["data"])
    data["result"]["estimates"][0]["admissible_if"] = [
        {"latex": "x > 0", "established": False, "how": "assumed"}
    ]
    assert statistics(data) == []


def test_the_posterior_and_the_steps_are_never_numbered(fixture: dict[str, Any]) -> None:
    data = copy.deepcopy(fixture["statistics_moments"]["data"])
    data["result"]["posterior"] = {
        "family": "beta",
        "parameters": {},
        "parameters_latex": {"alpha": r"\alpha + 1"},
        "dropped_factors_latex": [r"\frac{1}{B}"],
    }
    data["result"]["steps"] = [{"title": "s", "before_latex": "a", "after_latex": "b = c"}]
    assert [k for k, _ in statistics(data)] == EXPECTED_KEYS["statistics_moments"]


def test_the_fisher_matrix_of_one_observation_is_one_formula_the_panel_draws(
    fixture: dict[str, Any],
) -> None:
    data = fixture["statistics_fisher"]["data"]
    assert data["result"]["evidence"] == "identity" and data["result"]["premises"] == []
    assert dict(statistics(data)) == {
        "fisher": r"\begin{pmatrix}\frac{1}{v} & 0 \\ 0 & \frac{1}{2 v^{2}}\end{pmatrix}"
    }
    # the panel's own `matrixLatex` writes the same text (rows joined by ` \\ `, cells by ` & `)
    source = Path(FIXTURE.parents[3], "apps/web/src/components/statistics-panel.tsx").read_text()
    assert (
        '`\\\\begin{pmatrix}${rows.map((r) => r.join(" & ")).join(" \\\\\\\\ ")}\\\\end{pmatrix}`'
        in source
    )
    assert '<Numbered k="fisher">' in source


def test_a_fisher_matrix_that_holds_given_iid_or_was_not_accepted_is_not_numbered(
    fixture: dict[str, Any],
) -> None:
    base = fixture["statistics_fisher"]["data"]
    for edit in (
        {"evidence": "conditional"},
        {"premises": ["the observations are independent and identically distributed"]},
        {"conditions": [{"latex": "v > 0", "established": False, "how": "assumed"}]},
        {"status": "not_verified"},
        {"status": "unsupported"},
    ):
        data = copy.deepcopy(base)
        data["result"].update(edit)
        assert statistics(data) == [], edit


def test_a_matrix_that_is_not_a_rectangle_of_formulas_is_not_numbered(
    fixture: dict[str, Any],
) -> None:
    for bad in ([], [[]], [["a", "b"], ["c"]], [["a", ""]], [["a", 1]], "a", None):
        data = copy.deepcopy(fixture["statistics_fisher"]["data"])
        data["result"]["matrix_latex"] = bad
        assert statistics(data) == [], bad


def test_a_live_iid_fisher_matrix_and_every_live_posterior_are_unnumbered(ctx: ToolContext) -> None:
    out = execute(
        "statistical_model",
        {**NORMAL, "action": "fisher_information", "sample_size": "n", "iid": True},
        ctx,
    )
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "conditional"
    assert statistics(out.card.data) == []
    post = execute(
        "statistical_model",
        {
            "family": "binomial",
            "parameters": {"n": "n"},
            "observation": "k",
            "estimate": ["p"],
            "action": "posterior",
            "prior": {"family": "beta", "parameters": {"alpha": "alpha", "beta": "beta"}},
            "assumptions": {
                "n": ["integer", "positive"],
                "k": ["integer", "nonnegative"],
                "alpha": ["positive"],
                "beta": ["positive"],
            },
        },
        ctx,
    )
    assert post.status == "ok" and post.card.data["result"]["posterior"]["parameters_latex"]
    # the prior is an assumption and is always among the premises: a posterior holds given it
    assert post.card.data["result"]["premises"]
    assert statistics(post.card.data) == []


def test_statistics_rows_are_numbered_persist_and_a_conversation_from_before_gets_them(
    fixture: dict[str, Any],
) -> None:
    card = card_from("statistics", fixture["statistics_expressions"]["data"])
    table = number(card)
    assert set(card.numbers) == set(EXPECTED_KEYS["statistics_expressions"])
    assert all(table.items[n].part_of == "statistics" for n in card.numbers.values())
    old = card_from("statistics", fixture["statistics_moments"]["data"], "c2").model_dump(
        mode="json"
    )
    content = {"blocks": [{"kind": "card", "card": old}]}
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert set(content["blocks"][0]["card"]["numbers"]) == set(EXPECTED_KEYS["statistics_moments"])


# --- expansion ----------------------------------------------------------------------------------------------
def test_an_expansion_numbers_its_verified_rows_and_not_its_terms_or_matrices(
    fixture: dict[str, Any],
) -> None:
    data = fixture["expansion"]["data"]
    card = card_from("expansion", data)
    table = number(card)
    assert list(card.numbers) == EXPECTED_KEYS["expansion"]
    assert all(table.items[n].part_of == "expansion" for n in card.numbers.values())
    report = data["report"]
    texts = {latex for _, latex in expansion(data)}
    assert not texts & {t["latex"] for e in report["equations"] for t in e["terms"]}
    assert report["stiffness_latex"] not in texts
    assert card.numbers.keys() >= {"mode:0:vector"}
    assert (
        dict(expansion(data))["mode:0:vector"] == "v = " + report["modes"][0]["eigenvector_latex"]
    )
    assert expansion(data)[1][1].endswith(
        r"O\left(\delta^{4}\right)"
    )  # a truncation, still an equality


def test_an_unverified_expansion_or_mode_and_a_failed_equilibrium_are_not_numbered(
    fixture: dict[str, Any],
) -> None:
    base = fixture["expansion"]["data"]
    unverified = copy.deepcopy(base)
    unverified["report"]["equations"][0]["verification"]["passed"] = False
    keys = [k for k, _ in expansion(unverified)]
    assert "eq:x" not in keys and "char" not in keys and "mode:0:omega2" in keys
    mode = copy.deepcopy(base)
    mode["report"]["modes"][0]["verified"] = False
    assert [k for k, _ in expansion(mode)] == ["equil:x", "eq:x", "char"]  # no mode row at all
    no_vector = copy.deepcopy(base)
    no_vector["report"]["modes"][0]["eigenvector_latex"] = None  # the residual check did not pass
    assert "mode:0:vector" not in [k for k, _ in expansion(no_vector)]
    equilibrium = copy.deepcopy(base)
    equilibrium["report"]["equilibrium_verification"]["passed"] = False
    assert "equil:x" not in [k for k, _ in expansion(equilibrium)]
    no_omega = copy.deepcopy(base)
    no_omega["report"]["modes"][0]["omega_latex"] = None
    assert "mode:0:omega" not in [k for k, _ in expansion(no_omega)]
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit"):
        data = copy.deepcopy(base)
        data["report"]["status"] = status
        assert expansion(data) == [], status


def test_the_modes_are_numbered_in_a_canonical_order_whatever_the_solver_found(
    fixture: dict[str, Any],
) -> None:
    entry = fixture["expansion_two_modes"]
    data = entry["data"]
    assert dict(expansion(data)) == entry["rows"]
    assert [k for k, _ in expansion(data)] == entry["keys"]
    modes = data["report"]["modes"]
    # the solver's own order is the reverse of the canonical one here: the fixture would not catch an unsorted reader
    assert [m["omega_squared_latex"] for m in modes] == [r"\omega^2 = 4", r"\omega^2 = 2"]
    rows = dict(expansion(data))
    assert rows["mode:0:omega2"] == r"\omega^2 = 2" and rows["mode:1:omega2"] == r"\omega^2 = 4"
    for i, m in enumerate(canonical_modes(modes)):
        assert rows[f"mode:{i}:vector"] == "v = " + m["eigenvector_latex"]
    for _ in range(2):
        shuffled = copy.deepcopy(data)
        shuffled["report"]["modes"].reverse()
        assert expansion(shuffled) == expansion(data)


def test_an_unverified_mode_loses_its_own_rows_and_the_other_keeps_its_canonical_position(
    fixture: dict[str, Any],
) -> None:
    base = fixture["expansion_two_modes"]["data"]
    rows = dict(expansion(base))
    assert rows["mode:0:vector"] != rows["mode:1:vector"]
    data = copy.deepcopy(base)
    data["report"]["modes"][0]["verified"] = (
        False  # the solver's first mode is omega^2 = 4: canonical 1
    )
    modes = [k for k, _ in expansion(data) if k.startswith("mode:")]
    assert modes == ["mode:0:omega2", "mode:0:omega", "mode:0:vector"]
    assert dict(expansion(data))["mode:0:vector"] == rows["mode:0:vector"]
    data["report"]["modes"][0]["verified"] = True
    data["report"]["modes"][0]["eigenvector_latex"] = (
        None  # residual check did not pass: no vector row
    )
    keys = [k for k, _ in expansion(data)]
    assert "mode:1:vector" not in keys and "mode:0:vector" in keys


def test_the_canonical_mode_sort_is_by_text_and_stable() -> None:
    a = {"omega_squared_latex": "b", "eigenvector_latex": "2"}
    b = {"omega_squared_latex": "a", "eigenvector_latex": "9"}
    twin = {"omega_squared_latex": "a", "eigenvector_latex": "1"}
    assert canonical_modes([a, b, twin]) == [twin, b, a]
    assert canonical_modes("nothing") == [] and canonical_modes([1, a]) == [a]


def test_the_modes_of_a_two_coordinate_system_come_in_the_same_order_whatever_the_hash_seed() -> (
    None
):
    code = (
        "import json\n"
        "from minerva_engine import Engine\n"
        "from minerva_engine.testing import budget_limits\n"
        "def main():\n"
        "    r = Engine(budget_limits(max_concurrency=1)).expand_motion("
        "r'\\frac12 (\\dot{x}^2 + \\dot{y}^2) - \\frac12 (3 x^2 + 3 y^2 - 2 x y)', ['x', 'y'], 't', {}, {}, 1)\n"
        "    print(json.dumps([m.omega_squared_latex for m in r.modes]))\n"
        "if __name__ == '__main__':\n    main()\n"
    )
    outputs = set()
    for seed in ("1", "7", "13"):
        done = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            timeout=240,
        )
        outputs.add(done.stdout.strip().splitlines()[-1])
    assert len(outputs) == 1
    assert len(json.loads(next(iter(outputs)))) == 2


# --- numbers persist, are never reused, and a row is a name only --------------------------------------------
@pytest.mark.parametrize("name", ["variation", "statistics_expressions", "expansion"])
def test_numbers_survive_a_rebuild_and_a_retry_never_reuses_them(
    fixture: dict[str, Any], name: str
) -> None:
    card = card_from(CARD_TYPE[name], fixture[name]["data"])
    table = number(card)
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    rebuilt = build_table([stored], counter=table.counter + 3)
    assert {n: e.latex for n, e in rebuilt.items.items()} == {
        n: e.latex for n, e in table.items.items()
    }
    assert all(e.part_of == CARD_TYPE[name] for e in rebuilt.items.values())
    # the answer is replaced (retry or edit): the counter is a high-water mark, the next card starts above it
    other = "statistics_moments" if name != "statistics_expressions" else "variation_second"
    again = card_from(CARD_TYPE[other], fixture[other]["data"], "c2")
    number(again, rebuilt)
    assert min(again.numbers.values()) == table.counter + 4


@pytest.mark.parametrize(
    ("name", "key"),
    [("variation", "el:x"), ("statistics_moments", "moment:mean"), ("expansion", "eq:x")],
)
def test_a_row_reused_by_number_is_exact_latex_and_says_it_is_one_component(
    fixture: dict[str, Any], name: str, key: str
) -> None:
    card = card_from(CARD_TYPE[name], fixture[name]["data"])
    table = number(card)
    n = card.numbers[key]
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    resolved = store.resolve({"latex": f"@e{n}"})
    assert resolved.error is None and resolved.arguments["latex"] == table.items[n].latex
    assert resolved.unverified == [f"equation ({n}){COMPONENT_SUFFIX}"]


def test_a_relation_row_is_refused_where_an_expression_is_read_and_an_expression_row_is_not(
    fixture: dict[str, Any],
) -> None:
    equation = card_from("statistics", fixture["statistics_estimate"]["data"])
    values = card_from("statistics", fixture["statistics_moments"]["data"], "c2")
    table = number(equation)
    number(values, table)
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    n = equation.numbers["estimate:mu"]
    assert table.items[n].kind == "equation"
    refused = store.resolve({"latex": f"@e{n}"}, "simplify")
    assert refused.diagnostic is not None
    assert refused.diagnostic.code is InputCode.EQUATION_REFERENCE_NOT_EXPRESSION
    mean = values.numbers["moment:mean"]
    assert table.items[mean].kind == "expression"
    accepted = store.resolve({"latex": f"@e{mean}"}, "simplify")
    assert accepted.error is None and accepted.diagnostic is None
    # the same for the integrand of a variation: a relation is refused, nothing is rewritten
    refused = store.resolve({"integrand": f"@e{n}"}, "variation")
    assert refused.diagnostic is not None and refused.diagnostic.field == "integrand"


def test_a_call_built_on_a_variation_row_is_refused_in_the_loop_with_a_typed_diagnostic(
    ctx: ToolContext, fixture: dict[str, Any]
) -> None:
    card = card_from("variation", fixture["variation"]["data"])
    table = number(card)
    n = card.numbers["el:x"]
    message, _ = play(
        ctx,
        [call("simplify", {"latex": f"@e{n}"}), say("done")],
        equations=table,
    )
    step = message["trace"][0]
    assert step["status"] == "error"
    assert card_of(message)["data"]["diagnostic"]["code"] == "equation_reference_not_expression"


# --- the whole result of sums, expectation and distribution ---------------------------------------------------
LIVE_WHOLE = {
    "sums": (
        "sums",
        {
            "latex": r"\sum_{i=1}^{n}(x_i-\mu)^2",
            "assumptions": {"n": ["integer", "positive"], "x": ["real"], "mu": ["real"]},
        },
    ),
    "expectation": ("expectation", {"latex": r"\operatorname{Var}(X+Y)", "random": ["X", "Y"]}),
    "distribution": (
        "distribution",
        {
            "latex": r"x^2\operatorname{DiracDelta}(x^2-4)",
            "assumptions": {"x": ["real"]},
            "action": "integrate",
            "variable": "x",
            "lower": "-oo",
            "upper": "oo",
        },
    ),
}


@pytest.mark.parametrize("kind", ["sums", "expectation", "distribution"])
def test_the_whole_result_of_these_cards_gets_a_number_that_persists(
    ctx: ToolContext, kind: str
) -> None:
    assert kind in SINGLE_RESULT and kind in NESTED_RESULT_CARDS and kind not in LIST_READERS
    tool, args = LIVE_WHOLE[kind]
    out = execute(tool, args, ctx)
    assert out.status == "ok" and out.card.type == kind
    expected = out.card.data["result"]["result_latex"]
    assert result_formula(kind, out.card.data) == expected
    card = card_from(kind, out.card.data)
    table = number(card)
    assert card.number == 1 and card.numbers == {} and table.counter == 1
    # rebuilt from the stored message, and read from a conversation that had no number yet
    stored = Row("assistant", {"blocks": [{"kind": "card", "card": card.model_dump(mode="json")}]})
    assert build_table([stored]).items[1].latex == expected
    content = {
        "blocks": [
            {"kind": "card", "card": card_from(kind, out.card.data, "c2").model_dump(mode="json")}
        ]
    }
    assert number_stored([("assistant", content)], EquationTable()) is True
    assert content["blocks"][0]["card"]["number"] == 1
    # reusing it by number says it is a tool result with no result id: restrictions are not carried
    store = ResultStore.from_prior({}, {}, table.items, unverified={})
    assert store.resolve({"latex": "@e1"}).unverified == [f"equation (1){NO_ID_SUFFIX}"]


@pytest.mark.parametrize("kind", ["sums", "expectation", "distribution"])
def test_a_result_that_was_not_accepted_gets_no_number(ctx: ToolContext, kind: str) -> None:
    tool, args = LIVE_WHOLE[kind]
    data = copy.deepcopy(execute(tool, args, ctx).card.data)
    for status in ("not_verified", "unsupported", "parse_error", "resource_limit", "unchanged"):
        data["result"]["status"] = status
        assert result_formula(kind, data) is None, status
        card = card_from(kind, data)
        number(card)
        assert card.number is None


def test_an_expectation_that_holds_given_independence_has_no_number(ctx: ToolContext) -> None:
    out = execute(
        "expectation",
        {"latex": r"\operatorname{Var}(X+Y)", "random": ["X", "Y"], "independent": [["X", "Y"]]},
        ctx,
    )
    assert out.status == "ok" and out.card.data["result"]["evidence"] == "conditional"
    assert result_formula("expectation", out.card.data) is None
    card = card_from("expectation", out.card.data)
    number(card)
    assert card.number is None


def test_approximation_keeps_its_result_in_another_place_and_is_not_numbered_here() -> None:
    data = {"report": {"status": "ok", "result_latex": "x"}}
    assert "approximation" in SINGLE_RESULT and "approximation" not in NESTED_RESULT_CARDS
    assert result_formula("approximation", data) is None


def test_the_nested_cards_keep_a_top_level_result_first() -> None:
    assert (
        result_formula(
            "sums", {"result_latex": "a", "result": {"status": "ok", "result_latex": "b"}}
        )
        == "a"
    )
    assert result_formula("step", {"result": {"status": "ok", "result_latex": "b"}}) == "b"
