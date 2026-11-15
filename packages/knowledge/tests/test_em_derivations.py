"""Electromagnetic derivations: built from handbook identities and premises the user adopts, each step
checked, the result evaluated on exact solutions of its premises."""

import mpmath as mp
import pytest
from minerva_engine import FieldDecl, Limits, Status, VectorInput
from minerva_engine.dimensions import DimensionError
from minerva_engine.vector import em, vector_calculus
from minerva_engine.vector.dimensions import check_vector_equation
from minerva_knowledge import KnowledgeBase
from minerva_knowledge.checks import run_check
from minerva_knowledge.identities import (
    premise_options,
    resolve_premise_ids,
    vector_identity_rules,
)
from minerva_knowledge.models import VectorEquationCheck

KB = KnowledgeBase.load()
RULES = vector_identity_rules(KB)
L = Limits(timeout_seconds=120)
TIMED = ["x", "y", "z", "t"]
KINDS = {
    "E": "vector",
    "B": "vector",
    "J": "vector",
    "A": "vector",
    "rho": "scalar",
    "phi": "scalar",
    "chi": "scalar",
}


def fields(template: em.Template, smoothness: str | None = "c2") -> list[FieldDecl]:
    return [
        FieldDecl(
            name=n,
            kind=k,
            depends_on=TIMED,
            smoothness=smoothness if n in template.smooth else None,
        )
        for n, k in template.fields
    ]


def derive(
    template_id: str,
    premises: list[str] | None = None,
    smoothness: str | None = "c2",
    system: str = "cartesian",
    constants: list[str] | None = None,
):
    template = em.TEMPLATES[template_id]
    declared = fields(template, smoothness)
    if system != "cartesian":
        declared = [d.model_copy(update={"depends_on": None}) for d in declared]
    return vector_calculus(
        VectorInput(
            expression="",
            format="text",
            coordinates=system,  # type: ignore[arg-type]
            fields=declared,
            constants=list(template.constants) if constants is None else constants,
            premises=[{"equation": p} for p in (template.needs if premises is None else premises)],  # type: ignore[misc]
            action="em_derive",
            template=template_id,
            rules=RULES,
        ),
        L,
    )


# -- the handbook equations --------------------------------------------------------------------------
def test_every_premise_a_template_needs_can_be_adopted_from_the_handbook_as_written() -> None:
    adoptable = {o.equation for o in premise_options(KB)}
    for template in em.TEMPLATES.values():
        for need in template.needs:
            assert need in adoptable, (template.id, need)


def test_adoptable_equations_are_field_equations_never_identities() -> None:
    options = premise_options(KB)
    assert len(options) >= 15
    ids = {o.id for o in options}
    assert {"em.maxwell.faraday", "em.maxwell.ampere", "em.maxwell.gauss", "em.continuity"} <= ids
    assert "em.poynting_theorem" not in ids and "em.wave_equation.electric" not in ids  # derived
    with pytest.raises(KeyError):
        resolve_premise_ids(KB, ["vector.curl_curl"])  # an identity is not a premise


def test_the_equations_are_dimensionally_consistent_and_a_wrong_one_is_not() -> None:
    entries = [e for e in KB.entries if isinstance(e.check, VectorEquationCheck)]
    assert len(entries) >= 24
    for entry in entries:
        report = run_check(entry)
        assert report.passed, (entry.id, report.detail)
        assert "units only" in report.detail
    ampere = next(e for e in entries if e.id == "em.maxwell.ampere_vacuum")
    check = ampere.check
    assert isinstance(check, VectorEquationCheck)
    for wrong in ("mu0*dt(E)", "mu0*epsilon0*dt(dt(E))", "mu0*epsilon0*E", "epsilon0*dt(E)"):
        broken = ampere.model_copy(update={"check": check.model_copy(update={"rhs": wrong})})
        assert not run_check(broken).passed, wrong
    with pytest.raises(DimensionError):
        check_vector_equation("div(E)", "rho", {"E": "vector", "rho": "scalar"},
                              {"E": "kg*m/(A*s**3)", "rho": "A*s/m**3"})  # fmt: skip


def test_mu0_epsilon0_and_the_speed_of_light_agree_within_the_uncertainties() -> None:
    eps = KB.get("const.epsilon_0")
    mu = KB.get("const.mu_0")
    assert eps is not None and mu is not None and eps.constant and mu.constant
    c = 299792458
    product = mp.mpf(eps.constant.value) * mp.mpf(mu.constant.value) * c**2
    relative = float(abs(product - 1))
    bound = float(
        mp.mpf(eps.constant.uncertainty or 0) / mp.mpf(eps.constant.value)
        + mp.mpf(mu.constant.uncertainty or 0) / mp.mpf(mu.constant.value)
    )
    assert relative < 3 * bound, (relative, bound)


# -- the derivations ---------------------------------------------------------------------------------
@pytest.mark.parametrize("template_id", sorted(em.TEMPLATES))
def test_each_derivation_is_found_verified_and_consistent_with_exact_solutions(
    template_id: str,
) -> None:
    template = em.TEMPLATES[template_id]
    result = derive(template_id)
    assert result.status is Status.OK, result.message
    assert result.steps and all(s.chain is not None for s in result.steps)
    assert result.model_check is not None and result.model_check.status == "passed"
    assert result.model_check.models >= 2 and "not a verification" in result.model_check.note
    if template.needs:
        assert result.evidence == "conditional" and result.given
        assert all(g in template.needs for g in result.given)
        assert "not an identity" not in result.message or "conditional" in result.message
    else:
        assert result.evidence == "identity" and result.given == []
    assert len(result.equations_latex) == len(template.results)


def test_the_wave_equation_shows_which_laws_and_identities_were_used() -> None:
    result = derive("wave_equation_e")
    used = {s.rule_id for s in result.steps if s.rule_id}
    assert {"vector.curl_curl", "vector.time_curl"} <= used
    assert set(result.given) == {
        "div(E) = 0",
        "curl(E) = -dt(B)",
        "curl(B) = mu0*epsilon0*dt(E)",
    }
    conditional = [s for s in result.steps if s.evidence == "conditional"]
    assert len(conditional) == 3 and all(s.given for s in conditional)
    assert result.equations_latex == [
        r"\nabla^{2} \mathbf{E} = \epsilon_{0} \mu_{0} \frac{\partial^{2} \mathbf{E}}{\partial t^{2}}"
    ]


def test_gauge_invariance_is_an_identity_and_needs_no_premise() -> None:
    result = derive("gauge_invariance", premises=[])
    assert result.status is Status.OK and result.evidence == "identity" and result.given == []
    assert all(s.evidence == "identity" for s in result.steps)


def test_a_premise_that_was_not_supplied_is_asked_for_never_assumed() -> None:
    template = em.TEMPLATES["wave_equation_e"]
    result = derive("wave_equation_e", premises=list(template.needs[:2]))
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.needs_premises == [template.needs[2]]
    assert result.steps == [] and result.result_latex is None
    assert "do not assume" not in result.message and "does not assume" in result.message


def test_a_premise_that_differs_from_the_one_needed_counts_as_missing() -> None:
    changed = ["div(E) = 0", "curl(E) = dt(B)", "curl(B) = mu0*epsilon0*dt(E)"]  # a sign flipped
    result = derive("wave_equation_e", premises=changed)
    assert result.status is Status.CONDITIONS_NOT_SATISFIED
    assert result.needs_premises == ["curl(E) = -dt(B)"]


def test_without_declared_smoothness_the_identities_are_blocked() -> None:
    result = derive("wave_equation_e", smoothness=None)
    assert result.status is not Status.OK and result.result_latex is None
    chi = derive("gauge_invariance", premises=[], smoothness=None)
    assert chi.status is not Status.OK


def test_constants_must_be_declared() -> None:
    result = derive("wave_equation_e", constants=[])
    assert result.status is Status.PARSE_ERROR and result.undeclared


def test_the_fields_must_be_declared_with_the_right_kinds() -> None:
    result = vector_calculus(
        VectorInput(
            expression="",
            format="text",
            fields=[FieldDecl(name="E", kind="scalar", depends_on=TIMED)],
            constants=["mu0", "epsilon0"],
            action="em_derive",
            template="wave_equation_e",
            rules=RULES,
        ),
        L,
    )
    assert result.status is Status.PARSE_ERROR and result.diagnostics.reason == "bad_declaration"


def test_a_derivation_works_in_spherical_coordinates_but_the_model_check_is_not_run() -> None:
    result = derive("wave_equation_e", system="spherical")
    assert result.status is Status.OK, result.message
    assert result.model_check is not None and result.model_check.status == "not_run"
    assert "Cartesian" in result.model_check.note


def test_an_unknown_derivation_is_refused() -> None:
    result = vector_calculus(
        VectorInput(expression="", template="perpetual_motion", action="em_derive"), L
    )
    assert result.status is Status.UNSUPPORTED


def test_a_result_that_disagrees_with_exact_solutions_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = em._build_model

    def broken(kind, ctx, template, rng):  # type: ignore[no-untyped-def]
        fields, params = real(kind, ctx, template, rng)
        x = ctx.low.q[0]
        t = ctx.low.t
        fields["E_x"] = x**2 * t**2 + 1  # no longer a solution of the wave equation
        return fields, params

    monkeypatch.setattr(em, "_build_model", broken)
    result = derive("wave_equation_e")
    assert result.status is Status.NOT_VERIFIED
    assert result.diagnostics.reason == "model_check_failed" and result.result_latex is None
    assert result.model_check is not None and result.model_check.status == "failed"


def test_a_stated_combination_that_is_not_an_algebraic_consequence_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = em.TEMPLATES["wave_equation_e"]
    wrong = em.Template(
        **{
            **template.__dict__,
            "results": (em.Result("lap(E)", "-mu0*epsilon0*dt(dt(E))", (("1", 1), ("-1", 0))),),
        }
    )
    monkeypatch.setitem(em.TEMPLATES, "wave_equation_e", wrong)
    result = derive("wave_equation_e")
    assert result.status is Status.NOT_VERIFIED and "algebra" in result.message


def test_the_catalogue_does_not_promise_what_it_cannot_derive() -> None:
    assert em.PLANNED and all(
        "Lorenz" in p or "interface" in p or "matter" in p for p in em.PLANNED
    )
    assert {"wave_equation_e", "continuity", "poynting", "gauge_invariance"} <= set(em.TEMPLATES)
