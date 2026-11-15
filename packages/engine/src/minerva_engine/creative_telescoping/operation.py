"""Creative telescoping of a parameterised definite integral: the operation.

The route, each step reported as a stage and none skipped silently:

1. analyse    is the integrand hyperexponential in x and in the parameter?
2. search     bounded Almkvist-Zeilberger search for L and G with L f = dG/dx
3. verify     exact proof of that identity (plus two numerical legs that may only veto)
4. boundary   no interior singularity; convergence of f and its parameter derivatives; the
              limits of G at the ends. Nothing is assumed: what cannot be decided is reported
5. equation   sum_k p_k I^(k)(a) = G(upper) - G(lower)
6. solve      dsolve, checked by substitution; constants from verified initial values
7. value      shown only if the ODE, the initial values and quadrature all agree

A verified certificate alone is a partial result (`outcome="certificate"`); a derived equation is
`outcome="equation"`; only a fully verified value is `outcome="evaluated"`.
"""

from __future__ import annotations

import time

import sympy as sp

from ..build import Builder
from ..distributions.models import Condition
from ..errors import LimitError, MinervaMathError, UnsupportedInput
from ..integrate import _bound
from ..latexprint import to_latex, to_text
from ..models import ExpressionInput, Limits, Status
from ..ops import parse_expression
from . import boundary as bd
from . import certificate as cert
from . import evaluate as ev
from . import ode
from .integrand import analyse
from .models import (
    BoundaryOut,
    CertificateOut,
    EquationOut,
    OperatorOut,
    Reason,
    Refusal,
    SolutionOut,
    TelescopingInput,
    TelescopingResult,
    timed,
)
from .search import Candidate, _log_derivative_powers, discover

_STATUS_FOR_REASON: dict[Reason, Status] = {
    "not_hyperexponential": Status.UNSUPPORTED,
    "variable_missing": Status.PARSE_ERROR,
    "parameter_missing": Status.PARSE_ERROR,
    "parameter_domain_unknown": Status.CONDITIONS_NOT_SATISFIED,
    "bounds_required": Status.PARSE_ERROR,
    "bound_invalid": Status.PARSE_ERROR,
    "no_telescoper_found": Status.UNSUPPORTED,
    "certificate_rejected": Status.NOT_VERIFIED,
    "singularity_not_excluded": Status.CONDITIONS_NOT_SATISFIED,
    "branch_not_established": Status.CONDITIONS_NOT_SATISFIED,
    "convergence_not_established": Status.CONDITIONS_NOT_SATISFIED,
    "divergent": Status.UNSUPPORTED,
    "differentiation_not_justified": Status.CONDITIONS_NOT_SATISFIED,
    "boundary_not_established": Status.CONDITIONS_NOT_SATISFIED,
    "singular_point_in_domain": Status.OK,  # a verified equation, not solved over the domain
    "ode_not_solved": Status.OK,
    "initial_value_unavailable": Status.OK,
    "value_not_verified": Status.OK,
    "unsupported_input": Status.UNSUPPORTED,
    "resource_limit": Status.RESOURCE_LIMIT,
}

_PARTIAL: frozenset[Reason] = frozenset(
    {
        "singular_point_in_domain",
        "ode_not_solved",
        "initial_value_unavailable",
        "value_not_verified",
    }
)


def _derivative_symbol(k: int) -> sp.Symbol:
    names = {0: "I(a)", 1: "I'(a)", 2: "I''(a)"}
    return sp.Symbol(names.get(k, f"I^{{({k})}}(a)"))


def _equation_latex(coefficients: tuple[sp.Expr, ...], rhs: sp.Expr) -> str:
    terms: list[str] = []
    for k, c in enumerate(coefficients):
        if c == 0:
            continue
        text = to_latex(c)
        if isinstance(c, sp.Add):
            text = f"\\left({text}\\right)"
        name = _derivative_symbol(k).name
        terms.append(name if c == 1 else f"{text}\\,{name}")
    return f"{' + '.join(terms)} = {to_latex(rhs)}"


def creative_telescoping(request: TelescopingInput, limits: Limits) -> TelescopingResult:
    started = time.monotonic()
    deadline = started + 0.8 * limits.timeout_seconds
    result = TelescopingResult(
        status=Status.OK, variable=request.variable, parameter=request.parameter
    )
    stages = result.diagnostics.stages

    def finish(
        status: Status,
        outcome: str,
        reason: Reason | None,
        message: str,
    ) -> TelescopingResult:
        result.status = status
        result.outcome = outcome  # type: ignore[assignment]
        result.reason = reason
        result.diagnostics.reason = reason
        result.message = message
        result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    def refuse(r: Refusal, outcome: str = "none") -> TelescopingResult:
        if stages and stages[-1].status == "ok":
            stages[-1].status = "failed"
            stages[-1].reason = r.reason
        return finish(_STATUS_FOR_REASON[r.reason], outcome, r.reason, str(r))

    # ---- parse -------------------------------------------------------------------------------
    try:
        data = ExpressionInput(
            input=request.input, format=request.format, assumptions=request.assumptions
        )
        f = parse_expression(data, limits)
        builder = Builder(request.assumptions, limits)
        x = builder.symbol(request.variable)
        a = builder.symbol(request.parameter)
        if (request.lower is None) != (request.upper is None):
            raise Refusal("bounds_required", "Give both bounds, or neither.")
        try:
            lower = _bound(request.lower, data, limits) if request.lower is not None else None
            upper = _bound(request.upper, data, limits) if request.upper is not None else None
        except MinervaMathError as exc:
            raise Refusal("bound_invalid", str(exc)) from exc
    except LimitError as exc:
        return finish(Status.RESOURCE_LIMIT, "none", "resource_limit", str(exc))
    except UnsupportedInput as exc:
        return finish(Status.UNSUPPORTED, "none", "unsupported_input", str(exc))
    except Refusal as exc:
        return refuse(exc)
    except MinervaMathError as exc:
        return finish(Status.PARSE_ERROR, "none", None, str(exc))
    except ValueError as exc:
        return finish(Status.PARSE_ERROR, "none", None, str(exc))

    result.input_latex = to_latex(f)
    if x == a:
        return finish(
            Status.PARSE_ERROR, "none", None, "The variable and the parameter must differ."
        )
    if x.is_integer or x.is_rational:
        return finish(
            Status.UNSUPPORTED,
            "none",
            "unsupported_input",
            "The integration variable is declared integer or rational; declare it real or leave it.",
        )
    if lower is not None and upper is not None:
        result.lower_latex = to_latex(lower)
        result.upper_latex = to_latex(upper)
        if not (lower < upper):
            return finish(
                Status.PARSE_ERROR,
                "none",
                "bound_invalid",
                "The lower bound must be below the upper bound.",
            )
    domain = bd.parameter_domain(a)
    if domain is not None:
        result.parameter_domain_latex = f"{to_latex(a)} \\in {to_latex(sp.Interval.open(*domain))}"

    try:
        # ---- 1. analyse ----------------------------------------------------------------------
        with timed(stages, "analyse"):
            h = analyse(f, x, a)

        # ---- 2./3. search and verify -----------------------------------------------------------
        def accept(candidate: Candidate) -> bool:
            check = cert.verify(f, x, candidate.operator, candidate.y * f)
            result.certificate_check = check
            return check.passed

        with timed(stages, "search") as stage:
            found, families, rejected = discover(
                h,
                max_order=request.max_order,
                max_degree=request.max_degree,
                deadline=deadline,
                accept=accept,
            )
            result.families = families
            result.diagnostics.families_tried = len(families)
            result.diagnostics.candidates_rejected = rejected
            if found is None:
                stage.status = "failed"
        if found is None:
            result.certificate_check = result.certificate_check if rejected else None
            raise Refusal(
                "certificate_rejected" if rejected else "no_telescoper_found",
                "A candidate failed verification and was dropped."
                if rejected
                else "No telescoping relation was found within the searched families "
                f"(operator order up to {request.max_order}, certificate degree up to "
                f"{request.max_degree}). That says nothing about other families.",
            )
        stages.append(stages[-1].model_copy(update={"name": "verify", "duration_ms": 0}))
        op, y = found.operator, found.y
        G = sp.simplify(y * f)
        result.diagnostics.order, result.diagnostics.degree = found.order, found.degree
        result.operator = OperatorOut(
            order=found.order,
            coefficients_latex=[to_latex(c) for c in op.coefficients],
            latex=f"L = {op.latex()}",
        )
        result.certificate = CertificateOut(latex=to_latex(G), latex_ratio=to_latex(sp.factor(y)))

        if lower is None or upper is None:
            result.unresolved = [
                "No bounds were given, so no equation for the integral was derived."
            ]
            return finish(
                Status.OK,
                "certificate",
                None,
                "A telescoping certificate was found and proved: L f = dG/dx. No bounds were "
                "given, so nothing is claimed about an integral.",
            )

        # ---- 4. boundary -------------------------------------------------------------------------
        with timed(stages, "boundary") as stage:
            analysis = bd.analyse(h, _log_derivative_powers(h, found.order), y, lower, upper)
            result.conditions = analysis.conditions
            if not analysis.sound:
                stage.status = "failed"
        if analysis.divergent:
            raise Refusal(
                "divergent",
                "The integral diverges: the integrand is not integrable at an end of the interval "
                "(shown from its local behaviour, which has no oscillation there).",
            )
        if analysis.problems:
            problem = analysis.problems[0]
            result.unresolved = [p.message for p in analysis.problems]
            raise Refusal(
                problem.reason, problem.message + " The certificate above is still valid."
            )
        low_value = analysis.values["lower"]
        high_value = analysis.values["upper"]
        assert low_value is not None and high_value is not None  # noqa: S101 - set when sound
        contribution = sp.simplify(high_value - low_value)
        result.boundary = BoundaryOut(
            lower_latex=to_latex(low_value),
            upper_latex=to_latex(high_value),
            contribution_latex=to_latex(contribution),
            vanishes=contribution == 0,
        )

        # ---- 5. equation -------------------------------------------------------------------------
        with timed(stages, "equation"):
            leading = op.coefficients[found.order]
            nonzero: bool | None = None
            if domain is not None:
                nonzero = bd.sign_on_interval(sp.numer(sp.cancel(leading)), a, *domain) is not None
            y_param = [
                fac
                for fac, _m in sp.factor_list(sp.denom(sp.cancel(y)))[1]
                if x not in fac.free_symbols
            ]
            if nonzero and domain is not None:
                for fac in y_param:
                    if bd.sign_on_interval(fac, a, *domain) is None:
                        nonzero = False
            result.equation = EquationOut(
                latex=_equation_latex(op.coefficients, contribution),
                homogeneous=contribution == 0,
                rhs_latex=to_latex(contribution),
                leading_nonzero=nonzero,
            )
        result.conditions.append(
            Condition(
                latex=f"{to_latex(leading)} \\neq 0",
                established=bool(nonzero),
                how="the leading coefficient has a constant sign on the parameter domain"
                if nonzero
                else "not provable on the whole parameter domain",
            )
        )
        if not nonzero:
            result.unresolved = [
                "The leading coefficient of the equation may vanish inside the parameter "
                "domain, so the equation is not solved over it."
            ]
            raise Refusal(
                "singular_point_in_domain",
                "The equation holds, but its leading coefficient is not provably nonzero on the "
                "parameter domain, so Minerva stops here.",
            )

        # ---- 6./7. solve and evaluate -----------------------------------------------------------
        assert domain is not None  # noqa: S101 - `nonzero` is only True when it is known
        with timed(stages, "solve") as stage:
            if found.order == 0:
                general: sp.Expr | None = sp.simplify(contribution / leading)
            else:
                general = ode.general_solution(op, contribution)
            if general is None:
                stage.status = "failed"
        if general is None:
            result.unresolved = ["SymPy found no closed-form solution of the equation."]
            raise Refusal(
                "ode_not_solved",
                "The equation was derived and verified but could not be solved in closed form.",
            )
        initial = ev.initial_values(
            f, x, a, found.order, ev.candidate_points(domain), lower, upper, deadline
        )
        if initial is None:
            result.unresolved = [
                "The integral and its parameter derivatives could not be evaluated independently "
                "at any trial parameter value, so the integration constants stay undetermined."
            ]
            raise Refusal(
                "initial_value_unavailable",
                "The equation was derived but no independently verified initial values exist.",
            )
        fixed: tuple[sp.Expr, list[sp.Eq]] | None
        if found.order == 0:
            fixed = (general, [])
            initial = ev.Initial(sp.Integer(1), [])
        else:
            fixed = ev.fix_constants(general, a, initial)
        if fixed is None:
            result.unresolved = ["The integration constants could not be determined."]
            raise Refusal(
                "initial_value_unavailable", "The integration constants could not be determined."
            )
        value, constants = fixed
        with timed(stages, "value") as stage:
            check = ev.accept(f, x, a, op, contribution, value, initial, lower, upper, deadline)
            result.value_verification = check
            if not check.passed:
                stage.status = "failed"
        if not check.passed:
            stage.status = "failed"
        result.solution = SolutionOut(
            general_latex=f"I(a) = {to_latex(general)}",
            initial_point_latex=f"{to_latex(a)} = {to_latex(initial.point)}",
            initial_values_latex=[
                f"{_derivative_symbol(k).name} = {to_latex(v)}"
                for k, v in enumerate(initial.values)
            ],
            constants_latex=[to_latex(c) for c in constants],
        )
        if not check.passed:
            result.unresolved = [f"A candidate value was rejected and is not shown: {check.note}."]
            raise Refusal(
                "value_not_verified",
                "The equation was derived but the candidate value did not pass every check.",
            )
        # A solution that failed its checks is never shown, not even its constants.
        result.result_latex = to_latex(value)
        result.result_text = to_text(value)
        result.notes.append(
            "Derived from a proved telescoping certificate and a verified equation, with constants "
            "from independently verified initial values. The final value is confirmed by numerical "
            "quadrature at sampled parameter values: that is evidence, not a proof."
        )
        return finish(
            Status.OK,
            "evaluated",
            None,
            "The integral was evaluated from the derived differential equation and checked.",
        )
    except Refusal as exc:
        outcome = (
            "equation"
            if exc.reason in _PARTIAL
            else ("certificate" if result.certificate is not None else "none")
        )
        if exc.reason in _PARTIAL and result.equation is None:
            outcome = "certificate"
        if exc.reason in {"divergent"}:
            outcome = "none"
        return refuse(exc, outcome)
    except LimitError as exc:
        return finish(Status.RESOURCE_LIMIT, "none", "resource_limit", str(exc))
