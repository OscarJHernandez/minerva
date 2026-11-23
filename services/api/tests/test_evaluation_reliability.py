"""The reliability suite (evaluation/cases_reliability.yaml) without a real model.

Each case has an ideal trajectory (the calls and replies a correct agent produces), which must pass, and
adversarial ones, each of which must fail with the failure class it is built to trigger. The tools, the engine
and the claim checker are the real ones; only the model is scripted (`ReplayLLM`). This tests the harness: that
every check can fail, for the right reason, and that a correct agent is not failed.
"""

import asyncio
from typing import Any

import pytest
from minerva_api.agent.evaluation.cases import (
    EvalCase,
    cases_digest,
    load_cases,
    load_suite,
)
from minerva_api.agent.evaluation.runner import run_case
from minerva_api.agent.evaluation.score import CaseResult, FailureClass, combine
from minerva_api.agent.llm import LLMError
from minerva_api.agent.loop import AgentLimits
from minerva_api.agent.replay import ReplayLLM, call, say
from minerva_api.agent.tools import ToolContext
from minerva_engine import Engine
from minerva_engine.testing import budget_limits
from minerva_knowledge import KnowledgeBase

FAST = AgentLimits(retry_delay_seconds=0)
CASES = {c.id: c for c in [*load_suite("reliability"), *load_suite("heldout")]}
MATRICES = [{"name": "A", "rows": 2}, {"name": "B", "rows": 2}]
X_NEQ_1 = r"x \neq 1"


@pytest.fixture(scope="module")
def ctx() -> ToolContext:
    return ToolContext(engine=Engine(budget_limits(max_concurrency=2)), kb=KnowledgeBase.load())


def run(case_id: str, script: list[Any], ctx: ToolContext) -> CaseResult:
    async def go() -> CaseResult:
        return await run_case(ReplayLLM(script), ctx, CASES[case_id], FAST)

    return asyncio.run(go())


# The double pendulum Lagrangian of issue 137, as the cases write it.
DOUBLE_PENDULUM = (
    r"\frac12 (m_1+m_2) l_1^2 \dot{\theta}_1^2 + \frac12 m_2 l_2^2 \dot{\theta}_2^2 "
    r"+ m_2 l_1 l_2 \dot{\theta}_1 \dot{\theta}_2 \cos(\theta_1-\theta_2) "
    r"+ (m_1+m_2) g l_1 \cos\theta_1 + m_2 g l_2 \cos\theta_2"
)

# (case id) -> {"ideal": script, "adversarial": {failure class: script}}
SCRIPTS: dict[str, dict[str, Any]] = {
    "pole-cancelled-by-simplification-is-reported": {
        "ideal": [
            call("simplify", {"latex": r"\frac{x^2-4}{x-2}"}),
            say(
                r"The result is $x + 2$, valid for $x \neq 2$: the original is undefined at $x = 2$."
            ),
        ],
        "adversarial": {
            FailureClass.MISREPORTED_RESTRICTION: [
                call("simplify", {"latex": r"\frac{x^2-4}{x-2}"}),
                say("The result is $x + 2$."),
            ],
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "simplify", {"latex": r"\frac{x^2-4}{x-2}", "assumptions": {"x": ["positive"]}}
                ),
                say(r"The result is $x + 2$, valid for $x \neq 2$."),
            ],
        },
    },
    "reuse-and-restriction-after-simplify": {
        "ideal": [
            call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
            say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$ (the original is undefined at 1)."),
            call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
            say(
                rf"The derivative is $1$, and like the result it came from it holds only for ${X_NEQ_1}$."
            ),
        ],
        "adversarial": {
            # retyped instead of reused: the engine does not know the new call comes from @r1
            FailureClass.RESULT_REUSE: [
                call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
                say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$."),
                call("differentiate", {"latex": "x + 1", "variables": [{"name": "x"}]}),
                say(rf"The derivative is $1$, for ${X_NEQ_1}$."),
            ],
            FailureClass.MISREPORTED_RESTRICTION: [
                call("simplify", {"latex": r"\frac{x^2-1}{x-1}"}),
                say(rf"The result is $x + 1$, valid for ${X_NEQ_1}$."),
                call("differentiate", {"latex": "@r1", "variables": [{"name": "x"}]}),
                say("The derivative is $1$."),
            ],
        },
    },
    "followup-resolves-it-from-earlier-turns": {
        "ideal": [
            call("simplify", {"latex": r"\frac{x^2-9}{x-3}"}),
            say(r"The result is $x + 3$, valid for $x \neq 3$."),
            call(
                "integrate",
                {"latex": "@r1", "variable": "x", "lower": "0", "upper": "1"},
            ),
            say("Over 0 to 1 it is 7/2, using the simplified form from before."),
            say("We started from the expression (x^2 - 9)/(x - 3)."),
        ],
        "adversarial": {
            # the earlier result is retyped instead of reused as @r1
            FailureClass.RESULT_REUSE: [
                call("simplify", {"latex": r"\frac{x^2-9}{x-3}"}),
                say(r"The result is $x + 3$, valid for $x \neq 3$."),
                call("integrate", {"latex": "x + 3", "variable": "x", "lower": "0", "upper": "1"}),
                say("Over 0 to 1 it is 7/2."),
                say("We started from the expression (x^2 - 9)/(x - 3)."),
            ],
            # no tool at all, as when the model's view of its tools was cut away
            FailureClass.TOOL_SELECTION: [
                say("That simplifies to x + 3."),
                say("The integral is 7/2."),
                say("We started from (x^2 - 9)/(x - 3)."),
            ],
            # the conversation is there, but the model says it cannot see it
            FailureClass.MODEL_HONESTY: [
                call("simplify", {"latex": r"\frac{x^2-9}{x-3}"}),
                say(r"The result is $x + 3$, valid for $x \neq 3$."),
                call(
                    "integrate",
                    {"latex": "@r1", "variable": "x", "lower": "0", "upper": "1"},
                ),
                say("Over 0 to 1 it is 7/2."),
                say("I don't have access to the earlier conversation, so I cannot say."),
            ],
        },
    },
    "reciprocal-integral-invents-no-assumption": {
        "ideal": [
            call("integrate", {"latex": r"\frac{1}{x}", "variable": "x"}),
            say(r"An antiderivative is $\log{\left(x \right)}$, up to a constant."),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "integrate",
                    {"latex": r"\frac{1}{x}", "variable": "x", "assumptions": {"x": ["positive"]}},
                ),
                say(r"An antiderivative is $\log{\left(x \right)}$ for positive x."),
            ],
        },
    },
    "oscillating-integral-stops-when-not-verified": {
        "ideal": [
            call("integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}),
            say(
                "Minerva could not verify this integral: it does not converge, because the partial "
                "integrals oscillate and have no limit."
            ),
        ],
        "adversarial": {
            FailureClass.IGNORED_FAILED_VERIFICATION: [
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}
                ),
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "100"}
                ),
                say("It does not converge; I tried a finite upper limit instead."),
            ],
            FailureClass.MODEL_HONESTY: [
                call(
                    "integrate", {"latex": r"\sin x", "variable": "x", "lower": "0", "upper": "oo"}
                ),
                say("The integral equals 1."),
            ],
        },
    },
    "declared-matrices-then-false-expansion": {
        "ideal": [
            call(
                "matrix_calculus",
                {"latex": "(A+B)^2", "matrices": MATRICES, "action": "simplify"},
            ),
            say("Expanding keeps the order of the factors, so AB and BA stay separate terms."),
            say("No. That holds only if A and B commute; in general AB and BA differ."),
        ],
        "adversarial": {
            FailureClass.MODEL_HONESTY: [
                call(
                    "matrix_calculus",
                    {"latex": "(A+B)^2", "matrices": MATRICES, "action": "simplify"},
                ),
                say("Expanding keeps the order of the factors."),
                say("Yes, $(A+B)^2 = A^2 + 2AB + B^2$."),
            ],
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "matrix_calculus",
                    {
                        "latex": "(A+B)^2",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "commuting": [["A", "B"]],
                    },
                ),
                say("Expanding gives the usual binomial."),
                say("Yes, since they commute."),
            ],
        },
    },
    "declared-matrices-no-invented-commutation": {
        "ideal": [
            say(
                "Noted: A and B are 2x2 matrices. Nothing says whether they commute or are invertible."
            ),
            call(
                "matrix_calculus",
                {"latex": r"\exp(A)\exp(B)", "matrices": MATRICES, "action": "simplify"},
            ),
            say(
                "They cannot be merged into one exponential: that needs A and B to commute, which was not stated."
            ),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                say("Noted: A and B are 2x2 matrices."),
                call(
                    "matrix_calculus",
                    {
                        "latex": r"\exp(A)\exp(B)",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "commuting": [["A", "B"]],
                    },
                ),
                say("They commute, so the product is a single exponential."),
            ],
        },
    },
    "inverse-does-not-invent-invertibility": {
        "ideal": [
            call(
                "matrix_calculus",
                {"latex": "A B A^{-1}", "matrices": MATRICES, "action": "simplify"},
            ),
            say(
                "Minerva does not support matrix inverses, and that product cannot be reduced without assuming "
                "A is invertible and commutes with B; neither was stated."
            ),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "matrix_calculus",
                    {
                        "latex": "A B A^{-1}",
                        "matrices": MATRICES,
                        "action": "simplify",
                        "invertible": ["A"],
                    },
                ),
                say(
                    "Taking A invertible, the product is not supported either; an inverse needs assuming."
                ),
            ],
        },
    },
    "kepler-conserved-quantities-then-total-time-derivative": {
        "ideal": [
            call(
                "analyze_lagrangian",
                {
                    "lagrangian": r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}",
                    "coordinates": ["r", "theta"],
                },
            ),
            say(
                "Angular momentum and energy are conserved, and a third independent constant "
                "was found (a component of what is called the Laplace-Runge-Lenz vector)."
            ),
            call(
                "time_derivative", {"latex": r"\mu r^2 \dot{\theta}", "coordinates": ["r", "theta"]}
            ),
            say(
                r"Its total time derivative is $\mu (2 r \dot{r} \dot{\theta} + r^{2} \ddot{\theta})$."
            ),
        ],
        "adversarial": {
            # d/dt with `differentiate` holds the velocities fixed (it used to answer 0)
            FailureClass.TOOL_SELECTION: [
                call(
                    "analyze_lagrangian",
                    {
                        "lagrangian": r"\frac12 \mu (\dot{r}^2 + r^2 \dot{\theta}^2) + \frac{k}{r}",
                        "coordinates": ["r", "theta"],
                    },
                ),
                say("Angular momentum, energy and a third independent constant were found."),
                call(
                    "differentiate",
                    {"latex": r"\mu r^2 \dot{\theta}", "variables": [{"name": "t"}]},
                ),
                say("The derivative is 0."),
            ],
        },
    },
    "ode-bare-equation-asks-which-method": {
        "ideal": [
            call("ode_analyze", {"equations": ["y''(x) = y(x)"], "independent": "x"}),
            say(
                "Several methods apply: a closed form, a numerical solution and a series. "
                "Which method would you like me to use?"
            ),
            call(
                "ode_solve_symbolic",
                {"equations": ["y''(x) = y(x)"], "independent": "x", "method": "closed_form"},
            ),
            say(r"The general solution is $y(x) = C_{1} \mathrm{e}^{- x} + C_{2} \mathrm{e}^{x}$."),
        ],
        "adversarial": {
            # straight to a solver without asking: the server refuses it, and the attempt is the failure
            FailureClass.TOOL_SELECTION: [
                call("ode_analyze", {"equations": ["y''(x) = y(x)"], "independent": "x"}),
                call(
                    "ode_solve_symbolic",
                    {"equations": ["y''(x) = y(x)"], "independent": "x", "method": "closed_form"},
                ),
                say("The solution is y = C1 e^x + C2 e^-x."),
                call(
                    "ode_solve_symbolic",
                    {"equations": ["y''(x) = y(x)"], "independent": "x", "method": "closed_form"},
                ),
                say(
                    r"The general solution is $y(x) = C_{1} \mathrm{e}^{- x} + C_{2} \mathrm{e}^{x}$."
                ),
            ],
        },
    },
    "premise-stays-conditional": {
        "ideal": [
            call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
            say(r"Given your relation $y = 3x$, the result is $x^{2} + 3 x$."),
            say(
                "Not proved: it holds only given the premise $y = 3x$, which Minerva does not check."
            ),
        ],
        "adversarial": {
            FailureClass.MODEL_HONESTY: [
                call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
                say(r"Given your relation $y = 3x$, the result is $x^{2} + 3 x$."),
                say("Yes, it is proved that x^2 + y = x^2 + 3x."),
            ],
        },
    },
    "numbers-handed-over-are-chosen-and-said": {
        "ideal": [
            call(
                "ode_solve_numerical",
                {
                    "equations": ["x'' + x = 0"],
                    "independent": "t",
                    "conditions": ["x(0) = 1", "x'(0) = 0"],
                    "t_end": 10,
                },
            ),
            say("As you asked, I chose x(0) = 1, x'(0) = 0 and the interval up to t = 10."),
        ],
        "adversarial": {
            # the user handed the numbers over: asking again instead of running is not what they asked for
            FailureClass.TOOL_SELECTION: [say("Which initial values and end time would you like?")],
        },
    },
    "derivative-of-a-functional-is-refused-not-improvised": {
        "ideal": [
            say(
                "That is the derivative of a functional of an unknown curve, which is not supported "
                "here. `variation` can derive its Euler-Lagrange equation if you like."
            ),
        ],
        "adversarial": {
            FailureClass.MODEL_HONESTY: [
                say(r"The derivative is $\frac{d T}{d y_0} = \frac{1}{\sqrt{2 g}}$."),
            ],
        },
    },
    "linearise-about-a-circular-orbit-uses-expand-motion": {
        "ideal": [
            call(
                "expand_motion",
                {
                    "lagrangian": r"\frac12 m\dot{r}^2 - \frac{p^2}{2 m r^2} + \frac{G M m}{r}",
                    "coordinates": ["r"],
                    "equilibrium": {"r": "p^2/(G*M*m^2)"},
                    "order": 1,
                    "assumptions": {
                        "p": ["positive"],
                        "G": ["positive"],
                        "M": ["positive"],
                        "m": ["positive"],
                    },
                },
            ),
            say(
                "The tool checked the linearisation: the radial motion is harmonic (see the card)."
            ),
        ],
        "adversarial": {
            # substituting by hand instead of linearising with the tool
            FailureClass.TOOL_SELECTION: [
                call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
                say("I substituted the circular radius by hand."),
            ],
        },
    },
    "double-pendulum-both-angles-small-uses-expand-motion": {
        "ideal": [
            call(
                "expand_motion",
                {
                    "lagrangian": DOUBLE_PENDULUM,
                    "coordinates": ["theta_1", "theta_2"],
                    "equilibrium": {"theta_1": "0", "theta_2": "0"},
                    "order": 1,
                    "assumptions": {n: ["positive"] for n in ("m_1", "m_2", "g", "l_1", "l_2")},
                },
            ),
            say("The tool checked the linearisation about both angles at zero (see the card)."),
        ],
        "adversarial": {
            # substituting by hand instead of linearising with the tool
            FailureClass.TOOL_SELECTION: [
                call("substitute", {"latex": "x^2 + y", "relations": ["y = 3x"]}),
                say("I replaced the cosines by hand."),
            ],
        },
    },
    "double-pendulum-one-small-angle-is-not-widened": {
        "ideal": [
            call(
                "series",
                {"latex": DOUBLE_PENDULUM, "variable": "theta_1", "point": "0", "order": 2},
            ),
            say("Only $\\theta_1$ is taken as small here; $\\theta_2$ is left as it is."),
        ],
        "adversarial": {
            # expand_motion expands in every coordinate at once: it answers a different regime. The call
            # only has to be one to the forbidden tool, so it is a cheap one: the double pendulum itself
            # would spend seconds in the engine, and a slow runner would report a resource limit first.
            FailureClass.TOOL_SELECTION: [
                call(
                    "expand_motion",
                    {
                        "lagrangian": r"\frac12 m l^2\dot{\theta}_1^2 + m g l\cos\theta_1",
                        "coordinates": ["theta_1"],
                        "equilibrium": {"theta_1": "0"},
                        "order": 1,
                    },
                ),
                say("Both angles were taken as small."),
            ],
        },
    },
    "answer-without-thinking-aloud": {
        "ideal": [
            say(
                "The tautochrone problem asks for the curve on which a bead slides to the lowest point "
                "in the same time from any starting height. The answer is the cycloid."
            ),
        ],
        "adversarial": {
            # a conclusion, then a public retraction
            FailureClass.MODEL_HONESTY: [
                say(
                    "The derivative can never be zero. **Wait\u2014there is a subtle point here:** "
                    "the curve is not fixed."
                ),
            ],
        },
    },
    "spelled-velocities-are-read": {
        "ideal": [
            call(
                "analyze_lagrangian",
                {
                    "lagrangian": "(M+m)*(Xdot^2 + Ydot^2)/2 + m*sdot^2/2 + m*sdot*(Xdot*cos(alpha) + Ydot*sin(alpha)) - (M+m)*g*Y - m*g*s*sin(alpha)",
                    "coordinates": ["X", "Y", "s"],
                },
            ),
            say(
                r"The wedge and the block decouple: $\ddot{X} = 0$, $\ddot{Y} = -g$ and $\ddot{s} = 0$."
            ),
        ],
        "adversarial": {
            # the coordinates are missing: the call is refused before anything runs
            FailureClass.ARGUMENTS: [
                call("analyze_lagrangian", {"lagrangian": "m*Xdot^2/2"}),
                say("It moves freely."),
            ],
        },
    },
    "text-form-in-a-latex-field": {
        "ideal": [
            call("simplify", {"latex": "sin(x)^2 + cos(x)^2"}),
            say(r"The expression equals $1$."),
        ],
        "adversarial": {
            FailureClass.TOOL_SELECTION: [
                call(
                    "differentiate",
                    {"latex": r"\sin(x)^2 + \cos(x)^2", "variables": [{"name": "x"}]},
                ),
                say("It is zero."),
            ],
        },
    },
    "escaped-operators-in-a-derivative": {
        "ideal": [
            call(
                "differentiate",
                {"latex": "x\\^2\\*sin(x)", "variables": [{"name": "x"}]},
            ),
            say(r"The derivative is $x\left(x\cos(x) + 2\sin(x)\right)$."),
        ],
        "adversarial": {
            FailureClass.ARGUMENTS: [
                call("differentiate", {"latex": r"x^2 \sin(x)"}),
                say("The derivative is 2 x sin(x) + x^2 cos(x)."),
            ],
        },
    },
}

OSC = r"\frac12 m \dot{x}^2 - \frac12 k x^2"
PEND = r"\frac12 m \ell^2 \dot{\theta}^2 + m g \ell \cos\theta"
SQRT_X = r"\sqrt{x^2}"
SQRT_Y = r"\sqrt{y^2}"
PQ = [{"name": "P", "rows": 3}, {"name": "Q", "rows": 3}]
UNCHANGED_NOTE = "With x complex the root does not simplify without a sign assumption."
EXTRA_SCRIPTS: dict[str, dict[str, Any]] = {
    "assumption-added-then-dropped-in-followups": {
        "ideal": [
            call("simplify", {"latex": SQRT_X}),
            say(UNCHANGED_NOTE),
            call("simplify", {"latex": SQRT_X, "assumptions": {"x": ["positive"]}}),
            say("With x positive the root is unchanged: it is x."),
            call("simplify", {"latex": SQRT_X}),
            say(UNCHANGED_NOTE),
        ],
        "adversarial": {
            # the user's assumption never reaches the tool
            FailureClass.MISSING_ASSUMPTIONS: [
                call("simplify", {"latex": SQRT_X}),
                say(UNCHANGED_NOTE),
                call("simplify", {"latex": SQRT_X}),
                say("It is unchanged."),
                call("simplify", {"latex": SQRT_X}),
                say(UNCHANGED_NOTE),
            ],
            # ... and an assumption the user took back is still passed
            FailureClass.INVENTED_ASSUMPTION: [
                call("simplify", {"latex": SQRT_X}),
                say(UNCHANGED_NOTE),
                call("simplify", {"latex": SQRT_X, "assumptions": {"x": ["positive"]}}),
                say("With x positive the root is unchanged: it is x."),
                call("simplify", {"latex": SQRT_X, "assumptions": {"x": ["positive"]}}),
                say("It is unchanged."),
            ],
        },
    },
    "oscillator-equations-of-motion-then-conserved-quantity": {
        "ideal": [
            call("analyze_lagrangian", {"lagrangian": OSC, "coordinates": ["x"]}),
            say("The equation of motion follows from the Euler-Lagrange equation."),
            call("find_conserved_quantities", {"lagrangian": OSC, "coordinates": ["x"]}),
            say("The energy is conserved, within the families that were searched."),
        ],
        "adversarial": {
            FailureClass.TOOL_SELECTION: [
                call("analyze_lagrangian", {"lagrangian": OSC, "coordinates": ["x"]}),
                say("The equation of motion follows from the Euler-Lagrange equation."),
                say("The energy is conserved."),
            ],
        },
    },
    "heldout-reuse-and-restriction": {
        "ideal": [
            call("simplify", {"latex": r"\frac{t^2-25}{t-5}"}),
            say(r"The result is $t + 5$, valid for $t \neq 5$ (the original is undefined at 5)."),
            call("integrate", {"latex": "@r1", "variable": "t", "lower": "0", "upper": "2"}),
            say("The integral from 0 to 2 is 12, using the simplified form from before."),
        ],
        "adversarial": {
            FailureClass.RESULT_REUSE: [
                call("simplify", {"latex": r"\frac{t^2-25}{t-5}"}),
                say(r"The result is $t + 5$, valid for $t \neq 5$."),
                call("integrate", {"latex": "t + 5", "variable": "t", "lower": "0", "upper": "2"}),
                say("The integral from 0 to 2 is 12."),
            ],
        },
    },
    "heldout-ode-classify": {
        "ideal": [
            call("ode_analyze", {"equations": ["u'(t) = 3u(t)"], "independent": "t"}),
            say(
                "It is a first-order linear equation. A closed form, a numerical run or a series would work: which method do you want?"
            ),
        ],
        "adversarial": {
            FailureClass.TOOL_SELECTION: [
                say(
                    "It is a first-order linear equation; a closed form or a numerical method would work."
                ),
            ],
        },
    },
    "heldout-matrix-expansion": {
        "ideal": [
            call(
                "matrix_calculus",
                {"latex": "(P-Q)^2", "matrices": PQ, "action": "simplify"},
            ),
            say(
                "The square expands with both cross terms kept in order, because matrices need not commute."
            ),
        ],
        "adversarial": {
            FailureClass.INVENTED_ASSUMPTION: [
                call(
                    "matrix_calculus",
                    {
                        "latex": "(P-Q)^2",
                        "matrices": PQ,
                        "action": "simplify",
                        "commuting": [["P", "Q"]],
                    },
                ),
                say("Taking P and Q to commute, the cross terms combine."),
            ],
        },
    },
    "heldout-assumption-change": {
        "ideal": [
            call("simplify", {"latex": SQRT_Y}),
            say("With y complex the root does not simplify without a sign assumption."),
            call("simplify", {"latex": SQRT_Y, "assumptions": {"y": ["positive"]}}),
            say("With y positive the root is unchanged: it is y."),
        ],
        "adversarial": {
            FailureClass.MISSING_ASSUMPTIONS: [
                call("simplify", {"latex": SQRT_Y}),
                say("With y complex the root does not simplify without a sign assumption."),
                call("simplify", {"latex": SQRT_Y}),
                say("It is unchanged."),
            ],
        },
    },
    "heldout-pendulum-conserved": {
        "ideal": [
            call("analyze_lagrangian", {"lagrangian": PEND, "coordinates": ["theta"]}),
            say("The equation of motion follows from the Euler-Lagrange equation."),
            call("find_conserved_quantities", {"lagrangian": PEND, "coordinates": ["theta"]}),
            say("The energy is conserved, within the families that were searched."),
        ],
        "adversarial": {
            FailureClass.TOOL_SELECTION: [
                say("The equation of motion follows from the Euler-Lagrange equation."),
                say("The energy is conserved."),
            ],
        },
    },
    "heldout-series-exponential-cosine": {
        "ideal": [
            call("series", {"latex": r"\exp(x) \cos x", "variable": "x", "point": "0", "order": 4}),
            say("The expansion is given up to order 4."),
        ],
        "adversarial": {
            FailureClass.ARGUMENTS: [
                call(
                    "series",
                    {"latex": r"\exp(x) \cos x", "variable": "t", "point": "0", "order": 4},
                ),
                say("The expansion is given up to order 4."),
            ],
        },
    },
}
SCRIPTS.update(EXTRA_SCRIPTS)

IDEAL = [(case_id, s["ideal"]) for case_id, s in SCRIPTS.items()]
ADVERSARIAL = [
    (case_id, failure, script)
    for case_id, s in SCRIPTS.items()
    for failure, script in s["adversarial"].items()
]


def test_every_reliability_case_has_an_ideal_and_an_adversarial_trajectory() -> None:
    assert set(SCRIPTS) == set(CASES)
    assert all(s["adversarial"] for s in SCRIPTS.values())


def test_every_new_failure_class_is_exercised() -> None:
    exercised = {failure for _, failure, _ in ADVERSARIAL}
    assert exercised >= {
        FailureClass.INVENTED_ASSUMPTION,
        FailureClass.IGNORED_FAILED_VERIFICATION,
        FailureClass.MISREPORTED_RESTRICTION,
        FailureClass.RESULT_REUSE,
        FailureClass.MODEL_HONESTY,
    }


@pytest.mark.parametrize(("case_id", "script"), IDEAL, ids=[c for c, _ in IDEAL])
def test_the_ideal_trajectory_passes(case_id: str, script: list[Any], ctx: ToolContext) -> None:
    result = run(case_id, script, ctx)
    assert result.passed, (result.failure_class, result.notes)
    assert result.failure_class is None
    assert result.turns == 1 + len(CASES[case_id].followups)


@pytest.mark.parametrize(
    ("case_id", "failure", "script"),
    ADVERSARIAL,
    ids=[f"{c}:{f.value}" for c, f, _ in ADVERSARIAL],
)
def test_each_adversarial_trajectory_fails_with_its_own_class(
    case_id: str, failure: FailureClass, script: list[Any], ctx: ToolContext
) -> None:
    result = run(case_id, script, ctx)
    assert not result.passed
    assert result.failure_class is failure, (result.failure_class, result.notes)


def test_a_follow_up_is_scored_on_the_state_the_first_turn_really_produced(
    ctx: ToolContext,
) -> None:
    """The second call resolves @r1 to the engine's own result: the first answer reached the second turn."""
    llm = ReplayLLM(SCRIPTS["reuse-and-restriction-after-simplify"]["ideal"])
    asyncio.run(run_case(llm, ctx, CASES["reuse-and-restriction-after-simplify"], FAST))
    # the model's view of turn 2 includes the result id of turn 1 and the restriction note
    turn_two = next(
        m for m in llm.seen if any("result_id=r1" in str(x.get("content", "")) for x in m)
    )
    assert any("result_id=r1" in str(x.get("content", "")) for x in turn_two)
    last = llm.seen[-1]
    assert any("Inherited from @r1" in str(x.get("content", "")) for x in last)


def test_the_digest_of_the_original_cases_did_not_change() -> None:
    """Adding the reliability fields to the schema must not change which questions the core suite asks."""
    assert (
        cases_digest(load_cases()) == "b7d2202c3d87"
    )  # 28430901ca56 before check_identity was also accepted for false-pythagoras; 5ecf22d15a7d before the check_identity cases (two new false-claim cases and five also_ok lists); d2d3164e5b44 before the two ask-checks accepted a request for the missing expression; the core suite after the vector, electromagnetism, integral, creative telescoping, approximation, total-time-derivative, brachistochrone, ellipsoid, first-order-system, point-source (three-dimensional delta) and homepage example cases
    assert cases_digest(load_suite("reliability")) != cases_digest(load_cases())


def test_a_conversation_that_breaks_off_fails_and_a_combined_result_reports_the_first_failure() -> (
    None
):
    def result(passed: bool, failure: FailureClass | None, note: str = "") -> CaseResult:
        return CaseResult(
            case_id="c",
            category="symbolic",
            passed=passed,
            failure_class=failure,
            notes=[note] if note else [],
            expected_tool="simplify",
            tool_selected=True,
            tool_reached=True,
        )

    both = combine([result(True, None), result(False, FailureClass.RESULT_REUSE, "retyped")])
    assert not both.passed and both.failure_class is FailureClass.RESULT_REUSE
    assert both.turns == 2 and both.notes == ["turn 2: retyped"]
    earliest = combine(
        [
            result(False, FailureClass.INVENTED_ASSUMPTION, "a"),
            result(False, FailureClass.MODEL_HONESTY, "b"),
        ]
    )
    assert earliest.failure_class is FailureClass.INVENTED_ASSUMPTION


def test_a_follow_up_whose_earlier_turn_gave_no_answer_cannot_pass(ctx: ToolContext) -> None:
    """The model host fails on the very first call: there is no answer to build the next turn on."""
    case: EvalCase = CASES["reuse-and-restriction-after-simplify"]
    llm = ReplayLLM([])
    llm.error = LLMError("the model host is not reachable")
    result = asyncio.run(run_case(llm, ctx, case, FAST))
    assert not result.passed
    assert result.failure_class is FailureClass.MODEL_HOST
    assert len(llm.seen) <= 2  # it did not go on to play the follow-up turn
