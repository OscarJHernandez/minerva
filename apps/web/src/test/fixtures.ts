import type {
  HistoryItem,
  DifferentiateResult,
  IdentitiesResult,
  IdentityMatch,
  SeriesResult,
  IntegrateResult,
  SimplifyResult,
} from "@/lib/types";

export const okIntegral: IntegrateResult = {
  status: "ok",
  input_latex: "x \\cos{\\left(x \\right)}",
  variable: "x",
  definite: false,
  lower_latex: null,
  upper_latex: null,
  numeric: null,
  conditions_latex: [],
  quadrature: null,
  notation_latex: "\\int x \\cos{\\left(x \\right)}\\, dx",
  result_latex: "x \\sin{\\left(x \\right)} + \\cos{\\left(x \\right)}",
  result_text: "x*sin(x) + cos(x)",
  verification: {
    method: "central finite difference of the antiderivative, 40 digits",
    directions: ["real", "imaginary"],
    steps_checked: 1,
    branches_checked: 1,
    points_checked: 12,
    points_agreed: 12,
    passed: true,
    symbolic_zero: true,
    note: "agreed",
  },
  notes: ["This is one antiderivative: add an arbitrary constant."],
  message: "Integrated with SymPy; the antiderivative was verified by a numerical derivative.",
  duration_ms: 40,
};

export const okDerivative: DifferentiateResult = {
  status: "ok",
  input_latex: "x^{5}",
  variables: [{ name: "x", order: 2 }],
  total_order: 2,
  notation_latex: "\\frac{d^{2}}{d x^{2}} x^{5}",
  result_latex: "20 x^{3}",
  result_text: "20*x**3",
  steps: [
    { variable: "x", order_so_far: 1, latex: "5 x^{4}" },
    { variable: "x", order_so_far: 2, latex: "20 x^{3}" },
  ],
  verification: {
    method: "central finite difference of the previous expression, 40 digits",
    directions: ["real", "imaginary"],
    steps_checked: 2,
    points_checked: 24,
    points_agreed: 24,
    passed: true,
    note: "every step agreed with a numerical derivative at all sampled points",
  },
  notes: [],
  message: "Differentiated with SymPy; every step was verified numerically.",
  duration_ms: 30,
};

export const okSeries: SeriesResult = {
  status: "ok",
  input_latex: "e^{x}",
  variable: "x",
  point_latex: "0",
  order: 3,
  result_latex: "\\frac{x^{3}}{6} + \\frac{x^{2}}{2} + x + 1",
  result_text: "x**3/6 + x**2/2 + x + 1",
  remainder_latex: "O\\left(x^{4}\\right)",
  terms: [
    { power: 0, coefficient_latex: "1" },
    { power: 1, coefficient_latex: "1" },
    { power: 2, coefficient_latex: "\\frac{1}{2}" },
    { power: 3, coefficient_latex: "\\frac{1}{6}" },
  ],
  verification: {
    coefficient_method: "Cauchy integral",
    remainder_method: "remainder falls as h^(order+1)",
    directions: ["real", "imaginary"],
    parameter_sets_checked: 1,
    coefficients_passed: true,
    remainder_passed: true,
    radius: 1,
    passed: true,
    note: "coefficients agree with a Cauchy-integral estimate and the remainder shrinks at the claimed order",
  },
  measured_errors: [
    { distance: 0.5, absolute_error: "2.34e-3", relative_error: "1.42e-3" },
    { distance: 0.1, absolute_error: "4.2e-7", relative_error: "3.8e-7" },
  ],
  notes: [],
  message: "Expanded with SymPy; coefficients and remainder order verified numerically.",
  duration_ms: 40,
};

export const chainSimplify: SimplifyResult = {
  status: "ok",
  goal: "simplify",
  input_latex: "\\sin^{2}{x} + \\cos^{2}{x} + \\tan^{2}{x}",
  result_latex: "\\sec^{2}{x}",
  result_text: "sec(x)**2",
  verification: {
    symbolic_zero: true,
    numeric_points_checked: 12,
    numeric_points_agreed: 12,
    passed: true,
    note: "numeric check agreed at all sampled points",
  },
  steps: [
    {
      kind: "rule",
      name: "trig.pythagorean",
      direction: "forward",
      matched_latex: "\\sin^{2}{x} + \\cos^{2}{x}",
      before_latex: "\\sin^{2}{x} + \\cos^{2}{x} + \\tan^{2}{x}",
      after_latex: "\\tan^{2}{x} + 1",
      verification: {
        symbolic_zero: true,
        numeric_points_checked: 12,
        numeric_points_agreed: 12,
        passed: true,
        note: "",
      },
    },
    {
      kind: "builtin",
      name: "sympy.simplify",
      direction: null,
      matched_latex: null,
      before_latex: "\\tan^{2}{x} + 1",
      after_latex: "\\sec^{2}{x}",
      verification: {
        symbolic_zero: false,
        numeric_points_checked: 12,
        numeric_points_agreed: 12,
        passed: true,
        note: "",
      },
    },
  ],
  alternatives: [
    {
      result_latex: "\\tan^{2}{x} + 1",
      result_text: "tan(x)**2 + 1",
      steps: [],
      verification: {
        symbolic_zero: true,
        numeric_points_checked: 12,
        numeric_points_agreed: 12,
        passed: true,
        note: "",
      },
    },
  ],
  blocked: [],
  entries: {
    "trig.pythagorean": {
      id: "trig.pythagorean",
      title: "Pythagorean identity",
      kind: "identity",
      chapter: "math.trigonometry",
      statement_latex: "\\sin^2 a + \\cos^2 a = 1",
      validity: "All complex a.",
      verification: "symbolic",
      reference: null,
    },
  },
  message: "2 verified step(s).",
  duration_ms: 90,
};

export const okSimplify: SimplifyResult = {
  status: "ok",
  input_latex: "\\sin^{2}{\\left(x \\right)} + \\cos^{2}{\\left(x \\right)}",
  result_latex: "1",
  result_text: "1",
  verification: {
    symbolic_zero: true,
    numeric_points_checked: 12,
    numeric_points_agreed: 12,
    passed: true,
    note: "numeric check agreed at all sampled points",
  },
  message: "Simplified with SymPy and verified.",
  duration_ms: 42,
};

export function match(
  id: string,
  applicability: IdentityMatch["application"]["applicability"],
  extra: Partial<IdentityMatch["application"]> = {},
): IdentityMatch {
  return {
    entry: {
      id,
      title: `Title of ${id}`,
      kind: "identity",
      chapter: "math.exp_log",
      statement_latex: "\\log(ab) = \\log a + \\log b",
      validity: "Both factors positive.",
      verification: "symbolic",
      reference: { source: "DLMF", locator: "§4.8", url: null },
    },
    application: {
      rule_id: id,
      applicability,
      matched_subexpression_latex: null,
      rewritten_latex: null,
      unmet_conditions: [],
      verification: null,
      message: "",
      ...extra,
    },
  };
}

export function identities(matches: IdentityMatch[]): IdentitiesResult {
  return {
    status: "ok",
    input_latex: "\\log(xy)",
    message: "",
    matches,
    candidates_considered: matches.length,
    duration_ms: 5,
  };
}

export function step(
  id: string,
  operation: HistoryItem["operation"],
  input: string,
  result: Record<string, unknown>,
  extra: Partial<HistoryItem> = {},
): HistoryItem {
  return {
    id,
    operation,
    input,
    format: "text",
    assumptions: {},
    status: "ok",
    result: {
      status: "ok",
      message: "",
      duration_ms: 1,
      ...result,
    } as unknown as HistoryItem["result"],
    created_at: "2026-11-12T10:00:00Z",
    derivation_id: "d1",
    parent_id: null,
    ...extra,
  };
}

// --- controlled steps (substitute, rewrite one part, isolate) ---
import type { Diagnostics, IsolateResult, StepResult, Subterm } from "@/lib/types";

export function diagnostics(over: Partial<Diagnostics> = {}): Diagnostics {
  return {
    parse_ms: 1,
    candidates_ms: 0,
    transform_ms: 0,
    verify_ms: 2,
    candidates: 1,
    rejected: {},
    verification_method: "congruence",
    verification_outcome: "passed",
    symbolic_zero: false,
    restrictions_outcome: "none",
    assumptions_declared: 0,
    premises: 1,
    detail: [],
    ...over,
  };
}

export function stepResult(over: Partial<StepResult> = {}): StepResult {
  return {
    status: "ok",
    operation: "substitute",
    input_latex: "x^{2} + y^{2}",
    result_latex: "1",
    result_text: "1",
    steps: [
      {
        action: "substitute",
        name: "y^{2} = 1 - x^{2}",
        direction: null,
        matched_latex: "y^{2}",
        before_latex: "x^{2} + y^{2}",
        after_latex: "1",
        display_latex: "x^{2} + \\left(1 - x^{2}\\right)",
        evidence: "conditional",
        given: ["y^{2} = 1 - x^{2}"],
        conditions_latex: [],
        verification: null,
        note: "",
      },
    ],
    evidence: "conditional",
    given: [{ latex: "y^{2} = 1 - x^{2}", text: "y**2 = 1 - x**2" }],
    verification: {
      symbolic_zero: false,
      numeric_points_checked: 12,
      numeric_points_agreed: 12,
      passed: true,
      note: "",
    },
    restrictions_latex: [],
    restrictions_complete: true,
    conditions_latex: [],
    unmet_conditions: [],
    reason: null,
    order_dependent: false,
    subterms: [],
    truncated: false,
    diagnostics: diagnostics(),
    message: "Holds given y**2 = 1 - x**2.",
    duration_ms: 5,
    ...over,
  };
}

export function isolateResult(over: Partial<IsolateResult> = {}): IsolateResult {
  return {
    status: "ok",
    equation_latex: "a x^{2} = b",
    variable: "x",
    method: "polynomial",
    steps: [],
    solutions: [
      {
        latex: "x = \\sqrt{\\frac{b}{a}}",
        text: "sqrt(b/a)",
        equation_text: "x = sqrt(b/a)",
        conditions_latex: [],
        branch: "",
        verification: null,
      },
      {
        latex: "x = - \\sqrt{\\frac{b}{a}}",
        text: "-sqrt(b/a)",
        equation_text: "x = -sqrt(b/a)",
        conditions_latex: [],
        branch: "",
        verification: null,
      },
    ],
    conditions_latex: ["a \\neq 0"],
    excluded_latex: [],
    exceptional: ["a = 0"],
    verification: {
      symbolic_zero: false,
      numeric_points_checked: 12,
      numeric_points_agreed: 12,
      passed: true,
      note: "",
    },
    complete: true,
    completeness_note:
      "a random value of the unknown was returned by some solution in 8 of 8 probes",
    assumption_note: "",
    reason: null,
    diagnostics: diagnostics({ verification_method: "back_substitution" }),
    message: "2 solution(s), each substituted back into the equation and checked for completeness.",
    duration_ms: 9,
    ...over,
  };
}

export function subterm(index: number, latex: string, over: Partial<Subterm> = {}): Subterm {
  return {
    index,
    path: [index - 1],
    args: null,
    latex,
    text: latex,
    highlight_latex: `\\boxed{${latex}} + z`,
    occurrence: 1,
    occurrences: 1,
    blocked: false,
    ...over,
  };
}
