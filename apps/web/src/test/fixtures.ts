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
