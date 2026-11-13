import type {
  HistoryItem,
  DifferentiateResult,
  IdentitiesResult,
  IdentityMatch,
  SeriesResult,
  IntegrateResult,
  MatrixCalculusResult,
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

export const okMatrixIntegral: MatrixCalculusResult = {
  status: "ok",
  action: "integrate",
  input_latex: "\\exp\\left(w A\\right) \\exp\\left(- w B\\right)",
  structure:
    "A single term in the matrices A, B, with 2 matrix exponentials, commutation is not assumed for AB.",
  features: ["matrix:integrate", "matrix:exp", "matrix:noncommuting"],
  method: "sylvester_reduction",
  method_title: "Sylvester equation for the integral",
  result_kind: "explicit",
  result_latex:
    "\\left[\\begin{matrix}-1 + e & -1\\\\0 & \\frac{e^{2}}{2} - \\frac{1}{2}\\end{matrix}\\right]",
  result_entries: [
    ["-1 + E", "-1"],
    ["0", "-1/2 + exp(2)/2"],
  ],
  equation_latex: "A X - X B = \\exp\\left(A\\right) \\exp\\left(-B\\right) - I",
  evaluated: true,
  assumptions_used: ["the Sylvester operator is invertible", "all matrices are constant in w"],
  steps: [
    {
      method: "sylvester_reduction",
      title: "Integrate F' = aPF + cFQ over the interval",
      before_latex: null,
      after_latex: "A X - X B = \\exp\\left(A\\right) \\exp\\left(-B\\right) - I",
      note: "a Sylvester equation for X = the integral of F",
    },
  ],
  blocked: [],
  verification: {
    method: "adaptive quadrature (mpmath) of the integrand against the solved result",
    points_checked: 4,
    points_agreed: 4,
    passed: true,
    symbolic_zero: true,
    symbolic_method: "the explicit result substituted into the Sylvester equation",
    note: "",
  },
  notes: [],
  message: "Evaluated and verified.",
  duration_ms: 300,
  diagnostics: {
    trace_id: "req0123-card0123",
    method: "sylvester_reduction",
    methods_tried: [
      { id: "integrate_polynomial", outcome: "no_match" },
      { id: "sylvester_reduction", outcome: "selected" },
    ],
    dimensions: [[2, 2]],
    matrices: 2,
    explicit: 2,
    time_dependent: 0,
    commutation: "not_used",
    invertibility: "from_entries",
    stages: [
      { name: "parse", ms: 3 },
      { name: "transform", ms: 120 },
      { name: "verify_numeric", ms: 170 },
      { name: "verify_exact", ms: 2 },
      { name: "total", ms: 300 },
    ],
    digits: 16,
    tolerance: 1e-9,
    max_residual: 2.5e-14,
    points_checked: 4,
    points_agreed: 4,
    verification: "quadrature+residual",
    fallbacks: [],
    failure_reason: null,
  },
};

export const matrixRepresentation: MatrixCalculusResult = {
  ...okMatrixIntegral,
  method: "block_exponential",
  method_title: "Block-matrix exponential representation",
  result_kind: "representation",
  result_latex:
    "\\left[\\exp\\left(\\begin{pmatrix} -A & I \\\\ 0 & 0 \\end{pmatrix}\\right)\\right]_{12}",
  result_entries: null,
  evaluated: false,
  notes: ["X is an exact representation, but not an evaluated entrywise result"],
  message: "Written exactly through a block-matrix exponential and verified.",
};

export const matrixFrechet: MatrixCalculusResult = {
  ...okMatrixIntegral,
  action: "differentiate",
  input_latex: "\\exp\\left(t^{2} A + t B\\right)",
  method: "exp_frechet_derivative",
  method_title: "Frechet derivative of the exponential",
  result_kind: "representation",
  result_latex:
    "\\int_0^1 \\exp\\left((1-\\sigma) X\\right) \\left(2 t A + B\\right) \\exp\\left(\\sigma X\\right) d\\sigma",
  result_entries: null,
  equation_latex: null,
  evaluated: false,
  assumptions_used: ["A, B are constant in t"],
  blocked: [
    {
      method: "exp_derivative",
      title: "Use d/dt exp(X) = X' exp(X)",
      needs: "X X' = X' X for the exponent X (not established)",
      latex: null,
    },
  ],
  notes: ["The integral is an exact representation, not an evaluated matrix."],
  message: "Written exactly and verified; it is a representation, not an evaluated result.",
};

export const matrixNumericEvolution: MatrixCalculusResult = {
  ...okMatrixIntegral,
  action: "evolve",
  method: "evolution_numeric",
  method_title: "Numerical solution of dU/dt = A(t) U",
  result_kind: "numeric",
  result_latex: "\\begin{pmatrix} 0.68 & 0.47 \\\\ -0.92 & 0.84 \\end{pmatrix}",
  result_entries: [
    ["0.68", "0.47"],
    ["-0.92", "0.84"],
  ],
  equation_latex: null,
  evaluated: false,
  notes: ["numerical approximation on [0, 1], double precision"],
  message: "Computed numerically on the stated span and verified by independent checks.",
  verification: {
    ...okMatrixIntegral.verification!,
    symbolic_zero: null,
    points_checked: 8,
    points_agreed: 8,
  },
};

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
