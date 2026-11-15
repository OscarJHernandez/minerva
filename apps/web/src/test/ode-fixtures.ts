import type {
  MethodSuggestion,
  OdeAnalysis,
  OdeQualitative,
  OdeInput,
  OdeInterpretation,
  OdeNumericReport,
  OdeSolution,
  OdeSolutionItem,
} from "@/lib/ode-types";

export const diagnostics = { trace_id: null, stages: [], failure_reason: null };

export const meaning: OdeInterpretation = {
  unknowns: ["y"],
  independent: "t",
  parameters: ["c"],
  orders: { y: 2 },
  equations_latex: ["y'' + c y' + y = 0"],
  conditions_latex: [],
  notes: ["t is the independent variable, read as real unless you declared otherwise"],
};

export const request: OdeInput = {
  equations: ["y'' + c y' + y = 0"],
  format: "latex",
  unknowns: [],
  independent: "t",
  assumptions: {},
  conditions: [],
};

function method(
  over: Partial<MethodSuggestion> & Pick<MethodSuggestion, "id" | "title">,
): MethodSuggestion {
  return {
    applicable: true,
    available: true,
    rank: null,
    reason: "",
    requires: [],
    limitations: [],
    tool: null,
    action: null,
    ...over,
  };
}

export const analysis: OdeAnalysis = {
  status: "ok",
  message: "",
  interpretation: meaning,
  structure: {
    kind: "ode",
    n_unknowns: 1,
    n_equations: 1,
    orders: { y: 2 },
    order: 2,
    state_dimension: 2,
    linear: true,
    homogeneous: true,
    constant_coefficients: true,
    autonomous: true,
    solved_for_highest: true,
    position_only: false,
  },
  problem_kind: "general",
  conditions_needed: ["y(t_0)", "y'(t_0)"],
  classes: ["second_order_linear", "linear_constant_coefficients"],
  solved_latex: ["y'' = - c y' - y"],
  restrictions_latex: [],
  methods: [
    method({
      id: "symbolic_closed_form",
      title: "Closed-form solution (SymPy), verified by substitution",
      rank: 1,
      reason: "recognised as second-order linear",
      limitations: ["SymPy may fail"],
      tool: "ode_solve_symbolic",
      action: "closed_form",
    }),
    method({
      id: "numerical_ivp",
      title: "Numerical integration (initial-value problem)",
      rank: 2,
      reason: "the highest derivatives can be solved for",
      requires: ["initial values", "the interval"],
      limitations: ["an approximation on the interval you choose"],
      tool: "ode_solve_numerical",
      action: "ivp",
    }),
    method({
      id: "poincare_lindstedt",
      title: "Poincaré–Lindstedt",
      applicable: false,
      reason: "the acceleration depends on the velocity (damping or a gyroscopic term)",
    }),
    method({
      id: "multiple_scales",
      title: "Multiple scales",
      applicable: false,
      available: false,
      reason: "not implemented in Minerva",
    }),
  ],
  qualitative: null,
  notes: ["Classes marked by SymPy's classification are candidates."],
  diagnostics,
  duration_ms: 12,
};

const verification = {
  points_checked: 10,
  points_agreed: 10,
  points_skipped: 0,
  max_residual: 4e-62,
  symbolic_zero: true,
  independent_constants: 2,
  passed: true,
  note: "",
};

export const solutionItem: OdeSolutionItem = {
  kind: "explicit",
  expansion: null,
  equations_latex: ["y\\left(t\\right) = C_{1} e^{-t} + C_{2} e^{-2t}"],
  rhs_text: ["C1*exp(-t) + C2*exp(-2*t)"],
  constants: ["C1", "C2"],
  family: "general",
  evaluated: true,
  verification,
  conditions: [],
  restrictions_latex: [],
};

export const solution: OdeSolution = {
  status: "ok",
  message: "",
  method: "closed_form",
  interpretation: meaning,
  solutions: [solutionItem],
  verification,
  dropped: 0,
  notes: ["Singular solutions are not searched."],
  diagnostics,
  duration_ms: 30,
};

const t = [0, 1, 2, 3, 4];

export const numeric: OdeNumericReport = {
  status: "ok",
  message: "",
  interpretation: meaning,
  method_requested: "auto",
  method_used: "DOP853",
  method_reason: "DOP853 completed and showed no sign of stiffness",
  t_start: 0,
  t_end: 4,
  parameters: { c: 0.5 },
  initial: [
    { label_latex: "y", value: 1 },
    { label_latex: "y'", value: 0 },
  ],
  rtol: 1e-8,
  atol: 1e-10,
  steps: 38,
  evaluations: 572,
  t,
  series: [
    { label_latex: "y", unknown: "y", order: 0, values: [1, 0.7, 0.1, -0.3, -0.4] },
    { label_latex: "y'", unknown: "y", order: 1, values: [0, -0.8, -1, -0.5, 0.1] },
  ],
  verification: {
    checks: [
      { name: "initial_conditions", status: "passed", value: 0, tolerance: 1e-12, note: "" },
      { name: "independent_method", status: "passed", value: 2e-8, tolerance: 1e-5, note: "" },
      { name: "tolerance_refinement", status: "passed", value: 2e-8, tolerance: 1e-7, note: "" },
      { name: "defect", status: "passed", value: 0.2, tolerance: 1000, note: "" },
      { name: "first_integral", status: "skipped", value: null, tolerance: null, note: "none" },
    ],
    passed: true,
    independent_method: "Minerva's Dormand-Prince 5(4)",
    refinement_errors: [2e-8, 2e-9],
    first_integral_latex: null,
  },
  comparison: null,
  warnings: [],
  notes: ["A numerical approximation on the interval given."],
  diagnostics,
  duration_ms: 90,
};

export const qualitative: OdeQualitative = {
  status: "ok",
  message: "",
  states_latex: ["y", "y'"],
  field_latex: ["y' = y'", "y'' = - y^{3} - y - \\frac{y'}{5}"],
  equilibria: [
    {
      point_latex: ["y = 0", "y' = 0"],
      point_numeric: [0, 0],
      jacobian_latex: "\\left[\\begin{matrix}0 & 1\\\\-1 & - \\frac{1}{5}\\end{matrix}\\right]",
      eigenvalues_latex: [
        "- \\frac{1}{10} - \\frac{3 \\sqrt{11} i}{10}",
        "- \\frac{1}{10} + \\frac{3 \\sqrt{11} i}{10}",
      ],
      eigenvalues_numeric: ["-0.1 - 0.99498744i", "-0.1 + 0.99498744i"],
      characteristic_latex: "\\lambda^{2} + \\frac{\\lambda}{5} + 1",
      classification: "stable_spiral",
      classification_text: "linearly asymptotically stable (a sink: a spiral)",
      decided: true,
    },
  ],
  complete: true,
  jacobian_checked: true,
  direction_field: {
    window: [-1, 1, -1, 1],
    x: [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    y: [-1, -1, -1, 0, 0, 0, 1, 1, 1],
    dx: [-1, -1, -1, 0, 0, 0, 1, 1, 1],
    dy: [2.2, 1.2, 0.2, 1, 0, -1, -0.2, -1.2, -2.2],
    speed: [2.4, 1.6, 1, 1, 0, 1, 1, 1.6, 2.4],
  },
  notes: [],
  local_note:
    "Local and from the linearisation only: it describes orbits that start close to the point.",
};
