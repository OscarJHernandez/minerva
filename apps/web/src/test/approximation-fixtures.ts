import type {
  ApproxCandidate,
  ApproxComplexity,
  ApproximationReport,
} from "@/lib/approximation-types";

const NONE: ApproxComplexity = {
  degree: null,
  numerator_degree: null,
  denominator_degree: null,
  modes: null,
  segments: null,
  coefficients: null,
};

const BASE: ApproximationReport = {
  status: "ok",
  reason: null,
  method: "pade",
  message: "The [1/1] Padé approximant about the point.",
  input_latex: "\\mathrm{e}^{x}",
  variable: "x",
  domain_latex: "x \\to 0",
  result_latex: "\\frac{1 + \\frac{x}{2}}{1 - \\frac{x}{2}}",
  result_text: "(1 + x/2)/(1 - x/2)",
  remainder_latex: "O\\left(x^{3}\\right)",
  statement_latex: null,
  orders: [],
  extras: [{ label: "pole of the approximant", latex: "x = 2.0" }],
  evidence: "series_checked",
  checks: [
    {
      name: "f - P/Q falls like h^(m+n+1)",
      passed: true,
      points: 2,
      detail: "the residual falls like h^3",
    },
    { name: "coefficients against mpmath.pade", passed: null, points: 0, detail: "not run" },
  ],
  complexity: { ...NONE, numerator_degree: 1, denominator_degree: 1, coefficients: 3 },
  error: null,
  measured_errors: [],
  error_curve: [],
  tolerance_met: null,
  warnings: ["spurious_pole"],
  candidates: [],
  suggestions: [],
  questions: [],
  notes: ["The approximant has a pole at x = 2.0, although the original function does not."],
  assumptions_stated: [],
  duration_ms: 12,
};

export const pade: ApproximationReport = BASE;

const candidate = (over: Partial<ApproxCandidate>): ApproxCandidate => ({
  method: "chebyshev",
  label: "Chebyshev, degree 4",
  status: "ok",
  reason: null,
  result_latex: null,
  complexity: { ...NONE, degree: 4, coefficients: 5 },
  error: {
    samples: 2001,
    max_abs: "8.7e-4",
    rms: "3e-4",
    max_rel: null,
    relative_note: "",
    at: "0",
  },
  tolerance_met: true,
  warnings: [],
  error_curve: [
    { x: 0, error: 1e-3 },
    { x: 1, error: 5e-4 },
    { x: 2, error: 8e-4 },
  ],
  chosen: false,
  ...over,
});

export const comparison: ApproximationReport = {
  ...BASE,
  method: "compare",
  result_latex: "1.2\\,T_{0}(t) + 0.5\\,T_{1}(t)",
  remainder_latex: null,
  extras: [],
  warnings: [],
  notes: [],
  tolerance_met: true,
  error: {
    samples: 2001,
    max_abs: "8.7e-4",
    rms: "3e-4",
    max_rel: "1e-3",
    relative_note: "",
    at: "0",
  },
  candidates: [
    candidate({ label: "Taylor, degree 4", method: "taylor", tolerance_met: false, chosen: false }),
    candidate({ label: "Chebyshev, degree 4", chosen: true }),
    candidate({
      label: "Padé [2/2]",
      method: "pade",
      warnings: ["spurious_pole"],
      tolerance_met: false,
    }),
  ],
  error_curve: [
    { x: 0, error: 1e-3 },
    { x: 1, error: 5e-4 },
  ],
};

export const needsInterval: ApproximationReport = {
  ...BASE,
  status: "conditions_not_satisfied",
  reason: "domain_unspecified",
  method: "chebyshev",
  message: "The interval is yours to choose.",
  result_latex: null,
  result_text: null,
  remainder_latex: null,
  extras: [],
  checks: [],
  warnings: [],
  notes: [],
  questions: ["On which interval should the approximation hold?"],
};

export const notVerified: ApproximationReport = {
  ...needsInterval,
  status: "not_verified",
  reason: "verification_failed",
  message:
    "A Padé approximant was computed but its contact with the function could not be confirmed.",
  questions: [],
};
