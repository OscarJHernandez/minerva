import type { VectorRequest, VectorResult } from "@/lib/types";

export const divergenceRequest: VectorRequest = {
  expression: "\\nabla\\cdot\\mathbf{A}",
  coordinates: "cylindrical",
  fields: [{ name: "A", kind: "vector" }],
  constants: [],
  assumptions: {},
  action: "evaluate",
};

export const curl: VectorResult = {
  status: "ok",
  message: "numerically checked against an independent Cartesian finite-difference evaluation",
  action: "evaluate",
  coordinates: "cylindrical",
  to: null,
  result_kind: "vector",
  input_latex: "\\nabla \\times \\mathbf{A}",
  result_latex:
    "\\left(\\frac{\\frac{\\partial}{\\partial \\phi} A_{z}}{\\rho}\\right)\\,\\hat{\\rho}",
  result_text: "(…)",
  components_latex: ["\\frac{\\frac{\\partial}{\\partial \\phi} A_{z}}{\\rho}", "0", "0"],
  components_text: ["", "0", "0"],
  basis_latex: ["\\hat{\\rho}", "\\hat{\\phi}", "\\hat{z}"],
  scale_factors_latex: [],
  determinant_latex: null,
  verification: {
    method: "cartesian_finite_difference",
    numeric_passed: true,
    points_checked: 6,
    points_agreed: 6,
    points_skipped: 0,
    max_residual: 1e-20,
    symbolic_zero: null,
    checks: [{ name: "cartesian_finite_difference", passed: true, note: "agreed at 6 points" }],
    note: "",
  },
  conventions: ["Cylindrical coordinates (rho, phi, z), rho > 0."],
  assumptions_used: ["every field is differentiable as often as the expression differentiates it"],
  restrictions_latex: ["\\rho \\neq 0"],
  restrictions_complete: true,
  blocked: [],
  undeclared: [],
  steps: [],
  applicable: [],
  evidence: null,
  given: [],
  search_complete: null,
  duration_ms: 120,
};

export const blocked: VectorResult = {
  ...curl,
  status: "conditions_not_satisfied",
  message: "not shown: the result needs mixed partial derivatives of f to commute",
  result_kind: null,
  result_latex: null,
  result_text: null,
  components_latex: [],
  components_text: [],
  basis_latex: [],
  verification: null,
  restrictions_latex: [],
  blocked: [
    {
      needs: "c2",
      fields: ["f"],
      note: "the result contains mixed partial derivatives in both orders",
    },
  ],
};

export const jacobian: VectorResult = {
  ...curl,
  action: "jacobian",
  coordinates: "spherical",
  result_kind: "matrix",
  result_latex: "\\left[\\begin{matrix}1\\end{matrix}\\right]",
  components_latex: [],
  basis_latex: [],
  determinant_latex: "r^{2} \\sin{\\left(\\theta \\right)}",
  scale_factors_latex: ["1", "r", "r \\sin{\\left(\\theta \\right)}"],
};

export const derived: VectorResult = {
  ...curl,
  action: "derive",
  coordinates: "cartesian",
  result_kind: "vector",
  input_latex: "\\nabla \\times (\\nabla \\times \\mathbf{E})",
  result_latex: "-\\nabla \\times \\frac{\\partial \\mathbf{B}}{\\partial t}",
  components_latex: [],
  basis_latex: [],
  restrictions_latex: [],
  evidence: "conditional",
  given: ["curl(E) = -dt(B)"],
  steps: [
    {
      action: "derive",
      title: "Substitute premise 1",
      before_latex: "\\nabla \\times (\\nabla \\times \\mathbf{E})",
      after_latex: "\\nabla \\times (-\\frac{\\partial \\mathbf{B}}{\\partial t})",
      rule_id: null,
      reverse: false,
      path: [],
      part_latex: "\\nabla \\times \\mathbf{E}",
      evidence: "conditional",
      given: ["curl(E) = -dt(B)"],
      needs: [],
      verification: null,
      symbolic_zero: null,
    },
    {
      action: "derive",
      title: "vector.curl_curl",
      before_latex: "\\nabla \\times (\\nabla \\times \\mathbf{E})",
      after_latex: "\\nabla (\\nabla \\cdot \\mathbf{E}) - \\nabla^{2} \\mathbf{E}",
      rule_id: "vector.curl_curl",
      reverse: false,
      path: [],
      part_latex: "\\nabla \\times (\\nabla \\times \\mathbf{E})",
      evidence: "identity",
      given: [],
      needs: ["c2"],
      verification: curl.verification,
      symbolic_zero: true,
    },
  ],
};

export const listed: VectorResult = {
  ...curl,
  action: "identities",
  result_latex: null,
  components_latex: [],
  basis_latex: [],
  restrictions_latex: [],
  applicable: [
    {
      rule_id: "vector.curl_curl",
      reverse: false,
      path: [],
      part_latex: "\\nabla \\times (\\nabla \\times \\mathbf{A})",
      result_latex: "\\nabla (\\nabla \\cdot \\mathbf{A}) - \\nabla^{2} \\mathbf{A}",
      needs: ["c2"],
      blocked: true,
      note: "vector.curl_curl holds for twice continuously differentiable fields; the smoothness of these fields was not declared",
    },
  ],
};

export const waveEquation: VectorResult = {
  ...derived,
  derivation: "wave_equation_e",
  result_latex:
    "\\nabla^{2} \\mathbf{E} = \\epsilon_{0} \\mu_{0} \\frac{\\partial^{2} \\mathbf{E}}{\\partial t^{2}}",
  equations_latex: [
    "\\nabla^{2} \\mathbf{E} = \\epsilon_{0} \\mu_{0} \\frac{\\partial^{2} \\mathbf{E}}{\\partial t^{2}}",
  ],
  given: ["div(E) = 0", "curl(E) = -dt(B)", "curl(B) = mu0*epsilon0*dt(E)"],
  model_check: {
    status: "passed",
    kind: "vacuum_waves",
    models: 2,
    points: 8,
    max_residual: 1e-20,
    note: "consistent with 2 exact solutions of the premises at 8 points; this is not a verification",
  },
};

export const needsPremises: VectorResult = {
  ...blocked,
  derivation: "wave_equation_e",
  needs_premises: ["curl(B) = mu0*epsilon0*dt(E)"],
  blocked: [],
};

export const fluxResult: VectorResult = {
  ...curl,
  action: "integrate",
  coordinates: "spherical",
  result_kind: "scalar",
  result_latex: "\\iint_S \\left(\\mathbf{F}\\right) \\cdot d\\mathbf{S} = 4 \\pi",
  components_latex: [],
  basis_latex: [],
  restrictions_latex: [],
  integral: {
    kind: "flux",
    values: [
      {
        label: "\\iint_S \\left(\\mathbf{F}\\right) \\cdot d\\mathbf{S}",
        latex: "4 \\pi",
        text: "4*pi",
        numeric: "12.5663706143592",
        exact: true,
        quadrature_error: 1e-15,
        agreed: true,
        points: 81,
      },
    ],
    theorem: null,
    holds: null,
    hypotheses: [],
    orientation: "normal d_u r x d_v r (the order of the parameters is the orientation)",
    method: "symbolic",
  },
};

export const divergenceTheorem: VectorResult = {
  ...fluxResult,
  integral: {
    ...fluxResult.integral!,
    kind: "divergence_theorem",
    theorem: "the divergence theorem",
    holds: true,
    values: [
      fluxResult.integral!.values[0]!,
      { ...fluxResult.integral!.values[0]!, label: "\\oint_{\\partial V} F \\cdot d\\mathbf{S}" },
    ],
    hypotheses: [
      {
        text: "F is continuously differentiable on the closed region",
        status: "checked",
        note: "finite at every point of a grid; smoothness is not proved",
      },
      {
        text: "the region is a coordinate box",
        status: "established",
        note: "the faces and outward normals were generated",
      },
    ],
  },
};

export const approximation: VectorResult = {
  ...fluxResult,
  integral: {
    ...fluxResult.integral!,
    method: "numerical",
    values: [{ ...fluxResult.integral!.values[0]!, latex: null, text: null, exact: false }],
  },
};
