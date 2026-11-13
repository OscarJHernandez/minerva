import type { GeometryQuantityResult, GeometryRequest, GeometryResult } from "@/lib/types";

const numeric = (name: string) => ({
  name,
  kind: "numeric" as const,
  passed: true,
  points_checked: 8,
  points_agreed: 8,
  note: "agreed at every sampled point",
});
const symbolic = (name: string, passed: boolean | null) => ({
  name,
  kind: "symbolic" as const,
  passed,
  points_checked: 0,
  points_agreed: 0,
  note: passed === null ? "not run" : "difference simplifies to 0",
});

export const ricciScalar: GeometryQuantityResult = {
  kind: "ricci_scalar",
  status: "ok",
  title: "Ricci scalar R",
  variance: [],
  components: [],
  all_zero: false,
  scalar_latex: "\\frac{2}{a^{2}}",
  scalar_text: "2/a**2",
  symmetry_note: "",
  verification: {
    method: "finite differences of the lambdified metric",
    numeric_passed: true,
    symbolic_zero: true,
    checks: [numeric("agrees with the numerical Ricci scalar"), symbolic("contraction", true)],
    note: "",
  },
  notes: [],
  message: "",
};

export const christoffel: GeometryQuantityResult = {
  kind: "christoffel",
  status: "ok",
  title: "Christoffel symbols",
  variance: ["up", "down", "down"],
  components: [
    {
      indices: [0, 1, 1],
      label_latex: "\\Gamma^{\\theta}{}_{\\phi \\phi}",
      latex: "- \\sin{\\left(\\theta \\right)} \\cos{\\left(\\theta \\right)}",
      text: "-sin(theta)*cos(theta)",
    },
  ],
  all_zero: false,
  scalar_latex: null,
  scalar_text: null,
  symmetry_note: "Symmetric in the lower indices.",
  verification: {
    method: "finite differences of the lambdified metric",
    numeric_passed: true,
    symbolic_zero: null,
    checks: [numeric("metric compatibility"), symbolic("compat", null)],
    note: "",
  },
  notes: [],
  message: "",
};

export const sphere: GeometryResult = {
  status: "ok",
  dimension: 2,
  coordinates_latex: ["\\theta", "\\phi"],
  metric_latex:
    "\\left[\\begin{matrix}a^{2} & 0\\\\0 & a^{2} \\sin^{2}{\\theta}\\end{matrix}\\right]",
  signature: null,
  signature_note: "The sign of g is not decided by the declared assumptions.",
  conventions: ["Riemann tensor: R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ..."],
  assumptions_used: ["coordinates are real: theta, phi", "a: positive"],
  restrictions_latex: ["\\sin{\\left(\\theta \\right)} \\neq 0"],
  restrictions_complete: true,
  quantities: [ricciScalar, christoffel],
  diagnostics: [
    { stage: "christoffel", duration_ms: 11, ops: 6, components: 2, status: "ok", failure: null },
  ],
  peak_memory_mb: 120.5,
  message: "2 of 2 requested quantities computed and numerically verified.",
  duration_ms: 90,
};

export const sphereRequest: GeometryRequest = {
  coordinates: ["theta", "phi"],
  metric: [
    ["a^2", "0"],
    ["0", "a^2 \\sin^2\\theta"],
  ],
  format: "latex",
  assumptions: { a: ["positive"] },
  functions: [],
  quantities: ["ricci_scalar", "christoffel"],
  tensor: null,
  tensor_index: 0,
};
