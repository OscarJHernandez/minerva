import type { StatMechRequest, StatMechResult, StatToolArgs } from "@/lib/statmech-types";

const base: StatMechResult = {
  status: "ok",
  action: "derive",
  reason: null,
  state_variables: ["beta"],
  steps: [],
  quantities: [],
  conditions: [],
  methods: [],
  not_supported: [],
  questions: [],
  table: null,
  verification: {
    method: "enumeration",
    points_checked: 8,
    points_agreed: 8,
    symbolic_zero: null,
    passed: true,
    note: "numerical agreement is evidence, not a proof",
  },
  notes: [],
  message: "Verified against an independent numerical evaluation.",
  duration_ms: 12,
};

export const twoLevelRequest: StatMechRequest = {
  system: {
    kind: "levels",
    levels: [
      { energy: "0", degeneracy: "1" },
      { energy: "epsilon", degeneracy: "1" },
    ],
  },
  assumptions: { epsilon: ["positive"] },
  action: "derive",
  quantities: ["Z", "U"],
};

export const twoLevel: StatMechResult = {
  ...base,
  steps: [
    {
      name: "levels",
      title: "Energy levels",
      latex: "E_1 = 0",
      text: "",
      result_kind: "exact",
      conditions: [],
    },
    {
      name: "partition_function",
      title: "Partition function",
      latex: "Z = 1 + e^{-\\beta \\epsilon}",
      text: "1 + exp(-beta*epsilon)",
      result_kind: "exact",
      conditions: [{ latex: "\\beta > 0", established: true, how: "positive by definition" }],
    },
  ],
  quantities: [
    {
      name: "U",
      title: "Internal energy",
      latex: "\\frac{\\epsilon}{e^{\\beta \\epsilon} + 1}",
      text: "epsilon/(exp(beta*epsilon) + 1)",
      result_kind: "exact",
      held_fixed: [],
      variable: null,
      order: null,
      route: "U = -d ln Z / d beta",
      checks: [
        { name: "enumeration", subject: "U", passed: true, points: 8, agreed: 8, detail: "" },
        { name: "finite_difference", subject: "U", passed: true, points: 8, agreed: 8, detail: "" },
        { name: "symbolic_zero", subject: "U", passed: null, points: 0, agreed: 0, detail: "" },
      ],
    },
  ],
};

export const needsCell: StatMechResult = {
  ...base,
  status: "conditions_not_satisfied",
  reason: "convention_unconfirmed",
  message: "The phase-space normalisation is a convention only you can choose.",
  questions: ["What is the phase-space cell, the divisor of the phase-space measure?"],
  verification: { ...base.verification, passed: false, points_checked: 0, points_agreed: 0 },
};

export const refused: StatMechResult = {
  ...base,
  status: "conditions_not_satisfied",
  reason: "divergent",
  message: "the series diverges",
  verification: { ...base.verification, passed: false, points_checked: 0, points_agreed: 0 },
};

export const sweep: StatMechResult = {
  ...twoLevel,
  action: "evaluate",
  table: {
    variable: "beta",
    columns: ["beta", "U"],
    rows: [
      ["0.5", "0.4"],
      ["1", "0.3"],
      ["2", "0.1"],
    ],
  },
};

export const analysis: StatMechResult = {
  ...base,
  action: "analyze",
  methods: [
    {
      id: "closed_form_sum",
      title: "Closed-form partition function",
      applies: true,
      implemented: true,
      why: "finite sum",
    },
    {
      id: "high_temperature_expansion",
      title: "High-temperature expansion",
      applies: true,
      implemented: false,
      why: "beta -> 0",
    },
  ],
  not_supported: ["Monte Carlo sampling and many-body solvers"],
};

export const classicalArgs: StatToolArgs = {
  action: "derive",
  system: {
    kind: "classical",
    hamiltonian: "p**2/2",
    variables: [{ name: "p", domain_lower: "-oo", domain_upper: "oo" }],
  },
  quantities: ["U"],
};

export const approximated: StatMechResult = {
  ...twoLevel,
  action: "approximate",
  quantities: [
    {
      name: "lnZ_series",
      title: "ln Z, order 4 in beta",
      latex: "\\ln 2 - \\frac{\\beta \\epsilon}{2}",
      text: "log(2) - beta*epsilon/2",
      result_kind: "asymptotic",
      held_fixed: [],
      variable: null,
      order: null,
      route: "Taylor series about beta = 0",
      checks: [
        { name: "series_check", subject: "lnZ", passed: true, points: 4, agreed: 4, detail: "" },
        { name: "residual_order", subject: "lnZ", passed: true, points: 2, agreed: 2, detail: "" },
      ],
    },
  ],
  approximation: {
    method: "high_temperature",
    assumptions: ["beta is small", "ln Z is analytic at beta = 0"],
    parameter: "beta",
    order: "truncation error O(beta^5) for ln Z",
    validity: "inside the radius of convergence of the series",
    error_table: [
      {
        label: "beta = 0.3",
        exact: "0.437488",
        approximate: "0.437472",
        error: "-1.57699e-5",
        note: "ln Z",
      },
      {
        label: "beta = 0.15",
        exact: "0.554355",
        approximate: "0.554355",
        error: "-2.51409e-7",
        note: "ln Z",
      },
    ],
  },
};
