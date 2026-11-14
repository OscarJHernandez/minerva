import type {
  StatAction,
  StatKind,
  StatMechRequest,
  StatMethodName,
  StatPartial,
  StatQuantityName,
  StatStatistic,
  StatSystem,
} from "./statmech-types";
import type { Assumptions } from "./types";

/** The workbench form for a statistical-mechanics problem; the user chooses every convention here. */
export interface StatForm {
  kind: StatKind;
  levels: { energy: string; degeneracy: string }[];
  family: { index: string; energy: string; degeneracy: string; lower: string; upper: string };
  dos: { variable: string; density: string; lower: string; upper: string };
  classical: {
    hamiltonian: string;
    variables: { name: string; lower: string; upper: string }[];
    cell: string;
  };
  /** Rows on separate lines, entries separated by `;` (a matrix in text syntax). */
  quantum: { hamiltonian: string; observableName: string; observable: string };
  /** A periodic chain by the transfer matrix: site values separated by `,`, bond(s, t), site(s). */
  chain: { states: string; bond: string; site: string; length: string };
  /** The approximation to apply (its order is `order`). */
  method: StatMethodName;
  /** Particles in single-particle levels (the `levels` rows are reused); grand canonical. */
  occupation: { statistics: "bose" | "fermi" | "boltzmann"; mu: string };
  /** Count the microstates of fixed total energy (needs the copies); microcanonical. */
  microcanonical: { on: boolean; energy: string };
  copies: { on: boolean; count: string; identical: "yes" | "no" | "" };
  temperature: "beta" | "T";
  parameters: string;
  assumptions: Assumptions;
  quantities: StatQuantityName[];
  partials: {
    of: StatQuantityName;
    variable: string;
    held: string;
    scale: string;
    label: string;
  }[];
  observable: string;
  statistic: StatStatistic;
  order: number;
  values: string;
  sweep: {
    on: boolean;
    variable: string;
    start: string;
    stop: string;
    points: string;
    scale: "linear" | "log";
  };
}

export const BASE_QUANTITIES: { id: StatQuantityName; label: string }[] = [
  { id: "Z", label: "Partition function Z" },
  { id: "lnZ", label: "ln Z" },
  { id: "F", label: "Free energy F" },
  { id: "U", label: "Internal energy U" },
  { id: "S", label: "Entropy S" },
  { id: "Var_E", label: "Energy variance" },
];

export const GRAND_QUANTITIES: { id: StatQuantityName; label: string }[] = [
  { id: "Xi", label: "Grand partition function Ξ" },
  { id: "lnXi", label: "ln Ξ" },
  { id: "Omega", label: "Grand potential Ω" },
  { id: "N", label: "Mean particle number N" },
  { id: "U", label: "Internal energy U" },
  { id: "S", label: "Entropy S" },
  { id: "Var_N", label: "Number variance" },
];

export function quantityOptions(kind: StatKind): { id: StatQuantityName; label: string }[] {
  return kind === "occupation" ? GRAND_QUANTITIES : BASE_QUANTITIES;
}

export const STATISTICS: { id: StatStatistic; label: string }[] = [
  { id: "mean", label: "mean" },
  { id: "variance", label: "variance" },
  { id: "cumulant", label: "cumulant" },
  { id: "moment", label: "raw moment" },
  { id: "central_moment", label: "central moment" },
  { id: "covariance", label: "covariance" },
];

export function blankForm(): StatForm {
  return {
    kind: "levels",
    levels: [
      { energy: "0", degeneracy: "1" },
      { energy: "epsilon", degeneracy: "1" },
    ],
    family: { index: "n", energy: "", degeneracy: "1", lower: "0", upper: "oo" },
    dos: { variable: "En", density: "", lower: "0", upper: "oo" },
    classical: {
      hamiltonian: "",
      variables: [{ name: "p", lower: "-oo", upper: "oo" }],
      cell: "",
    },
    quantum: { hamiltonian: "-h; -d\n-d; h", observableName: "sz", observable: "1; 0\n0; -1" },
    chain: { states: "1, -1", bond: "-J*s*t", site: "-h*s", length: "N" },
    method: "high_temperature",
    occupation: { statistics: "fermi", mu: "mu" },
    microcanonical: { on: false, energy: "" },
    copies: { on: false, count: "N", identical: "" },
    temperature: "beta",
    parameters: "",
    assumptions: { epsilon: ["positive"] },
    quantities: ["Z", "F", "U", "S"],
    partials: [],
    observable: "E",
    statistic: "mean",
    order: 1,
    values: "",
    sweep: { on: false, variable: "beta", start: "1/10", stop: "5", points: "11", scale: "log" },
  };
}

export const EXAMPLES: { id: string; label: string; form: StatForm }[] = [
  { id: "two-level", label: "Two-level system", form: blankForm() },
  {
    id: "qho",
    label: "Quantum harmonic oscillator",
    form: {
      ...blankForm(),
      kind: "level_family",
      family: {
        index: "n",
        energy: "hbar*omega*(n + 1/2)",
        degeneracy: "1",
        lower: "0",
        upper: "oo",
      },
      assumptions: { hbar: ["positive"], omega: ["positive"] },
      quantities: ["Z", "U", "Var_E"],
      partials: [{ of: "U", variable: "T", held: "", scale: "1", label: "C" }],
    },
  },
  {
    id: "spin-tilted-field",
    label: "Spin-1/2 in a tilted field (operators that do not commute)",
    form: {
      ...blankForm(),
      kind: "quantum",
      assumptions: { h: ["positive"], d: ["positive"] },
      quantities: ["Z", "U", "S"],
    },
  },
  {
    id: "fermi-levels",
    label: "Fermions in three levels (grand canonical)",
    form: {
      ...blankForm(),
      kind: "occupation",
      levels: [
        { energy: "0", degeneracy: "1" },
        { energy: "eps", degeneracy: "1" },
        { energy: "2*eps", degeneracy: "2" },
      ],
      assumptions: { eps: ["positive"] },
      quantities: ["lnXi", "Omega", "N", "U", "S"],
    },
  },
  {
    id: "two-level-counting",
    label: "Counting microstates: N two-level units at fixed energy",
    form: {
      ...blankForm(),
      levels: [
        { energy: "0", degeneracy: "1" },
        { energy: "1", degeneracy: "1" },
      ],
      assumptions: {},
      copies: { on: true, count: "10", identical: "no" },
      microcanonical: { on: true, energy: "4" },
    },
  },
  {
    id: "ising-chain",
    label: "Ising ring in a field (transfer matrix)",
    form: {
      ...blankForm(),
      kind: "chain",
      assumptions: { J: ["real"], h: ["real"], N: ["integer", "positive"] },
      quantities: ["Z", "U"],
      method: "thermodynamic_limit",
    },
  },
  {
    id: "classical-oscillator",
    label: "Classical harmonic oscillator",
    form: {
      ...blankForm(),
      kind: "classical",
      classical: {
        hamiltonian: "p**2/(2*m) + m*omega**2*x**2/2",
        variables: [
          { name: "x", lower: "-oo", upper: "oo" },
          { name: "p", lower: "-oo", upper: "oo" },
        ],
        cell: "h",
      },
      assumptions: { m: ["positive"], omega: ["positive"], h: ["positive"] },
      quantities: ["Z", "U", "Var_E"],
    },
  },
  {
    id: "ideal-gas",
    label: "Classical ideal gas (N identical particles)",
    form: {
      ...blankForm(),
      kind: "classical",
      classical: {
        hamiltonian: "(px**2 + py**2 + pz**2)/(2*m)",
        variables: [
          { name: "x", lower: "0", upper: "V**(1/3)" },
          { name: "y", lower: "0", upper: "V**(1/3)" },
          { name: "z", lower: "0", upper: "V**(1/3)" },
          { name: "px", lower: "-oo", upper: "oo" },
          { name: "py", lower: "-oo", upper: "oo" },
          { name: "pz", lower: "-oo", upper: "oo" },
        ],
        cell: "h**3",
      },
      copies: { on: true, count: "N", identical: "yes" },
      parameters: "V",
      assumptions: {
        m: ["positive"],
        h: ["positive"],
        V: ["positive"],
        N: ["integer", "positive"],
      },
      quantities: ["U", "S"],
      partials: [{ of: "F", variable: "V", held: "T", scale: "-1", label: "p" }],
    },
  },
];

export type BuildResult = { ok: true; request: StatMechRequest } | { ok: false; error: string };

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,15}$/;

/** `a; b` per line into rows of entries. */
export function parseMatrix(text: string): string[][] | string {
  const rows = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(";").map((cell) => cell.trim()));
  if (rows.length === 0) return "Give the matrix, one row per line, entries separated by ;.";
  if (rows.some((r) => r.length !== rows.length || r.some((c) => c === ""))) {
    return "A matrix must be square with every entry filled in (rows on lines, entries separated by ;).";
  }
  if (rows.length > 4) return "At most 4 x 4.";
  return rows;
}

export function splitNames(text: string): string[] {
  return text
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/** `name = number` lines, one per symbol: the numbers are the user's. */
export function parseValues(text: string): Record<string, string> | string {
  const out: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const m = /^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$/.exec(line);
    if (!m) return `Write each value as name = number, not “${line}”.`;
    out[m[1]!] = m[2]!.trim();
  }
  return out;
}

function system(form: StatForm): StatSystem | string {
  switch (form.kind) {
    case "levels": {
      const rows = form.levels.filter((l) => l.energy.trim() !== "");
      if (rows.length === 0) return "Give at least one energy level.";
      return {
        kind: "levels",
        levels: rows.map((l) => ({
          energy: l.energy.trim(),
          degeneracy: l.degeneracy.trim() || "1",
        })),
      };
    }
    case "level_family": {
      const f = form.family;
      if (!f.energy.trim()) return "Give the energy E(n) of the level with index n.";
      return {
        kind: "level_family",
        index: f.index.trim() || "n",
        energy: f.energy.trim(),
        degeneracy: f.degeneracy.trim() || "1",
        lower: f.lower.trim() || "0",
        upper: f.upper.trim() || "oo",
      };
    }
    case "density_of_states": {
      const d = form.dos;
      if (!d.density.trim()) return "Give the density of states.";
      return {
        kind: "density_of_states",
        variable: d.variable.trim() || "En",
        density: d.density.trim(),
        lower: d.lower.trim() || "0",
        upper: d.upper.trim() || "oo",
      };
    }
    case "chain": {
      const states = splitNames(form.chain.states.replace(/,/g, " "));
      if (states.length < 2 || states.length > 3) return "Give two or three site values.";
      if (!form.chain.bond.trim()) return "Give the bond energy as an expression in s and t.";
      if (!form.chain.length.trim())
        return "Give the length N (a symbol or a number of at most 14).";
      return {
        kind: "chain",
        states,
        bond: form.chain.bond.trim(),
        site: form.chain.site.trim() || "0",
        length: form.chain.length.trim(),
      };
    }
    case "occupation": {
      const rows = form.levels.filter((l) => l.energy.trim() !== "");
      if (rows.length === 0) return "Give at least one single-particle level.";
      const mu = form.occupation.mu.trim() || "mu";
      if (!NAME.test(mu)) return "Invalid name for the chemical potential.";
      return {
        kind: "occupation",
        statistics: form.occupation.statistics,
        levels: rows.map((l) => ({
          energy: l.energy.trim(),
          degeneracy: l.degeneracy.trim() || "1",
        })),
        mu,
      };
    }
    case "quantum": {
      const h = parseMatrix(form.quantum.hamiltonian);
      if (typeof h === "string") return h;
      const name = form.quantum.observableName.trim();
      if (name === "" && form.quantum.observable.trim() === "") {
        return { kind: "quantum", hamiltonian: h };
      }
      const a = parseMatrix(form.quantum.observable);
      if (typeof a === "string") return a;
      if (!NAME.test(name) || name === "E")
        return "Name the operator (not E, which is the energy).";
      if (a.length !== h.length) return "The operator must have the size of H.";
      return { kind: "quantum", hamiltonian: h, observables: { [name]: a } };
    }
    case "classical": {
      const c = form.classical;
      if (!c.hamiltonian.trim()) return "Give the Hamiltonian.";
      const variables = c.variables.filter((v) => v.name.trim() !== "");
      if (variables.length === 0) return "Name the phase-space variables.";
      if (variables.some((v) => !NAME.test(v.name.trim()))) return "Invalid variable name.";
      return {
        kind: "classical",
        hamiltonian: c.hamiltonian.trim(),
        variables: variables.map((v) => ({
          name: v.name.trim(),
          domain_lower: v.lower.trim() || "-oo",
          domain_upper: v.upper.trim() || "oo",
        })),
        phase_space_cell: c.cell.trim() || null,
      };
    }
  }
}

function partials(form: StatForm): StatPartial[] | string {
  const out: StatPartial[] = [];
  for (const p of form.partials) {
    if (!p.variable.trim()) return "Name the variable of each derivative.";
    out.push({
      of: p.of,
      variable: p.variable.trim(),
      held_fixed: splitNames(p.held),
      scale: p.scale.trim() || "1",
      label: p.label.trim(),
    });
  }
  return out;
}

export function buildRequest(form: StatForm, action: StatAction): BuildResult {
  const sys = system(form);
  if (typeof sys === "string") return { ok: false, error: sys };
  const parts = partials(form);
  if (typeof parts === "string") return { ok: false, error: parts };
  const parameters = splitNames(form.parameters);
  if (parameters.some((p) => !NAME.test(p))) return { ok: false, error: "Invalid parameter name." };
  const micro = form.kind === "levels" && form.microcanonical.on;
  if (micro && !form.copies.on) {
    return {
      ok: false,
      error: "Counting microstates needs the number of copies (N units share the energy).",
    };
  }
  if (micro && form.microcanonical.energy.trim() === "") {
    return { ok: false, error: "Give the fixed total energy." };
  }
  const request: StatMechRequest = {
    ensemble:
      form.kind === "occupation" ? "grand_canonical" : micro ? "microcanonical" : "canonical",
    ...(micro ? { total_energy: form.microcanonical.energy.trim() } : {}),
    system: sys,
    format: "text",
    temperature: form.temperature,
    parameters,
    assumptions: form.assumptions,
    action,
    copies: form.copies.on
      ? {
          count: form.copies.count.trim(),
          identical: form.copies.identical === "" ? null : form.copies.identical === "yes",
        }
      : null,
  };
  if (action === "derive" || action === "evaluate") {
    request.quantities = form.quantities;
    request.partials = action === "derive" ? parts : [];
  }
  if (action === "derive" && form.quantities.length === 0 && parts.length === 0) {
    return { ok: false, error: "Choose at least one quantity or derivative." };
  }
  if (action === "expectation") {
    request.observable = form.observable.trim() || "E";
    request.statistic = form.statistic;
    request.order = form.order;
  }
  if (action === "approximate") {
    request.method = form.method;
    request.order = form.order;
  }
  if (action === "evaluate") {
    const values = parseValues(form.values);
    if (typeof values === "string") return { ok: false, error: values };
    request.values = values;
    if (form.sweep.on) {
      const points = Number(form.sweep.points);
      if (!Number.isInteger(points) || points < 2 || points > 60) {
        return { ok: false, error: "A sweep has 2 to 60 points." };
      }
      request.sweep = {
        variable: form.sweep.variable.trim(),
        start: form.sweep.start.trim(),
        stop: form.sweep.stop.trim(),
        points,
        scale: form.sweep.scale,
      };
    }
  }
  return { ok: true, request };
}

/** Loads a saved problem back into the form so that it can be changed and run again. */
export function restoreForm(request: StatMechRequest): StatForm {
  const form = blankForm();
  const sys = request.system;
  form.kind = sys.kind;
  if (request.ensemble === "microcanonical") {
    form.microcanonical = { on: true, energy: request.total_energy ?? "" };
  }
  if (request.method) form.method = request.method;
  if (sys.kind === "chain") {
    form.chain = {
      states: sys.states.join(", "),
      bond: sys.bond,
      site: sys.site ?? "0",
      length: sys.length,
    };
  } else if (sys.kind === "occupation") {
    form.levels = sys.levels.map((l) => ({ energy: l.energy, degeneracy: l.degeneracy }));
    form.occupation = { statistics: sys.statistics, mu: sys.mu };
  } else if (sys.kind === "levels") {
    form.levels = sys.levels.map((l) => ({ energy: l.energy, degeneracy: l.degeneracy }));
  } else if (sys.kind === "level_family") {
    form.family = {
      index: sys.index,
      energy: sys.energy,
      degeneracy: sys.degeneracy,
      lower: sys.lower,
      upper: sys.upper,
    };
  } else if (sys.kind === "density_of_states") {
    form.dos = {
      variable: sys.variable,
      density: sys.density,
      lower: sys.lower,
      upper: sys.upper,
    };
  } else if (sys.kind === "quantum") {
    const [name, matrix] = Object.entries(sys.observables ?? {})[0] ?? ["", [] as string[][]];
    form.quantum = {
      hamiltonian: sys.hamiltonian.map((r) => r.join("; ")).join("\n"),
      observableName: name,
      observable: matrix.map((r) => r.join("; ")).join("\n"),
    };
  } else {
    form.classical = {
      hamiltonian: sys.hamiltonian,
      variables: sys.variables.map((v) => ({
        name: v.name,
        lower: v.domain_lower,
        upper: v.domain_upper,
      })),
      cell: sys.phase_space_cell ?? "",
    };
  }
  form.copies = request.copies
    ? {
        on: true,
        count: request.copies.count,
        identical: request.copies.identical === null ? "" : request.copies.identical ? "yes" : "no",
      }
    : { on: false, count: "N", identical: "" };
  form.temperature = request.temperature ?? "beta";
  form.parameters = (request.parameters ?? []).join(", ");
  form.assumptions = (request.assumptions ?? {}) as Assumptions;
  form.quantities = request.quantities ?? [];
  form.partials = (request.partials ?? []).map((p) => ({
    of: p.of,
    variable: p.variable,
    held: p.held_fixed.join(", "),
    scale: p.scale ?? "1",
    label: p.label ?? "",
  }));
  form.observable = request.observable ?? "E";
  form.statistic = request.statistic ?? "mean";
  form.order = request.order ?? 1;
  form.values = Object.entries(request.values ?? {})
    .map(([k, v]) => `${k} = ${v}`)
    .join("\n");
  if (request.sweep) {
    form.sweep = {
      on: true,
      variable: request.sweep.variable,
      start: request.sweep.start,
      stop: request.sweep.stop,
      points: String(request.sweep.points),
      scale: request.sweep.scale,
    };
  }
  return form;
}
