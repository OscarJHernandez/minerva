import type { Assumptions } from "./types";
import type {
  NumericMethod,
  OdeCondition,
  OdeInput,
  OdeInterpretation,
  OdeNumericInput,
} from "./ode-types";

/** The text the user types for a differential equation; nothing here computes or guesses. */
export interface OdeForm {
  /** One equation per line, `left = right`. */
  equations: string;
  format: "latex" | "text";
  independent: string;
  /** One per line or separated by semicolons: `y(0) = 1`. Optional for the analysis. */
  conditions: string;
  assumptions: Assumptions;
}

export const BLANK_ODE_FORM: OdeForm = {
  equations: "",
  format: "latex",
  independent: "",
  conditions: "",
  assumptions: {},
};

export interface OdeExample {
  id: string;
  title: string;
  form: OdeForm;
}

export const ODE_EXAMPLES: OdeExample[] = [
  {
    id: "growth",
    title: "First order, separable: y′ = x y",
    form: { ...BLANK_ODE_FORM, equations: "y' = x y", independent: "x", conditions: "y(0) = 2" },
  },
  {
    id: "linear2",
    title: "Second order, linear: y″ + 3y′ + 2y = 0",
    form: {
      ...BLANK_ODE_FORM,
      equations: "y'' + 3 y' + 2 y = 0",
      independent: "t",
      conditions: "y(0) = 1\ny'(0) = 0",
    },
  },
  {
    id: "duffing",
    title: "Nonlinear damped oscillator",
    form: { ...BLANK_ODE_FORM, equations: "y'' + 0.2 y' + y + y^3 = 0", independent: "t" },
  },
  {
    id: "robertson",
    title: "A stiff system (Robertson)",
    form: {
      ...BLANK_ODE_FORM,
      equations:
        "\\dot{a} = -0.04 a + 10^4 b c\n\\dot{b} = 0.04 a - 10^4 b c - 3 \\cdot 10^7 b^2\n\\dot{c} = 3 \\cdot 10^7 b^2",
      independent: "t",
      conditions: "a(0) = 1\nb(0) = 0\nc(0) = 0",
    },
  },
  {
    id: "lotka",
    title: "A coupled system (predator and prey)",
    form: {
      ...BLANK_ODE_FORM,
      equations: "\\dot{x} = a x - b x y\n\\dot{y} = -c y + d x y",
      independent: "t",
      conditions: "x(0) = 2\ny(0) = 1",
    },
  },
  {
    id: "riccati",
    title: "No closed form: y′ = y² + x",
    form: { ...BLANK_ODE_FORM, equations: "y' = y^2 + x", independent: "x" },
  },
];

function lines(text: string): string[] {
  return text
    .split(/\n|;/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export type Built<T> = { ok: true; request: T } | { ok: false; error: string };

/** The request for the analysis and the closed-form solver. */
export function buildOdeInput(form: OdeForm): Built<OdeInput> {
  const equations = form.equations
    .split("\n")
    .map((e) => e.trim())
    .filter((e) => e.length > 0);
  if (equations.length === 0) return { ok: false, error: "Write an equation, like y' = x y." };
  if (equations.length > 8) return { ok: false, error: "At most 8 equations." };
  if (equations.some((e) => !e.includes("="))) {
    return { ok: false, error: "Each equation needs an equals sign: left = right." };
  }
  const conditions = lines(form.conditions);
  if (conditions.some((c) => !c.includes("="))) {
    return { ok: false, error: "Write each condition like y(0) = 1." };
  }
  const independent = form.independent.trim();
  return {
    ok: true,
    request: {
      equations,
      format: form.format,
      unknowns: [],
      independent: independent === "" ? null : independent,
      assumptions: form.assumptions,
      conditions,
    },
  };
}

/** What the numerical form asks the user for, from how the equation was read. */
export interface StateField {
  unknown: string;
  order: number;
  /** For the label, e.g. `y` or `y'`. */
  key: string;
}

export function stateFields(meaning: OdeInterpretation): StateField[] {
  const out: StateField[] = [];
  for (const unknown of meaning.unknowns) {
    const count = meaning.orders[unknown] ?? 0;
    for (let k = 0; k < count; k++)
      out.push({ unknown, order: k, key: `${unknown}${"'".repeat(k)}` });
  }
  return out;
}

export interface NumericForm {
  /** Where the initial values are given. */
  start: string;
  /** One entry per state, keyed `unknown:order`. */
  initial: Record<string, string>;
  parameters: Record<string, string>;
  tEnd: string;
  method: NumericMethod;
  rtol: string;
  atol: string;
}

export const stateKey = (f: StateField): string => `${f.unknown}:${f.order}`;

export function blankNumericForm(meaning: OdeInterpretation): NumericForm {
  return {
    start: "",
    initial: Object.fromEntries(stateFields(meaning).map((f) => [stateKey(f), ""])),
    parameters: Object.fromEntries(meaning.parameters.map((p) => [p, ""])),
    tEnd: "",
    method: "auto",
    rtol: "1e-8",
    atol: "1e-10",
  };
}

/** A decimal the user typed; fractions and expressions are not accepted here on purpose. */
export function parseNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "" || !/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(trimmed)) return null;
  const value = Number(trimmed);
  return Number.isFinite(value) ? value : null;
}

/**
 * The numerical request: the equations as they were sent for the analysis, with the initial values,
 * parameter values and interval the user typed. Nothing is defaulted that changes the problem.
 */
export function buildNumericRequest(
  base: OdeInput,
  meaning: OdeInterpretation,
  form: NumericForm,
): Built<OdeNumericInput> {
  const start = parseNumber(form.start);
  if (start === null)
    return { ok: false, error: "Give the point where the initial values hold, as a number." };
  const conditions: OdeCondition[] = [];
  for (const field of stateFields(meaning)) {
    const value = parseNumber(form.initial[stateKey(field)] ?? "");
    if (value === null) {
      return { ok: false, error: `Give the initial value of ${field.key} as a number.` };
    }
    conditions.push({
      unknown: field.unknown,
      order: field.order,
      at: String(start),
      value: String(value),
    });
  }
  const parameters: Record<string, number> = {};
  for (const name of meaning.parameters) {
    const value = parseNumber(form.parameters[name] ?? "");
    if (value === null) return { ok: false, error: `Give a number for the parameter ${name}.` };
    parameters[name] = value;
  }
  const end = parseNumber(form.tEnd);
  if (end === null) return { ok: false, error: "Give the end of the interval as a number." };
  if (end === start)
    return { ok: false, error: "The interval is empty: the end equals the start." };
  const rtol = parseNumber(form.rtol);
  const atol = parseNumber(form.atol);
  if (rtol === null || rtol < 1e-12 || rtol > 1e-3) {
    return { ok: false, error: "The relative tolerance must be between 1e-12 and 1e-3." };
  }
  if (atol === null || atol < 1e-14 || atol > 1e-3) {
    return { ok: false, error: "The absolute tolerance must be between 1e-14 and 1e-3." };
  }
  return {
    ok: true,
    request: {
      ...base,
      conditions,
      parameters,
      t_end: end,
      method: form.method,
      rtol,
      atol,
      samples: 200,
    },
  };
}

/** The conditions for a closed-form request: every initial value left empty is simply not given. */
export function buildSymbolicRequest(
  base: OdeInput,
  meaning: OdeInterpretation,
  start: string,
  initial: Record<string, string>,
): Built<OdeInput> {
  const filled = stateFields(meaning).filter((f) => (initial[stateKey(f)] ?? "").trim() !== "");
  if (filled.length === 0) return { ok: true, request: { ...base, conditions: [] } };
  const point = parseNumber(start);
  if (point === null)
    return { ok: false, error: "Give the point where the initial values hold, as a number." };
  const conditions: OdeCondition[] = [];
  for (const field of filled) {
    const value = parseNumber(initial[stateKey(field)] ?? "");
    if (value === null) return { ok: false, error: `${field.key} must be a number.` };
    conditions.push({
      unknown: field.unknown,
      order: field.order,
      at: String(point),
      value: String(value),
    });
  }
  return { ok: true, request: { ...base, conditions } };
}

const SIMPLE_CONDITION =
  /^\s*([A-Za-z]\w*)('*)\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*$/;

/**
 * Initial values the user already typed with the equation (`y(0) = 1`, `y'(0) = 0`), so that the
 * numerical form does not ask for them twice. Only plain numbers are taken over; anything else is
 * left blank for the user to fill in.
 */
export function prefillFromConditions(
  conditions: (OdeCondition | string)[],
  meaning: OdeInterpretation,
): Pick<NumericForm, "start" | "initial"> {
  const initial = blankNumericForm(meaning).initial;
  let start = "";
  for (const condition of conditions) {
    if (typeof condition === "string") {
      const m = SIMPLE_CONDITION.exec(condition);
      if (!m) continue;
      const key = `${m[1]}:${(m[2] ?? "").length}`;
      if (key in initial) {
        initial[key] = m[4] ?? "";
        start = start || (m[3] ?? "");
      }
    } else {
      const key = `${condition.unknown}:${condition.order}`;
      if (key in initial && parseNumber(condition.value) !== null) {
        initial[key] = condition.value;
        start = start || condition.at;
      }
    }
  }
  return { start, initial };
}

/** The text of a condition as the user would write it (`y'(0) = 1`). */
export function conditionText(condition: OdeCondition | string): string {
  if (typeof condition === "string") return condition;
  return `${condition.unknown}${"'".repeat(condition.order)}(${condition.at}) = ${condition.value}`;
}

/** A saved request, back into the form, so that a saved problem can be changed and run again. */
export function restoreOdeForm(request: OdeInput): OdeForm {
  return {
    equations: request.equations.join("\n"),
    format: request.format,
    independent: request.independent ?? "",
    conditions: request.conditions.map(conditionText).join("\n"),
    assumptions: request.assumptions,
  };
}
