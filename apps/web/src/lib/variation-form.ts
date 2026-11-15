import type { Assumptions } from "./types";
import type {
  EndpointCondition,
  GivenFunction,
  TrialPath,
  VariationAction,
  VariationEnd,
  VariationRequest,
} from "./variation-types";

export const MAX_FUNCTIONS = 4;
export const MAX_GIVEN = 8;
export const MAX_CONDITIONS = 64;
export const MAX_ORDER = 7;
const NAME_RE = /^[A-Za-z\\][A-Za-z0-9_\\{}]{0,31}$/;
const GIVEN_RE = /^([A-Za-z][A-Za-z0-9_]{0,31})\(([^()]*)\)$/;

/** Actions that need numbers only the user can choose. */
export const NUMERIC_ACTIONS: VariationAction[] = ["evaluate", "verify", "solve"];

/** One endpoint condition as typed. Nothing is filled in for the user. */
export interface ConditionForm {
  function: string;
  order: string;
  end: VariationEnd;
  kind: "fixed" | "free";
  value: string;
}

/** The editable state of the variational form: text exactly as the user typed it. */
export interface VariationForm {
  integrand: string;
  format: "latex" | "text";
  variable: string;
  /** Unknown functions of the variable, separated by commas. */
  functions: string;
  /** Given functions such as `V(x), W(x, t)`, separated by commas outside the brackets. */
  givenFunctions: string;
  lower: string;
  upper: string;
  lowerTerm: string;
  upperTerm: string;
  conditions: ConditionForm[];
  assumptions: Assumptions;
  /** Numbers (numerical actions only). */
  values: Record<string, string>;
  definitions: Record<string, string>;
  /** One trial path (or starting guess) expression per function. */
  paths: Record<string, string>;
  sweepName: string;
  sweepValues: string;
  withSecond: boolean;
}

export function blankCondition(): ConditionForm {
  return { function: "", order: "0", end: "lower", kind: "fixed", value: "" };
}

export function blankForm(): VariationForm {
  return {
    integrand: "",
    format: "latex",
    variable: "t",
    functions: "x",
    givenFunctions: "",
    lower: "",
    upper: "",
    lowerTerm: "",
    upperTerm: "",
    conditions: [],
    assumptions: {},
    values: {},
    definitions: {},
    paths: {},
    sweepName: "",
    sweepValues: "",
    withSecond: false,
  };
}

function splitTop(text: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let current = "";
  for (const ch of text) {
    if (ch === "(") depth += 1;
    if (ch === ")") depth -= 1;
    if ((ch === "," || ch === ";") && depth === 0) {
      out.push(current);
      current = "";
    } else current += ch;
  }
  out.push(current);
  return out.map((s) => s.trim()).filter(Boolean);
}

export function functionNames(form: Pick<VariationForm, "functions">): string[] {
  return splitTop(form.functions);
}

/** `V(x), W(x, t)` to declarations, or an error message. */
export function parseGiven(text: string): GivenFunction[] | string {
  const out: GivenFunction[] = [];
  for (const part of splitTop(text)) {
    const m = GIVEN_RE.exec(part);
    if (!m) return `“${part}” is not a given function; write it like V(x).`;
    const args = m[2]!
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    if (args.length === 0) return `Name the arguments of ${m[1]}, for example ${m[1]}(x).`;
    out.push({ name: m[1]!, args });
  }
  return out;
}

export type BuiltVariation = { ok: true; request: VariationRequest } | { ok: false; error: string };

function emptyToNull(text: string): string | null {
  const t = text.trim();
  return t === "" ? null : t;
}

function buildConditions(
  form: VariationForm,
  functions: string[],
  needValues: boolean,
): EndpointCondition[] | string {
  const seen = new Set<string>();
  const out: EndpointCondition[] = [];
  for (const c of form.conditions) {
    const fn = c.function.trim();
    if (!functions.includes(fn))
      return `A condition names ${fn || "no function"}, which is not one of the unknown functions.`;
    const orderText = c.order.trim();
    if (!/^\d+$/.test(orderText))
      return `The derivative order of a condition on ${fn} must be a whole number 0 to ${MAX_ORDER}.`;
    const order = Number(orderText);
    if (order > MAX_ORDER)
      return `The derivative order of a condition on ${fn} must be at most ${MAX_ORDER}.`;
    const key = `${fn}|${order}|${c.end}`;
    if (seen.has(key))
      return `${fn} has two conditions for derivative ${order} at the ${c.end} end.`;
    seen.add(key);
    const value = c.value.trim();
    if (c.kind === "fixed" && needValues && value === "") {
      return `Give the value of derivative ${order} of ${fn} at the ${c.end} end: only you can choose it.`;
    }
    out.push({
      function: fn,
      order,
      end: c.end,
      kind: c.kind,
      ...(c.kind === "fixed" && value !== "" ? { value } : {}),
    });
  }
  if (out.length > MAX_CONDITIONS) return `At most ${MAX_CONDITIONS} conditions.`;
  return out;
}

/**
 * Validates the form and builds the request. The server validates again; this only saves a round trip.
 * It never supplies an interval, a condition, a parameter value or a path the user did not type.
 */
export function buildVariationRequest(
  form: VariationForm,
  action: VariationAction,
): BuiltVariation {
  if (form.integrand.trim() === "")
    return { ok: false, error: "Enter the integrand L of the action." };
  if (!NAME_RE.test(form.variable.trim()))
    return { ok: false, error: "Name the independent variable, for example t." };
  const functions = functionNames(form);
  if (functions.length === 0)
    return { ok: false, error: "Name the unknown function(s), for example x." };
  if (functions.length > MAX_FUNCTIONS)
    return { ok: false, error: `At most ${MAX_FUNCTIONS} unknown functions.` };
  for (const f of functions) {
    if (!NAME_RE.test(f)) return { ok: false, error: `“${f}” is not a valid function name.` };
  }
  if (new Set(functions).size !== functions.length)
    return { ok: false, error: "The unknown functions must have different names." };
  const given = parseGiven(form.givenFunctions);
  if (typeof given === "string") return { ok: false, error: given };
  if (given.length > MAX_GIVEN)
    return { ok: false, error: `At most ${MAX_GIVEN} given functions.` };
  for (const g of given) {
    if (functions.includes(g.name))
      return { ok: false, error: `${g.name} is both an unknown and a given function.` };
  }
  // The second variation is symbolic unless the user gave a path to examine it along.
  const hasPath = Object.values(form.paths).some((e) => e.trim() !== "");
  const numeric = NUMERIC_ACTIONS.includes(action) || (action === "second" && hasPath);
  const conditions = buildConditions(form, functions, action === "evaluate" || action === "solve");
  if (typeof conditions === "string") return { ok: false, error: conditions };
  const assumptions: Assumptions = {};
  for (const [name, items] of Object.entries(form.assumptions)) {
    if (items.length > 0) assumptions[name] = items;
  }

  const request: VariationRequest = {
    action,
    integrand: form.integrand.trim(),
    format: form.format,
    variable: form.variable.trim(),
    functions,
    given_functions: given,
    lower: emptyToNull(form.lower),
    upper: emptyToNull(form.upper),
    lower_term: emptyToNull(form.lowerTerm),
    upper_term: emptyToNull(form.upperTerm),
    conditions,
    assumptions,
    values: {},
    definitions: {},
    paths: [],
    sweep: null,
    perturbations: [],
    automatic_perturbations: true,
    with_second: false,
  };
  if (!numeric) return { ok: true, request };

  if (request.lower === null || request.upper === null) {
    return {
      ok: false,
      error: "Give both ends of the interval: only you can choose the numbers for this step.",
    };
  }
  if (action === "solve" && conditions.length === 0) {
    return {
      ok: false,
      error:
        "Declare the endpoint conditions first (fixed with their values, or free): a stationary path needs them.",
    };
  }
  for (const [name, v] of Object.entries(form.values)) {
    if (v.trim() !== "") request.values[name] = v.trim();
  }
  for (const g of given) {
    const d = (form.definitions[g.name] ?? "").trim();
    if (d !== "") request.definitions[g.name] = d;
    else {
      return {
        ok: false,
        error: `Define the given function ${g.name}(${g.args.join(", ")}) by an expression: only you can choose it.`,
      };
    }
  }
  const analytic: Record<string, string> = {};
  for (const f of functions) {
    const e = (form.paths[f] ?? "").trim();
    if (e !== "") analytic[f] = e;
  }
  const given_count = Object.keys(analytic).length;
  if (given_count > 0 && given_count < functions.length) {
    return { ok: false, error: "Give a path for every unknown function, or for none." };
  }
  if (given_count === 0 && action !== "solve") {
    return {
      ok: false,
      error: "Give a trial path for each unknown function: the model never chooses one for you.",
    };
  }
  if (given_count > 0) request.paths = [{ analytic, sampled: {} } satisfies TrialPath];
  const name = form.sweepName.trim();
  if (action === "evaluate" && name !== "") {
    const items = form.sweepValues
      .split(/[,;]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (items.length < 2 || items.length > 6)
      return { ok: false, error: "A trial family needs 2 to 6 values for its parameter." };
    request.sweep = { name, values: items };
  }
  request.with_second = action === "solve" && form.withSecond;
  return { ok: true, request };
}

/** Rebuilds the form from a saved request (what `result.request` holds), so a problem can be reopened. */
export function restoreVariationForm(request: Partial<VariationRequest>): VariationForm {
  const blank = blankForm();
  const analytic = request.paths?.[0]?.analytic ?? {};
  return {
    ...blank,
    integrand: request.integrand ?? "",
    format: request.format ?? "latex",
    variable: request.variable ?? "t",
    functions: (request.functions ?? ["x"]).join(", "),
    givenFunctions: (request.given_functions ?? [])
      .map((g) => `${g.name}(${g.args.join(", ")})`)
      .join(", "),
    lower: request.lower ?? "",
    upper: request.upper ?? "",
    lowerTerm: request.lower_term ?? "",
    upperTerm: request.upper_term ?? "",
    conditions: (request.conditions ?? []).map((c) => ({
      function: c.function,
      order: String(c.order),
      end: c.end,
      kind: c.kind,
      value: c.value ?? "",
    })),
    assumptions: request.assumptions ?? {},
    values: { ...(request.values ?? {}) },
    definitions: { ...(request.definitions ?? {}) },
    paths: { ...analytic },
    sweepName: request.sweep?.name ?? "",
    sweepValues: (request.sweep?.values ?? []).join(", "),
    withSecond: request.with_second ?? false,
  };
}

/**
 * The form of a saved derivation: the functional comes from the first step that recorded a request,
 * and the numbers (parameters, definitions, paths) from the latest step that had any.
 */
export function formFromSteps(steps: { result: unknown }[]): VariationForm {
  const requests = steps
    .map((s) => (s.result as { request?: Partial<VariationRequest> } | null)?.request)
    .filter((r): r is Partial<VariationRequest> => !!r && typeof r.integrand === "string");
  const first = requests[0];
  if (!first) return blankForm();
  const form = restoreVariationForm(first);
  const withNumbers = [...requests]
    .reverse()
    .find(
      (r) =>
        Object.keys(r.values ?? {}).length > 0 ||
        Object.keys(r.definitions ?? {}).length > 0 ||
        (r.paths ?? []).length > 0,
    );
  if (!withNumbers) return form;
  const numbers = restoreVariationForm({ ...first, ...withNumbers });
  return { ...form, ...pickNumbers(numbers), lower: numbers.lower, upper: numbers.upper };
}

function pickNumbers(f: VariationForm) {
  return {
    values: f.values,
    definitions: f.definitions,
    paths: f.paths,
    sweepName: f.sweepName,
    sweepValues: f.sweepValues,
    withSecond: f.withSecond,
    conditions: f.conditions,
  };
}

export interface VariationExample {
  id: string;
  label: string;
  form: VariationForm;
}

const fixed = (end: VariationEnd, value: string, order = "0", fn = "x"): ConditionForm => ({
  function: fn,
  order,
  end,
  kind: "fixed",
  value,
});
const free = (end: VariationEnd, order = "0", fn = "x"): ConditionForm => ({
  function: fn,
  order,
  end,
  kind: "free",
  value: "",
});

/** Starting points the user chooses. They are ordinary input: nothing is assumed beyond what they state. */
export const EXAMPLES: VariationExample[] = [
  {
    id: "free-particle",
    label: "Free particle",
    form: { ...blankForm(), integrand: "\\frac{m}{2}\\dot{x}^2", lower: "a", upper: "b" },
  },
  {
    id: "harmonic-oscillator",
    label: "Harmonic oscillator, fixed ends",
    form: {
      ...blankForm(),
      integrand: "\\frac12\\dot{x}^2-\\frac12 x^2",
      lower: "0",
      upper: "1",
      conditions: [fixed("lower", "0"), fixed("upper", "1")],
    },
  },
  {
    id: "higher-derivative",
    label: "Higher derivative with a given potential V(x)",
    form: {
      ...blankForm(),
      integrand: "\\frac{m}{2}\\dot{x}^2+\\frac{\\alpha}{2}\\ddot{x}^2-V(x)",
      givenFunctions: "V(x)",
      lower: "a",
      upper: "b",
    },
  },
  {
    id: "cantilever",
    label: "Cantilever (clamped at 0, free at 1)",
    form: {
      ...blankForm(),
      integrand: "\\frac12\\ddot{x}^2-qx",
      lower: "0",
      upper: "1",
      conditions: [
        fixed("lower", "0"),
        fixed("lower", "0", "1"),
        free("upper"),
        free("upper", "1"),
      ],
    },
  },
  {
    id: "endpoint-term",
    label: "Endpoint term at the upper end",
    form: {
      ...blankForm(),
      integrand: "\\frac12\\dot{x}^2",
      lower: "0",
      upper: "1",
      upperTerm: "\\frac{k}{2}x^2",
      conditions: [fixed("lower", "0"), free("upper")],
    },
  },
];
