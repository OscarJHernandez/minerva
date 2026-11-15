import type {
  ApproximationMethod,
  ApproximationNumbers,
  ApproximationToolArgs,
} from "./approximation-types";

/** Every field is text, as typed. Nothing here has a default: the numbers are the user's. */
export interface ApproximationForm {
  method: ApproximationMethod;
  point: string;
  direction: "" | "+" | "-";
  order: string;
  small: string;
  lower: string;
  upper: string;
  degree: string;
  denominatorDegree: string;
  period: string;
  modes: string;
  segments: string;
  kind: string;
  tolerance: string;
  toleranceKind: "" | "absolute" | "relative";
  maxTerms: string;
  variables: string;
  points: string;
}

export function blankForm(method: ApproximationMethod = "auto"): ApproximationForm {
  return {
    method,
    point: "",
    direction: "",
    order: "",
    small: "",
    lower: "",
    upper: "",
    degree: "",
    denominatorDegree: "",
    period: "",
    modes: "",
    segments: "",
    kind: "",
    tolerance: "",
    toleranceKind: "",
    maxTerms: "",
    variables: "",
    points: "",
  };
}

const text = (v: unknown): string =>
  typeof v === "string" || typeof v === "number" ? String(v) : "";

/** The form starts with exactly what the model's call (which the user's words produced) contained. */
export function restoreApproximationForm(args: Partial<ApproximationNumbers>): ApproximationForm {
  return {
    method: args.method ?? "auto",
    point: text(args.point),
    direction: args.direction ?? "",
    order: text(args.order),
    small: text(args.small),
    lower: text(args.lower),
    upper: text(args.upper),
    degree: text(args.degree),
    denominatorDegree: text(args.denominator_degree),
    period: text(args.period),
    modes: text(args.modes),
    segments: text(args.segments),
    kind: text(args.kind),
    tolerance: text(args.tolerance),
    toleranceKind: args.tolerance_kind ?? "",
    maxTerms: text(args.max_terms),
    variables: (args.variables ?? []).join(", "),
    points: (args.points ?? []).join(", "),
  };
}

type Built = { ok: true; numbers: ApproximationNumbers } | { ok: false; error: string };

function integer(label: string, raw: string, max: number, min = 0): number | string | undefined {
  const s = raw.trim();
  if (s === "") return undefined;
  if (!/^\d+$/.test(s)) return `${label} must be a whole number.`;
  const n = Number(s);
  if (n < min || n > max) return `${label} must be between ${min} and ${max}.`;
  return n;
}

const SHORT = 64;
const list = (raw: string): string[] =>
  raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "");

/** Turn the typed form into the user-owned fields of a request; refuse what the server would. */
export function buildApproximationNumbers(form: ApproximationForm): Built {
  const out: ApproximationNumbers = { method: form.method };
  const texts: [keyof ApproximationNumbers, string, string][] = [
    ["point", form.point, "The point"],
    ["small", form.small, "The small parameter"],
    ["lower", form.lower, "The lower end"],
    ["upper", form.upper, "The upper end"],
    ["period", form.period, "The period"],
    ["kind", form.kind, "The kind"],
  ];
  for (const [key, raw, label] of texts) {
    const v = raw.trim();
    if (v === "") continue;
    if (v.length > SHORT) return { ok: false, error: `${label} is too long.` };
    (out as Record<string, unknown>)[key] = v;
  }
  if ((out.lower === undefined) !== (out.upper === undefined)) {
    return { ok: false, error: "Give both ends of the interval, or neither." };
  }
  const ints: [keyof ApproximationNumbers, string, string, number, number?][] = [
    ["order", form.order, "The order", 64],
    ["degree", form.degree, "The degree", 64],
    ["denominator_degree", form.denominatorDegree, "The denominator degree", 32],
    ["modes", form.modes, "The number of modes", 128],
    ["segments", form.segments, "The number of segments", 512, 1],
    ["max_terms", form.maxTerms, "The term limit", 128, 1],
  ];
  for (const [key, raw, label, max, min] of ints) {
    const n = integer(label, raw, max, min ?? 0);
    if (typeof n === "string") return { ok: false, error: n };
    if (n !== undefined) (out as Record<string, unknown>)[key] = n;
  }
  const tol = form.tolerance.trim();
  if (tol !== "") {
    const value = Number(tol);
    if (!Number.isFinite(value) || value <= 0) {
      return { ok: false, error: "The tolerance must be a positive number, e.g. 1e-6." };
    }
    if (form.toleranceKind === "") {
      return { ok: false, error: "Say whether the tolerance is an absolute or a relative error." };
    }
    out.tolerance = tol;
    out.tolerance_kind = form.toleranceKind;
  } else if (form.toleranceKind !== "") {
    return { ok: false, error: "A tolerance kind needs a tolerance." };
  }
  if (form.direction !== "") out.direction = form.direction;
  const variables = list(form.variables);
  const points = list(form.points);
  if (variables.length !== points.length) {
    return { ok: false, error: "Give one point for each variable." };
  }
  if (variables.length > 4) return { ok: false, error: "At most four variables." };
  if (variables.length > 0) {
    out.variables = variables;
    out.points = points;
  }
  return { ok: true, numbers: out };
}

/** The model's call with the user's typed values on top; fields the user cleared are removed. */
export function buildToolArgs(
  base: ApproximationToolArgs,
  form: ApproximationForm,
): { ok: true; args: ApproximationToolArgs } | { ok: false; error: string } {
  const built = buildApproximationNumbers(form);
  if (!built.ok) return built;
  const keep: ApproximationToolArgs = { latex: base.latex, assumptions: base.assumptions };
  if (base.variable !== undefined) keep.variable = base.variable;
  return { ok: true, args: { ...keep, ...built.numbers } };
}

/** What the form needs to show, per method: which fields apply (the rest are hidden, not defaulted). */
export const FIELDS_FOR: Record<ApproximationMethod, (keyof ApproximationForm)[]> = {
  auto: [
    "point",
    "order",
    "small",
    "lower",
    "upper",
    "degree",
    "denominatorDegree",
    "period",
    "modes",
    "tolerance",
  ],
  taylor: ["point", "order"],
  perturbation: ["small", "order"],
  laurent: ["point", "order"],
  asymptotic: ["point", "order"],
  multivariable: ["variables", "points", "order"],
  leading: ["point", "direction"],
  limit: ["point", "direction"],
  pade: ["point", "degree", "denominatorDegree", "lower", "upper"],
  chebyshev: ["lower", "upper", "degree", "tolerance", "maxTerms"],
  legendre: ["lower", "upper", "degree", "tolerance", "maxTerms"],
  fourier: ["lower", "upper", "period", "modes", "kind", "tolerance", "maxTerms"],
  piecewise: ["lower", "upper", "segments", "kind", "tolerance", "maxTerms"],
  continued_fraction: ["order", "tolerance"],
  euler_maclaurin: ["lower", "upper", "order"],
  integral_expansion: ["small", "lower", "upper", "order"],
  root_expansion: ["small", "point", "order"],
  reversion: ["small", "point", "order"],
  laplace: ["small", "lower", "upper", "order"],
  accelerate: ["lower", "maxTerms"],
  compare: ["lower", "upper", "degree", "point", "tolerance", "maxTerms"],
};
