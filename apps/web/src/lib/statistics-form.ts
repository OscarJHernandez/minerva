import type { Assumptions, StatAction, StatModelRequest } from "./types";

/** The families the engine knows, with the names of their parameters (same names as the engine). */
export interface FamilyInfo {
  id: string;
  title: string;
  /** Parametrisation key -> parameter names. The first is the default. */
  parametrizations: Record<string, string[]>;
  /** Parameters that have a standard value when left empty. */
  defaults?: Record<string, string>;
}

export const FAMILIES: FamilyInfo[] = [
  {
    id: "normal",
    title: "Normal",
    parametrizations: { sd: ["mu", "sigma"], variance: ["mu", "v"] },
  },
  { id: "bernoulli", title: "Bernoulli", parametrizations: { default: ["p"] } },
  { id: "binomial", title: "Binomial", parametrizations: { default: ["n", "p"] } },
  { id: "poisson", title: "Poisson", parametrizations: { default: ["lam"] } },
  { id: "beta", title: "Beta", parametrizations: { default: ["alpha", "beta"] } },
  {
    id: "gamma",
    title: "Gamma (shape and rate)",
    parametrizations: { default: ["alpha", "beta"] },
  },
  {
    id: "cauchy",
    title: "Cauchy",
    parametrizations: { default: ["loc", "scale"] },
    defaults: { loc: "0", scale: "1" },
  },
  {
    id: "uniform_0_theta",
    title: "Uniform on (0, theta)",
    parametrizations: { default: ["theta"] },
  },
];

/** Families that can be a prior for a parameter (their support is the parameter's). */
export const PRIORS = ["beta", "gamma", "normal"];

export const STAT_ACTIONS: { id: StatAction; label: string; hint: string }[] = [
  {
    id: "moments",
    label: "Moments",
    hint: "mean, variance and second moment, derived and checked",
  },
  { id: "log_likelihood", label: "Log-likelihood", hint: "of one or several observations" },
  { id: "score", label: "Score", hint: "derivative of the log-likelihood" },
  { id: "mle", label: "Maximum likelihood", hint: "solve the score equations, check the maximum" },
  {
    id: "fisher_information",
    label: "Fisher information",
    hint: "minus the expected Hessian, checked against the variance of the score",
  },
  {
    id: "posterior",
    label: "Posterior",
    hint: "found by matching the kernel of prior times likelihood",
  },
];

export function familyInfo(id: string): FamilyInfo {
  return FAMILIES.find((f) => f.id === id) ?? FAMILIES[0]!;
}

export interface StatParamForm {
  value: string;
  /** This parameter is the unknown to estimate / update / differentiate by. */
  unknown: boolean;
}

export interface StatForm {
  action: StatAction;
  family: string;
  parametrization: string;
  params: Record<string, StatParamForm>;
  observation: string;
  sampleSize: string;
  /** Off by default: independence is never assumed, only stated. */
  iid: boolean;
  data: string;
  priorFamily: string;
  priorParametrization: string;
  priorParams: Record<string, string>;
  assumptions: Assumptions;
}

export function paramNames(family: string, parametrization: string): string[] {
  const info = familyInfo(family);
  return info.parametrizations[parametrization] ?? Object.values(info.parametrizations)[0] ?? [];
}

export function blankStatForm(): StatForm {
  return {
    action: "mle",
    family: "normal",
    parametrization: "variance",
    params: { mu: { value: "", unknown: true }, v: { value: "", unknown: true } },
    observation: "x",
    sampleSize: "n",
    iid: false,
    data: "",
    priorFamily: "beta",
    priorParametrization: "default",
    priorParams: {},
    assumptions: {},
  };
}

/** Switching family or parametrisation keeps the values of parameters with the same name. */
export function withFamily(form: StatForm, family: string, parametrization?: string): StatForm {
  const info = familyInfo(family);
  const key =
    parametrization && parametrization in info.parametrizations
      ? parametrization
      : Object.keys(info.parametrizations)[0]!;
  const params: Record<string, StatParamForm> = {};
  for (const name of paramNames(family, key)) {
    params[name] = form.params[name] ?? { value: "", unknown: false };
  }
  return { ...form, family, parametrization: key, params };
}

export type BuiltStat = { ok: true; request: StatModelRequest } | { ok: false; error: string };

const NAME = /^[A-Za-z][A-Za-z0-9_]{0,31}$/;
const NUMBER = /^-?\d+(\.\d+)?(\/\d+)?$/;

/** Validates the form and builds the request. The server validates again; this only saves a round trip. */
export function buildStatRequest(form: StatForm): BuiltStat {
  const info = familyInfo(form.family);
  const names = paramNames(form.family, form.parametrization);
  const unknown = names.filter((n) => form.params[n]?.unknown);
  const needsUnknown = ["score", "mle", "fisher_information", "posterior"].includes(form.action);
  if (form.action === "moments" && unknown.length > 0) {
    return { ok: false, error: "Moments need a value for every parameter: clear “unknown”." };
  }
  if (needsUnknown && unknown.length === 0) {
    return {
      ok: false,
      error: "Mark the parameter to estimate, differentiate by or update as unknown.",
    };
  }
  if (form.action === "posterior" && unknown.length !== 1) {
    return { ok: false, error: "A posterior updates exactly one parameter: mark one unknown." };
  }
  const parameters: Record<string, string> = {};
  for (const n of names) {
    if (form.params[n]?.unknown) continue;
    const value = (form.params[n]?.value ?? "").trim();
    if (value === "" && info.defaults?.[n] === undefined) {
      return { ok: false, error: `Give a value for ${n}, or mark it as unknown.` };
    }
    if (value !== "") parameters[n] = value;
  }
  const dataValues = form.data
    .split(/[\s,;]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (dataValues.some((t) => !NUMBER.test(t))) {
    return {
      ok: false,
      error: "Write the observations as numbers separated by commas, for example 1, 0, 1.",
    };
  }
  const usesData = ["log_likelihood", "score", "mle", "posterior"].includes(form.action);
  const sample = form.sampleSize.trim();
  if (sample !== "" && !NAME.test(sample) && !/^\d+$/.test(sample)) {
    return { ok: false, error: "The sample size is a name such as n, or a number." };
  }
  const request: StatModelRequest = {
    family: form.family,
    parametrization: Object.keys(info.parametrizations).length > 1 ? form.parametrization : null,
    parameters,
    estimate: unknown,
    action: form.action,
    assumptions: Object.fromEntries(
      Object.entries(form.assumptions).filter(([, v]) => v.length > 0),
    ),
  };
  if (usesData || form.action === "fisher_information") {
    if (form.observation.trim() && usesData) request.observation = form.observation.trim();
    if (dataValues.length > 0 && usesData) {
      request.data = dataValues;
      request.sample_size = sample || String(dataValues.length);
    } else if (sample !== "") {
      request.sample_size = sample;
    }
    // sent exactly as the user set it: when it is off and there are several observations, the
    // engine refuses with independence_not_declared and the panel says so
    if (request.sample_size) request.iid = form.iid ? true : null;
  }
  if (form.action === "posterior") {
    const prior = form.priorParams;
    const priorNames = paramNames(form.priorFamily, form.priorParametrization);
    for (const n of priorNames) {
      if (!(prior[n] ?? "").trim()) return { ok: false, error: `Give the prior's ${n}.` };
    }
    request.prior = {
      family: form.priorFamily,
      parametrization:
        Object.keys(familyInfo(form.priorFamily).parametrizations).length > 1
          ? form.priorParametrization
          : null,
      parameters: Object.fromEntries(priorNames.map((n) => [n, prior[n]!.trim()])),
    };
  }
  return { ok: true, request };
}

/** Rebuilds the form from a saved request, so an earlier problem can be reopened and changed. */
export function restoreStatForm(request: StatModelRequest): StatForm {
  const info = familyInfo(request.family);
  const key = request.parametrization ?? Object.keys(info.parametrizations)[0]!;
  const params: Record<string, StatParamForm> = {};
  for (const n of paramNames(request.family, key)) {
    params[n] = {
      value: request.parameters[n] ?? "",
      unknown: request.estimate.includes(n),
    };
  }
  return {
    action: request.action,
    family: request.family,
    parametrization: key,
    params,
    observation: request.observation ?? "x",
    sampleSize: request.sample_size ?? "",
    iid: request.iid === true,
    data: (request.data ?? []).join(", "),
    priorFamily: request.prior?.family ?? "beta",
    priorParametrization: request.prior?.parametrization ?? "default",
    priorParams: { ...(request.prior?.parameters ?? {}) },
    assumptions: request.assumptions ?? {},
  };
}

export interface StatExample {
  id: string;
  label: string;
  form: StatForm;
}

const base = blankStatForm();
export const STAT_EXAMPLES: StatExample[] = [
  {
    id: "gaussian-mle",
    label: "Maximum likelihood: Gaussian mean and variance",
    form: { ...base, iid: true, assumptions: { n: ["integer", "positive"], x: ["real"] } },
  },
  {
    id: "bernoulli-mle",
    label: "Maximum likelihood: Bernoulli proportion",
    form: {
      ...withFamily(base, "bernoulli"),
      params: { p: { value: "", unknown: true } },
      iid: true,
      assumptions: { n: ["integer", "positive"] },
    },
  },
  {
    id: "normal-fisher",
    label: "Fisher information: Normal mean and standard deviation",
    form: {
      ...withFamily(base, "normal", "sd"),
      action: "fisher_information",
      params: { mu: { value: "", unknown: true }, sigma: { value: "", unknown: true } },
      sampleSize: "",
      assumptions: { sigma: ["positive"] },
    },
  },
  {
    id: "beta-binomial",
    label: "Posterior: Beta prior, Binomial data",
    form: {
      ...withFamily(base, "binomial"),
      action: "posterior",
      params: { n: { value: "n", unknown: false }, p: { value: "", unknown: true } },
      observation: "k",
      sampleSize: "",
      priorFamily: "beta",
      priorParametrization: "default",
      priorParams: { alpha: "alpha", beta: "beta" },
      assumptions: {
        n: ["integer", "positive"],
        k: ["integer", "nonnegative"],
        alpha: ["positive"],
        beta: ["positive"],
      },
    },
  },
  {
    id: "cauchy-moments",
    label: "Moments of the Cauchy family (refused: no mean)",
    form: {
      ...withFamily(base, "cauchy"),
      action: "moments",
      params: { loc: { value: "", unknown: false }, scale: { value: "", unknown: false } },
    },
  },
];
