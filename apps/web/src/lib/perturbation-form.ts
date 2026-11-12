import type { PerturbationRequest } from "./chat-types";

/** What the form holds: every number as the text the user typed, so nothing is guessed or defaulted. */
export interface PerturbationFormValues {
  parameters: Record<string, string>;
  epsilon: string;
  amplitude: string;
  tEnd: string;
}

export type PerturbationBuildResult = { request: PerturbationRequest } | { error: string };

function num(text: string, what: string): number | string {
  const trimmed = text.trim();
  if (trimmed === "") return `Give a value for ${what}.`;
  const value = Number(trimmed);
  if (!Number.isFinite(value)) return `${what} must be a number.`;
  return value;
}

/** Turns the typed values into the request with `compare`, or says which field is wrong. */
export function buildPerturbationRequest(
  base: PerturbationRequest,
  parameterNames: string[],
  form: PerturbationFormValues,
): PerturbationBuildResult {
  const parameters: Record<string, number> = {};
  for (const name of parameterNames) {
    const v = num(form.parameters[name] ?? "", name);
    if (typeof v === "string") return { error: v };
    parameters[name] = v;
  }
  const small = base.small_parameter;
  const epsilon = num(form.epsilon, small ?? "the initial displacement");
  if (typeof epsilon === "string") return { error: epsilon };
  if (epsilon <= 0) {
    return {
      error: small
        ? `${small} must be greater than 0.`
        : "The initial displacement must be greater than 0.",
    };
  }
  let amplitude: number | undefined;
  if (small) {
    const a = num(form.amplitude, "the initial displacement A");
    if (typeof a === "string") return { error: a };
    amplitude = a;
  }
  const tEnd = num(form.tEnd, "the end time");
  if (typeof tEnd === "string") return { error: tEnd };
  if (tEnd <= 0) return { error: "The end time must be greater than 0." };
  return {
    request: {
      ...base,
      compare: {
        parameters,
        epsilon,
        ...(amplitude === undefined ? {} : { amplitude }),
        t_end: tEnd,
      },
    },
  };
}
