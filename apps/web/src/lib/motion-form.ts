import type { MotionRequest } from "./chat-types";

/** What the form holds: every number as the text the user typed, so nothing is guessed or defaulted. */
export interface MotionFormValues {
  parameters: Record<string, string>;
  positions: string[];
  velocities: string[];
  tStart: string;
  tEnd: string;
  sweep: { name: string; from: string; to: string; count: string } | null;
}

export type BuildResult = { request: MotionRequest } | { error: string };

function num(text: string, what: string): number | string {
  const trimmed = text.trim();
  if (trimmed === "") return `Give a value for ${what}.`;
  const value = Number(trimmed);
  if (!Number.isFinite(value)) return `${what} must be a number.`;
  return value;
}

/** Turns the typed values into the request, or says which field is wrong. */
export function buildMotionRequest(
  base: Pick<MotionRequest, "lagrangian" | "coordinates" | "time" | "assumptions">,
  parameterNames: string[],
  form: MotionFormValues,
): BuildResult {
  const parameters: Record<string, number> = {};
  for (const name of parameterNames) {
    if (form.sweep?.name === name) continue;
    const v = num(form.parameters[name] ?? "", name);
    if (typeof v === "string") return { error: v };
    parameters[name] = v;
  }
  const positions: number[] = [];
  const velocities: number[] = [];
  for (let i = 0; i < base.coordinates.length; i++) {
    const c = base.coordinates[i] ?? "";
    const q = num(form.positions[i] ?? "", `${c}(start)`);
    if (typeof q === "string") return { error: q };
    const v = num(form.velocities[i] ?? "", `the initial velocity of ${c}`);
    if (typeof v === "string") return { error: v };
    positions.push(q);
    velocities.push(v);
  }
  const tStart = num(form.tStart, "the start time");
  if (typeof tStart === "string") return { error: tStart };
  const tEnd = num(form.tEnd, "the end time");
  if (typeof tEnd === "string") return { error: tEnd };
  if (tEnd <= tStart) return { error: "The end time must be greater than the start time." };

  let sweep: MotionRequest["sweep"] = null;
  if (form.sweep) {
    const from = num(form.sweep.from, `the first ${form.sweep.name}`);
    if (typeof from === "string") return { error: from };
    const to = num(form.sweep.to, `the last ${form.sweep.name}`);
    if (typeof to === "string") return { error: to };
    const count = Number(form.sweep.count);
    if (!Number.isInteger(count) || count < 2 || count > 6) {
      return { error: "A sweep takes between 2 and 6 values." };
    }
    const values = Array.from({ length: count }, (_, k) =>
      Number((from + ((to - from) * k) / (count - 1)).toPrecision(12)),
    );
    sweep = { name: form.sweep.name, values };
  }
  return {
    request: {
      lagrangian: base.lagrangian,
      coordinates: base.coordinates,
      time: base.time,
      assumptions: base.assumptions,
      parameters,
      initial_positions: positions,
      initial_velocities: velocities,
      t_start: tStart,
      t_end: tEnd,
      sweep,
    },
  };
}
