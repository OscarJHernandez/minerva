import { describe, expect, it } from "vitest";
import type { PerturbationRequest } from "./chat-types";
import { buildPerturbationRequest, type PerturbationFormValues } from "./perturbation-form";

const base: PerturbationRequest = {
  lagrangian: "L",
  coordinates: ["x"],
  time: "t",
  assumptions: {},
  small_parameter: "b",
  equilibrium: "0",
  order: 2,
};
const amplitudeBase: PerturbationRequest = { ...base, small_parameter: null };
const form = (over: Partial<PerturbationFormValues> = {}): PerturbationFormValues => ({
  parameters: { m: "1", g: "9.81" },
  epsilon: "0.1",
  amplitude: "1",
  tEnd: "30",
  ...over,
});

describe("buildPerturbationRequest", () => {
  it("sends exactly what was typed, with the amplitude in parameter mode", () => {
    const r = buildPerturbationRequest(base, ["m", "g"], form());
    expect(r).toEqual({
      request: {
        ...base,
        compare: { parameters: { m: 1, g: 9.81 }, epsilon: 0.1, amplitude: 1, t_end: 30 },
      },
    });
  });

  it("has no amplitude in amplitude mode (the small parameter is the displacement)", () => {
    const r = buildPerturbationRequest(amplitudeBase, ["m", "g"], form({ amplitude: "" }));
    expect("request" in r && r.request.compare).toEqual({
      parameters: { m: 1, g: 9.81 },
      epsilon: 0.1,
      t_end: 30,
    });
  });

  it("names the field that is missing instead of defaulting it", () => {
    expect(buildPerturbationRequest(base, ["m", "g"], form({ parameters: { m: "1" } }))).toEqual({
      error: "Give a value for g.",
    });
    expect(buildPerturbationRequest(base, [], form({ epsilon: "" }))).toEqual({
      error: "Give a value for b.",
    });
    expect(buildPerturbationRequest(base, [], form({ amplitude: " " }))).toEqual({
      error: "Give a value for the initial displacement A.",
    });
    expect(buildPerturbationRequest(base, [], form({ tEnd: "" }))).toEqual({
      error: "Give a value for the end time.",
    });
  });

  it("rejects non-numbers, non-positive sizes and non-positive times", () => {
    expect(buildPerturbationRequest(base, ["m"], form({ parameters: { m: "abc" } }))).toEqual({
      error: "m must be a number.",
    });
    for (const epsilon of ["0", "-0.1"]) {
      expect("error" in buildPerturbationRequest(base, [], form({ epsilon }))).toBe(true);
    }
    expect("error" in buildPerturbationRequest(amplitudeBase, [], form({ epsilon: "0" }))).toBe(
      true,
    );
    expect("error" in buildPerturbationRequest(base, [], form({ tEnd: "0" }))).toBe(true);
    expect("error" in buildPerturbationRequest(base, [], form({ tEnd: "Infinity" }))).toBe(true);
  });

  it("keeps a negative initial displacement A, which is a valid start", () => {
    const r = buildPerturbationRequest(base, [], form({ amplitude: "-0.5" }));
    expect("request" in r && r.request.compare?.amplitude).toBe(-0.5);
  });
});
