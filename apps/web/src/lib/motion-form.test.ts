import { describe, expect, it } from "vitest";
import { buildMotionRequest, type MotionFormValues } from "./motion-form";

const base = {
  lagrangian: "\\frac12 m\\dot{x}^2 - \\frac12 k x^2",
  coordinates: ["x"],
  time: "t",
  assumptions: {},
};

const filled: MotionFormValues = {
  parameters: { m: "1", k: "4" },
  positions: ["1"],
  velocities: ["0"],
  tStart: "0",
  tEnd: "10",
  sweep: null,
};

describe("buildMotionRequest", () => {
  it("builds a single-run request from typed values", () => {
    const r = buildMotionRequest(base, ["k", "m"], filled);
    expect(r).toEqual({
      request: {
        ...base,
        parameters: { k: 4, m: 1 },
        initial_positions: [1],
        initial_velocities: [0],
        t_start: 0,
        t_end: 10,
        sweep: null,
      },
    });
  });

  it("never fills in a missing number", () => {
    const r = buildMotionRequest(base, ["k", "m"], { ...filled, parameters: { m: "1" } });
    expect(r).toEqual({ error: "Give a value for k." });
    expect(buildMotionRequest(base, ["k", "m"], { ...filled, velocities: [""] })).toHaveProperty(
      "error",
    );
    expect(buildMotionRequest(base, ["k", "m"], { ...filled, tEnd: "" })).toHaveProperty("error");
  });

  it("rejects text that is not a finite number", () => {
    const r = buildMotionRequest(base, ["k", "m"], { ...filled, parameters: { m: "1", k: "abc" } });
    expect(r).toEqual({ error: "k must be a number." });
    expect(buildMotionRequest(base, [], { ...filled, positions: ["Infinity"] })).toHaveProperty(
      "error",
    );
  });

  it("requires an end time after the start time", () => {
    expect(buildMotionRequest(base, ["k", "m"], { ...filled, tEnd: "0" })).toEqual({
      error: "The end time must be greater than the start time.",
    });
  });

  it("spreads a sweep evenly and leaves the swept parameter out of the single values", () => {
    const r = buildMotionRequest(base, ["k", "m"], {
      ...filled,
      parameters: { m: "1" },
      sweep: { name: "k", from: "1", to: "4", count: "4" },
    });
    expect(r).toHaveProperty("request.sweep", { name: "k", values: [1, 2, 3, 4] });
    expect(r).toHaveProperty("request.parameters", { m: 1 });
  });

  it("limits a sweep to 2 to 6 values", () => {
    for (const count of ["1", "7", "2.5", "x"]) {
      const r = buildMotionRequest(base, ["k", "m"], {
        ...filled,
        parameters: { m: "1" },
        sweep: { name: "k", from: "1", to: "4", count },
      });
      expect(r).toEqual({ error: "A sweep takes between 2 and 6 values." });
    }
  });
});
