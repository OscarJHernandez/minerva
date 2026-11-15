import { describe, expect, it } from "vitest";
import {
  FIELDS_FOR,
  blankForm,
  buildApproximationNumbers,
  buildToolArgs,
  restoreApproximationForm,
} from "./approximation-form";
import { METHOD_LABEL } from "./approximation-types";

describe("the approximation form", () => {
  it("starts empty: no number is pre-filled", () => {
    const form = blankForm();
    expect(Object.entries(form).filter(([k, v]) => k !== "method" && v !== "")).toEqual([]);
    const built = buildApproximationNumbers(form);
    expect(built).toEqual({ ok: true, numbers: { method: "auto" } });
  });

  it("starts from exactly what the model's call contained", () => {
    const form = restoreApproximationForm({
      method: "pade",
      point: "0",
      degree: 1,
      denominator_degree: 1,
      tolerance: "1e-6",
      tolerance_kind: "absolute",
    });
    expect(form.method).toBe("pade");
    expect(form.degree).toBe("1");
    expect(form.lower).toBe("");
    const built = buildApproximationNumbers(form);
    expect(built).toEqual({
      ok: true,
      numbers: {
        method: "pade",
        point: "0",
        degree: 1,
        denominator_degree: 1,
        tolerance: "1e-6",
        tolerance_kind: "absolute",
      },
    });
  });

  it("refuses what the server would refuse, with a reason", () => {
    const error = (patch: Partial<ReturnType<typeof blankForm>>) => {
      const built = buildApproximationNumbers({ ...blankForm(), ...patch });
      return built.ok ? null : built.error;
    };
    expect(error({ lower: "0" })).toMatch(/both ends/);
    expect(error({ upper: "1" })).toMatch(/both ends/);
    expect(error({ order: "2.5" })).toMatch(/whole number/);
    expect(error({ order: "65" })).toMatch(/between 0 and 64/);
    expect(error({ segments: "0" })).toMatch(/between 1 and 512/);
    expect(error({ tolerance: "abc", toleranceKind: "absolute" })).toMatch(/positive number/);
    expect(error({ tolerance: "-1", toleranceKind: "absolute" })).toMatch(/positive number/);
    expect(error({ tolerance: "1e-3" })).toMatch(/absolute or a relative/);
    expect(error({ toleranceKind: "relative" })).toMatch(/needs a tolerance/);
    expect(error({ variables: "x, y", points: "0" })).toMatch(/one point for each/);
    expect(error({ point: "x".repeat(65) })).toMatch(/too long/);
  });

  it("builds the multivariable lists", () => {
    const built = buildApproximationNumbers({
      ...blankForm("multivariable"),
      variables: "x, y",
      points: "0, 1",
      order: "2",
    });
    expect(built).toEqual({
      ok: true,
      numbers: { method: "multivariable", variables: ["x", "y"], points: ["0", "1"], order: 2 },
    });
  });

  it("puts the typed values on the model's call and drops what was cleared", () => {
    const base = {
      latex: "\\exp(x)",
      method: "pade" as const,
      point: "0",
      degree: 1,
      variable: "x",
    };
    const form = { ...restoreApproximationForm(base), point: "", lower: "-1", upper: "3" };
    const built = buildToolArgs(base, form);
    expect(built).toEqual({
      ok: true,
      args: {
        latex: "\\exp(x)",
        variable: "x",
        method: "pade",
        degree: 1,
        lower: "-1",
        upper: "3",
      },
    });
  });

  it("offers fields for every method, and a label for each", () => {
    for (const method of Object.keys(METHOD_LABEL) as (keyof typeof METHOD_LABEL)[]) {
      expect(FIELDS_FOR[method].length).toBeGreaterThan(0);
    }
    expect(FIELDS_FOR.pade).toContain("denominatorDegree");
    expect(FIELDS_FOR.fourier).toContain("period");
  });
});
