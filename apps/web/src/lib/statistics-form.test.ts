import { describe, expect, it } from "vitest";
import {
  STAT_EXAMPLES,
  blankStatForm,
  buildStatRequest,
  restoreStatForm,
  withFamily,
  type StatForm,
} from "./statistics-form";

describe("buildStatRequest", () => {
  it("builds the Gaussian maximum-likelihood request and leaves independence off until it is ticked", () => {
    const form = {
      ...blankStatForm(),
      assumptions: { n: ["integer", "positive"] } as StatForm["assumptions"],
    };
    const built = buildStatRequest(form);
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request).toMatchObject({
      family: "normal",
      parametrization: "variance",
      estimate: ["mu", "v"],
      action: "mle",
      observation: "x",
      sample_size: "n",
      parameters: {},
    });
    expect(built.request.iid).toBeNull(); // never defaulted to independent
    const ticked = buildStatRequest({ ...form, iid: true });
    expect(ticked.ok && ticked.request.iid).toBe(true);
  });

  it("sends a known parameter and omits the unknown one", () => {
    const form = withFamily(blankStatForm(), "normal", "variance");
    form.params = { mu: { value: "", unknown: true }, v: { value: "v", unknown: false } };
    const built = buildStatRequest(form);
    expect(built.ok && built.request.parameters).toEqual({ v: "v" });
    expect(built.ok && built.request.estimate).toEqual(["mu"]);
  });

  it("asks for what is missing instead of sending an incomplete model", () => {
    const form = { ...blankStatForm(), action: "moments" as const };
    expect(buildStatRequest(form)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/every parameter/),
    });
    const noUnknown = withFamily({ ...blankStatForm(), action: "mle" }, "bernoulli");
    expect(buildStatRequest(noUnknown)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/unknown/),
    });
    const noValue = withFamily({ ...blankStatForm(), action: "moments" }, "poisson");
    expect(buildStatRequest(noValue)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/lam/),
    });
  });

  it("uses the standard values of the Cauchy family when they are left empty", () => {
    const form = { ...withFamily(blankStatForm(), "cauchy"), action: "moments" as const };
    const built = buildStatRequest(form);
    expect(built.ok && built.request.parameters).toEqual({});
  });

  it("takes concrete data as numbers and sets the sample size from them", () => {
    const form = withFamily(
      { ...blankStatForm(), sampleSize: "", data: "1, 0, 1, 1", iid: true },
      "bernoulli",
    );
    form.params = { p: { value: "", unknown: true } };
    const built = buildStatRequest(form);
    expect(built.ok && built.request).toMatchObject({
      data: ["1", "0", "1", "1"],
      sample_size: "4",
      iid: true,
    });
    const bad = buildStatRequest({ ...form, data: "1, two" });
    expect(bad.ok).toBe(false);
  });

  it("needs exactly one unknown and a complete prior for a posterior", () => {
    const example = STAT_EXAMPLES.find((x) => x.id === "beta-binomial")!.form;
    const built = buildStatRequest(example);
    expect(built.ok && built.request.prior).toEqual({
      family: "beta",
      parametrization: null,
      parameters: { alpha: "alpha", beta: "beta" },
    });
    expect(buildStatRequest({ ...example, priorParams: { alpha: "alpha" } })).toMatchObject({
      ok: false,
      error: expect.stringMatching(/prior's beta/),
    });
    const two = withFamily({ ...example, action: "posterior" }, "normal", "variance");
    two.params = { mu: { value: "", unknown: true }, v: { value: "", unknown: true } };
    expect(buildStatRequest(two)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/exactly one/),
    });
  });

  it("restores a saved request into the form", () => {
    const built = buildStatRequest(STAT_EXAMPLES[0]!.form);
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    const restored = restoreStatForm(built.request);
    expect(restored.params.mu).toEqual({ value: "", unknown: true });
    expect(restored.family).toBe("normal");
    const again = buildStatRequest(restored);
    expect(again.ok && again.request).toEqual(built.request);
  });

  it("keeps every example buildable", () => {
    for (const example of STAT_EXAMPLES) {
      const built = buildStatRequest(example.form);
      expect(built.ok, example.id).toBe(true);
    }
  });
});
