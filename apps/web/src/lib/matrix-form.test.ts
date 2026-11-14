import { describe, expect, it } from "vitest";
import {
  EXAMPLES,
  blankForm,
  blankMatrix,
  buildMatrixRequest,
  resize,
  restoreMatrixForm,
  setKind,
} from "./matrix-form";

const filled = () => ({ ...blankForm(), input: "\\exp(A)\\exp(B)" });

describe("buildMatrixRequest", () => {
  it("declares sizes only and states nothing about commutation or invertibility", () => {
    const built = buildMatrixRequest(filled());
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.matrices).toEqual([
      { name: "A", rows: 2, cols: 2, depends_on: null },
      { name: "B", rows: 2, cols: 2, depends_on: null },
    ]);
    expect(built.request.commuting).toEqual([]);
    expect(built.request.invertible).toEqual([]);
    expect(built.request.action).toBe("simplify");
    expect(built.request.variable).toBeNull();
  });

  it("sends only what the user declared", () => {
    const built = buildMatrixRequest({
      ...filled(),
      commuting: "A,B",
      invertible: "A",
      selfCommuting: "",
    });
    expect(built.ok && built.request.commuting).toEqual([["A", "B"]]);
    expect(built.ok && built.request.invertible).toEqual(["A"]);
  });

  it("refuses incomplete explicit matrices instead of filling in zeros", () => {
    const form = filled();
    form.matrices = [setKind(blankMatrix("A"), "explicit"), blankMatrix("B")];
    const built = buildMatrixRequest(form);
    expect(built).toEqual({ ok: false, error: expect.stringMatching(/Fill in every entry of A/) });
  });

  it("explains what is missing", () => {
    expect(buildMatrixRequest(blankForm())).toEqual({
      ok: false,
      error: expect.stringMatching(/Enter the expression/),
    });
    const dup = filled();
    dup.matrices = [blankMatrix("A"), blankMatrix("A")];
    expect(buildMatrixRequest(dup)).toEqual({
      ok: false,
      error: expect.stringMatching(/declared twice/),
    });
    expect(buildMatrixRequest({ ...filled(), commuting: "A,C" })).toEqual({
      ok: false,
      error: expect.stringMatching(/C is used in a declaration/),
    });
    expect(buildMatrixRequest({ ...filled(), commuting: "A,B,C" })).toEqual({
      ok: false,
      error: expect.stringMatching(/not a pair/),
    });
    expect(buildMatrixRequest({ ...filled(), action: "differentiate" })).toEqual({
      ok: false,
      error: expect.stringMatching(/variable/),
    });
  });

  it("never picks the end of a numerical span", () => {
    const form = { ...filled(), action: "evolve" as const, variable: "t" };
    expect(buildMatrixRequest(form)).toEqual({
      ok: false,
      error: expect.stringMatching(/only you can choose/),
    });
    expect(buildMatrixRequest({ ...form, upper: "2" }).ok).toBe(true);
  });
});

describe("matrix form helpers", () => {
  it("keeps typed entries when resizing and clamps to what each kind allows", () => {
    let m = setKind(blankMatrix("A"), "explicit");
    m = {
      ...m,
      entries: [
        ["1", "2"],
        ["3", "4"],
      ],
    };
    m = resize(m, 3, 3);
    expect(m.entries[0]).toEqual(["1", "2", ""]);
    expect(resize(m, 9, 9).rows).toBe(4); // explicit matrices go up to 4x4
    expect(resize(blankMatrix("A"), 9, 9).rows).toBe(8); // abstract ones up to 8x8
  });

  it("restores a saved problem into the form and builds the same request again", () => {
    for (const example of EXAMPLES) {
      const built = buildMatrixRequest(example.form);
      expect(built.ok).toBe(true);
      if (!built.ok) continue;
      const restored = restoreMatrixForm(
        built.request,
        built.input,
        built.format,
        built.assumptions,
      );
      const again = buildMatrixRequest(restored);
      expect(again).toEqual(built);
    }
  });
});

describe("gradients, Jacobians and Hessians", () => {
  const gaussian = () => {
    const form = {
      ...blankForm(),
      format: "text" as const,
      input: "-1/2*logdet(S) - 1/2*transpose(x - mu)*inv(S)*(x - mu)",
      matrices: [
        { ...blankMatrix("x"), rows: 3, cols: 1, entries: [[""], [""], [""]] },
        { ...blankMatrix("S"), rows: 3, cols: 3, positiveDefinite: true },
      ],
      action: "gradient" as const,
      variable: "S",
    };
    return form;
  };

  it("sends the declarations the user stated, and only those", () => {
    const built = buildMatrixRequest({
      ...gaussian(),
      matrices: [
        { ...blankMatrix("x"), rows: 3, cols: 1 },
        { ...blankMatrix("S"), rows: 3, cols: 3, symmetric: true },
        { ...blankMatrix("T"), rows: 3, cols: 3, positiveDefinite: true },
      ],
    });
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    const [x, s, t] = built.request.matrices;
    expect(x).not.toHaveProperty("symmetric");
    expect(s).toMatchObject({ symmetric: true });
    expect(s).not.toHaveProperty("positive_definite");
    expect(t).toMatchObject({ positive_definite: true });
    expect(built.request).toMatchObject({
      action: "gradient",
      variable: "S",
      lower: null,
      upper: null,
    });
    expect(built.request.invertible).toEqual([]); // invertibility is never declared for the user
  });

  it("requires plain text and a declared matrix to differentiate by", () => {
    expect(buildMatrixRequest({ ...gaussian(), format: "latex" })).toMatchObject({
      ok: false,
      error: expect.stringMatching(/plain-text syntax/),
    });
    expect(buildMatrixRequest({ ...gaussian(), variable: "Q" })).toMatchObject({
      ok: false,
      error: expect.stringMatching(/declared matrices/),
    });
  });

  it("refuses to call a non-square matrix symmetric", () => {
    const form = gaussian();
    form.matrices[1] = { ...form.matrices[1]!, cols: 2, symmetric: true };
    expect(buildMatrixRequest(form)).toMatchObject({
      ok: false,
      error: expect.stringMatching(/square/),
    });
  });

  it("restores the declarations of a saved problem", () => {
    const built = buildMatrixRequest(gaussian());
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    const restored = restoreMatrixForm(built.request, built.input, built.format, {});
    expect(restored.matrices[1]).toMatchObject({ positiveDefinite: true, symmetric: false });
    expect(restored.action).toBe("gradient");
  });

  it("offers the Gaussian examples with the plain-text syntax", () => {
    const example = EXAMPLES.find((x) => x.id === "gaussian-gradient-covariance")!;
    const built = buildMatrixRequest(example.form);
    expect(built.ok && built.format).toBe("text");
    expect(built.ok && built.request.matrices.some((m) => m.positive_definite)).toBe(true);
  });
});
