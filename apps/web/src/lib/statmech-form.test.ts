import { describe, expect, it } from "vitest";
import {
  EXAMPLES,
  blankForm,
  buildRequest,
  parseValues,
  restoreForm,
  splitNames,
} from "./statmech-form";

describe("statmech form", () => {
  it("builds a derive request for the two-level example", () => {
    const built = buildRequest(blankForm(), "derive");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.system).toEqual({
      kind: "levels",
      levels: [
        { energy: "0", degeneracy: "1" },
        { energy: "epsilon", degeneracy: "1" },
      ],
    });
    expect(built.request.quantities).toEqual(["Z", "F", "U", "S"]);
    expect(built.request.assumptions).toEqual({ epsilon: ["positive"] });
    expect(built.request.copies).toBeNull();
  });

  it("asks for what is missing instead of sending it", () => {
    const empty = { ...blankForm(), levels: [{ energy: " ", degeneracy: "1" }] };
    expect(buildRequest(empty, "partition")).toEqual({
      ok: false,
      error: "Give at least one energy level.",
    });
    const none = { ...blankForm(), quantities: [] };
    const built = buildRequest(none, "derive");
    expect(built.ok).toBe(false);
  });

  it("never fills in a convention the user left open", () => {
    const gas = EXAMPLES.find((e) => e.id === "ideal-gas")!.form;
    const open = {
      ...gas,
      classical: { ...gas.classical, cell: "" },
      copies: { ...gas.copies, identical: "" as const },
    };
    const built = buildRequest(open, "derive");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.copies).toEqual({ count: "N", identical: null });
    expect(built.request.system.kind === "classical" && built.request.system.phase_space_cell).toBe(
      null,
    );
  });

  it("splits partial derivatives with their held-fixed names", () => {
    const form = {
      ...blankForm(),
      partials: [{ of: "F" as const, variable: "V", held: "T, h", scale: "-1", label: "p" }],
    };
    const built = buildRequest(form, "derive");
    expect(built.ok && built.request.partials).toEqual([
      { of: "F", variable: "V", held_fixed: ["T", "h"], scale: "-1", label: "p" },
    ]);
    expect(splitNames("a, b  c")).toEqual(["a", "b", "c"]);
  });

  it("parses the user's numbers, line by line", () => {
    expect(parseValues("beta = 1\n k_B=1 \n")).toEqual({ beta: "1", k_B: "1" });
    expect(typeof parseValues("beta 1")).toBe("string");
  });

  it("validates a sweep", () => {
    const form = {
      ...blankForm(),
      values: "epsilon = 1\nk_B = 1",
      sweep: { ...blankForm().sweep, on: true, points: "1" },
    };
    expect(buildRequest(form, "evaluate")).toEqual({
      ok: false,
      error: "A sweep has 2 to 60 points.",
    });
  });

  it("restores any example it builds", () => {
    for (const example of EXAMPLES) {
      const built = buildRequest(example.form, "derive");
      if (!built.ok) throw new Error(`${example.id}: ${built.error}`);
      const again = buildRequest(restoreForm(built.request), "derive");
      expect(again).toEqual(built);
    }
  });
});
