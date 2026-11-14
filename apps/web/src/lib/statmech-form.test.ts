import { describe, expect, it } from "vitest";
import {
  EXAMPLES,
  blankForm,
  buildRequest,
  parseMatrix,
  parseValues,
  quantityOptions,
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

  it("reads a matrix from lines and semicolons, and refuses a malformed one", () => {
    expect(parseMatrix("1; 0\n0; -1")).toEqual([
      ["1", "0"],
      ["0", "-1"],
    ]);
    expect(typeof parseMatrix("1; 0")).toBe("string"); // not square
    expect(typeof parseMatrix("1; \n0; 1")).toBe("string"); // an empty entry
    expect(typeof parseMatrix("")).toBe("string");
    expect(typeof parseMatrix("1;0;0;0;0\n".repeat(5))).toBe("string"); // larger than 4 x 4
  });

  it("builds a quantum system with an operator that is not assumed to commute", () => {
    const example = EXAMPLES.find((e) => e.id === "spin-tilted-field")!.form;
    const built = buildRequest(example, "expectation");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.system).toEqual({
      kind: "quantum",
      hamiltonian: [
        ["-h", "-d"],
        ["-d", "h"],
      ],
      observables: {
        sz: [
          ["1", "0"],
          ["0", "-1"],
        ],
      },
    });
    const wrong = { ...example, quantum: { ...example.quantum, observableName: "E" } };
    expect(buildRequest(wrong, "expectation")).toEqual({
      ok: false,
      error: "Name the operator (not E, which is the energy).",
    });
    const size = { ...example, quantum: { ...example.quantum, observable: "1" } };
    expect(buildRequest(size, "expectation").ok).toBe(false);
  });

  it("builds a grand canonical request for particles, with the chemical potential and statistics", () => {
    const form = EXAMPLES.find((e) => e.id === "fermi-levels")!.form;
    const built = buildRequest(form, "derive");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.ensemble).toBe("grand_canonical");
    expect(built.request.system).toMatchObject({
      kind: "occupation",
      statistics: "fermi",
      mu: "mu",
    });
    expect(built.request.quantities).toEqual(["lnXi", "Omega", "N", "U", "S"]);
    expect(quantityOptions("occupation").map((q) => q.id)).toContain("Var_N");
    expect(quantityOptions("levels").map((q) => q.id)).not.toContain("Omega");
  });

  it("builds a microcanonical count only with the copies and a fixed energy given", () => {
    const form = EXAMPLES.find((e) => e.id === "two-level-counting")!.form;
    const built = buildRequest(form, "partition");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.ensemble).toBe("microcanonical");
    expect(built.request.total_energy).toBe("4");
    expect(built.request.copies).toEqual({ count: "10", identical: false });
    const noCopies = { ...form, copies: { ...form.copies, on: false } };
    expect(buildRequest(noCopies, "partition").ok).toBe(false);
    const noEnergy = { ...form, microcanonical: { on: true, energy: " " } };
    expect(buildRequest(noEnergy, "partition")).toEqual({
      ok: false,
      error: "Give the fixed total energy.",
    });
    const restored = restoreForm(built.request);
    expect(restored.microcanonical).toEqual({ on: true, energy: "4" });
  });

  it("builds a chain and an approximation request with the method and order the user chose", () => {
    const form = EXAMPLES.find((e) => e.id === "ising-chain")!.form;
    const built = buildRequest({ ...form, order: 3 }, "approximate");
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.system).toEqual({
      kind: "chain",
      states: ["1", "-1"],
      bond: "-J*s*t",
      site: "-h*s",
      length: "N",
    });
    expect(built.request.action).toBe("approximate");
    expect(built.request.method).toBe("thermodynamic_limit");
    expect(built.request.order).toBe(3);
    expect(restoreForm(built.request).method).toBe("thermodynamic_limit");
    const one = { ...form, chain: { ...form.chain, states: "1" } };
    expect(buildRequest(one, "partition").ok).toBe(false);
    const noBond = { ...form, chain: { ...form.chain, bond: " " } };
    expect(buildRequest(noBond, "partition")).toEqual({
      ok: false,
      error: "Give the bond energy as an expression in s and t.",
    });
  });
});
