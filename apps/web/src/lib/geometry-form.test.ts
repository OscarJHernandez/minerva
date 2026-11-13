import { describe, expect, it } from "vitest";
import {
  EXAMPLES,
  blankForm,
  buildGeometryRequest,
  parseFunctions,
  restoreForm,
  setDimension,
  setMetricCell,
  setRank,
} from "./geometry-form";

function filled() {
  let form = setDimension(blankForm(2), 2);
  form = { ...form, coordinates: ["r", "\\theta"] };
  form = setMetricCell(form, 0, 0, "1");
  form = setMetricCell(form, 1, 1, "r^2");
  return form;
}

describe("geometry form", () => {
  it("builds a request, filling an empty off-diagonal entry with 0", () => {
    const built = buildGeometryRequest(filled());
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.request.metric).toEqual([
      ["1", "0"],
      ["0", "r^2"],
    ]);
    expect(built.request.coordinates).toEqual(["r", "\\theta"]);
    expect(built.request.tensor).toBeNull();
  });

  it("mirrors an off-diagonal entry so the metric is symmetric", () => {
    const form = setMetricCell(filled(), 0, 1, "x");
    expect(form.metric[0]![1]).toBe("x");
    expect(form.metric[1]![0]).toBe("x");
  });

  it("refuses an empty diagonal entry, repeated coordinates and bad names", () => {
    expect(buildGeometryRequest(setMetricCell(filled(), 1, 1, ""))).toMatchObject({ ok: false });
    expect(buildGeometryRequest({ ...filled(), coordinates: ["r", "r"] })).toMatchObject({
      ok: false,
      error: "Coordinate names must be distinct.",
    });
    expect(buildGeometryRequest({ ...filled(), coordinates: ["r", "2x"] })).toMatchObject({
      ok: false,
    });
    expect(buildGeometryRequest({ ...filled(), quantities: [] })).toMatchObject({ ok: false });
  });

  it("keeps what was typed when the dimension changes", () => {
    const bigger = setDimension(filled(), 3);
    expect(bigger.metric).toHaveLength(3);
    expect(bigger.metric[1]![1]).toBe("r^2");
    expect(bigger.coordinates.slice(0, 2)).toEqual(["r", "\\theta"]);
    expect(setDimension(bigger, 99).coordinates).toHaveLength(4); // capped
  });

  it("parses function declarations and reports the first bad one", () => {
    expect(parseFunctions("f(r), h(r, t)")).toEqual([
      { name: "f", args: ["r"] },
      { name: "h", args: ["r", "t"] },
    ]);
    expect(parseFunctions("")).toEqual([]);
    expect(parseFunctions("f(r), g")).toMatch(/not a function declaration/);
    expect(parseFunctions("f()")).toMatch(/not a function declaration|at least one/);
  });

  it("requires every tensor component only when a tensor quantity is chosen", () => {
    const base = { ...filled(), quantities: ["covariant_derivative" as const] };
    expect(buildGeometryRequest(base)).toMatchObject({ ok: false });
    const full = { ...base, tensorComponents: ["1", "0"] };
    const built = buildGeometryRequest(full);
    expect(built.ok && built.request.tensor).toEqual({ variance: ["up"], components: ["1", "0"] });
    const rank2 = setRank(full, 2);
    expect(rank2.tensorComponents).toHaveLength(4);
  });

  it("restores the form from a saved request and builds the same request again", () => {
    for (const example of EXAMPLES) {
      const built = buildGeometryRequest(example.form);
      expect(built.ok).toBe(true);
      if (!built.ok) continue;
      const again = buildGeometryRequest(restoreForm(built.request));
      expect(again).toEqual(built);
    }
  });

  it("declares nothing about the signature or the sign of a symbol on its own", () => {
    for (const example of EXAMPLES) {
      const built = buildGeometryRequest(example.form);
      if (!built.ok) continue;
      const declared = Object.values(built.request.assumptions).flat();
      expect(declared.every((a) => a === "positive")).toBe(true);
    }
    const schwarzschild = EXAMPLES.find((e) => e.id === "schwarzschild")!;
    expect(schwarzschild.form.assumptions).toEqual({ M: ["positive"], r: ["positive"] });
  });
});
