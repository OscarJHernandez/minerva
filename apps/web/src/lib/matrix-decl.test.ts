import { describe, expect, it } from "vitest";
import { formatMatrices, formatPairs, parseMatrices, parseNames, parsePairs } from "./matrix-decl";

describe("parseMatrices", () => {
  it("reads abstract sizes and explicit entries", () => {
    const r = parseMatrices("A 2x2\nC 2×3\nD 3\nB = [1, 0; 0, \\frac{1}{2}]");
    expect(r).toEqual({
      ok: true,
      value: [
        { name: "A", rows: 2, cols: 2 },
        { name: "C", rows: 2, cols: 3 },
        { name: "D", rows: 3, cols: 3 },
        {
          name: "B",
          entries: [
            ["1", "0"],
            ["0", "\\frac{1}{2}"],
          ],
        },
      ],
    });
  });

  it("round-trips through formatMatrices", () => {
    const text = "A 2x3\nB = [0, 1; 0, 0]";
    const parsed = parseMatrices(text);
    expect(parsed.ok && formatMatrices(parsed.value)).toBe(text);
  });

  it("rejects what the engine would reject, with a readable reason", () => {
    const bad = [
      "",
      "A",
      "A 0x2",
      "A 9x9",
      "A 2x2\nA 3x3",
      "A = [1, 2; 3]",
      "A = [1, ; 3, 4]",
      "A = [1, 2, 3, 4, 5]",
      "7 2x2",
      "A 2x2\nB 2x2\nC 2x2\nD 2x2\nE 2x2\nF 2x2\nG 2x2",
    ];
    for (const text of bad) {
      const r = parseMatrices(text);
      expect(r.ok, text).toBe(false);
      if (!r.ok) expect(r.error.length).toBeGreaterThan(0);
    }
  });
});

describe("pairs and names", () => {
  it("parses commuting pairs", () => {
    const r = parsePairs("A~B, C~D");
    expect(r).toEqual({
      ok: true,
      value: [
        ["A", "B"],
        ["C", "D"],
      ],
    });
    expect(r.ok && formatPairs(r.value)).toBe("A~B, C~D");
    expect(parsePairs("")).toEqual({ ok: true, value: [] });
    expect(parsePairs("A~A").ok).toBe(false);
    expect(parsePairs("AB").ok).toBe(false);
    expect(parsePairs("A~B~C").ok).toBe(false);
  });

  it("parses names", () => {
    expect(parseNames("A, B")).toEqual({ ok: true, value: ["A", "B"] });
    expect(parseNames("A, 2")).toEqual({ ok: false, error: "“2” is not a matrix name." });
  });
});
