import {
  approvedExplorations,
  EXPLORATIONS,
  explorePath,
  findExploration,
} from "@minerva/ui/explorations";
import { describe, expect, it } from "vitest";
import { safeNext } from "./next-path";

describe("exploration registry", () => {
  it("has the three Schwarzschild explorations, in order", () => {
    expect(EXPLORATIONS.map((e) => e.id)).toEqual([
      "schwarzschild-curvature",
      "schwarzschild-ricci",
      "schwarzschild-kretschmann",
    ]);
  });

  it("has unique ids and prompts, and every prompt states the metric, coordinates and assumptions", () => {
    expect(new Set(EXPLORATIONS.map((e) => e.id)).size).toBe(EXPLORATIONS.length);
    expect(new Set(EXPLORATIONS.map((e) => e.prompt)).size).toBe(EXPLORATIONS.length);
    for (const e of EXPLORATIONS) {
      expect(e.prompt).toContain("(1-2M/r)^{-1}dr^2");
      expect(e.prompt).toContain("r^2\\sin^2\\theta");
      expect(e.prompt).toContain("t, r, theta, phi");
      expect(e.prompt).toContain("M and r positive");
      expect(e.prompt.length).toBeLessThan(4000);
    }
  });

  it("gives no result away in a prompt", () => {
    for (const e of EXPLORATIONS) expect(e.prompt).not.toMatch(/48|Ricci-flat|vanishes/);
  });

  it("finds an exploration by id and refuses an unknown one", () => {
    expect(findExploration("schwarzschild-ricci")?.button).toBe("Explore the Ricci tensor");
    expect(findExploration("../etc/passwd")).toBeUndefined();
  });

  it("shows a button only for an approved exploration", () => {
    const list = EXPLORATIONS.map((e, i) => ({ ...e, approved: i === 0 }));
    expect(approvedExplorations("geometry-schwarzschild", list).map((e) => e.id)).toEqual([
      "schwarzschild-curvature",
    ]);
    expect(approvedExplorations("something-else", list)).toEqual([]);
  });

  it("is an address that login accepts as a destination", () => {
    for (const e of EXPLORATIONS) expect(safeNext(explorePath(e.id))).toBe(`/explore/${e.id}`);
  });
});
