import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { formatTick, LinePlot, niceTicks } from "./line-plot";

describe("niceTicks", () => {
  it("uses 1, 2 or 5 times a power of ten and stays inside the range", () => {
    expect(niceTicks(0, 10)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks(-1, 1)).toEqual([-1, -0.5, 0, 0.5, 1]);
    const ticks = niceTicks(0.013, 0.97);
    expect(ticks[0]).toBeGreaterThanOrEqual(0.013);
    expect(ticks[ticks.length - 1]).toBeLessThanOrEqual(0.97);
  });
  it("copes with an empty range", () => {
    expect(niceTicks(3, 3)).toEqual([3]);
  });
});

describe("formatTick", () => {
  it("is compact for large and tiny values", () => {
    expect(formatTick(0)).toBe("0");
    expect(formatTick(0.5)).toBe("0.5");
    expect(formatTick(12345)).toBe("1.2e4");
    expect(formatTick(0.00002)).toBe("2.0e-5");
  });
});

describe("LinePlot", () => {
  it("draws one path per series and a legend entry for each", () => {
    render(
      <LinePlot
        label="x against time"
        xLabel="t"
        yLabel="x"
        series={[
          { label: "k = 1", x: [0, 1, 2], y: [0, 1, 0] },
          { label: "k = 4", x: [0, 1, 2], y: [1, 0, 1] },
        ]}
      />,
    );
    const chart = screen.getByRole("img", { name: "x against time" });
    expect(chart.querySelectorAll("path")).toHaveLength(2);
    expect(screen.getByText("k = 1")).toBeInTheDocument();
    expect(screen.getByText("k = 4")).toBeInTheDocument();
  });

  it("draws a constant series and skips non-finite points", () => {
    render(
      <LinePlot
        label="flat"
        xLabel="t"
        yLabel="y"
        series={[{ label: "c", x: [0, 1, 2], y: [2, Number.NaN, 2] }]}
      />,
    );
    const d = screen.getByRole("img", { name: "flat" }).querySelector("path")?.getAttribute("d");
    expect(d).toBeTruthy();
    expect(d).not.toContain("NaN");
  });
});

describe("LinePlot yFloor", () => {
  it("anchors the axis at the floor, also when every value equals it", () => {
    render(
      <LinePlot
        label="drift"
        xLabel="t"
        yLabel="drift"
        yFloor={0}
        series={[{ label: "Q", x: [0, 1, 2], y: [0, 0, 0] }]}
      />,
    );
    const labels = Array.from(
      screen.getByRole("img", { name: "drift" }).querySelectorAll("text"),
    ).map((t) => t.textContent);
    expect(labels).toContain("0");
    expect(labels).not.toContain("-1");
  });
});
