import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { buildTelescopingRequest, TelescopingPanel } from "./telescoping-panel";

vi.mock("./math-input", () => ({
  MathInput: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <input aria-label="Integrand" value={value} onChange={(e) => onChange(e.target.value)} />
  ),
}));

const form = {
  integrand: "\\exp(-a x^2)",
  variable: "x",
  parameter: "a",
  lower: "-oo",
  upper: "oo",
  parameterSign: "positive" as const,
};

describe("buildTelescopingRequest", () => {
  it("declares only what the user chose", () => {
    expect(buildTelescopingRequest(form)).toEqual({
      input: "\\exp(-a x^2)",
      format: "latex",
      variable: "x",
      parameter: "a",
      lower: "-oo",
      upper: "oo",
      assumptions: { x: ["real"], a: ["positive"] },
    });
    const undeclared = buildTelescopingRequest({ ...form, parameterSign: "none" });
    expect(typeof undeclared === "object" && undeclared.assumptions).toEqual({ x: ["real"] });
  });

  it("refuses an incomplete or inconsistent form", () => {
    expect(buildTelescopingRequest({ ...form, integrand: " " })).toMatch(/integrand/);
    expect(buildTelescopingRequest({ ...form, upper: "" })).toMatch(/both bounds/);
    expect(buildTelescopingRequest({ ...form, parameter: "x" })).toMatch(/differ/);
    expect(buildTelescopingRequest({ ...form, variable: "1x" })).toMatch(/names/);
  });
});

describe("TelescopingPanel", () => {
  it("shows the engine's refusal as an error", async () => {
    render(<TelescopingPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Find the relation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter the integrand.");
    await waitFor(() => expect(screen.queryByText("Searching…")).toBeNull());
  });
});
