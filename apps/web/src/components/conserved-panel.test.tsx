import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { buildConservedRequest, ConservedPanel } from "./conserved-panel";

vi.mock("./math-input", () => ({
  MathInput: ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <textarea
      aria-label="Mathematical expression"
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  ),
}));

const base = {
  system: "lagrangian" as const,
  fn: "x",
  coordinates: "r, theta",
  momenta: "",
  budget: "standard" as const,
  advanced: false,
  degree: "",
  power: "",
  harmonics: "",
  timeDegree: "",
};

describe("buildConservedRequest", () => {
  it("sends the function and coordinates only: no ansatz, angles or expected invariant", () => {
    expect(buildConservedRequest(base)).toEqual({
      function: "x",
      system: "lagrangian",
      coordinates: ["r", "theta"],
      momenta: null,
      budget: "standard",
    });
  });

  it("pins one family only when asked in the advanced section", () => {
    const r = buildConservedRequest({ ...base, advanced: true, degree: "2", power: "3" });
    expect(r).toMatchObject({ pins: { degree: 2, power_bound: 3, harmonics: null } });
    expect(buildConservedRequest({ ...base, advanced: true })).not.toHaveProperty("pins");
  });

  it("refuses bad input with a message", () => {
    expect(buildConservedRequest({ ...base, fn: " " })).toMatch(/Enter the/);
    expect(buildConservedRequest({ ...base, coordinates: "a,b,c,d" })).toMatch(/between 1 and 3/);
    expect(buildConservedRequest({ ...base, system: "hamiltonian", momenta: "p_r" })).toMatch(
      /one momentum name per coordinate/,
    );
    expect(buildConservedRequest({ ...base, advanced: true, degree: "1.5" })).toMatch(/whole/);
  });
});

describe("ConservedPanel", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("runs the search and shows an error from the server", async () => {
    const spy = vi.spyOn(api, "conserved").mockRejectedValue(new Error("boom"));
    render(<ConservedPanel />);
    fireEvent.change(screen.getByLabelText("Mathematical expression"), { target: { value: "x" } });
    fireEvent.change(screen.getByPlaceholderText("r, theta"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Find conserved quantities" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(await screen.findByRole("alert")).toHaveTextContent("The search failed.");
  });

  it("asks for the missing input before calling the server", () => {
    const spy = vi.spyOn(api, "conserved");
    render(<ConservedPanel />);
    fireEvent.click(screen.getByRole("button", { name: "Find conserved quantities" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Enter the Lagrangian or Hamiltonian.");
    expect(spy).not.toHaveBeenCalled();
  });
});
