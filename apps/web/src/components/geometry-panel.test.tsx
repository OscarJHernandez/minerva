import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { christoffel, ricciScalar, sphere } from "@/test/geometry-fixtures";
import type { GeometryQuantityResult } from "@/lib/types";
import { GeometryPanel } from "./geometry-panel";

describe("GeometryPanel", () => {
  it("shows the metric, the restriction, the quantities and how they were checked", () => {
    render(<GeometryPanel result={sphere} />);
    expect(screen.getByLabelText("Metric")).toBeInTheDocument();
    expect(screen.getByTestId("restrictions")).toHaveTextContent(/valid only where/);
    expect(screen.getByTestId("restrictions")).toHaveTextContent(
      /not whether the geometry is singular/,
    );
    const scalar = screen.getByTestId("quantity-ricci_scalar");
    expect(within(scalar).getByText(/Proved by a symbolic zero test/)).toBeInTheDocument();
    const gamma = screen.getByTestId("quantity-christoffel");
    expect(within(gamma).getByText(/Numerically checked; no symbolic proof/)).toBeInTheDocument();
    expect(within(gamma).queryByText(/Proved/)).not.toBeInTheDocument();
    expect(screen.getByText(/Peak memory 120.5 MB/)).toBeInTheDocument();
  });

  it("says the signature is not determined instead of guessing one", () => {
    render(<GeometryPanel result={sphere} />);
    expect(screen.getByTestId("signature")).toHaveTextContent("not determined");
    render(<GeometryPanel result={{ ...sphere, signature: "(-, +, +, +)", signature_note: "" }} />);
    expect(screen.getAllByTestId("signature")[1]).toHaveTextContent("(-, +, +, +)");
  });

  it("warns when the restrictions could not be fully determined", () => {
    render(<GeometryPanel result={{ ...sphere, restrictions_complete: false }} />);
    expect(screen.getByTestId("restrictions")).toHaveTextContent(/there may be more/);
  });

  it("never draws a value for a quantity that was not verified", () => {
    const failed = {
      ...ricciScalar,
      status: "not_verified" as const,
      scalar_latex: null,
      message: "A computed candidate failed its independent numerical check, so it is not shown.",
    };
    render(<GeometryPanel result={{ ...sphere, status: "not_verified", quantities: [failed] }} />);
    const block = screen.getByTestId("quantity-ricci_scalar");
    expect(block).toHaveTextContent(/not shown/);
    expect(block.querySelector(".katex")).toBeNull();
  });

  it("lists the checks with their outcome", async () => {
    render(<GeometryPanel result={{ ...sphere, quantities: [christoffel] }} />);
    await userEvent.click(screen.getByText(/Checks \(2\)/));
    expect(screen.getByText(/metric compatibility: passed/)).toBeInTheDocument();
    expect(screen.getByText(/compat: not run/)).toBeInTheDocument();
  });

  it("offers the quantities not yet computed and reports the one chosen", async () => {
    const onMore = vi.fn();
    render(<GeometryPanel result={sphere} onMore={onMore} />);
    expect(screen.queryByRole("button", { name: "Christoffel symbols" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Riemann tensor" }));
    expect(onMore).toHaveBeenCalledWith("riemann");
    expect(screen.queryByRole("button", { name: /Covariant|Raise|Lower/ })).not.toBeInTheDocument();
  });

  it("offers no follow-up buttons outside a conversation", () => {
    render(<GeometryPanel result={sphere} />);
    expect(screen.queryByText(/Also compute/)).not.toBeInTheDocument();
  });

  it("labels each conserved charge with where it comes from and keeps the search's wording", () => {
    const charges: GeometryQuantityResult = {
      kind: "geodesic_constants",
      status: "ok",
      title: "Conserved charges of the geodesics from the Killing vectors",
      variance: [],
      components: [
        {
          indices: [1],
          label_latex: "Q_{1}",
          latex: "\\dot{\\phi} r^{2}",
          text: "phi_dot*r**2",
          origin: "cyclic coordinate",
        },
        {
          indices: [2],
          label_latex: "Q_{2}",
          latex: "\\dot{\\phi}^{2} r^{2} + \\dot{r}^{2}",
          text: "phi_dot**2*r**2 + r_dot**2",
          origin: "normalisation",
        },
      ],
      all_zero: false,
      scalar_latex: null,
      scalar_text: null,
      symmetry_note: "",
      verification: null,
      notes: ["Killing tensors (hidden quadratic symmetries) are not searched."],
      message: "",
    };
    render(<GeometryPanel result={{ ...sphere, quantities: [charges] }} />);
    const block = screen.getByTestId("quantity-geodesic_constants");
    const origins = within(block).getAllByTestId("charge-origin");
    expect(origins.map((o) => o.textContent)).toEqual(["cyclic coordinate", "normalisation"]);
    expect(block).toHaveTextContent(/Killing tensors .* are not searched/);
  });
});
