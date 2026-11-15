import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  analyzed,
  cantilever,
  conditionsMissing,
  evaluated,
  higherDerivative,
  notVerified,
  secondVariation,
  solved,
  verified,
} from "@/test/variation-fixtures";
import type { VariationReport } from "@/lib/variation-types";
import { VariationPanel } from "./variation-panel";

const panel = () => screen.getByRole("region", { name: /Variational calculus/ });

/** Visible text with the server's own statements removed (they say "not a proof"). */
function ownText(report: VariationReport): string {
  let text = panel().textContent ?? "";
  const statements = [
    report.first?.checks.statement,
    report.second?.checks.statement,
    report.evaluation?.statement,
    report.verification?.statement,
    report.solution?.statement,
  ];
  for (const s of statements) if (s) text = text.split(s).join("");
  return text;
}

describe("VariationPanel: first variation", () => {
  it("shows the derivation in order for a higher-derivative functional", () => {
    const { container } = render(<VariationPanel report={higherDerivative} />);
    const headings = [...container.querySelectorAll("h3")].map((h) => h.textContent);
    expect(headings).toEqual([
      "Functional",
      "Varied functions",
      "First variation of the integrand",
      "Integration by parts",
      "Euler–Lagrange equations",
      "Boundary contributions",
      "Stationarity conditions",
      "Numerical evidence",
      "What you can do next",
    ]);
    // how derivatives were read
    expect(screen.getByText(/derivatives are written/)).toBeInTheDocument();
    // the EL equation and the boundary form are rendered as math from the server's LaTeX
    const eq = higherDerivative.first!.equations[0]!.equation_latex;
    expect(eq).toContain("x^{(4)}");
    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(10);
    expect(container.textContent).toContain("x¨"); // \ddot{x} in the boundary form
  });

  it("lists every integration by parts with its smoothness need and check mark", () => {
    render(<VariationPanel report={higherDerivative} />);
    const steps = within(screen.getByRole("region", { name: "Integration by parts" })).getAllByRole(
      "listitem",
    );
    expect(steps).toHaveLength(higherDerivative.first!.stages.length);
    expect(steps[0]).toHaveTextContent("Before any integration by parts");
    expect(steps[1]).toHaveTextContent(
      /Integration by parts 1: derivative 2 of the perturbation of x moved/,
    );
    expect(steps[1]).toHaveTextContent(/derivatives up to order 3/);
    expect(steps[1]).toHaveTextContent("✓ checked");
  });

  it("keeps the boundary terms of undeclared conditions and says nothing is assumed", () => {
    render(<VariationPanel report={higherDerivative} />);
    const table = screen.getByRole("table", { name: "Boundary terms" });
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows).toHaveLength(4);
    for (const row of rows) {
      expect(row).toHaveTextContent("Kept: no condition was declared, nothing is assumed");
      expect(row).toHaveTextContent("unspecified");
    }
    expect(screen.getByText(/4 endpoint conditions are not declared/)).toBeInTheDocument();
  });

  it("shows natural conditions at the free end of the cantilever and removes the clamped end", () => {
    render(<VariationPanel report={cantilever} />);
    const rows = within(screen.getByRole("table", { name: "Boundary terms" }))
      .getAllByRole("row")
      .slice(1);
    const upper = rows.filter((r) => r.textContent?.startsWith("upper"));
    expect(upper).toHaveLength(2);
    for (const row of upper) {
      expect(row).toHaveTextContent("Free: this coefficient must vanish (natural condition)");
      expect(row.querySelector("[data-effect='natural_condition']")).not.toBeNull();
      expect(row.querySelectorAll(".katex").length).toBeGreaterThanOrEqual(3); // incl. the condition
    }
    const lower = rows.filter((r) => r.textContent?.startsWith("lower"));
    for (const row of lower)
      expect(row).toHaveTextContent("Removed: the perturbation is zero here");
    expect(screen.queryByText(/not declared/)).not.toBeInTheDocument();
  });

  it("states the server's statement verbatim as numerical evidence", () => {
    render(<VariationPanel report={cantilever} />);
    expect(screen.getByTestId("checks-statement")).toHaveTextContent(
      cantilever.first!.checks.statement,
    );
    expect(screen.getByTestId("checks-statement")).toHaveTextContent("It is not a proof");
    expect(screen.getByText(/Deciding comparison:/).closest("p")).toHaveTextContent(
      "Deciding comparison: passed",
    );
  });

  it("offers next steps in words, with what the user must supply", () => {
    render(<VariationPanel report={analyzed} />);
    const list = within(screen.getByRole("region", { name: "What you can do next" }));
    expect(list.getByText(/Derive the first variation/)).toBeInTheDocument();
    const solve = list
      .getAllByRole("listitem")
      .find((li) => /stationary path/.test(li.textContent!));
    expect(solve).toHaveTextContent(
      "Look for a stationary path (needs from you: the interval, values for the parameters, the endpoint conditions, a starting guess)",
    );
  });
});

describe("VariationPanel: other results", () => {
  it("shows the second variation with its convention and evidence", () => {
    render(<VariationPanel report={secondVariation} />);
    expect(screen.getByRole("region", { name: "Second variation" })).toBeInTheDocument();
    expect(screen.getByText("Legendre matrix")).toBeInTheDocument();
    expect(screen.getByTestId("checks-statement")).toBeInTheDocument();
  });

  it("shows the evaluated action, the declared-condition check and the ordering caveat", () => {
    render(<VariationPanel report={evaluated} />);
    const row = within(screen.getByRole("table", { name: "Trial paths" })).getAllByRole("row")[1]!;
    expect(row).toHaveTextContent("0.321046307967165");
    expect(screen.getByTestId("evaluation-statement")).toHaveTextContent(
      evaluated.evaluation!.statement,
    );
  });

  it("shows the finite-difference verification with its verdict and statement", () => {
    render(<VariationPanel report={verified} />);
    expect(screen.getByText("passed", { selector: "span.font-medium" })).toBeInTheDocument();
    expect(
      within(screen.getByRole("table", { name: "Perturbations" })).getAllByRole("row").length,
    ).toBeGreaterThan(3);
    expect(screen.getByTestId("verification-statement")).toHaveTextContent(
      "numerical evidence, not a proof",
    );
  });

  it("shows the stationary path with methods, agreement, a plot and the statement", () => {
    render(<VariationPanel report={solved} />);
    const methods = within(screen.getByRole("table", { name: "Methods" })).getAllByRole("row");
    expect(methods).toHaveLength(3);
    expect(methods[1]).toHaveTextContent("hermite action");
    expect(screen.getByRole("img", { name: /Stationary path found/ })).toBeInTheDocument();
    expect(screen.getByText(/Both methods found the path/)).toBeInTheDocument();
    expect(screen.getByTestId("solution-statement")).toHaveTextContent(solved.solution!.statement);
    expect(screen.getByTestId("second-numeric")).toBeInTheDocument();
    expect(screen.getByText(/not claimed to be unique/)).toBeInTheDocument();
  });
});

describe("VariationPanel: refusals and wording", () => {
  it("shows only the message and the status for a path that was not verified", () => {
    const { container } = render(<VariationPanel report={notVerified} />);
    expect(screen.getByTestId("variation-message")).toHaveTextContent(notVerified.message);
    expect(screen.getByText("Could not verify a result")).toBeInTheDocument();
    expect(container.querySelectorAll(".katex")).toHaveLength(0);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("asks for the missing numbers without showing a value", () => {
    const { container } = render(<VariationPanel report={conditionsMissing} />);
    expect(screen.getByTestId("variation-message")).toHaveTextContent(
      "Numerical values are needed for: m",
    );
    expect(container.querySelectorAll(".katex")).toHaveLength(0);
  });

  it.each([
    ["first", cantilever],
    ["first with undeclared ends", higherDerivative],
    ["analysis", analyzed],
    ["second", secondVariation],
    ["evaluation", evaluated],
    ["verification", verified],
    ["solution", solved],
  ])("never claims a proof, a unique or an optimal path itself (%s)", (_name, report) => {
    render(<VariationPanel report={report} />);
    const own = ownText(report);
    expect(own).not.toMatch(/proved|proof|unique|optimal|minimiz|minimis/i);
    // "minimum" appears only in the enumerated second-variation facts, never as a conclusion
    if (!report.solution?.second && !report.second?.numeric) expect(own).not.toMatch(/minimum/i);
  });
});
