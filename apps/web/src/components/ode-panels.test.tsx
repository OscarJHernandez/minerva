import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { analysis, meaning, numeric, solution, solutionItem } from "@/test/ode-fixtures";
import { OdeAnalysisPanel } from "./ode-analysis-panel";
import { OdeNumericPanel } from "./ode-numeric-panel";
import { OdeSolutionPanel } from "./ode-solution-panel";

describe("OdeAnalysisPanel", () => {
  it("shows how the equation was read before anything else", () => {
    render(<OdeAnalysisPanel report={analysis} />);
    expect(screen.getByText("As Minerva read it")).toBeInTheDocument();
    expect(screen.getByText("Independent variable").nextSibling).toHaveTextContent("t");
    expect(screen.getByText("Parameters").nextSibling).toHaveTextContent("c");
    expect(screen.getByText(/read as real/)).toBeInTheDocument();
  });

  it("lists the applicable approaches in order, with what they need", async () => {
    render(<OdeAnalysisPanel report={analysis} />);
    const items = screen.getAllByRole("listitem").filter((li) => li.hasAttribute("data-method"));
    const applicable = items.filter((li) => li.getAttribute("data-applicable") === "true");
    expect(applicable.map((li) => li.getAttribute("data-method"))).toEqual([
      "symbolic_closed_form",
      "numerical_ivp",
    ]);
    expect(applicable[0]).toHaveTextContent("1.");
    expect(applicable[1]).toHaveTextContent("2.");
    await userEvent.click(within(applicable[1] as HTMLElement).getByText(/What it needs/));
    expect(within(applicable[1] as HTMLElement).getByText("the interval")).toBeVisible();
    expect(
      screen.getByText(/does not mean that it will\s+succeed|not that it will succeed/),
    ).toBeInTheDocument();
  });

  it("keeps methods that do not apply, and the ones Minerva does not have, out of the ranking", () => {
    render(<OdeAnalysisPanel report={analysis} />);
    const other = screen.getByText(/Not applicable here \(2\)/).closest("details");
    expect(other).not.toBeNull();
    expect(within(other as HTMLElement).getByText(/damping or a gyroscopic/)).toBeInTheDocument();
    expect(
      within(other as HTMLElement).getByText("not implemented in Minerva"),
    ).toBeInTheDocument();
  });

  it("says what the classes are and that they are candidates", () => {
    render(<OdeAnalysisPanel report={analysis} />);
    expect(
      screen.getByText(/second-order linear, linear with constant coefficients/),
    ).toBeInTheDocument();
    expect(screen.getByText(/each solver checks its own result/)).toBeInTheDocument();
  });

  it("shows a refusal as the message and nothing else", () => {
    render(
      <OdeAnalysisPanel
        report={{
          ...analysis,
          status: "parse_error",
          message: "say which variable is independent",
          interpretation: null,
          structure: null,
          methods: [],
          notes: [],
        }}
      />,
    );
    expect(screen.getByText("say which variable is independent")).toBeInTheDocument();
    expect(screen.queryByText("As Minerva read it")).not.toBeInTheDocument();
    expect(screen.queryByText(/Approaches that can be tried/)).not.toBeInTheDocument();
  });

  it("puts the action for a method next to it", () => {
    render(
      <OdeAnalysisPanel
        report={analysis}
        actions={(m) => (m.id === "numerical_ivp" ? <button>Go</button> : null)}
      />,
    );
    expect(screen.getAllByRole("button", { name: "Go" })).toHaveLength(1);
  });
});

describe("OdeSolutionPanel", () => {
  it("says proved only when a symbolic zero test passed", () => {
    const { rerender } = render(<OdeSolutionPanel report={solution} />);
    expect(screen.getByText("Solved, checked numerically and proved")).toBeInTheDocument();
    const numericOnly = {
      ...solution,
      verification: { ...solution.verification!, symbolic_zero: null },
      solutions: [
        { ...solutionItem, verification: { ...solutionItem.verification, symbolic_zero: null } },
      ],
    };
    rerender(<OdeSolutionPanel report={numericOnly} />);
    expect(screen.getByText("Solved, checked numerically")).toBeInTheDocument();
    expect(screen.queryByText(/and proved/)).not.toBeInTheDocument();
  });

  it("describes the family honestly", () => {
    const { rerender } = render(<OdeSolutionPanel report={solution} />);
    expect(screen.getByText(/The general solution: 2 independent constants/)).toBeInTheDocument();
    const partial = { ...solution, solutions: [{ ...solutionItem, family: "partial" as const }] };
    rerender(<OdeSolutionPanel report={partial} />);
    expect(screen.getByText(/not claimed to be the general solution/)).toBeInTheDocument();
    const particular = {
      ...solution,
      solutions: [{ ...solutionItem, family: "particular" as const, constants: [] }],
    };
    rerender(<OdeSolutionPanel report={particular} />);
    expect(screen.getByText(/every constant is fixed by the conditions/)).toBeInTheDocument();
  });

  it("labels an implicit result and one with an unevaluated integral as what they are", () => {
    const { rerender } = render(
      <OdeSolutionPanel
        report={{ ...solution, solutions: [{ ...solutionItem, kind: "implicit" }] }}
      />,
    );
    expect(screen.getByText(/implicit \(a relation between the variables\)/)).toBeInTheDocument();
    rerender(
      <OdeSolutionPanel
        report={{
          ...solution,
          solutions: [{ ...solutionItem, kind: "unevaluated", evaluated: false }],
        }}
      />,
    );
    expect(screen.getByText(/a representation, not a formula/)).toBeInTheDocument();
  });

  it("shows conditions that hold and ones that do not", () => {
    render(
      <OdeSolutionPanel
        report={{
          ...solution,
          solutions: [
            {
              ...solutionItem,
              conditions: [
                { latex: "y(0) = 1", satisfied: true, residual: 0 },
                { latex: "y'(0) = 0", satisfied: false, residual: 0.2 },
              ],
            },
          ],
        }}
      />,
    );
    expect(screen.getByText(/holds$/)).toBeInTheDocument();
    expect(screen.getByText(/does not hold/)).toBeInTheDocument();
  });

  it("states a refusal without a solution, and that nothing is claimed", () => {
    render(
      <OdeSolutionPanel
        report={{
          ...solution,
          status: "unsupported",
          message: "SymPy found no closed form. That does not show that none exists.",
          solutions: [],
          verification: null,
          notes: [],
        }}
      />,
    );
    expect(screen.getByText(/does not show that none exists/)).toBeInTheDocument();
    expect(screen.queryByRole("list")).not.toBeInTheDocument();
  });

  it("reports candidates that failed the check", () => {
    render(<OdeSolutionPanel report={{ ...solution, dropped: 2 }} />);
    expect(
      screen.getByText(/2 candidates from SymPy failed the check and are not shown/),
    ).toBeInTheDocument();
  });

  it("offers the right-hand side to Expression mode only for a plain explicit result", async () => {
    const onUse = vi.fn();
    const explicit = {
      ...solution,
      solutions: [{ ...solutionItem, constants: [], family: "particular" as const }],
    };
    const { rerender } = render(<OdeSolutionPanel report={explicit} onUse={onUse} />);
    await userEvent.click(screen.getByRole("button", { name: /Use the right-hand side/ }));
    expect(onUse).toHaveBeenCalledWith(
      "C1*exp(-t) + C2*exp(-2*t)",
      solutionItem.equations_latex[0],
    );
    rerender(<OdeSolutionPanel report={solution} onUse={onUse} />);
    expect(
      screen.queryByRole("button", { name: /Use the right-hand side/ }),
    ).not.toBeInTheDocument();
  });
});

describe("OdeNumericPanel", () => {
  it("calls the result an approximation and plots it", () => {
    render(<OdeNumericPanel report={numeric} />);
    expect(screen.getByText("Numerical approximation, checked independently")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "The solution against t" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Phase portrait: y' against y/ })).toBeInTheDocument();
    expect(screen.getByText(/Initial values \(yours\)/)).toBeInTheDocument();
    expect(screen.getByText("y = 1, y' = 0")).toBeInTheDocument();
    expect(screen.getByText("c = 0.5")).toBeInTheDocument();
    expect(screen.getByText(/measurements, not a bound on the error/)).toBeInTheDocument();
  });

  it("lists every check with its measurement", () => {
    render(<OdeNumericPanel report={numeric} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows).toHaveLength(5);
    expect(
      within(rows[1] as HTMLElement).getByText("An independent integrator agrees"),
    ).toBeInTheDocument();
    expect(within(rows[1] as HTMLElement).getByText("passed")).toBeInTheDocument();
    expect(within(rows[4] as HTMLElement).getByText("not run")).toBeInTheDocument();
  });

  it("shows no plot when the run was not verified, but says which check failed", () => {
    render(
      <OdeNumericPanel
        report={{
          ...numeric,
          status: "not_verified",
          message: "The check 'independent method' failed. No trajectory is shown.",
          t: [],
          series: [],
          warnings: ["not_converged"],
          verification: {
            ...numeric.verification!,
            passed: false,
            checks: [
              {
                name: "independent_method",
                status: "failed",
                value: 0.3,
                tolerance: 1e-5,
                note: "",
              },
            ],
          },
        }}
      />,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText(/No trajectory is shown/)).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByText("Refining the tolerance did not converge.")).toBeInTheDocument();
  });

  it("explains a stiffness warning as measured, not predicted", () => {
    render(<OdeNumericPanel report={{ ...numeric, warnings: ["stiffness_indicated"] }} />);
    expect(screen.getByText(/judged from measured behaviour, not predicted/)).toBeInTheDocument();
  });

  it("draws one phase portrait only for exactly two states", () => {
    const three = {
      ...numeric,
      series: [
        ...numeric.series,
        { label_latex: "z", unknown: "z", order: 0, values: [0, 1, 2, 3, 4] },
      ],
    };
    render(<OdeNumericPanel report={three} />);
    expect(screen.getAllByRole("img")).toHaveLength(1);
  });

  it("uses the interpretation's variable for the axis", () => {
    render(
      <OdeNumericPanel report={{ ...numeric, interpretation: { ...meaning, independent: "s" } }} />,
    );
    expect(screen.getByRole("img", { name: "The solution against s" })).toBeInTheDocument();
  });
});

describe("series and perturbation results", () => {
  const expansion = {
    kind: "taylor" as const,
    order: 6,
    point_latex: "t = 0",
    small_latex: null,
    secular_orders: [],
    checks: [
      { name: "residual_order" as const, passed: true, value: 1e-16, note: "" },
      { name: "initial_conditions" as const, passed: true, value: null, note: "" },
    ],
    validity: "A local approximation about the expansion point.",
  };
  const numericOnly = { ...solutionItem.verification, symbolic_zero: null };
  const series = {
    ...solutionItem,
    verification: numericOnly,
    kind: "series" as const,
    constants: [],
    family: "particular" as const,
    expansion,
  };

  it("describes a series as local, with its order and what was checked", () => {
    render(
      <OdeSolutionPanel
        report={{ ...solution, method: "series", verification: numericOnly, solutions: [series] }}
      />,
    );
    expect(screen.getByText(/Taylor polynomial to order 6 about t = 0/)).toBeInTheDocument();
    expect(
      screen.getByText("A local approximation about the expansion point."),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the residual of the equation falls at the claimed order/),
    ).toBeInTheDocument();
    expect(screen.getByText(/confirmations agree/)).toBeInTheDocument();
    expect(screen.queryByText(/and proved/)).not.toBeInTheDocument();
  });

  it("warns about secular terms and shows the check that failed", () => {
    const secular = {
      ...series,
      kind: "perturbation" as const,
      expansion: {
        ...expansion,
        kind: "regular_perturbation" as const,
        secular_orders: [1, 2],
        checks: [
          ...expansion.checks,
          {
            name: "boundedness" as const,
            passed: false,
            value: null,
            note: "terms grow with the variable",
          },
        ],
      },
    };
    render(
      <OdeSolutionPanel report={{ ...solution, method: "perturbation", solutions: [secular] }} />,
    );
    expect(
      screen.getByText(/Terms that grow with the variable appear at order 1, 2/),
    ).toBeInTheDocument();
    expect(screen.getByText(/only on a bounded interval/)).toBeInTheDocument();
    expect(screen.getByText(/no term grows with the variable/)).toHaveTextContent("✗");
  });

  it("names the periodic method", () => {
    const periodic = {
      ...series,
      kind: "perturbation" as const,
      expansion: { ...expansion, kind: "poincare_lindstedt" as const, point_latex: null },
    };
    render(
      <OdeSolutionPanel report={{ ...solution, method: "perturbation", solutions: [periodic] }} />,
    );
    expect(
      screen.getByText(/Poincaré–Lindstedt expansion \(periodic motion\) to order 6/),
    ).toBeInTheDocument();
  });
});

describe("a boundary-value result", () => {
  it("calls the given values conditions and lists the boundary checks", () => {
    render(
      <OdeNumericPanel
        report={{
          ...numeric,
          method_used: "solve_bvp",
          method_reason:
            "a solution found from the starting guess 'zero'; other solutions may exist",
          initial: [
            { label_latex: "y(0)", value: 0 },
            { label_latex: "y(1)", value: 1 },
          ],
          verification: {
            ...numeric.verification!,
            checks: [
              {
                name: "boundary_conditions",
                status: "passed",
                value: 0,
                tolerance: 1e-6,
                note: "",
              },
              { name: "residual", status: "passed", value: 1e-8, tolerance: 1e-6, note: "" },
              {
                name: "mesh_refinement",
                status: "passed",
                value: 1e-11,
                tolerance: 1e-5,
                note: "",
              },
              { name: "shooting", status: "skipped", value: null, tolerance: null, note: "" },
            ],
          },
        }}
      />,
    );
    expect(screen.getByText("Conditions (yours)")).toBeInTheDocument();
    expect(screen.getByText(/other solutions may exist/)).toBeInTheDocument();
    expect(screen.getByText("The conditions hold at both ends")).toBeInTheDocument();
    expect(screen.getByText("A refined mesh gives the same solution")).toBeInTheDocument();
    expect(screen.getByText("not run")).toBeInTheDocument();
  });
});
