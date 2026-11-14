import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { OdeInput, OdeNumericInput } from "@/lib/ode-types";
import { analysis, numeric, request, solution } from "@/test/ode-fixtures";
import { CardView } from "./card-view";
import { OdeAnalysisCard } from "./ode-card";
import { MotionRunnerProvider } from "./motion-runner";

const numericForm = () => within(screen.getByRole("form", { name: "Numerical solution" }));
const symbolicForm = () => within(screen.getByRole("form", { name: "Closed-form solution" }));

function withRunner(
  node: React.ReactNode,
  ode: {
    symbolic?: (r: OdeInput) => Promise<void>;
    numeric?: (r: OdeNumericInput) => Promise<void>;
  } = {},
) {
  const symbolic = vi.fn<(r: OdeInput) => Promise<void>>(ode.symbolic ?? (async () => {}));
  const numericRun = vi.fn<(r: OdeNumericInput) => Promise<void>>(ode.numeric ?? (async () => {}));
  render(
    <MotionRunnerProvider value={{ run: vi.fn(), ode: { symbolic, numeric: numericRun } }}>
      {node}
    </MotionRunnerProvider>,
  );
  return { symbolic, numeric: numericRun };
}

describe("OdeAnalysisCard", () => {
  it("offers no form outside a conversation", () => {
    render(<OdeAnalysisCard report={analysis} request={request} />);
    expect(screen.queryByRole("form", { name: "Numerical solution" })).not.toBeInTheDocument();
    expect(screen.queryByRole("form", { name: "Closed-form solution" })).not.toBeInTheDocument();
    expect(screen.getByText(/As Minerva read it/)).toBeInTheDocument();
  });

  it("offers the forms of the methods that apply, and only those", () => {
    withRunner(<OdeAnalysisCard report={analysis} request={request} />);
    expect(screen.getByText(/checked by substituting it back/)).toBeInTheDocument();
    expect(screen.getByText(/Solve numerically/)).toBeInTheDocument();
    const methods = analysis.methods.map((m) =>
      m.id === "numerical_ivp" ? { ...m, applicable: false } : m,
    );
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), ode: { symbolic: vi.fn(), numeric: vi.fn() } }}>
        <OdeAnalysisCard report={{ ...analysis, methods }} request={request} />
      </MotionRunnerProvider>,
    );
    expect(screen.getAllByText(/Solve numerically/)).toHaveLength(1);
  });

  it("offers nothing to run when the equation could not be read", () => {
    withRunner(
      <OdeAnalysisCard
        report={{
          ...analysis,
          status: "parse_error",
          message: "no",
          interpretation: null,
          methods: [],
        }}
        request={request}
      />,
    );
    expect(screen.queryByText(/Solve numerically/)).not.toBeInTheDocument();
  });

  it("refuses to submit a missing number and names it, then sends exactly what was typed", async () => {
    const { numeric: run } = withRunner(<OdeAnalysisCard report={analysis} request={request} />);
    await userEvent.click(screen.getByText(/Solve numerically/));
    const submit = numericForm().getByRole("button", { name: "Solve and check" });
    await userEvent.click(submit);
    expect(screen.getByRole("alert")).toHaveTextContent("point where the initial values hold");
    expect(run).not.toHaveBeenCalled();

    await userEvent.type(numericForm().getByLabelText("at t ="), "0");
    await userEvent.type(numericForm().getByLabelText("y(t₀)"), "1");
    await userEvent.click(submit);
    expect(screen.getByRole("alert")).toHaveTextContent("initial value of y'");
    await userEvent.type(numericForm().getByLabelText("y'(t₀)"), "0");
    await userEvent.click(submit);
    expect(screen.getByRole("alert")).toHaveTextContent("parameter c");
    await userEvent.type(numericForm().getByLabelText("c"), "0.5");
    await userEvent.click(submit);
    expect(screen.getByRole("alert")).toHaveTextContent("end of the interval");
    await userEvent.type(numericForm().getByLabelText(/t from 0 to/), "10");
    await userEvent.click(submit);
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls[0]?.[0]).toMatchObject({
      equations: request.equations,
      independent: "t",
      conditions: [
        { unknown: "y", order: 0, at: "0", value: "1" },
        { unknown: "y", order: 1, at: "0", value: "0" },
      ],
      parameters: { c: 0.5 },
      t_end: 10,
      method: "auto",
    });
  });

  it("does not ask again for initial values the user typed with the equation", async () => {
    withRunner(
      <OdeAnalysisCard
        report={analysis}
        request={{ ...request, conditions: ["y(0) = 1", "y'(0) = 0"] }}
      />,
    );
    await userEvent.click(screen.getByText(/Solve numerically/));
    expect(numericForm().getByLabelText("at t =")).toHaveValue("0");
    expect(numericForm().getByLabelText("y(t₀)")).toHaveValue("1");
    expect(numericForm().getByLabelText("y'(t₀)")).toHaveValue("0");
    expect(numericForm().getByLabelText("c")).toHaveValue("");
  });

  it("shows the server's own message when the run fails", async () => {
    const run = async () => {
      throw new ApiError(429, "Too many requests; slow down.");
    };
    withRunner(
      <OdeAnalysisCard
        report={analysis}
        request={{ ...request, conditions: ["y(0) = 1", "y'(0) = 0"] }}
      />,
      { numeric: run },
    );
    await userEvent.click(screen.getByText(/Solve numerically/));
    await userEvent.type(numericForm().getByLabelText("c"), "0.5");
    await userEvent.type(numericForm().getByLabelText(/t from 0 to/), "5");
    await userEvent.click(numericForm().getByRole("button", { name: "Solve and check" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many requests");
    expect(numericForm().getByRole("button", { name: "Solve and check" })).toBeEnabled();
  });

  it("asks for a closed form with no conditions unless some are filled in", async () => {
    const { symbolic } = withRunner(<OdeAnalysisCard report={analysis} request={request} />);
    await userEvent.click(screen.getByText(/checked by substituting it back/));
    await userEvent.click(symbolicForm().getByRole("button", { name: "Find a closed form" }));
    expect(symbolic).toHaveBeenCalledWith({ ...request, conditions: [] });
    await userEvent.type(symbolicForm().getByLabelText("y(t₀)"), "1");
    await userEvent.click(symbolicForm().getByRole("button", { name: "Find a closed form" }));
    expect(screen.getByRole("alert")).toHaveTextContent("point where the initial values hold");
  });
});

describe("CardView for differential equations", () => {
  it("renders each of the three cards", () => {
    const base = { id: "a", tool: "ode", title: "t" };
    const { rerender } = render(
      <CardView card={{ ...base, type: "ode_analysis", data: { report: analysis, request } }} />,
    );
    expect(
      screen.getByRole("region", { name: "Differential equation analysis" }),
    ).toBeInTheDocument();
    rerender(
      <CardView card={{ ...base, type: "ode_solution", data: { report: solution, request } }} />,
    );
    expect(screen.getByRole("region", { name: "Closed-form solution" })).toBeInTheDocument();
    rerender(
      <CardView card={{ ...base, type: "ode_numeric", data: { report: numeric, request } }} />,
    );
    expect(screen.getByRole("region", { name: "Numerical solution" })).toBeInTheDocument();
  });
});
