import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { OdeInput, OdeNumericInput } from "@/lib/ode-types";
import { analysis, numeric, qualitative, request, solution } from "@/test/ode-fixtures";
import { OdeWorkbench } from "./ode-workbench";

const analyze = vi.fn<(r: OdeInput) => Promise<typeof analysis>>();
const solve = vi.fn<(r: OdeInput) => Promise<typeof solution>>();
const numericRun = vi.fn<(r: OdeNumericInput) => Promise<typeof numeric>>();
vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    api: {
      odeAnalyze: (r: OdeInput) => analyze(r),
      odeSolve: (r: OdeInput) => solve(r),
      odeNumeric: (r: OdeNumericInput) => numericRun(r),
    },
  };
});

const noop = () => undefined;
const mount = (props: Partial<React.ComponentProps<typeof OdeWorkbench>> = {}) =>
  render(<OdeWorkbench restore={null} onUnauthorized={noop} onSaved={noop} {...props} />);

async function analyse(equations = "y'' + c y' + y = 0") {
  await userEvent.type(screen.getByLabelText("Equations"), equations);
  await userEvent.type(screen.getByLabelText("Independent variable"), "t");
  await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
}

beforeEach(() => {
  analyze.mockReset().mockResolvedValue(analysis);
  solve.mockReset().mockResolvedValue(solution);
  numericRun.mockReset().mockResolvedValue(numeric);
});

describe("OdeWorkbench", () => {
  it("sends the equation as typed, with the independent variable the user named", async () => {
    mount();
    await analyse();
    await waitFor(() => expect(analyze).toHaveBeenCalledOnce());
    expect(analyze.mock.calls[0]![0]).toEqual({
      equations: ["y'' + c y' + y = 0"],
      format: "latex",
      unknowns: [],
      independent: "t",
      assumptions: {},
      conditions: [],
    });
    expect(
      await screen.findByRole("region", { name: "Differential equation analysis" }),
    ).toBeInTheDocument();
  });

  it("leaves the independent variable empty when the user did not give one", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Equations"), "y' = y");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    await waitFor(() => expect(analyze).toHaveBeenCalledOnce());
    expect(analyze.mock.calls[0]![0].independent).toBeNull();
  });

  it("states what is missing instead of sending an incomplete problem", async () => {
    mount();
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/Write an equation/);
    await userEvent.type(screen.getByLabelText("Equations"), "y' + y");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    expect(screen.getByRole("alert")).toHaveTextContent(/equals sign/);
    expect(analyze).not.toHaveBeenCalled();
  });

  it("offers an action only for the methods that exist, next to each", async () => {
    mount();
    await analyse();
    const region = await screen.findByRole("region", { name: "Differential equation analysis" });
    const closed = region.querySelector('[data-method="symbolic_closed_form"]') as HTMLElement;
    const ivp = region.querySelector('[data-method="numerical_ivp"]') as HTMLElement;
    expect(within(closed).getByRole("button", { name: "Find a closed form" })).toBeInTheDocument();
    expect(within(ivp).getByRole("button", { name: "Solve numerically…" })).toBeInTheDocument();
  });

  it("finds a closed form from the analysed request and stacks the result under the analysis", async () => {
    mount();
    await userEvent.type(screen.getByLabelText("Equations"), "y'' + c y' + y = 0");
    await userEvent.type(screen.getByLabelText("Independent variable"), "t");
    await userEvent.type(screen.getByLabelText(/Conditions/), "y(0) = 1");
    await userEvent.click(screen.getByRole("button", { name: "Analyse" }));
    await userEvent.click(await screen.findByRole("button", { name: "Find a closed form" }));
    await waitFor(() => expect(solve).toHaveBeenCalledOnce());
    expect(solve.mock.calls[0]![0].conditions).toEqual(["y(0) = 1"]);
    expect(await screen.findByRole("region", { name: "Closed-form solution" })).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Differential equation analysis" }),
    ).toBeInTheDocument();
  });

  it("asks for every number before a numerical run and sends what was typed", async () => {
    mount();
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Solve numerically…" }));
    const form = screen.getByRole("form", { name: "Numerical solution" });
    await userEvent.click(within(form).getByRole("button", { name: "Solve and check" }));
    expect(within(form).getByRole("alert")).toHaveTextContent(
      /point where the initial values hold/,
    );
    expect(numericRun).not.toHaveBeenCalled();
    await userEvent.type(within(form).getByLabelText("at t ="), "0");
    await userEvent.type(within(form).getByLabelText("y(t₀)"), "1");
    await userEvent.type(within(form).getByLabelText("y'(t₀)"), "0");
    await userEvent.type(within(form).getByLabelText("c"), "0.5");
    await userEvent.type(within(form).getByLabelText(/t from 0 to/), "10");
    await userEvent.click(within(form).getByRole("button", { name: "Solve and check" }));
    await waitFor(() => expect(numericRun).toHaveBeenCalledOnce());
    expect(numericRun.mock.calls[0]![0]).toMatchObject({
      equations: ["y'' + c y' + y = 0"],
      parameters: { c: 0.5 },
      t_end: 10,
      conditions: [
        { unknown: "y", order: 0, at: "0", value: "1" },
        { unknown: "y", order: 1, at: "0", value: "0" },
      ],
    });
    expect(await screen.findByRole("region", { name: "Numerical solution" })).toBeInTheDocument();
  });

  it("keeps the results of several methods on screen so that they can be compared", async () => {
    mount();
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Find a closed form" }));
    await screen.findByRole("region", { name: "Closed-form solution" });
    await userEvent.click(screen.getByRole("button", { name: "Find a closed form" }));
    await waitFor(() => expect(solve).toHaveBeenCalledTimes(2));
    expect(await screen.findAllByRole("region", { name: "Closed-form solution" })).toHaveLength(2);
    await userEvent.click(screen.getByRole("button", { name: /Clear the results/ }));
    expect(screen.queryByRole("region", { name: /analysis|solution/i })).not.toBeInTheDocument();
  });

  it("shows an error from the server and keeps the form", async () => {
    analyze.mockRejectedValueOnce(new ApiError(429, "Too many requests; slow down."));
    mount();
    await analyse();
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many requests");
    expect(screen.getByLabelText("Equations")).toHaveValue("y'' + c y' + y = 0");
  });

  it("goes to the login page on 401", async () => {
    const onUnauthorized = vi.fn();
    analyze.mockRejectedValueOnce(new ApiError(401, "Not signed in"));
    mount({ onUnauthorized });
    await analyse();
    await waitFor(() => expect(onUnauthorized).toHaveBeenCalledOnce());
  });

  it("reports each saved run so that the sidebar refreshes", async () => {
    const onSaved = vi.fn();
    mount({ onSaved });
    await analyse();
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
  });

  it("loads a saved problem into the form, including structured conditions", async () => {
    const saved: OdeInput = {
      ...request,
      equations: ["x' = -y", "y' = x"],
      independent: null,
      conditions: [{ unknown: "x", order: 0, at: "0", value: "1" }],
      assumptions: { k: ["positive"] },
    };
    mount({ restore: { request: saved, nonce: 1 } });
    expect(screen.getByLabelText("Equations")).toHaveValue("x' = -y\ny' = x");
    expect(screen.getByLabelText("Independent variable")).toHaveValue("");
    expect(screen.getByLabelText(/Conditions/)).toHaveValue("x(0) = 1");
  });

  it("starts from an example", async () => {
    mount();
    await userEvent.selectOptions(
      screen.getByLabelText("Start from an example"),
      screen.getByRole("option", { name: /Nonlinear damped oscillator/ }),
    );
    expect(screen.getByLabelText("Equations")).toHaveValue("y'' + 0.2 y' + y + y^3 = 0");
    expect(screen.getByLabelText("Independent variable")).toHaveValue("t");
  });

  it("offers a closed form's right-hand side to Expression mode", async () => {
    const onUse = vi.fn();
    const particular = {
      ...solution,
      solutions: [
        {
          ...solution.solutions[0]!,
          constants: [],
          family: "particular" as const,
          rhs_text: ["exp(-t)"],
        },
      ],
    };
    solve.mockResolvedValue(particular);
    mount({ onUseExpression: onUse });
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Find a closed form" }));
    await userEvent.click(await screen.findByRole("button", { name: /Use the right-hand side/ }));
    expect(onUse).toHaveBeenCalledWith("exp(-t)", expect.any(String));
  });

  it("opens the equilibria form from the method's action and shows the stability result", async () => {
    const withMethods = {
      ...analysis,
      methods: [
        ...analysis.methods,
        {
          ...analysis.methods[0]!,
          id: "equilibrium_analysis" as const,
          title: "Equilibria",
          rank: 3,
        },
      ],
    };
    analyze
      .mockResolvedValueOnce(withMethods)
      .mockResolvedValueOnce({ ...withMethods, qualitative });
    mount();
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Equilibria and stability…" }));
    const form = screen.getByRole("form", { name: "Equilibria and stability" });
    await userEvent.type(within(form).getByLabelText("c"), "0.5");
    await userEvent.click(within(form).getByRole("button", { name: "Find the equilibria" }));
    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(2));
    expect(analyze.mock.calls[1]![0]).toMatchObject({
      detail: "qualitative",
      parameters: { c: 0.5 },
    });
    expect(await screen.findByText("sink (spiral)")).toBeInTheDocument();
  });

  it("offers a verified closed form of the same equation to compare a numerical run with", async () => {
    const particular = {
      ...solution,
      solutions: [
        {
          ...solution.solutions[0]!,
          constants: [],
          family: "particular" as const,
          rhs_text: ["exp(-t)"],
        },
      ],
    };
    solve.mockResolvedValue(particular);
    mount();
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Find a closed form" }));
    await screen.findByRole("region", { name: "Closed-form solution" });
    await userEvent.click(screen.getByRole("button", { name: "Solve numerically…" }));
    const form = screen.getByRole("form", { name: "Numerical solution" });
    await userEvent.selectOptions(within(form).getByLabelText("Compare with a closed form"), "0");
    await userEvent.type(within(form).getByLabelText("at t ="), "0");
    await userEvent.type(within(form).getByLabelText("y(t₀)"), "1");
    await userEvent.type(within(form).getByLabelText("y'(t₀)"), "0");
    await userEvent.type(within(form).getByLabelText("c"), "0.5");
    await userEvent.type(within(form).getByLabelText(/t from 0 to/), "5");
    await userEvent.click(within(form).getByRole("button", { name: "Solve and check" }));
    await waitFor(() => expect(numericRun).toHaveBeenCalledOnce());
    expect(numericRun.mock.calls[0]![0].compare_with).toEqual(["exp(-t)"]);
  });

  it("does not offer a comparison when there is no closed form to compare with", async () => {
    mount();
    await analyse();
    await userEvent.click(await screen.findByRole("button", { name: "Solve numerically…" }));
    const form = screen.getByRole("form", { name: "Numerical solution" });
    expect(within(form).queryByLabelText("Compare with a closed form")).not.toBeInTheDocument();
  });
});
