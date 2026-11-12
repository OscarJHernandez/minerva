import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api";
import type { PerturbationReport, PerturbationRequest } from "@/lib/chat-types";
import { MotionRunnerProvider } from "./motion-runner";
import { PerturbationCard } from "./perturbation-card";

const base: PerturbationReport = {
  status: "ok",
  lagrangian_latex: "L",
  coordinates: ["x"],
  equilibrium: [{ name: "x", latex: "0", name_latex: "x" }],
  mode: "parameter",
  parameters: [],
  small_latex: "b",
  amplitude_latex: "A",
  order: 2,
  linear_frequency_latex: "\\omega_0 = 1",
  frequency_latex: "\\omega = 1 + \\frac{3 b A^{2}}{8}",
  period_ratio_latex: "\\frac{T}{T_0} = 1 - \\frac{3 A^{2} b}{8} + O\\left(b^{3}\\right)",
  solution_latex: "u(t) = A \\cos{\\left(\\omega t \\right)} + O\\left(b^{3}\\right)",
  orders: [
    {
      order: 0,
      frequency_latex: null,
      shape_latex: "w_{0}(\\tau) = A \\cos{\\left(\\tau \\right)}",
    },
    {
      order: 1,
      frequency_latex: "\\omega_{1} = \\frac{3 A^{2}}{8}",
      shape_latex: "w_{1}(\\tau) = 0",
    },
  ],
  verification: {
    residual_method: "",
    residual_passed: true,
    parameter_sets_checked: 4,
    initial_conditions_passed: true,
    bounded: true,
    passed: true,
    note: "",
  },
  comparison: null,
  notes: [],
  message: "Poincare-Lindstedt solution to order 2.",
  duration_ms: 20,
};

const comparison = {
  status: "ok" as const,
  message: "",
  epsilon: 0.1,
  predicted_order: 3,
  t: [0, 1, 2],
  numerical: [1, 0.5, -0.3],
  perturbative: [1, 0.5, -0.31],
  max_error: 0.00106,
  max_error_half: 0.00014,
  observed_order: 2.92,
  scaling: "max |u_numerical - u_perturbative| over the samples",
  notes: ["These are measurements on this span for these numbers."],
};

describe("PerturbationCard", () => {
  it("shows the verified frequency, period, solution and the checks", () => {
    render(<PerturbationCard report={base} />);
    expect(screen.getByText("Solved and verified")).toBeInTheDocument();
    expect(screen.getByLabelText("Frequency")).toBeInTheDocument();
    expect(screen.getByLabelText("Period ratio")).toBeInTheDocument();
    expect(screen.getByLabelText("Solution")).toBeInTheDocument();
    expect(
      screen.getByText(/falls at the claimed order in the unexpanded equation/),
    ).toBeInTheDocument();
    expect(screen.getByText(/none: every order is a finite sum/)).toBeInTheDocument();
    expect(screen.getByText(/Each order \(2\)/)).toBeInTheDocument();
  });

  it("asks for numbers instead of comparing when none were given", () => {
    render(<PerturbationCard report={base} />);
    expect(
      screen.queryByLabelText("Comparison with the numerical solution"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText(/give parameter values, the small parameter or amplitude/),
    ).toBeInTheDocument();
    expect(screen.getByText(/local series in the small parameter/)).toBeInTheDocument();
  });

  it("shows the comparison as a measurement with its observed order and a plot", () => {
    render(<PerturbationCard report={{ ...base, comparison }} />);
    const region = screen.getByLabelText("Comparison with the numerical solution");
    expect(region).toHaveTextContent("2.9 (predicted 3)");
    expect(region).toHaveTextContent("1.06e-3");
    expect(
      screen.getByLabelText("Perturbative and numerical solutions against time"),
    ).toBeInTheDocument();
    expect(screen.getByText(/measurements on this span/)).toBeInTheDocument();
    expect(screen.queryByText(/give parameter values/)).not.toBeInTheDocument();
  });

  it("shows no comparison when the numerical run did not pass its own checks", () => {
    render(
      <PerturbationCard
        report={{
          ...base,
          comparison: {
            ...comparison,
            status: "resource_limit",
            message: "The numerical solution did not pass its own checks.",
            t: [],
            numerical: [],
            perturbative: [],
            max_error: null,
            max_error_half: null,
            observed_order: null,
          },
        }}
      />,
    );
    expect(
      screen.getByText(/No comparison is shown: The numerical solution did not pass/),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText("Perturbative and numerical solutions against time"),
    ).not.toBeInTheDocument();
  });

  it("never shows formulas for a refused or unverified solution", () => {
    for (const status of [
      "unsupported",
      "not_verified",
      "parse_error",
      "resource_limit",
    ] as const) {
      const { unmount } = render(
        <PerturbationCard
          report={{
            ...base,
            status,
            message: "Explained here.",
            comparison,
            verification: { ...base.verification!, residual_passed: false, passed: false },
          }}
        />,
      );
      expect(screen.queryByLabelText("Frequency")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Period ratio")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Solution")).not.toBeInTheDocument();
      expect(screen.queryByText(/Each order/)).not.toBeInTheDocument();
      expect(
        screen.queryByLabelText("Comparison with the numerical solution"),
      ).not.toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      expect(screen.getByText(/does not fall at the claimed order/)).toBeInTheDocument();
      unmount();
    }
  });
});

describe("PerturbationCard comparison form", () => {
  const request = {
    lagrangian: "L",
    coordinates: ["x"],
    time: "t",
    assumptions: {},
    small_parameter: "b",
    equilibrium: "0",
    order: 2,
  };
  const withParameters = { ...base, parameters: ["k"] };

  it("offers no form outside a conversation", () => {
    render(<PerturbationCard report={withParameters} request={request} />);
    expect(screen.queryByRole("form", { name: "Comparison numbers" })).not.toBeInTheDocument();
  });

  it("offers no form for a refused solution", () => {
    const perturb = vi.fn<(r: PerturbationRequest) => Promise<void>>().mockResolvedValue();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), perturb }}>
        <PerturbationCard report={{ ...withParameters, status: "unsupported" }} request={request} />
      </MotionRunnerProvider>,
    );
    expect(screen.queryByRole("form", { name: "Comparison numbers" })).not.toBeInTheDocument();
  });

  it("refuses to submit with a missing number, then sends the typed values", async () => {
    const perturb = vi.fn<(r: PerturbationRequest) => Promise<void>>().mockResolvedValue();
    const user = userEvent.setup();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), perturb }}>
        <PerturbationCard report={withParameters} request={request} />
      </MotionRunnerProvider>,
    );
    await user.click(screen.getByText(/Compare with the numerical solution/));
    await user.click(screen.getByRole("button", { name: "Compare" }));
    expect(screen.getByRole("alert")).toHaveTextContent("Give a value for k.");
    expect(perturb).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText("k"), "1");
    await user.type(screen.getByLabelText("b (the small parameter)"), "0.1");
    await user.type(screen.getByLabelText("initial displacement A"), "1");
    await user.type(screen.getByLabelText("end time"), "30");
    await user.click(screen.getByRole("button", { name: "Compare" }));
    expect(perturb).toHaveBeenCalledTimes(1);
    expect(perturb.mock.calls[0]![0]).toEqual({
      ...request,
      compare: { parameters: { k: 1 }, epsilon: 0.1, amplitude: 1, t_end: 30 },
    });
  });

  it("shows the server's refusal instead of failing silently", async () => {
    const perturb = vi
      .fn<(r: PerturbationRequest) => Promise<void>>()
      .mockRejectedValue(new ApiError(429, "Too many requests; slow down."));
    const user = userEvent.setup();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), perturb }}>
        <PerturbationCard report={{ ...base, parameters: [] }} request={request} />
      </MotionRunnerProvider>,
    );
    await user.click(screen.getByText(/Compare with the numerical solution/));
    await user.type(screen.getByLabelText("b (the small parameter)"), "0.1");
    await user.type(screen.getByLabelText("initial displacement A"), "1");
    await user.type(screen.getByLabelText("end time"), "5");
    await user.click(screen.getByRole("button", { name: "Compare" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many requests; slow down.");
  });

  it("asks for the displacement itself when the small parameter is the amplitude", async () => {
    const perturb = vi.fn<(r: PerturbationRequest) => Promise<void>>().mockResolvedValue();
    const user = userEvent.setup();
    render(
      <MotionRunnerProvider value={{ run: vi.fn(), perturb }}>
        <PerturbationCard
          report={{ ...base, mode: "amplitude", parameters: [] }}
          request={{ ...request, small_parameter: null }}
        />
      </MotionRunnerProvider>,
    );
    expect(screen.queryByLabelText("initial displacement A")).not.toBeInTheDocument();
    await user.click(screen.getByText(/Compare with the numerical solution/));
    await user.type(screen.getByLabelText("initial displacement"), "0.3");
    await user.type(screen.getByLabelText("end time"), "5");
    await user.click(screen.getByRole("button", { name: "Compare" }));
    expect(perturb.mock.calls[0]![0].compare).toEqual({ parameters: {}, epsilon: 0.3, t_end: 5 });
  });
});
