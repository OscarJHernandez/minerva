import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card, MvnReason, MvnResult } from "@/lib/chat-types";
import { CardView } from "./card-view";
import { MVN_REASON_TEXT, MvnCard } from "./mvn-card";

const ok: MvnResult = {
  status: "ok",
  action: "density",
  reason: null,
  message: "numerically checked",
  variables: ["x", "y"],
  mean_text: ["1", "-2"],
  mean_latex: ["1", "-2"],
  covariance_text: [
    ["4", "1"],
    ["1", "2"],
  ],
  covariance_latex: [
    ["4", "1"],
    ["1", "2"],
  ],
  density_text: "",
  density_latex: "\\frac{1}{2 \\pi \\sqrt{7}}",
  correlation_text: "",
  correlation_latex: "",
  input_mean_text: ["1", "-2"],
  input_covariance_text: [
    ["4", "1"],
    ["1", "2"],
  ],
  checks: [
    {
      kind: "normalisation",
      method: "SciPy quadrature of the printed density over the plane",
      points_checked: 1,
      points_agreed: 1,
      passed: true,
      note: "",
    },
  ],
  conditions: [],
  premises: [],
  evidence: "identity",
  duration_ms: 10,
};

describe("MvnCard", () => {
  it("shows the mean, the covariance, the density and how many points agreed", () => {
    render(<MvnCard result={ok} />);
    const text = document.body.textContent ?? "";
    expect(text).toContain("Numerically checked");
    expect(screen.getByTestId("distribution")).toBeInTheDocument();
    expect(screen.getByLabelText("Mean")).toBeInTheDocument();
    expect(screen.getByLabelText("Covariance")).toBeInTheDocument();
    expect(screen.getByLabelText("Density")).toBeInTheDocument();
    expect(screen.getByTestId("checks")).toHaveTextContent(/1 of 1 agree/);
    expect(screen.getByTestId("checks")).toHaveTextContent(/not a proof/);
    expect(screen.queryByTestId("premises")).not.toBeInTheDocument();
    expect(screen.queryByTestId("assumed-note")).not.toBeInTheDocument();
    expect(text).not.toContain("proved");
  });

  it("the moments card shows the correlation", () => {
    render(
      <MvnCard
        result={{ ...ok, action: "moments", density_latex: "", correlation_latex: "\\frac{1}{2}" }}
      />,
    );
    expect(screen.getByLabelText("Correlation")).toBeInTheDocument();
    expect(screen.queryByLabelText("Density")).not.toBeInTheDocument();
  });

  it("a one-dimensional result is a variance, not a matrix", () => {
    render(
      <MvnCard
        result={{
          ...ok,
          action: "marginal",
          variables: ["x"],
          mean_latex: ["1"],
          covariance_latex: [["4"]],
          density_latex: "",
        }}
      />,
    );
    expect(document.body.textContent).toContain("Marginal distribution".toLowerCase());
    expect(screen.getByLabelText("Covariance").innerHTML).not.toContain("pmatrix");
  });

  it("a result that holds given a condition says so and lists the condition", () => {
    const conditional: MvnResult = {
      ...ok,
      evidence: "conditional",
      premises: ["\\det\\Sigma = s_1 s_2 - c^2 > 0 (not established by the declared assumptions)"],
      conditions: [
        {
          latex: "\\det\\Sigma = s_{1} s_{2} - c^{2} > 0",
          established: false,
          how: "positive definiteness is not decided by the declared assumptions",
        },
      ],
    };
    render(<MvnCard result={conditional} />);
    expect(document.body.textContent).toContain("Holds given a premise");
    expect(screen.getByTestId("premises")).toHaveTextContent(/Holds only given/);
    expect(screen.getByTestId("premises")).toHaveTextContent(/not decided by the declared/);
  });

  it("a mean and covariance the user never wrote are labelled as the assistant's", () => {
    render(
      <MvnCard
        result={ok}
        provenance={{
          who: "assistant",
          values: ["0"],
          label: "mean and covariance assumed by the assistant",
        }}
      />,
    );
    const note = screen.getByTestId("assumed-note");
    expect(note).toHaveTextContent(/assumed by the assistant/);
    expect(note).toHaveTextContent(/Not written by you: 0/);
  });

  it("values the user wrote carry no label", () => {
    render(<MvnCard result={ok} provenance={{ who: "user", values: [], label: null }} />);
    expect(screen.queryByTestId("assumed-note")).not.toBeInTheDocument();
  });

  it("every reason has plain words, and a refusal shows no formula of a result", () => {
    const reasons: MvnReason[] = [
      "dimension_unsupported",
      "not_symmetric",
      "not_positive_definite",
      "singular_covariance",
      "map_singular",
      "variables_missing",
      "index_invalid",
      "parse_error",
      "verification_failed",
      "resource_limit",
    ];
    expect(Object.keys(MVN_REASON_TEXT).sort()).toEqual([...reasons].sort());
    for (const reason of reasons) {
      const { unmount } = render(
        <MvnCard
          result={{
            ...ok,
            status: "unsupported",
            reason,
            message: "the engine's own words",
            variables: [],
            mean_latex: [],
            covariance_latex: [],
            density_latex: "",
            checks: [],
            evidence: null,
          }}
        />,
      );
      const refusal = screen.getByTestId("refusal");
      expect(refusal).toHaveTextContent(MVN_REASON_TEXT[reason]);
      expect(refusal).toHaveTextContent(/No result is shown/);
      expect(screen.queryByTestId("distribution")).not.toBeInTheDocument();
      expect(screen.queryByTestId("checks")).not.toBeInTheDocument();
      unmount();
    }
  });

  it("is drawn for a card of type multivariate_normal", () => {
    const card: Card = {
      id: "c1",
      tool: "multivariate_normal",
      type: "multivariate_normal",
      title: "Bivariate Normal",
      data: { result: ok, request: {} },
    };
    render(<CardView card={card} />);
    expect(screen.getByRole("region")).toBeInTheDocument();
    expect(screen.getByTestId("distribution")).toBeInTheDocument();
  });
});
