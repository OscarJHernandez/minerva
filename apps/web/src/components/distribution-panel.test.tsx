import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { okDistributionIntegral, okDistributionSimplify } from "@/test/fixtures";
import type { DistributionResult } from "@/lib/types";
import { DistributionPanel } from "./distribution-panel";

describe("DistributionPanel", () => {
  it("shows each step with its handbook identity, conditions and the evidence", () => {
    render(<DistributionPanel result={okDistributionIntegral} />);
    expect(screen.getByText("Evaluated, test-function checked")).toBeInTheDocument();
    expect(screen.getByLabelText("Result")).toBeInTheDocument();
    const links = screen.getAllByRole("link");
    expect(links.map((l) => l.getAttribute("href"))).toEqual([
      "/reference/distributions.delta_composite_simple_roots",
      "/reference/distributions.delta_sifting",
    ]);
    expect(screen.getAllByText("established").length).toBe(2);
    expect(screen.getByText("Established exactly")).toBeInTheDocument();
    expect(
      screen.getByText(/1 of 1 pairings agree \(supporting evidence, not a proof\)/),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("distributional-note")).not.toBeInTheDocument();
  });

  it("says that a distributional result is not a pointwise statement", () => {
    render(<DistributionPanel result={okDistributionSimplify} />);
    expect(screen.getByText("Equal as distributions", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByTestId("distributional-note")).toHaveTextContent(
      /not a pointwise statement/,
    );
    expect(screen.queryByText(/proved/i)).not.toBeInTheDocument();
  });

  it("names the endpoint convention that was used", () => {
    render(<DistributionPanel result={{ ...okDistributionIntegral, endpoint: "half" }} />);
    expect(screen.getByTestId("endpoint-note")).toHaveTextContent("half");
  });

  it("never shows a result for a refusal and names the reason", () => {
    const refusals: [DistributionResult["status"], NonNullable<DistributionResult["reason"]>][] = [
      ["unsupported", "singular_root"],
      ["conditions_not_satisfied", "assumptions_insufficient"],
      ["conditions_not_satisfied", "integration_domain_ambiguous"],
      ["not_verified", "verification_failed"],
      ["resource_limit", "resource_limit"],
    ];
    for (const [status, reason] of refusals) {
      const { unmount } = render(
        <DistributionPanel
          result={{
            ...okDistributionIntegral,
            status,
            reason,
            result_latex: "SHOULD-NOT-APPEAR",
            result_text: "SHOULD-NOT-APPEAR",
            equality: null,
            steps: [],
            message: "Explained here.",
          }}
        />,
      );
      expect(screen.queryByLabelText("Result")).not.toBeInTheDocument();
      expect(screen.queryByText(/SHOULD-NOT-APPEAR/)).not.toBeInTheDocument();
      expect(screen.getByTestId("refusal-reason")).toBeInTheDocument();
      expect(screen.getByText("Explained here.")).toBeInTheDocument();
      unmount();
    }
  });

  it("lists what was not applied", () => {
    render(
      <DistributionPanel
        result={{
          ...okDistributionIntegral,
          status: "conditions_not_satisfied",
          reason: "assumptions_insufficient",
          result_latex: null,
          result_text: null,
          steps: [],
          message: "The slope a is not proved nonzero.",
          blocked: [
            {
              entry_id: "distributions.delta_scaling",
              reason: "assumptions_insufficient",
              detail: "The slope a is not proved nonzero.",
            },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Not applied/)).toBeInTheDocument();
    // said once, not again as the general message
    expect(screen.getAllByText(/The slope a is not proved nonzero\./)).toHaveLength(1);
  });
});
