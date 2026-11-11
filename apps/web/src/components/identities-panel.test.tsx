import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { identities, match } from "@/test/fixtures";
import { IdentitiesPanel } from "./identities-panel";

describe("IdentitiesPanel", () => {
  it("groups matches by applicability and explains unmet conditions", () => {
    render(
      <IdentitiesPanel
        result={identities([
          match("a.ok", "applicable", {
            matched_subexpression_latex: "\\log(xy)",
            rewritten_latex: "\\log x + \\log y",
          }),
          match("b.blocked", "conditions_not_satisfied", {
            matched_subexpression_latex: "\\log(xy)",
            unmet_conditions: [
              { variable: "a", assumption: "positive" },
              { variable: "b", assumption: "positive" },
            ],
          }),
          match("c.related", "no_match", { message: "Related, but the pattern does not occur." }),
        ])}
      />,
    );
    expect(screen.getByRole("heading", { name: /Applicable/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /Needs assumptions/ })).toBeInTheDocument();
    expect(screen.getByText(/Related, not applicable/)).toBeInTheDocument(); // collapsed group
    expect(screen.getByText(/Rewritten expression, verified/)).toBeInTheDocument();
    expect(screen.getByText(/not applied: these conditions cannot be shown/)).toBeInTheDocument();
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(within(items[0]!).getByText(/must be positive/)).toBeInTheDocument();
  });

  it("does not present non-applicable entries as results", () => {
    render(<IdentitiesPanel result={identities([match("c.related", "no_match")])} />);
    expect(screen.getByText(/No identity from the knowledge base applies/)).toBeInTheDocument();
    expect(screen.queryByText(/Rewritten expression, verified/)).not.toBeInTheDocument();
  });

  it("links entries to the reference", () => {
    render(<IdentitiesPanel result={identities([match("log.product", "no_match")])} />);
    expect(screen.getByRole("link", { name: "Title of log.product" })).toHaveAttribute(
      "href",
      "/reference/log.product",
    );
  });
});
