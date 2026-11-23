import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Card } from "@/lib/chat-types";
import { okDerivative } from "@/test/fixtures";
import { CardView } from "./card-view";

const entries = (list: unknown[]): Card =>
  ({
    id: "c1",
    tool: "search_handbook",
    type: "entries",
    title: "Handbook: curve of fastest descent",
    data: { query: "curve of fastest descent", entries: list },
  }) as unknown as Card;

describe("CardView, handbook search", () => {
  it("says in one line that nothing matched, with no empty card around it", () => {
    render(<CardView card={entries([])} />);
    const line = screen.getByLabelText("Handbook: curve of fastest descent");
    expect(line).toHaveTextContent("no entry matched");
    expect(line).toHaveTextContent("curve of fastest descent");
    expect(screen.queryByRole("heading", { name: "Handbook" })).not.toBeInTheDocument();
  });
});

it("keeps a derivative card's number and copy actions with the formula they actually reference", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.assign(navigator, { clipboard: { writeText } });
  render(
    <CardView
      card={{
        id: "d1",
        type: "differentiate",
        tool: "time_derivative",
        title: "Total time derivative",
        number: 9,
        data: {
          ...okDerivative,
          zero_on_motion: true,
          on_motion_latex: "0",
          symbolic_zero_on_motion: true,
        },
      }}
    />,
  );
  expect(screen.getByRole("heading", { name: "Total time derivative" })).toBeVisible();
  expect(screen.getByRole("button", { name: "Copy LaTeX" })).not.toBeVisible();
  expect(screen.getByTestId("equation-number")).not.toBeVisible();
  fireEvent.click(screen.getByText("Full derivative before using the equations of motion"));
  expect(screen.getAllByRole("button", { name: "Copy LaTeX" })).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Copy LaTeX" }));
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(okDerivative.result_latex));
  expect(screen.getByTestId("equation-number")).toHaveTextContent("(9)");
});

describe("CardView, qualifications stored on a card", () => {
  const card = (extra: Record<string, unknown>): Card => ({
    id: "q",
    type: "differentiate",
    tool: "time_derivative",
    title: "Total time derivative",
    data: { ...okDerivative, ...extra },
  });
  const restrictions = ["x \\neq 1"];
  const written = "equation (2)";
  const part = "equation (3), a component of an earlier result";
  const noId = "equation (4), a tool result with no result id";

  it("shows inherited restrictions alone, as the stored formula", () => {
    render(<CardView card={card({ inherited_restrictions_latex: restrictions })} />);
    const box = screen.getByLabelText("Inherited conditions");
    expect(box).toBeVisible();
    expect(box.querySelector("annotation")?.textContent).toBe("x \\neq 1");
    expect(screen.queryByLabelText("Input not verified")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Input taken from a list")).not.toBeInTheDocument();
  });

  it("says a model-written input was never verified and checking the computation does not establish it", () => {
    render(<CardView card={card({ unverified_inputs: [written] })} />);
    const box = screen.getByLabelText("Input not verified");
    expect(box).toHaveTextContent("equation (2)");
    expect(box).toHaveTextContent("Minerva never verified");
    expect(box).toHaveTextContent("does not establish the input");
    expect(screen.queryByLabelText("Input taken from a list")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Inherited conditions")).not.toBeInTheDocument();
  });

  it("says a list component's verification and restrictions were not carried forward, never that it was unverified", () => {
    render(<CardView card={card({ unverified_inputs: [part] })} />);
    const box = screen.getByLabelText("Input taken from a list");
    expect(box).toHaveTextContent(part);
    expect(box).toHaveTextContent("verification and restrictions were not carried forward");
    expect(box).not.toHaveTextContent(/never verified/i);
    expect(screen.queryByLabelText("Input not verified")).not.toBeInTheDocument();
  });

  it("says a tool result without a result id was computed by a tool but its verification and restrictions were not carried, never that the model wrote it", () => {
    render(<CardView card={card({ unverified_inputs: [noId] })} />);
    const box = screen.getByLabelText("Input without a result id");
    expect(box).toHaveTextContent(noId);
    expect(box).toHaveTextContent("A tool computed this equation");
    expect(box).toHaveTextContent("does not carry that tool's verification or restrictions");
    expect(box).not.toHaveTextContent(/model wrote/i);
    expect(box).not.toHaveTextContent(/never verified/i);
    expect(screen.queryByLabelText("Input not verified")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Input taken from a list")).not.toBeInTheDocument();
  });

  it("shows all three input kinds together, each entry under its own heading only", () => {
    render(<CardView card={card({ unverified_inputs: [written, part, noId] })} />);
    const w = screen.getByLabelText("Input not verified");
    const p = screen.getByLabelText("Input taken from a list");
    const n = screen.getByLabelText("Input without a result id");
    expect(w).toHaveTextContent("equation (2)");
    expect(w).not.toHaveTextContent("equation (4)");
    expect(p).toHaveTextContent("equation (3)");
    expect(p).not.toHaveTextContent("equation (4)");
    expect(n).toHaveTextContent("equation (4)");
    expect(n).not.toHaveTextContent("equation (2)");
  });

  it("shows every kind together, each entry under its own heading", () => {
    render(
      <CardView
        card={card({
          inherited_restrictions_latex: restrictions,
          unverified_inputs: [written, part],
        })}
      />,
    );
    expect(screen.getByLabelText("Input not verified")).toHaveTextContent("equation (2)");
    expect(screen.getByLabelText("Input not verified")).not.toHaveTextContent("component");
    expect(screen.getByLabelText("Input taken from a list")).toHaveTextContent("equation (3)");
    expect(screen.getByLabelText("Inherited conditions")).toBeVisible();
  });

  it("shows nothing extra for an old card without the fields, or with empty or null ones", () => {
    for (const extra of [
      {},
      { inherited_restrictions_latex: [], unverified_inputs: [] },
      {
        inherited_restrictions_latex: null,
        unverified_inputs: null,
      },
    ]) {
      const { unmount } = render(<CardView card={card(extra)} />);
      expect(screen.queryByLabelText("Inherited conditions")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Input not verified")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Input taken from a list")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Input without a result id")).not.toBeInTheDocument();
      expect(screen.getByRole("heading", { name: "Total time derivative" })).toBeVisible();
      unmount();
    }
  });

  it("shows them on a card of another kind and beside a compact summary", () => {
    const compact = entries([]);
    Object.assign(compact.data, { unverified_inputs: [written] });
    render(<CardView card={compact} />);
    expect(screen.getByLabelText("Input not verified")).toBeVisible();
    expect(screen.getByLabelText("Handbook: curve of fastest descent")).toBeVisible();
  });
});

describe("CardView, a premise carried to a later result", () => {
  it("says the result holds given a premise, never that the model wrote it", () => {
    const premise = "$y^{2} = 1 - x^{2}$, a relation you gave as a premise";
    render(
      <CardView
        card={
          {
            id: "p",
            type: "differentiate",
            tool: "time_derivative",
            title: "Total time derivative",
            data: { ...okDerivative, unverified_inputs: [premise] },
          } as unknown as Card
        }
      />,
    );
    const box = screen.getByLabelText("Holds given a premise");
    expect(box).toHaveTextContent(premise);
    expect(box).toHaveTextContent("given, not proved");
    expect(box).not.toHaveTextContent(/model wrote/i);
    expect(screen.queryByLabelText("Input not verified")).not.toBeInTheDocument();
  });
});
