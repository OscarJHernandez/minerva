import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ExpressionEditor } from "./expression-editor";

vi.mock("../math-input", () => ({
  MathInput: ({ value }: { value: string }) => (
    <textarea aria-label="Expression" value={value} readOnly />
  ),
}));

const props = {
  latex: "\\frac{x^2-1}{x-1}",
  text: "",
  onLatex: vi.fn(),
  onText: vi.fn(),
  onFormat: vi.fn(),
  onSubmit: vi.fn(),
  continuing: null,
  onClear: vi.fn(),
  focusSignal: 0,
};

describe("ExpressionEditor format drafts", () => {
  it("keeps the LaTeX expression visible when switching to an empty text draft", () => {
    render(<ExpressionEditor {...props} format="text" />);
    expect(screen.getByRole("status")).toHaveTextContent("Your expression is still in LaTeX");
    expect(screen.getByRole("status")).toHaveTextContent("x");
  });

  it("explains that two populated representations are separate drafts", () => {
    render(<ExpressionEditor {...props} text="x+1" format="text" />);
    expect(screen.getByText(/does not convert it/)).toBeInTheDocument();
  });
});
