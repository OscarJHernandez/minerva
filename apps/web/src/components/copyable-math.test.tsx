import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FormulaActionsProvider } from "@/components/chat/formula-actions";
import { CopyMath } from "./copyable-math";

describe("CopyMath", () => {
  it("copies the exact LaTeX of a displayed result and can send it to the chat box", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    const editInChat = vi.fn();
    const latex = "C3 = -GM\\cos(\\theta) + r^{3}\\dot{\\theta}^{2}\\cos(\\theta)";
    render(
      <FormulaActionsProvider value={{ editInChat }}>
        <CopyMath latex={latex} display />
      </FormulaActionsProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Copy LaTeX" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(latex));
    fireEvent.click(screen.getByRole("button", { name: "Edit in chat" }));
    expect(editInChat).toHaveBeenCalledWith(latex);
  });

  it("adds no buttons to an inline formula", () => {
    render(<CopyMath latex="x+1" />);
    expect(screen.queryByRole("button")).toBeNull();
  });
});
