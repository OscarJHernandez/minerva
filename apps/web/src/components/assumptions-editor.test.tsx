import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AssumptionsEditor } from "./assumptions-editor";

describe("AssumptionsEditor", () => {
  it("adds a valid symbol with a default assumption", async () => {
    const onChange = vi.fn();
    render(<AssumptionsEditor value={{}} onChange={onChange} />);
    await userEvent.type(screen.getByLabelText("Symbol name"), "x");
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(onChange).toHaveBeenCalledWith({ x: ["real"] });
  });

  it("rejects invalid or duplicate symbol names", async () => {
    const onChange = vi.fn();
    render(<AssumptionsEditor value={{ x: ["real"] }} onChange={onChange} />);
    for (const bad of ["1x", "x", "a b", "$"]) {
      await userEvent.clear(screen.getByLabelText("Symbol name"));
      await userEvent.type(screen.getByLabelText("Symbol name"), bad);
      await userEvent.click(screen.getByRole("button", { name: "Add" }));
    }
    expect(onChange).not.toHaveBeenCalled();
  });

  it("toggles and removes assumptions", async () => {
    const onChange = vi.fn();
    render(<AssumptionsEditor value={{ x: ["real"] }} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "positive" }));
    expect(onChange).toHaveBeenLastCalledWith({ x: ["real", "positive"] });
    await userEvent.click(screen.getByRole("button", { name: "real" }));
    expect(onChange).toHaveBeenLastCalledWith({ x: [] });
    await userEvent.click(screen.getByRole("button", { name: /Remove assumptions for x/ }));
    expect(onChange).toHaveBeenLastCalledWith({});
  });
});
