import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DebugPanel } from "./debug-panel";

describe("DebugPanel", () => {
  it("lists the arguments the model sent and what it read back", () => {
    render(
      <DebugPanel
        getReport={() => ""}
        content={{
          text: "",
          trace: [
            {
              id: "t1",
              name: "simplify",
              arguments: { latex: "cos x" },
              for_model: "status=ok",
              duration_ms: 12,
              status: "ok",
            },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Trace · 1 tool call · no timing recorded/)).toBeInTheDocument();
    expect(screen.getByText(/"latex": "cos x"/)).toBeInTheDocument();
    expect(screen.getByText("status=ok")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy this turn" })).toBeInTheDocument();
  });
});
