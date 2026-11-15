import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ParamsInfo } from "@/lib/chat-types";
import { MessageView } from "./message-view";
import { ParamsPanel } from "./params-panel";

const info: ParamsInfo = {
  defaults: {
    num_ctx: 32768,
    num_predict: null,
    temperature: 0.2,
    top_p: null,
    top_k: null,
    think: false,
    history_messages: 20,
    history_tokens: null,
    toolset: "auto",
  },
  editable: ["temperature", "toolset", "num_predict"],
  num_ctx_max: 65536,
};

describe("ParamsPanel", () => {
  it("shows the server default as the hint and sends only what the user changed", () => {
    const onChange = vi.fn();
    render(<ParamsPanel info={info} value={{}} onChange={onChange} />);
    const temperature = screen.getByLabelText("Temperature");
    expect(temperature).toHaveAttribute("placeholder", "0.2");
    fireEvent.change(temperature, { target: { value: "0.7" } });
    expect(onChange).toHaveBeenCalledWith({ temperature: 0.7 });
  });

  it("shows a field this account may not change as read-only", () => {
    render(<ParamsPanel info={info} value={{}} onChange={vi.fn()} />);
    expect(screen.getByLabelText("Context window (tokens)")).toBeDisabled();
    expect(screen.getByLabelText("Longest reply (0: model's own)")).toBeEnabled();
  });

  it("changing the tool set and resetting", () => {
    const onChange = vi.fn();
    const { rerender } = render(<ParamsPanel info={info} value={{}} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("Tools sent"), { target: { value: "core" } });
    expect(onChange).toHaveBeenLastCalledWith({ toolset: "core" });
    expect(screen.getByRole("button", { name: "Reset to server defaults" })).toBeDisabled();
    rerender(<ParamsPanel info={info} value={{ toolset: "core" }} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Reset to server defaults" }));
    expect(onChange).toHaveBeenLastCalledWith({});
  });
});

describe("MessageView: trimmed context", () => {
  const reply = (context_trimmed?: number) => ({
    id: "1",
    role: "assistant" as const,
    created_at: "2026-11-15T00:00:00Z",
    content: { text: "ok", blocks: [], truncated: false, context_trimmed },
  });

  it("says when earlier messages did not fit", () => {
    render(<MessageView message={reply(4)} />);
    expect(screen.getByTestId("context-trimmed")).toHaveTextContent(
      "Earlier messages were left out",
    );
  });

  it("says nothing when everything fit", () => {
    render(<MessageView message={reply(0)} />);
    expect(screen.queryByTestId("context-trimmed")).not.toBeInTheDocument();
  });
});
