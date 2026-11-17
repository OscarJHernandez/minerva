import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { StoredMessage, TurnDebug } from "@/lib/chat-types";
import { conversationUsage, formatGB, memoryLine } from "@/lib/usage";
import { ContextMeter } from "./context-meter";

type Call = { prompt_tokens?: number; completion_tokens?: number };

function reply(calls: Call[], over: Partial<TurnDebug> & { trimmed?: number } = {}): StoredMessage {
  const { trimmed, ...debug } = over;
  return {
    id: Math.random().toString(),
    role: "assistant",
    created_at: "2026-01-01T00:00:00Z",
    content: {
      text: "ok",
      blocks: [],
      truncated: false,
      context_trimmed: trimmed ?? 0,
      debug: {
        provider: "ollama",
        model: "m",
        context_window: 10000,
        llm_calls: calls.map((c) => ({ ms: 1, text_chars: 1, tool_calls: 0, ...c })),
        ...debug,
      } as unknown as TurnDebug,
    },
  };
}

describe("conversation usage", () => {
  it("uses the provider's last prompt count and sums every reply", () => {
    const u = conversationUsage([
      reply([{ prompt_tokens: 100, completion_tokens: 10 }]),
      reply([
        { prompt_tokens: 200, completion_tokens: 20 },
        { prompt_tokens: 300, completion_tokens: 5 },
      ]),
    ]);
    expect(u?.used).toBe(300);
    expect(u?.estimated).toBe(false);
    expect(u?.window).toBe(10000);
    expect(u?.lastTurn).toEqual({ prompt: 500, completion: 25, unreported: 0 });
    expect(u?.total).toEqual({ prompt: 600, completion: 35, unreported: 0 });
  });

  it("labels the estimate and never invents a count the provider did not report", () => {
    const u = conversationUsage([reply([{}], { prompt_estimate_tokens: 4200 })]);
    expect(u?.used).toBe(4200);
    expect(u?.estimated).toBe(true);
    expect(u?.total).toEqual({ prompt: 0, completion: 0, unreported: 1 });
    expect(conversationUsage([reply([{}])])?.used).toBeNull();
  });

  it("has no window for a hosted model", () => {
    const u = conversationUsage([reply([{ prompt_tokens: 50 }], { provider: "openrouter" })]);
    expect(u?.window).toBeNull();
  });

  it("is null before a reply carries data", () => {
    expect(conversationUsage([])).toBeNull();
  });

  it("formats sizes only when the host reported them", () => {
    expect(formatGB(6_200_000_000)).toBe("6.2 GB");
    expect(formatGB(undefined)).toBeNull();
    expect(formatGB(0)).toBeNull();
    expect(memoryLine({ name: "a", recommended: false, note: "" })).toBeNull();
    expect(
      memoryLine({
        name: "a",
        recommended: false,
        note: "",
        size_bytes: 9e9,
        size_vram_bytes: 6e9,
        loaded: true,
      }),
    ).toBe("9.0 GB in memory (6.0 GB on the GPU)");
    expect(memoryLine({ name: "a", recommended: false, note: "", size_bytes: 4e9 })).toBe(
      "4.0 GB on disk, not loaded",
    );
  });
});

describe("ContextMeter", () => {
  it("shows the window and what the last request used", () => {
    render(<ContextMeter messages={[reply([{ prompt_tokens: 2500, completion_tokens: 7 }])]} />);
    expect(screen.getByTestId("context-label")).toHaveTextContent("2,500 / 10,000 tokens");
    expect(screen.getByRole("meter")).toHaveAttribute("aria-valuenow", "2500");
    expect(screen.queryByTestId("context-usage-warning")).toBeNull();
  });

  it("says estimate when the count is an estimate", () => {
    render(<ContextMeter messages={[reply([{}], { prompt_estimate_tokens: 1000 })]} />);
    expect(screen.getByTestId("context-label")).toHaveTextContent("(estimate)");
  });

  it("warns near the window and when messages were left out", () => {
    const { rerender } = render(<ContextMeter messages={[reply([{ prompt_tokens: 8500 }])]} />);
    expect(screen.getByTestId("context-usage-warning")).toHaveTextContent(/close to/);
    rerender(<ContextMeter messages={[reply([{ prompt_tokens: 100 }], { trimmed: 4 })]} />);
    expect(screen.getByTestId("context-usage-warning")).toHaveTextContent(/4 earlier messages/);
    rerender(<ContextMeter messages={[reply([{ prompt_tokens: 100 }])]} />);
    expect(screen.queryByTestId("context-usage-warning")).toBeNull();
  });

  it("renders nothing without data", () => {
    const { container } = render(<ContextMeter messages={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
