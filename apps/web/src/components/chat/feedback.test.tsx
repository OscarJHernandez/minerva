import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Segment, StoredMessage } from "@/lib/chat-types";
import { replyToMarkdown } from "@/lib/reply-markdown";
import { TOOL_LABEL } from "@/lib/tool-labels";
import { Composer } from "./composer";
import { MarksLegend } from "./marks-legend";
import { MessageView } from "./message-view";
import { type Pending, PendingView, statusText } from "./pending-view";
import { RichText } from "./rich-text";

const seg = (type: Segment["type"], value: string, status: Segment["status"] = null): Segment => ({
  type,
  value,
  status,
});

const reply = (content: Partial<StoredMessage["content"]> & { text?: string }): StoredMessage => ({
  id: "m",
  role: "assistant",
  created_at: "2026-11-15T00:00:00Z",
  content: { text: "", blocks: [], truncated: false, ...content },
});

describe("marks are readable without a mouse", () => {
  it("opens the explanation of an unchecked formula from its mark", () => {
    render(<RichText segments={[seg("math", "x = 1", "unchecked")]} />);
    expect(screen.queryByTestId("mark-explanation")).not.toBeInTheDocument();
    const mark = screen.getByRole("button", { name: /unchecked: what this mark means/ });
    expect(mark).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(mark);
    expect(mark).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("mark-explanation")).toHaveTextContent("could not check it");
    fireEvent.click(mark);
    expect(screen.queryByTestId("mark-explanation")).not.toBeInTheDocument();
  });

  it("copies the LaTeX of a displayed formula", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    render(<RichText segments={[seg("display_math", "x^2 + 1", "from_tool")]} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy LaTeX" }));
    expect(writeText).toHaveBeenCalledWith("x^2 + 1");
    expect(await screen.findByText("Copied")).toBeInTheDocument(); // the label shows it; the name stays
  });

  it("the legend explains every mark in the words of the marks themselves", () => {
    render(<MarksLegend onClose={() => undefined} />);
    for (const word of ["quoted", "checked", "conditional", "contradicted", "unchecked"]) {
      expect(screen.getAllByText(new RegExp(word)).length).toBeGreaterThan(0);
    }
    expect(screen.getByText(/says nothing about whether it was verified/)).toBeInTheDocument();
  });

  it("the composer links to the legend", () => {
    render(<Composer disabled={false} busy={false} onSend={vi.fn()} onStop={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "What do the marks mean?" }));
    expect(screen.getByRole("region", { name: "What the marks mean" })).toBeInTheDocument();
  });
});

describe("a reply tells what it is", () => {
  const withFormula = [
    { kind: "text" as const, segments: [seg("math", "x = 1", "unchecked")], card: null },
  ];

  it("says when no tool was run for an answer that contains formulas", () => {
    render(<MessageView message={reply({ blocks: withFormula, trace: [] })} />);
    expect(screen.getByTestId("no-tool")).toHaveTextContent("No tool was run");
  });

  it("says nothing of the kind when a tool ran, for plain prose, or for an older message", () => {
    const traced = reply({
      blocks: withFormula,
      trace: [{ id: "t", name: "simplify", arguments: {}, for_model: "" }],
    });
    const { rerender } = render(<MessageView message={traced} />);
    expect(screen.queryByTestId("no-tool")).not.toBeInTheDocument();
    const prose = [
      { kind: "text" as const, segments: [seg("text", "Which variable?")], card: null },
    ];
    rerender(<MessageView message={reply({ blocks: prose, trace: [] })} />);
    expect(screen.queryByTestId("no-tool")).not.toBeInTheDocument();
    rerender(<MessageView message={reply({ blocks: withFormula })} />); // stored before traces existed
    expect(screen.queryByTestId("no-tool")).not.toBeInTheDocument();
  });

  it("marks a reply that was stopped", () => {
    render(<MessageView message={reply({ stop_reason: "stopped", truncated: true })} />);
    expect(screen.getByTestId("stopped")).toBeInTheDocument();
  });

  it("copies the reply as Markdown with its equation numbers", () => {
    const blocks = [
      {
        kind: "text" as const,
        segments: [
          seg("text", "So "),
          seg("math", "a", "unchecked"),
          seg("text", " and:"),
          { ...seg("display_math", "b = c", "unchecked"), number: 2 },
        ],
        card: null,
      },
      {
        kind: "card" as const,
        segments: [],
        card: {
          id: "c",
          tool: "simplify",
          type: "error" as const,
          title: "t",
          data: { message: "m", result_latex: "x + 1" },
          number: 3,
        },
      },
    ];
    expect(replyToMarkdown({ blocks })).toBe("So $a$ and:\n\n$$b = c$$ (2)\n\n$$x + 1$$ (3)");
    expect(replyToMarkdown({ blocks: [] })).toBe("");
  });
});

describe("progress while a reply arrives", () => {
  const pending = (over: Partial<Pending> = {}): Pending => ({
    items: [],
    tools: [],
    cards: 0,
    startedAt: Date.now(),
    ...over,
  });

  it("names the tool that is running, with a label for every tool a chat can call", () => {
    const running = pending({ tools: [{ id: "1", name: "differentiate", arguments: {} }] });
    expect(statusText(running, 1)).toBe(`${TOOL_LABEL.differentiate}…`);
    for (const name of [
      "integrate",
      "series",
      "vector_calculus",
      "variation",
      "check_units",
      "load_tools",
    ]) {
      expect(TOOL_LABEL[name]).toBeTruthy();
    }
  });

  it("says the formulas are being checked once the server says so", () => {
    expect(statusText(pending({ phase: "checking" }), 5)).toContain("Checking the formulas");
  });

  it("explains a long silence at the start: the model may be loading", () => {
    expect(statusText(pending(), 2)).toBe("Thinking…");
    expect(statusText(pending(), 25)).toContain("may be loading it");
    expect(statusText(pending({ items: [{ kind: "text", text: "Hello" }] }), 25)).toBe("Working…");
  });

  describe("the seconds counter", () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it("counts seconds while waiting", () => {
      render(<PendingView pending={pending({ startedAt: Date.now() })} />);
      expect(screen.getByTestId("pending-status")).not.toHaveTextContent(" s");
      act(() => {
        vi.advanceTimersByTime(7000);
      });
      expect(screen.getByTestId("pending-status")).toHaveTextContent("7 s");
    });
  });
});

describe("retry and edit", () => {
  it("offers Regenerate on the last reply and calls back", () => {
    const onRetry = vi.fn();
    render(<MessageView message={reply({ blocks: [] })} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Regenerate" }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("says Retry when the reply ended early", () => {
    render(<MessageView message={reply({})} onRetry={vi.fn()} retryLabel="Retry" />);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("offers Edit and Retry under the last user message", () => {
    const onEdit = vi.fn();
    const onRetry = vi.fn();
    const user: StoredMessage = {
      id: "u",
      role: "user",
      created_at: "2026-11-15T00:00:00Z",
      content: { text: "hello" },
    };
    render(<MessageView message={user} onEdit={onEdit} onRetry={onRetry} />);
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(onEdit).toHaveBeenCalled();
    expect(onRetry).toHaveBeenCalled();
  });

  it("offers nothing on older messages", () => {
    render(<MessageView message={reply({ blocks: withBlocks() })} />);
    expect(screen.queryByRole("button", { name: /Regenerate|Retry|Edit/ })).not.toBeInTheDocument();
  });
});

function withBlocks() {
  return [{ kind: "text" as const, segments: [seg("text", "hello")], card: null }];
}

describe("Composer: replacing the text (editing the last message)", () => {
  it("puts the old message back in the box once per request", () => {
    const props = { disabled: false, busy: false, onSend: vi.fn(), onStop: vi.fn() };
    const { rerender } = render(<Composer {...props} />);
    const area = screen.getByLabelText("Message") as HTMLTextAreaElement;
    fireEvent.change(area, { target: { value: "half written" } });
    rerender(<Composer {...props} replaceRequest={{ id: 1, text: "the old message" }} />);
    expect(area.value).toBe("the old message");
    fireEvent.change(area, { target: { value: "changed it" } });
    rerender(<Composer {...props} replaceRequest={{ id: 1, text: "the old message" }} />);
    expect(area.value).toBe("changed it"); // the same request does not undo what was typed since
    rerender(<Composer {...props} replaceRequest={{ id: 2, text: "" }} />);
    expect(area.value).toBe("");
  });
});
