import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Card, Segment, StoredMessage } from "@/lib/chat-types";
import { sphere, sphereRequest } from "@/test/geometry-fixtures";
import { CardView } from "./card-view";
import { Composer } from "./composer";
import { EquationNumber, InsertReferenceProvider } from "./equation-number";
import { MessageView } from "./message-view";
import { RichText } from "./rich-text";

const display = (value: string, number?: number): Segment => ({
  type: "display_math",
  value,
  status: "from_tool",
  number,
});

describe("EquationNumber", () => {
  it("is plain text outside the chat", () => {
    render(<EquationNumber n={3} />);
    expect(screen.getByTestId("equation-number")).toHaveTextContent("(3)");
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("is a button in the chat that asks to refer to the equation", () => {
    const insert = vi.fn();
    render(
      <InsertReferenceProvider value={insert}>
        <EquationNumber n={3} />
      </InsertReferenceProvider>,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Equation 3: refer to it in your message" }),
    );
    expect(insert).toHaveBeenCalledWith(3);
  });
});

describe("numbers in the conversation", () => {
  it("shows the number of a displayed formula and none for an unnumbered one", () => {
    render(<RichText segments={[display("x+1", 2), display("y", undefined)]} />);
    expect(screen.getAllByTestId("equation-number")).toHaveLength(1);
    expect(screen.getByTestId("equation-number")).toHaveTextContent("(2)");
  });

  it("shows the number of a card result", () => {
    const card: Card = {
      id: "c",
      tool: "error",
      type: "error",
      title: "t",
      data: { message: "m" },
      number: 4,
    };
    render(<CardView card={card} />);
    expect(screen.getByTestId("equation-number")).toHaveTextContent("(4)");
  });

  it("numbers each listed equation of a card by its key, and a card without numbers shows none", () => {
    const card: Card = {
      id: "g",
      tool: "geometry",
      type: "geometry",
      title: "Differential geometry",
      data: { result: sphere, request: sphereRequest },
      numbers: { christoffel: 7, "christoffel:0,1,1": 8, ricci_scalar: 9 },
    };
    const { unmount } = render(<CardView card={card} />);
    // the Christoffel component and the Ricci scalar are numbered; a key the server did not give is not
    expect(screen.getAllByTestId("equation-number").map((e) => e.textContent)).toEqual([
      "(9)", // the Ricci scalar is shown first
      "(8)",
    ]);
    unmount();
    render(<CardView card={{ ...card, numbers: {} }} />);
    expect(screen.queryByTestId("equation-number")).not.toBeInTheDocument();
  });

  it("makes a listed equation's number a reference to it", () => {
    const insert = vi.fn();
    const card: Card = {
      id: "g",
      tool: "geometry",
      type: "geometry",
      title: "Differential geometry",
      data: { result: sphere, request: sphereRequest },
      numbers: { "christoffel:0,1,1": 8 },
    };
    render(
      <InsertReferenceProvider value={insert}>
        <CardView card={card} />
      </InsertReferenceProvider>,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Equation 8: refer to it in your message" }),
    );
    expect(insert).toHaveBeenCalledWith(8);
  });

  it("numbers the displayed formulas of the user's own message in order", () => {
    const message: StoredMessage = {
      id: "u",
      role: "user",
      created_at: "2026-11-15T00:00:00Z",
      content: {
        text: "Take $$x+1$$ and then $$y^2$$",
        equations: [
          { number: 5, latex: "x+1" },
          { number: 6, latex: "y^2" },
        ],
      },
    };
    render(<MessageView message={message} />);
    expect(screen.getAllByTestId("equation-number").map((e) => e.textContent)).toEqual([
      "(5)",
      "(6)",
    ]);
  });

  it("shows what a reference in the user's message was taken to mean, and says when there is none", () => {
    const message: StoredMessage = {
      id: "u",
      role: "user",
      created_at: "2026-11-15T00:00:00Z",
      content: {
        text: "integrate (2) and (9)",
        references: [
          { number: 2, latex: "x+1", source: "written_by_model" },
          { number: 9, latex: null, source: "unknown" },
        ],
      },
    };
    render(<MessageView message={message} />);
    const list = screen.getByTestId("equation-references");
    expect(list).toHaveTextContent("Taken as equation (2)");
    expect(list).toHaveTextContent("written by the model, not verified");
    expect(list).toHaveTextContent("Equation (9): there is no such equation");
  });

  it("says when a number is the same formula as a lower one (a conversation numbered before labels were removed)", () => {
    const message: StoredMessage = {
      id: "u",
      role: "user",
      created_at: "2026-11-15T00:00:00Z",
      content: {
        text: "what is equation 7?",
        references: [{ number: 7, latex: "x+1", source: "written_by_model", alias_of: 2 }],
      },
    };
    render(<MessageView message={message} />);
    expect(screen.getByTestId("equation-references")).toHaveTextContent("the same formula as (2)");
  });
});

describe("Composer: inserting a reference", () => {
  const props = { disabled: false, busy: false, onSend: vi.fn(), onStop: vi.fn() };

  it("puts the reference at the cursor, set apart from the word before it", () => {
    const { rerender } = render(<Composer {...props} />);
    const area = screen.getByLabelText("Message") as HTMLTextAreaElement;
    fireEvent.change(area, { target: { value: "Integrate" } });
    area.setSelectionRange(9, 9);
    rerender(<Composer {...props} referenceRequest={{ id: 1, text: "(3)" }} />);
    expect(area.value).toBe("Integrate (3)");
    // the same request does not insert twice; a new id does
    rerender(<Composer {...props} referenceRequest={{ id: 1, text: "(3)" }} />);
    expect(area.value).toBe("Integrate (3)");
    area.setSelectionRange(area.value.length, area.value.length);
    rerender(<Composer {...props} referenceRequest={{ id: 2, text: "(4)" }} />);
    expect(area.value).toBe("Integrate (3) (4)");
  });
});
