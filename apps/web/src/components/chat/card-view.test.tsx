import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card } from "@/lib/chat-types";
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
