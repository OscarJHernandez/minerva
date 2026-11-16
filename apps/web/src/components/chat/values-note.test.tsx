import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ValuesNote } from "./values-note";

describe("ValuesNote", () => {
  it("says nothing about numbers the user wrote or typed on the form", () => {
    const { container } = render(<ValuesNote chosenBy="user" values={[]} />);
    expect(container).toBeEmptyDOMElement();
    const none = render(<ValuesNote />);
    expect(none.container).toBeEmptyDOMElement();
  });

  it("says the assistant chose the values at the user's request, and lists them", () => {
    render(<ValuesNote chosenBy="assistant_delegated" values={["t_end = 100", "c = 0.5"]} />);
    expect(screen.getByRole("note")).toHaveTextContent(
      "Values chosen by the assistant at your request",
    );
    expect(screen.getByRole("note")).toHaveTextContent("t_end = 100, c = 0.5");
  });

  it("flags values the model supplied without being told to", () => {
    render(<ValuesNote chosenBy="assistant" values={["k = 4"]} />);
    expect(screen.getByRole("note")).toHaveTextContent(
      "Some values were supplied by the assistant, not by you",
    );
  });
});
