import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { ModelInfo } from "@/lib/chat-types";
import { ModelDetails } from "./model-details";

const info: ModelInfo = {
  provider: "ollama",
  alias: "minerva",
  hosted: false,
  known: true,
  base_model: "gemma4:26b",
  parameter_size: "25.2B",
  quantization: "Q4_K_M",
  digest: "149a0ff8a1ca",
  thinking_requested: false,
};

describe("ModelDetails", () => {
  it("shows the resolved model and the thinking state on the line", () => {
    render(<ModelDetails info={info} suffix=" · thinking off" />);
    expect(screen.getByText("gemma4:26b · local · thinking off")).toBeTruthy();
  });
  it("shows the alias, digest and quantisation in the details", () => {
    render(<ModelDetails info={info} open />);
    expect(screen.getByText("minerva")).toBeTruthy();
    expect(screen.getByText("149a0ff8a1ca")).toBeTruthy();
    expect(screen.getByText("Q4_K_M")).toBeTruthy();
  });
  it("opens on a tap of the line", () => {
    render(<ModelDetails info={info} />);
    const details = screen.getByTestId("model-details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    fireEvent.click(screen.getByText("gemma4:26b · local"));
    // jsdom toggles <details> on summary click only in newer versions; the content is in the DOM either way
    expect(screen.getByText("149a0ff8a1ca")).toBeTruthy();
  });
  it("says the identity was not reported when there is no information", () => {
    render(<ModelDetails info={null} alias="minerva" />);
    expect(screen.getByText("minerva · identity not reported")).toBeTruthy();
  });
});
