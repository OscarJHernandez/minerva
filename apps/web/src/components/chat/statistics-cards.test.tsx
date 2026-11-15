import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Card } from "@/lib/chat-types";
import {
  expectationConditional,
  gradientSymmetric,
  mleGaussian,
  momentsCauchy,
  sumsOk,
} from "@/test/stat-fixtures";
import { CardView } from "./card-view";

describe("CardView: statistics", () => {
  it("renders a sums card with how it was checked", () => {
    const card: Card = {
      id: "t1",
      tool: "sums",
      type: "sums",
      title: "Sums",
      data: { result: sumsOk, request: {} },
    };
    render(<CardView card={card} />);
    expect(screen.getByRole("region", { name: "Finite sums" })).toBeInTheDocument();
    expect(screen.getByTestId("checks")).toHaveTextContent(/not proved for every size/);
  });

  it("renders an expectation card with the premise it holds under", () => {
    const card: Card = {
      id: "t2",
      tool: "expectation",
      type: "expectation",
      title: "Expectation",
      data: { result: expectationConditional, request: {} },
    };
    render(<CardView card={card} />);
    expect(screen.getByTestId("premises")).toHaveTextContent(/X and Y are independent/);
  });

  it("renders a statistical model card, and a refusal without a result", () => {
    const ok: Card = {
      id: "t3",
      tool: "statistical_model",
      type: "statistics",
      title: "Statistics",
      data: { result: mleGaussian, request: {} },
    };
    const { unmount } = render(<CardView card={ok} />);
    expect(screen.getByTestId("estimates")).toBeInTheDocument();
    unmount();
    render(<CardView card={{ ...ok, id: "t4", data: { result: momentsCauchy, request: {} } }} />);
    expect(screen.getByTestId("refusal")).toBeInTheDocument();
    expect(screen.queryByTestId("estimates")).toBeNull();
  });

  it("renders a gradient on the matrix card with its convention", () => {
    const card: Card = {
      id: "t5",
      tool: "matrix_calculus",
      type: "matrix",
      title: "Matrix",
      data: { result: gradientSymmetric, methods: [], request: {} as never },
    };
    render(<CardView card={card} />);
    expect(screen.getByTestId("derivative-convention")).toBeInTheDocument();
    expect(screen.getByTestId("entrywise-alternative")).toBeInTheDocument();
  });
});
