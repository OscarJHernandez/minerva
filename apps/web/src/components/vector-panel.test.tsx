import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  blocked,
  curl,
  derived,
  divergenceRequest,
  jacobian,
  listed,
  needsPremises,
  waveEquation,
} from "@/test/vector-fixtures";
import { VectorPanel } from "./vector-panel";

describe("VectorPanel", () => {
  it("shows the result by component, where the coordinates break down, and how it was checked", () => {
    render(<VectorPanel result={curl} request={divergenceRequest} />);
    expect(screen.getByText("Numerically checked; no symbolic proof")).toBeInTheDocument();
    expect(screen.queryByText(/Proved/)).not.toBeInTheDocument();
    expect(screen.getByTestId("components").children).toHaveLength(3);
    expect(screen.getByTestId("restrictions")).toHaveTextContent(
      /where the coordinates break down/,
    );
    expect(screen.getByText(/How it was checked \(6 of 6 points\)/)).toBeInTheDocument();
    expect(screen.getByText(/evidence, not a proof/)).toBeInTheDocument();
  });

  it("says a symbolic proof only when the engine reports one", () => {
    const proved = {
      ...curl,
      verification: { ...curl.verification!, symbolic_zero: true },
    };
    render(<VectorPanel result={proved} />);
    expect(screen.getByText(/Proved by a symbolic zero test/)).toBeInTheDocument();
  });

  it("explains a withheld result and draws no formula for it", () => {
    const { container } = render(<VectorPanel result={blocked} />);
    expect(screen.getByRole("note")).toHaveTextContent(/twice continuously differentiable f/);
    expect(screen.getByText(/does not assume smoothness/)).toBeInTheDocument();
    expect(screen.queryByTestId("components")).not.toBeInTheDocument();
    expect(container.querySelector("[data-testid=restrictions]")).toBeNull();
  });

  it("lists symbols that were never declared", () => {
    render(
      <VectorPanel
        result={{ ...blocked, blocked: [], undeclared: ["k"], status: "parse_error" }}
      />,
    );
    expect(screen.getByRole("note")).toHaveTextContent(/Not declared: k/);
  });

  it("shows a Jacobian with its determinant and scale factors", () => {
    render(<VectorPanel result={jacobian} />);
    expect(screen.getByLabelText("Determinant")).toBeInTheDocument();
    expect(screen.getByLabelText("Scale factors")).toBeInTheDocument();
    expect(screen.queryByTestId("components")).not.toBeInTheDocument();
  });

  it("shows a derivation step by step and says it is conditional on the premises", () => {
    render(<VectorPanel result={derived} />);
    expect(screen.getByText("Holds given the premises you supplied")).toBeInTheDocument();
    expect(screen.getByTestId("step-0")).toHaveTextContent("Conditional");
    expect(screen.getByTestId("step-0")).toHaveTextContent("curl(E) = -dt(B)");
    expect(screen.getByTestId("step-1")).toHaveTextContent("Identity");
    expect(screen.getByTestId("step-1")).toHaveTextContent("Needs c2 fields (declared by you)");
    expect(screen.getByTestId("given")).toHaveTextContent(/not an identity/);
    expect(screen.queryByText(/Proved/)).not.toBeInTheDocument();
  });

  it("lists matching identities without claiming any was applied", () => {
    render(<VectorPanel result={listed} />);
    const list = screen.getByTestId("applicable");
    expect(list).toHaveTextContent("vector.curl_curl");
    expect(list).toHaveTextContent(/Blocked: .*smoothness of these fields was not declared/);
    expect(screen.queryByTestId("steps")).not.toBeInTheDocument();
  });

  it("shows a derived equation, the premises it is conditional on and the check on exact solutions", () => {
    render(<VectorPanel result={waveEquation} />);
    expect(screen.getByTestId("equations")).toBeInTheDocument();
    expect(screen.getByTestId("given")).toHaveTextContent("curl(B) = mu0*epsilon0*dt(E)");
    expect(screen.getByTestId("given")).toHaveTextContent(/not an identity/);
    expect(screen.getByTestId("model-check")).toHaveTextContent(/consistent with 2 of them/);
    expect(screen.getByTestId("model-check")).toHaveTextContent(/Evidence, not a verification/);
  });

  it("asks for the premises a derivation needs and shows no equation", () => {
    render(<VectorPanel result={needsPremises} />);
    expect(screen.getByTestId("needs")).toHaveTextContent("curl(B) = mu0*epsilon0*dt(E)");
    expect(screen.getByTestId("needs")).toHaveTextContent(/Minerva does not assume them/);
    expect(screen.queryByTestId("equations")).not.toBeInTheDocument();
  });
});
