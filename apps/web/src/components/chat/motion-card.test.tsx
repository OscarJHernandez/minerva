import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { MotionRequest } from "@/lib/chat-types";
import { failedRun, okRun, report } from "@/test/motion-fixtures";
import { MotionCard } from "./motion-card";

const request = {} as MotionRequest;

describe("MotionCard", () => {
  it("shows the plots and the verification numbers of a verified run", () => {
    render(<MotionCard report={report([okRun()])} request={request} />);
    expect(screen.getByText("Computed and verified")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "x against time" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Phase portrait of x" })).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Drift of the constants of motion" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/differs by 1\.7e-9/)).toBeInTheDocument();
    expect(screen.getByText(/drift by at most 5\.0e-7/)).toBeInTheDocument();
  });

  it("shows no plot and no data for a run that failed verification", () => {
    render(<MotionCard report={report([failedRun()])} request={request} />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText(/cross-check differs by 4\.09e\+00/)).toBeInTheDocument();
    expect(screen.getByText(/No data is shown for this run/)).toBeInTheDocument();
    expect(screen.getByText("not verified")).toBeInTheDocument();
  });

  it("overlays the verified runs of a sweep and lists the failed one without data", async () => {
    render(
      <MotionCard report={report([okRun(1), okRun(4), failedRun(9)], "k")} request={request} />,
    );
    const plot = screen.getByRole("img", { name: "x against time" });
    expect(plot.querySelectorAll("path")).toHaveLength(2);
    const list = screen.getByRole("list");
    expect(within(list).getByText("k = 9")).toBeInTheDocument();
    expect(within(list).getAllByText("verified")).toHaveLength(2);
    await userEvent.selectOptions(screen.getByLabelText("Run"), "1");
    expect(
      screen.getByRole("img", { name: "Drift of the constants of motion" }),
    ).toBeInTheDocument();
  });

  it("says when there is no constant of motion to follow", () => {
    const run = { ...okRun(), conserved: [] };
    render(<MotionCard report={report([run])} request={request} />);
    expect(screen.getByText(/No verified constant of motion exists/)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: "Drift of the constants of motion" })).toBeNull();
  });
});
