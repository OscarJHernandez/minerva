import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { TraceEvent, TurnDebug } from "@/lib/chat-types";
import { DebugPanel } from "./debug-panel";

const event = (e: Partial<TraceEvent> & { event: string; t_ms: number }): TraceEvent => ({
  trace_id: "trace1234",
  seq: e.t_ms,
  step: null,
  ...e,
});

const withEvents = (events: TraceEvent[], over: Partial<TurnDebug> = {}): TurnDebug => ({
  request_id: "req1",
  api_version: "1",
  mode: "ollama",
  model: "some-model",
  prompt_sha: "abc",
  max_steps: 6,
  max_seconds: 180,
  duration_ms: 900,
  llm_calls: [{ ms: 100, text_chars: 0, tool_calls: 1 }],
  claims: [],
  error: null,
  trace_id: "trace1234",
  evidence_pct: 50,
  formulas: 2,
  formulas_with_evidence: 1,
  events,
  ...over,
});

describe("DebugPanel", () => {
  it("lists the arguments the model sent and what it read back", () => {
    render(
      <DebugPanel
        getReport={() => ""}
        content={{
          text: "",
          trace: [
            {
              id: "t1",
              name: "simplify",
              arguments: { latex: "cos x" },
              for_model: "status=ok",
              duration_ms: 12,
              status: "ok",
            },
          ],
        }}
      />,
    );
    expect(screen.getByText(/Trace · 1 tool call · no timing recorded/)).toBeInTheDocument();
    expect(screen.getByText(/"latex": "cos x"/)).toBeInTheDocument();
    expect(screen.getByText("status=ok")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy this turn" })).toBeInTheDocument();
  });

  it("shows a chronological timeline, why the turn stopped, and what is conversation content", () => {
    render(
      <DebugPanel
        getReport={() => ""}
        content={{
          text: "",
          stop_reason: "repeated_calls",
          trace: [
            {
              id: "c1",
              name: "simplify",
              arguments: { latex: "@r1" },
              model_arguments: { latex: "@r1" },
              for_model: "status=ok",
              status: "ok",
              duration_ms: 3,
              result_id: "r2",
            },
          ],
          debug: withEvents(
            [
              event({ event: "agent.started", t_ms: 0, model: "some-model", max_steps: 6 }),
              event({
                event: "tool.started",
                t_ms: 20,
                tool: "simplify",
                call_id: "c1",
                arguments_valid: true,
              }),
              event({
                event: "tool.completed",
                t_ms: 25,
                tool: "simplify",
                call_id: "c1",
                status: "ok",
              }),
              event({
                event: "verification.completed",
                t_ms: 26,
                scope: "tool",
                tool: "simplify",
                accepted: true,
                reason: "verified",
                check: "numeric",
              }),
              event({
                event: "agent.stopped",
                t_ms: 90,
                reason: "repeated_calls",
                steps: 3,
                tool_calls: 3,
              }),
              event({
                event: "agent.completed",
                t_ms: 91,
                status: "ok",
                total_ms: 91,
                llm_ms: 40,
                tool_ms: 5,
              }),
            ],
            { stop_reason: "repeated_calls" },
          ),
        }}
      />,
    );
    const rows = screen.getByTestId("trace-timeline").querySelectorAll("li");
    expect([...rows].map((r) => r.getAttribute("data-kind"))).toEqual([
      "start",
      "tool",
      "verify",
      "stop",
      "end",
    ]);
    expect(screen.getByText("Verification of simplify: accepted")).toBeInTheDocument();
    expect(screen.getByText(/Ended early: the model kept repeating calls/)).toBeInTheDocument();
    expect(
      screen.getByText(/Trace · 1 tool call · 900 ms · some-model · stopped: repeated calls/),
    ).toBeInTheDocument();
    expect(screen.getByText(/trace trace1234/)).toBeInTheDocument();
    expect(screen.getByText(/1\/2 formulas with engine evidence/)).toBeInTheDocument();
    expect(screen.getByText(/are your conversation content/)).toBeInTheDocument();
    expect(screen.getByText("After replacing references to earlier results")).toBeInTheDocument();
  });

  it("still renders messages stored before tracing existed", () => {
    render(
      <DebugPanel
        getReport={() => ""}
        content={{
          text: "",
          trace: [{ id: "t", name: "simplify", arguments: {}, for_model: "status=ok" }],
          debug: {
            request_id: "r",
            api_version: "1",
            mode: "ollama",
            model: "m",
            prompt_sha: "p",
            max_steps: 6,
            max_seconds: 180,
            duration_ms: 5,
            llm_calls: [],
            claims: [],
            error: null,
          },
        }}
      />,
    );
    expect(screen.queryByTestId("trace-timeline")).not.toBeInTheDocument();
    expect(screen.getByText("status=ok")).toBeInTheDocument();
  });
});
