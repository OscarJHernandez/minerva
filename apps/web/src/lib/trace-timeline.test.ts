import { describe, expect, it } from "vitest";
import type { TraceEvent, TraceStep, TurnDebug } from "./chat-types";
import {
  REASON_EXPLANATION,
  STOP_EXPLANATION,
  buildTimeline,
  stageTimings,
} from "./trace-timeline";

let seq = 0;
const ev = (event: string, t_ms: number, fields: Record<string, unknown> = {}): TraceEvent => ({
  event,
  trace_id: "trace1",
  seq: ++seq,
  t_ms,
  step: null,
  ...fields,
});

const debug = (events: TraceEvent[]): TurnDebug => ({
  request_id: "r",
  api_version: "1",
  mode: "ollama",
  model: "m",
  prompt_sha: "p",
  max_steps: 6,
  max_seconds: 180,
  duration_ms: 900,
  llm_calls: [],
  claims: [],
  error: null,
  trace_id: "trace1",
  events,
});

const step = (over: Partial<TraceStep> = {}): TraceStep => ({
  id: "c1",
  name: "simplify",
  arguments: { latex: "SECRET-EXPRESSION" },
  for_model: "status=ok; result=$SECRET-RESULT$",
  ...over,
});

describe("buildTimeline", () => {
  it("is empty for messages stored before tracing existed", () => {
    expect(buildTimeline(null, [step()])).toEqual([]);
    expect(buildTimeline(debug([]), undefined)).toEqual([]);
  });

  it("tells a whole successful turn in order, with the reason for each decision", () => {
    const rows = buildTimeline(
      debug([
        ev("agent.started", 0, { model: "m", max_steps: 6, max_tool_calls: 12, max_seconds: 180 }),
        ev("llm.request", 1, { step: 1 }),
        ev("llm.response", 120, {
          step: 1,
          latency_ms: 119,
          first_token_ms: 40,
          finish_reason: "tool_calls",
          tool_calls: 1,
          proposed_tool: "simplify",
          tokens_in: 50,
          tokens_out: 9,
        }),
        ev("tool.started", 121, {
          step: 1,
          tool: "simplify",
          call_id: "c1",
          arguments_valid: true,
        }),
        ev("tool.completed", 300, {
          step: 1,
          tool: "simplify",
          call_id: "c1",
          status: "ok",
          duration_ms: 179,
          engine_ms: 170,
          sandbox_calls: 1,
          result_id: "r1",
        }),
        ev("verification.completed", 301, {
          step: 1,
          scope: "tool",
          tool: "simplify",
          check: "symbolic_zero",
          accepted: true,
          reason: "verified",
          points_checked: 20,
          points_agreed: 20,
          symbolic_zero: true,
          restrictions: 1,
          restrictions_complete: true,
        }),
        ev("verification.completed", 700, {
          scope: "claims",
          check: "claims",
          accepted: true,
          claims: 2,
          claims_holding: 1,
          claims_conditional: 0,
          claims_contradicted: 1,
          claims_unchecked: 0,
          verify_ms: 30,
        }),
        ev("agent.stopped", 710, { reason: "completed", steps: 2, tool_calls: 1 }),
        ev("agent.completed", 720, {
          status: "ok",
          total_ms: 720,
          llm_ms: 400,
          tool_ms: 179,
          engine_ms: 170,
          verify_ms: 30,
          formulas: 4,
          formulas_with_evidence: 3,
          evidence_pct: 75,
        }),
      ]),
      [step()],
    );
    expect(rows.map((r) => r.kind)).toEqual([
      "start",
      "model",
      "tool",
      "verify",
      "verify",
      "stop",
      "end",
    ]);
    expect(rows.map((r) => r.t_ms)).toEqual([...rows.map((r) => r.t_ms)].sort((a, b) => a - b));
    const [, model, tool, verify, claims, stop, end] = rows;
    expect(model!.title).toContain("asked for 1 tool call");
    expect(model!.detail).toContain("first proposed tool: simplify");
    expect(model!.detail).toContain("tokens 50 in, 9 out");
    expect(model!.detail).toContain("the tool results go back to the model");
    expect(tool!.title).toBe("Tool simplify: status ok");
    expect(tool!.detail).toContain("arguments valid");
    expect(tool!.detail).toContain("170 ms in the sandbox");
    expect(tool!.detail).toContain("result id r1");
    expect(verify!.title).toBe("Verification of simplify: accepted");
    expect(verify!.detail).toContain("20/20 sample points agreed");
    expect(verify!.detail).toContain("symbolic zero test passed");
    expect(verify!.detail).toContain(REASON_EXPLANATION.verified);
    expect(claims!.title).toContain("2 tested");
    expect(claims!.detail).toContain("1 contradicted");
    expect(stop!.title).toBe("Stopped: completed");
    expect(stop!.detail).toContain(STOP_EXPLANATION.completed);
    expect(end!.detail).toContain("formulas with engine evidence 3/4 (75%)");
  });

  it("explains a rejected result, a refused call, a retry and an early stop", () => {
    const rows = buildTimeline(
      debug([
        ev("tool.started", 5, { tool: "simplify", call_id: "c1", arguments_valid: false }),
        ev("tool.completed", 6, {
          tool: "simplify",
          call_id: "c1",
          status: "error",
          error_category: "invalid_arguments",
          duration_ms: 1,
        }),
        ev("verification.completed", 7, {
          scope: "tool",
          tool: "integrate",
          check: "none",
          accepted: false,
          reason: "unsupported",
        }),
        ev("agent.retry", 8, { reason: "model_host", attempt: 1 }),
        ev("llm.response", 9, { step: 2, finish_reason: "timeout", latency_ms: 5 }),
        ev("agent.stopped", 10, { reason: "repeated_calls", steps: 3, tool_calls: 3 }),
      ]),
      [],
    );
    expect(rows[0]).toMatchObject({ tone: "bad" });
    expect(rows[0]!.detail).toContain("arguments INVALID");
    expect(rows[0]!.detail).toContain("category invalid_arguments");
    expect(rows[1]!.title).toBe("Verification of integrate: rejected");
    expect(rows[1]!.detail).toContain(REASON_EXPLANATION.unsupported);
    expect(rows[2]!.title).toBe("Retry 1: model host");
    expect(rows[2]!.detail).toContain("no computation is repeated");
    expect(rows[3]!.title).toContain("failed (timeout)");
    expect(rows[4]!.title).toBe("Stopped: repeated calls");
    expect(rows[4]!.detail).toContain(STOP_EXPLANATION.repeated_calls);
  });

  it("shows repeated and skipped calls and dropped assumptions", () => {
    const rows = buildTimeline(
      debug([
        ev("tool.started", 1, {
          tool: "simplify",
          call_id: "c1",
          arguments_valid: true,
          repeated: true,
        }),
        ev("tool.completed", 2, { tool: "simplify", call_id: "c1", status: "skipped" }),
        ev("tool.started", 3, {
          tool: "integrate",
          call_id: "c2",
          arguments_valid: true,
          references: 2,
        }),
        ev("tool.completed", 4, { tool: "integrate", call_id: "c2", status: "ok" }),
      ]),
      [step({ id: "c2", assumptions_dropped: ["x"] })],
    );
    expect(rows[0]!.title).toContain("not run again");
    expect(rows[0]!.tone).toBe("warn");
    expect(rows[1]!.detail).toContain("2 references to earlier results replaced");
    expect(rows[1]!.detail).toContain("declared earlier but not repeated here: x");
  });

  it("never puts arguments or results into the timeline", () => {
    const rows = buildTimeline(
      debug([
        ev("tool.started", 1, { tool: "simplify", call_id: "c1", arguments_valid: true }),
        ev("tool.completed", 2, { tool: "simplify", call_id: "c1", status: "ok" }),
      ]),
      [step()],
    );
    const text = JSON.stringify(rows);
    expect(text).not.toContain("SECRET");
  });
});

describe("stageTimings", () => {
  it("reads the stage timings from the final event", () => {
    const d = debug([
      ev("agent.completed", 9, {
        total_ms: 900,
        llm_ms: 500,
        tool_ms: 300,
        engine_ms: 250,
        verify_ms: 40,
      }),
    ]);
    expect(stageTimings(d)).toEqual([
      ["total", 900],
      ["model calls", 500],
      ["tool calls", 300],
      ["  of which in the sandbox", 250],
      ["claim check", 40],
    ]);
    expect(stageTimings(null)).toEqual([]);
    expect(stageTimings(debug([]))).toEqual([]);
  });
});
