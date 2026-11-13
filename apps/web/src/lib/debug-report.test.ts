import { describe, expect, it } from "vitest";
import { matrixFrechet, okDistributionIntegral } from "@/test/fixtures";
import type { StoredMessage, TraceEvent } from "./chat-types";
import { sphere, sphereRequest } from "../test/geometry-fixtures";
import {
  CONTENT_NOTICE,
  JSON_END,
  REPORT_FORMAT,
  JSON_START,
  buildDebugReport,
  buildTurnReport,
  fence,
  reportData,
} from "./debug-report";

const NOW = new Date("2026-01-02T03:04:05Z");

const user: StoredMessage = {
  id: "u1",
  role: "user",
  content: { text: "Take the 10th derivative of $e^{\\cos x}$" },
  created_at: "2026-01-02T03:00:00Z",
};

const assistant: StoredMessage = {
  id: "a1",
  role: "assistant",
  created_at: "2026-01-02T03:00:05Z",
  content: {
    text: "Verified: $x$",
    truncated: false,
    trace: [
      {
        id: "t1",
        name: "simplify",
        arguments: { latex: "e^{cosx} sx" },
        for_model: "status=ok; input_as_read=$c o s x$ with `ticks` and ```fences```",
        duration_ms: 1436,
        status: "ok",
      },
    ],
    debug: {
      request_id: "req123",
      api_version: "0.1.0",
      mode: "ollama",
      model: "some-model",
      prompt_sha: "abcdef123456",
      max_steps: 6,
      max_seconds: 180,
      duration_ms: 2000,
      llm_calls: [{ ms: 500, text_chars: 0, tool_calls: 1 }],
      claims: [
        { value: "x = 2", status: "fails", detail: "differs at x=1" },
        {
          value: "y = 3",
          status: "conditional",
          detail: "holds given z = 3",
          kind: "relation",
          given: ["z = 3"],
        },
      ],
      error: null,
    },
    blocks: [
      {
        kind: "card",
        segments: [],
        card: {
          id: "t1",
          tool: "simplify",
          type: "simplify",
          title: "Simplify",
          data: {
            status: "ok",
            input_latex: "c o s x",
            result_latex: "x",
            result_text: "x",
            verification: {
              symbolic_zero: true,
              numeric_points_checked: 12,
              numeric_points_agreed: 12,
              passed: true,
              note: "",
            },
            message: "Simplified with SymPy and verified.",
            duration_ms: 1436,
          },
        },
      },
      {
        kind: "text",
        card: null,
        segments: [{ type: "math", value: "x = 2", status: "contradicted", note: null }],
      },
    ],
  },
};

const input = {
  title: "Derivative",
  messages: [user, assistant],
  status: { enabled: true, mode: "ollama", model: "some-model" },
  now: NOW,
};

describe("distribution cards in a report", () => {
  it("lists the status, the evidence and the stages", () => {
    const withCard: StoredMessage = {
      ...assistant,
      content: {
        ...assistant.content,
        trace: [
          {
            id: "d1",
            name: "distribution",
            arguments: { latex: "\\operatorname{DiracDelta}(x)" },
            for_model: "status=ok",
            duration_ms: 90,
            status: "ok",
          },
        ],
        blocks: [
          {
            kind: "card",
            segments: [],
            card: {
              id: "d1",
              tool: "distribution",
              type: "distribution",
              title: "Delta and step functions",
              data: {
                result: okDistributionIntegral,
                methods: [],
                request: {
                  action: "integrate",
                  variable: "x",
                  lower: "-oo",
                  upper: "oo",
                  endpoint: null,
                  latex: String.raw`x^2 \operatorname{DiracDelta}(x^2-4)`,
                  assumptions: { x: ["real"] },
                },
              },
            },
          },
        ],
      },
    };
    const out = buildDebugReport({ ...input, messages: [user, withCard] });
    expect(out).toContain("- card: distribution");
    expect(out).toContain("1/1 test-function pairings, conditions established true");
    expect(out).toContain("- stages: parsed > completed");
  });
});

describe("fence", () => {
  it("is longer than any backtick run in the text", () => {
    const text = "a ```` b";
    const out = fence(text, "json");
    expect(out.startsWith("`````json\n")).toBe(true);
    expect(out.endsWith("\n`````")).toBe(true);
  });
  it("uses three backticks for plain text", () => {
    expect(fence("plain")).toBe("```\nplain\n```");
  });
});

describe("buildDebugReport", () => {
  const report = buildDebugReport(input);

  it("shows what the model sent and what the engine read", () => {
    expect(report).toContain("Tool call 1: simplify (ok, 1436 ms)");
    expect(report).toContain('"latex": "e^{cosx} sx"');
    expect(report).toContain("- input as read: c o s x");
    expect(report).toContain("numeric 12/12, symbolic zero true");
    expect(report).toContain("request req123");
    expect(report).toContain("claim check fails: `x = 2` (differs at x=1)");
    expect(report).toContain("- contradicted: `x = 2`");
    expect(report).toContain(
      "claim check conditional [relation]: `y = 3` (holds given z = 3) given: z = 3",
    );
  });

  it("keeps turns in order: user first, then assistant", () => {
    expect(report.indexOf("### 1. User")).toBeLessThan(report.indexOf("### 2. Assistant"));
  });

  it("ends with a JSON block that parses back to the same data", () => {
    const start = report.indexOf(JSON_START);
    const end = report.indexOf(JSON_END);
    expect(start).toBeGreaterThan(0);
    const block = report.slice(start + JSON_START.length, end).trim();
    const lines = block.split("\n");
    const parsed = JSON.parse(lines.slice(1, -1).join("\n"));
    expect(parsed).toEqual(JSON.parse(JSON.stringify(reportData(input))));
    expect(parsed.messages[1].trace[0].arguments).toEqual({ latex: "e^{cosx} sx" });
  });

  it("does not let backticks in tool output close the code fence", () => {
    const after = report.slice(report.indexOf("What the model read back"));
    expect(after).toContain("````\nstatus=ok");
  });

  it("includes a failed turn and works for messages stored without debug data", () => {
    const old: StoredMessage = {
      ...assistant,
      content: { text: "old", blocks: [], truncated: false },
    };
    const out = buildDebugReport({
      ...input,
      messages: [user, old],
      lastError: { message: "The model host is not reachable." },
    });
    expect(out).toContain("## Last turn failed");
    expect(out).toContain("The model host is not reachable.");
    expect(out).toContain("### 2. Assistant");
  });

  it("contains no email or host name of the user", () => {
    expect(report).not.toMatch(/@|https?:\/\//);
  });
});

describe("buildTurnReport", () => {
  it("pairs an assistant message with the user message before it", () => {
    const out = buildTurnReport({ title: "t", status: null, now: NOW }, [user, assistant], 1);
    expect(out).toContain("### 1. User");
    expect(out).toContain("### 2. Assistant");
  });

  it("summarises a geometry card by quantity and stage, without the metric", () => {
    const slow = {
      ...sphere,
      diagnostics: [
        {
          stage: "riemann",
          duration_ms: 4100,
          ops: 2300,
          components: 36,
          status: "ok" as const,
          failure: null,
        },
        {
          stage: "einstein",
          duration_ms: 0,
          ops: null,
          components: null,
          status: "failed" as const,
          failure: "resource_limit",
        },
      ],
    };
    const message: StoredMessage = {
      id: "g1",
      role: "assistant",
      created_at: "2026-01-02T03:00:05Z",
      content: {
        text: "",
        truncated: false,
        trace: [
          {
            id: "c1",
            name: "geometry",
            arguments: { ...sphereRequest },
            for_model: "status=ok",
            duration_ms: 4200,
            status: "ok",
          },
        ],
        blocks: [
          {
            kind: "card",
            segments: [],
            card: {
              id: "c1",
              tool: "geometry",
              type: "geometry",
              title: "g",
              data: { result: slow, request: sphereRequest },
            },
          },
        ],
      },
    };
    const report = buildDebugReport({ title: "t", messages: [message], status: null, now: NOW });
    expect(report).toContain("- card: geometry");
    expect(report).toContain("- quantity ricci_scalar: ok (checks n:true s:true)");
    expect(report).toContain("- stage riemann: 4100 ms, 2300 ops, ok");
    expect(report).toContain("- stage einstein: 0 ms, resource_limit");
    expect(report).toContain("signature: not determined; restrictions: 1");
  });
});

const events: TraceEvent[] = [
  { event: "agent.started", trace_id: "tr1", seq: 1, t_ms: 0, step: null, model: "some-model" },
  {
    event: "tool.started",
    trace_id: "tr1",
    seq: 2,
    t_ms: 10,
    step: 1,
    tool: "simplify",
    call_id: "t1",
    arguments_valid: true,
  },
  {
    event: "tool.completed",
    trace_id: "tr1",
    seq: 3,
    t_ms: 40,
    step: 1,
    tool: "simplify",
    call_id: "t1",
    status: "ok",
    duration_ms: 30,
  },
  {
    event: "agent.stopped",
    trace_id: "tr1",
    seq: 4,
    t_ms: 90,
    step: null,
    reason: "repeated_calls",
    steps: 2,
    tool_calls: 1,
  },
  {
    event: "agent.completed",
    trace_id: "tr1",
    seq: 5,
    t_ms: 95,
    step: null,
    status: "ok",
    total_ms: 95,
    llm_ms: 50,
    tool_ms: 30,
    engine_ms: 25,
    verify_ms: 5,
  },
];

const traced: StoredMessage = {
  ...assistant,
  content: {
    ...assistant.content,
    stop_reason: "repeated_calls",
    truncated: true,
    trace: [
      {
        id: "t1",
        name: "simplify",
        arguments: { latex: "x+1" },
        model_arguments: { latex: "@r1" },
        for_model: "status=ok",
        duration_ms: 30,
        status: "ok",
        arguments_valid: true,
        engine_ms: 25,
        result_id: "r2",
        error_category: null,
        evidence: {
          check: "symbolic_zero",
          accepted: true,
          reason: "verified",
          points_checked: 20,
          points_agreed: 20,
          symbolic_zero: true,
          restrictions: 1,
        },
        assumptions_dropped: ["x"],
      },
    ],
    debug: {
      ...assistant.content.debug!,
      trace_id: "tr1",
      stop_reason: "repeated_calls",
      events,
      evidence_pct: 50,
      formulas: 2,
      formulas_with_evidence: 1,
      claim_check: "timeout",
      llm_calls: [
        {
          ms: 50,
          text_chars: 0,
          tool_calls: 1,
          prompt_tokens: 120,
          completion_tokens: 9,
          finish_reason: "tool_calls",
        },
      ],
    },
  },
};

describe("buildDebugReport, format 2", () => {
  const report = buildDebugReport({ ...input, messages: [user, traced] });

  it("says what it contains and how it is shaped", () => {
    expect(REPORT_FORMAT).toBe(2);
    expect(report).toContain("format 2 ·");
    expect(report).toContain(CONTENT_NOTICE);
    expect(CONTENT_NOTICE).toMatch(/conversation content/);
    const json = JSON.parse(
      report
        .slice(report.indexOf(JSON_START) + JSON_START.length, report.indexOf(JSON_END))
        .replace(/```json|```/g, ""),
    );
    expect(json.format).toBe(2);
    expect(json.contains_conversation_content).toBe(true);
    expect(json.messages[1].debug.events).toHaveLength(5);
    expect(json.messages[1].stop_reason).toBe("repeated_calls");
    // The prose is rebuilt with its formula delimiters, so a turn can be replayed from the report.
    expect(json.messages[1].prose).toEqual(["$x = 2$"]);
    expect(json.messages[0].prose).toEqual([]);
  });

  it("gives the trace id, stop reason, stage timings, tokens and evidence", () => {
    expect(report).toContain("trace tr1 · request req123");
    expect(report).toContain("stop reason: repeated_calls (the model kept repeating calls");
    expect(report).toContain("| total | 95 |");
    expect(report).toContain("| model calls | 50 |");
    expect(report).toContain("| of which in the sandbox | 25 |");
    expect(report).toContain("| claim check | 5 |");
    expect(report).toContain("tokens: 120 in, 9 out (as reported by the model host)");
    expect(report).toContain("formulas with engine evidence: 1/2 (50%)");
    expect(report).toContain("claim check: timeout");
  });

  it("tells the story of the turn in order", () => {
    const at = (needle: string) => report.indexOf(needle);
    expect(at("Timeline:")).toBeGreaterThan(-1);
    expect(at("- 0 ms · Turn started")).toBeLessThan(at("- 40 ms · Tool simplify: status ok"));
    expect(at("- 40 ms · Tool simplify: status ok")).toBeLessThan(
      at("- 90 ms · Stopped: repeated calls"),
    );
    expect(report).toContain("this turn ended early (repeated_calls)");
  });

  it("describes each tool call: validity, category, evidence, references, dropped assumptions", () => {
    expect(report).toContain("- arguments valid: true");
    expect(report).toContain("result id r2");
    expect(report).toContain("25 ms in the sandbox");
    expect(report).toContain(
      "- evidence: symbolic_zero, accepted (verified), 20/20 points, 1 restriction(s)",
    );
    expect(report).toContain("- declared earlier but not repeated in this call: x");
    expect(report).toContain('"latex": "@r1"'); // what the model wrote
    expect(report).toContain("After replacing references to earlier results:");
    expect(report).toContain('"latex": "x+1"');
  });

  it("holds no email address or host name", () => {
    expect(report).not.toMatch(/@(?!r\d)[a-z]|https?:\/\//);
  });

  it("still reports messages stored before tracing, with no invented fields", () => {
    const old = buildDebugReport({ ...input, messages: [user, assistant] });
    expect(old).toContain("format 2");
    expect(old).not.toContain("stop reason:");
    expect(old).not.toContain("Timeline:");
    expect(old).not.toContain("- evidence:");
    expect(old).toContain("trace - · request req123");
  });
});

describe("matrix cards in a debug report", () => {
  it("states the method, evaluated flag, checks and the content-free diagnostics", () => {
    const matrix: StoredMessage = {
      ...assistant,
      content: {
        ...assistant.content,
        blocks: [
          {
            kind: "card",
            segments: [],
            card: {
              id: "t1",
              tool: "matrix_calculus",
              type: "matrix",
              title: "Matrix calculus",
              data: { result: matrixFrechet, methods: [], request: {} as never },
            },
          },
        ],
      },
    };
    const report = buildDebugReport({ ...input, messages: [user, matrix] });
    expect(report).toContain("- card: matrix");
    expect(report).toContain(
      "- method: exp_frechet_derivative; kind: representation; evaluated: false",
    );
    expect(report).toContain("- verification: numeric 4/4, exact leg true");
    expect(report).toContain("trace req0123-card0123; tried [integrate_polynomial:no_match");
    expect(report).toContain("fallbacks []");
  });
});
