import { describe, expect, it } from "vitest";
import { okDistributionIntegral } from "@/test/fixtures";
import type { StoredMessage } from "./chat-types";
import {
  JSON_END,
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
});
