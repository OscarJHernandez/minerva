import type { EntrySummary, IdentitiesResult, SimplifyResult } from "./types";

/**
 * Formulas only. from_tool: appears in a tool result or handbook entry of this conversation.
 * checked: an equation the model wrote that Minerva tested numerically and found true.
 * contradicted: an equation the model wrote that Minerva tested and found false.
 * unchecked: written by the model; Minerva could not check it.
 */
export type FormulaStatus = "from_tool" | "checked" | "contradicted" | "unchecked";

export interface Segment {
  type: "text" | "math" | "display_math";
  value: string;
  status: FormulaStatus | null;
}

export type Card =
  | { id: string; tool: string; type: "simplify"; title: string; data: SimplifyResult }
  | { id: string; tool: string; type: "identities"; title: string; data: IdentitiesResult }
  | {
      id: string;
      tool: string;
      type: "entries";
      title: string;
      data: { query: string; entries: EntrySummary[] };
    }
  | { id: string; tool: string; type: "entry"; title: string; data: EntrySummary }
  | {
      id: string;
      tool: string;
      type: "tensor";
      title: string;
      data: {
        status: string;
        input_text: string | null;
        result_text: string | null;
        result_latex: string | null;
        message: string;
      };
    }
  | { id: string; tool: string; type: "error"; title: string; data: { message: string } };

/** One item of an assistant turn in the order it happened: prose, or a tool result card. */
export type Block =
  | { kind: "text"; segments: Segment[]; card: null }
  | { kind: "card"; segments: Segment[]; card: Card };

export interface AssistantContent {
  text: string;
  blocks: Block[];
  truncated: boolean;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: { text: string } & Partial<AssistantContent>;
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
}

export interface ChatStatus {
  enabled: boolean;
  mode: string;
  model: string;
}

export interface ToolRun {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export type StreamEvent =
  | { event: "text"; data: { delta: string } }
  | { event: "tool_call"; data: ToolRun }
  | { event: "card"; data: Card }
  | { event: "message"; data: AssistantContent }
  | { event: "error"; data: { message: string } }
  | { event: "done"; data: Record<string, never> };
