import { Math } from "@minerva/ui/math";
import type { ReactNode } from "react";
import type { FormulaStatus, Segment } from "@/lib/chat-types";

const MARKS: Record<
  FormulaStatus,
  { label: string | null; title: string; underline: string; box: string; tone: string }
> = {
  from_tool: {
    label: null,
    title: "Matches a formula from a tool result or the handbook",
    underline: "",
    box: "border-rule bg-raised",
    tone: "",
  },
  checked: {
    label: "checked",
    title:
      "Written by the language model. Minerva tested this equation numerically at sample points and it held. Conditions written beside it are not tested.",
    underline: "border-b border-dotted border-ok",
    box: "border-ok/40 bg-ok-soft/40",
    tone: "text-ok",
  },
  conditional: {
    label: "conditional",
    title:
      "Written by the language model. Minerva tested this equation: it is not true in general, but it holds under an extra assumption (see the note).",
    underline: "border-b border-dashed border-warn",
    box: "border-warn/50 bg-warn-soft/40",
    tone: "text-warn",
  },
  contradicted: {
    label: "contradicted",
    title:
      "Written by the language model. Minerva tested this equation, as an identity (true for every value of its symbols) or against definitions stated earlier, and it does not hold. See the note for which.",
    underline: "border-b border-solid border-bad",
    box: "border-bad/50 bg-bad-soft/50",
    tone: "text-bad",
  },
  unchecked: {
    label: "unchecked",
    title:
      "Written by the language model. Minerva did not produce this formula and could not check it. Not being checked is not a finding that it is wrong.",
    underline: "border-b border-dotted border-warn",
    box: "border-warn/50 bg-warn-soft/40",
    tone: "text-warn",
  },
};

/** The mark's explanation plus the reason Minerva gave for this particular formula, if any. */
const explain = (seg: Segment) => {
  const mark = MARKS[seg.status ?? "unchecked"];
  return seg.note ? `${mark.title} ${seg.note}` : mark.title;
};

/** An equation or inequality is a claim Minerva can test; a lone symbol such as `r` or `\\theta` is not. */
const RELATION = /=|<|>|\\(?:le|ge|leq|geq|neq|ne|approx|equiv|sim|propto|to|implies)\b/;
export const isClaim = (latex: string) => RELATION.test(latex);

type Node = { kind: "text"; text: string } | { kind: "math"; seg: Segment; key: number };
type Line = { nodes: Node[] } | { display: Segment; key: number };

/** Lines of prose and formulas; a display formula is its own block. Blank lines are dropped. */
function toLines(segments: Segment[]): Line[] {
  const lines: Line[] = [];
  let current: Node[] = [];
  const flush = () => {
    if (current.some((n) => n.kind === "math" || n.text.trim() !== ""))
      lines.push({ nodes: current });
    current = [];
  };
  segments.forEach((s, key) => {
    if (s.type === "display_math") {
      flush();
      lines.push({ display: s, key });
    } else if (s.type === "math") {
      current.push({ kind: "math", seg: s, key });
    } else {
      s.value.split("\n").forEach((part, i) => {
        if (i > 0) flush();
        if (part !== "") current.push({ kind: "text", text: part });
      });
    }
  });
  flush();
  return lines;
}

/** `**bold**` (which may enclose formulas) and `code`: the only markup the assistant may use. */
function renderNodes(nodes: Node[], prefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let bold = false;
  nodes.forEach((n, ni) => {
    if (n.kind === "math") {
      out.push(<Formula key={`${prefix}-m${n.key}`} seg={n.seg} />);
      return;
    }
    n.text.split(/(\*\*|`[^`]+`)/).forEach((piece, pi) => {
      const key = `${prefix}-${ni}-${pi}`;
      if (piece === "**") bold = !bold;
      else if (piece.startsWith("`") && piece.length > 2) {
        out.push(
          <code key={key} className="rounded bg-sunk px-1 font-mono text-[0.9em]">
            {piece.slice(1, -1)}
          </code>,
        );
      } else if (piece !== "") {
        out.push(bold ? <strong key={key}>{piece}</strong> : piece);
      }
    });
  });
  return out;
}

const BULLET = /^\s*[*-]\s+/;

function Formula({ seg }: { seg: Segment }) {
  // `null`: written by the user or still streaming. Minerva has made no claim about it.
  if (seg.status === null) return <Math latex={seg.value} />;
  const status = seg.status ?? "unchecked";
  const mark = MARKS[status];
  // A lone symbol names something; it asserts nothing, so there is nothing to verify or flag.
  const plain = status === "unchecked" && !isClaim(seg.value);
  return (
    <span
      title={plain ? undefined : explain(seg)}
      data-status={plain ? "symbol" : status}
      className={plain ? undefined : mark.underline || undefined}
    >
      <Math latex={seg.value} />
      {!plain && mark.label && (
        <sup className={`ml-0.5 text-[10px] font-medium ${mark.tone}`}>
          {status === "unchecked" ? "?" : mark.label}
        </sup>
      )}
    </span>
  );
}

function DisplayFormula({ seg }: { seg: Segment }) {
  const mark = MARKS[seg.status ?? "from_tool"];
  return (
    <div data-status={seg.status} className={`my-2 rounded-md border px-3 py-1 ${mark.box}`}>
      <Math latex={seg.value} display />
      {mark.label && (
        <div className={`pb-1 text-xs ${mark.tone}`} title={mark.title}>
          {seg.status === "checked"
            ? "Written by the model; checked numerically by Minerva"
            : seg.status === "conditional"
              ? `Written by the model; ${seg.note ?? "true only under extra assumptions"}`
              : seg.status === "contradicted"
                ? `Written by the model; ${seg.note ?? "Minerva's check says this does not hold"}`
                : `Not verified by Minerva (written by the model)${seg.note ? `: ${seg.note}` : ""}`}
        </div>
      )}
    </div>
  );
}

export function RichText({ segments }: { segments: Segment[] }) {
  const blocks: ReactNode[] = [];
  let items: ReactNode[] = [];
  const closeList = (at: number) => {
    if (items.length > 0) {
      blocks.push(
        <ul key={`ul${at}`} className="ml-5 list-disc space-y-1.5 marker:text-muted">
          {items}
        </ul>,
      );
      items = [];
    }
  };
  toLines(segments).forEach((line, li) => {
    if ("display" in line) {
      closeList(li);
      blocks.push(<DisplayFormula key={`d${line.key}`} seg={line.display} />);
      return;
    }
    const first = line.nodes[0];
    if (first?.kind === "text" && BULLET.test(first.text)) {
      const nodes = [{ ...first, text: first.text.replace(BULLET, "") }, ...line.nodes.slice(1)];
      items.push(<li key={`li${li}`}>{renderNodes(nodes, `l${li}`)}</li>);
      return;
    }
    closeList(li);
    blocks.push(<p key={`p${li}`}>{renderNodes(line.nodes, `l${li}`)}</p>);
  });
  closeList(blocks.length);
  return <div className="space-y-2.5 leading-relaxed">{blocks}</div>;
}
