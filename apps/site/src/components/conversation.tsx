import { MARK_TEXT, type FormulaStatus } from "@minerva/ui/marks";
import { Math } from "@minerva/ui/math";
import type { ReactNode } from "react";
import type {
  Session,
  SessionCard,
  SessionSegment,
  SessionText,
} from "@/lib/landing-sessions";

/** The look of a mark on the landing page (its words come from `@minerva/ui/marks`, the chat's own table). */
const MARK_TONE: Record<FormulaStatus, string> = {
  from_tool: "",
  quoted: "text-muted",
  checked: "text-ok",
  conditional: "text-warn",
  contradicted: "text-bad",
  unchecked: "text-warn",
};

function Mark({ status }: { status?: FormulaStatus }) {
  const s = status ?? "unchecked";
  const text = MARK_TEXT[s];
  if (!text.label) return null;
  return (
    <span
      title={text.title}
      className={`ml-1 align-super text-[0.65rem] font-medium uppercase tracking-wide ${MARK_TONE[s]}`}
    >
      {text.label}
    </span>
  );
}

function Num({ n }: { n?: number | null }) {
  if (n == null) return null;
  return <span className="shrink-0 text-sm tabular-nums text-muted">({n})</span>;
}

// --- prose: the chat's small Markdown subset, over text and formula segments ----------------------------------
type Node = { kind: "text"; text: string } | { kind: "math"; seg: SessionSegment };
type Line = { nodes: Node[] } | { display: SessionSegment };

function toLines(segments: SessionSegment[]): Line[] {
  const lines: Line[] = [];
  let current: Node[] = [];
  const flush = () => {
    lines.push({ nodes: current });
    current = [];
  };
  for (const seg of segments) {
    if (seg.type === "display_math") {
      if (current.length) flush();
      lines.push({ display: seg });
    } else if (seg.type === "math") {
      current.push({ kind: "math", seg });
    } else {
      const parts = seg.value.split("\n");
      parts.forEach((part, i) => {
        if (i > 0) flush();
        if (part) current.push({ kind: "text", text: part });
      });
    }
  }
  if (current.length) flush();
  return lines;
}

const plain = (nodes: Node[]) => nodes.map((n) => (n.kind === "text" ? n.text : "x")).join("");

function Bold({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <>
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") && p.length > 4 ? (
          <strong key={i} className="font-semibold text-ink">
            {p.slice(2, -2)}
          </strong>
        ) : (
          <span key={i}>{p.replace(/`/g, "")}</span>
        ),
      )}
    </>
  );
}

function Inline({ nodes }: { nodes: Node[] }) {
  return (
    <>
      {nodes.map((n, i) =>
        n.kind === "text" ? (
          <Bold key={i} text={n.text} />
        ) : (
          <span key={i} className="whitespace-nowrap">
            <Math latex={n.seg.value} />
            {n.seg.status !== "from_tool" && <Mark status={n.seg.status} />}
          </span>
        ),
      )}
    </>
  );
}

/** Table cells: split on `|` in text only, so a formula such as `|x|` stays whole. */
function cells(nodes: Node[]): Node[][] {
  const out: Node[][] = [[]];
  for (const n of nodes) {
    if (n.kind !== "text") {
      out[out.length - 1]!.push(n);
      continue;
    }
    n.text.split("|").forEach((piece, i) => {
      if (i > 0) out.push([]);
      if (piece.trim()) out[out.length - 1]!.push({ kind: "text", text: piece.trim() });
    });
  }
  return out.slice(1, -1).length ? out.slice(1, -1) : out;
}

function Prose({ block }: { block: SessionText }) {
  const lines = toLines(block.segments);
  const out: ReactNode[] = [];
  let table: Node[][][] = [];
  let list: Node[][] = [];
  const end = (key: number) => {
    if (table.length) {
      const [head, ...body] = table;
      out.push(
        <div key={`t${key}`} className="my-3 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead>
              <tr className="border-b border-rule">
                {head!.map((c, i) => (
                  <th key={i} className="px-2 py-1 font-medium">
                    <Inline nodes={c} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {body.map((row, r) => (
                <tr key={r} className="border-b border-rule/60 align-top">
                  {row.map((c, i) => (
                    <td key={i} className="px-2 py-1">
                      <Inline nodes={c} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      table = [];
    }
    if (list.length) {
      out.push(
        <ul key={`l${key}`} className="my-2 list-disc space-y-1 pl-5">
          {list.map((item, i) => (
            <li key={i}>
              <Inline nodes={item} />
            </li>
          ))}
        </ul>,
      );
      list = [];
    }
  };
  lines.forEach((line, i) => {
    if ("display" in line) {
      end(i);
      out.push(
        <div key={i} className="my-3 flex items-center gap-3">
          <div className="min-w-0 flex-1 overflow-x-auto py-1">
            <Math latex={line.display.value} display />
          </div>
          {line.display.status !== "from_tool" && <Mark status={line.display.status} />}
          <Num n={line.display.number} />
        </div>,
      );
      return;
    }
    const text = plain(line.nodes).trim();
    if (text.startsWith("|")) {
      if (/^\|?[\s:|-]+\|?$/.test(text)) return; // the `|---|` separator row
      if (list.length) end(i);
      table.push(cells(line.nodes));
      return;
    }
    const bullet = /^\s*(?:[-*]|\d+\.)\s+/;
    const first = line.nodes[0];
    if (first?.kind === "text" && bullet.test(first.text)) {
      if (table.length) end(i);
      list.push([{ kind: "text", text: first.text.replace(bullet, "") }, ...line.nodes.slice(1)]);
      return;
    }
    end(i);
    if (!text) return;
    if (first?.kind === "text" && /^#{1,6}\s/.test(first.text)) {
      const rest = [{ kind: "text" as const, text: first.text.replace(/^#{1,6}\s+/, "") }, ...line.nodes.slice(1)];
      out.push(
        <p key={i} className="mt-3 font-medium text-ink">
          <Inline nodes={rest} />
        </p>,
      );
      return;
    }
    out.push(
      <p key={i} className="my-2">
        <Inline nodes={line.nodes} />
      </p>,
    );
  });
  end(lines.length);
  return <div className="text-sm leading-relaxed text-ink/90">{out}</div>;
}

// --- a tool result ----------------------------------------------------------------------------------------------
const TOOL_NAMES: Record<string, string> = {
  search_handbook: "Handbook search",
  analyze_lagrangian: "Lagrangian analysis",
  find_conserved_quantities: "Search for conserved quantities",
  search_invariants: "Search for hidden constants of motion",
  geometry: "Differential geometry",
};

function evidenceTone(evidence: string): string {
  return evidence.startsWith("Not verified") ? "bg-warn-soft text-warn" : "bg-ok-soft text-ok";
}

function Result({ card }: { card: SessionCard }) {
  const name = TOOL_NAMES[card.tool] ?? card.title;
  if (card.sections.length === 0) {
    return (
      <p className="my-2 text-xs text-muted">
        <span className="font-medium">{name}</span>
        {card.status === "no_results" ? ": no handbook entry matched." : `: ${card.status.replaceAll("_", " ")}.`}
      </p>
    );
  }
  return (
    <div className="my-3 rounded-md border border-rule bg-paper p-3 sm:p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{name}</p>
      <div className="mt-2 space-y-3">
        {card.sections.map((section) => (
          <div key={section.title}>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-sm font-medium">{section.title}</p>
              <span
                className={`rounded-full px-2 py-0.5 text-[0.7rem] font-medium ${evidenceTone(section.evidence)}`}
              >
                {section.evidence}
              </span>
            </div>
            {section.note && <p className="mt-1 text-sm">{section.note}</p>}
            {section.rows.map((row) => (
              <div key={row.key} className="mt-1 flex items-center gap-3">
                <div className="min-w-0 flex-1 overflow-x-auto py-0.5">
                  <Math latex={row.latex} />
                </div>
                <Num n={card.numbers[row.key]} />
              </div>
            ))}
          </div>
        ))}
      </div>
      {card.restrictions_latex.length > 0 && (
        <p className="mt-3 text-xs text-muted">
          Valid where{" "}
          {card.restrictions_latex.map((r, i) => (
            <span key={r}>
              {i > 0 && ", "}
              <Math latex={r} />
            </span>
          ))}{" "}
          (where the coordinates break down, not a statement about the physics).
        </p>
      )}
    </div>
  );
}

/** One recorded research session: the person's questions and Minerva's replies, as they were. */
export function Conversation({ session, runs }: { session: Session; runs: number }) {
  return (
    <figure className="mt-8 overflow-hidden rounded-lg border border-rule bg-raised">
      <figcaption className="border-b border-rule px-5 py-4 sm:px-8">
        <span className="font-serif text-xl sm:text-2xl">{session.title}</span>
        <p className="mt-1 text-sm text-muted">
          {session.passed_runs} of {runs} recorded runs passed every turn. Shown: run{" "}
          {session.shown_run}, the first that did.
        </p>
      </figcaption>
      <ol className="space-y-6 p-5 sm:p-8">
        {session.turns.map((turn, i) => (
          <li key={i} className="space-y-3">
            <div className="flex justify-end">
              <p className="max-w-[85%] rounded-lg bg-sunk px-4 py-2 text-sm sm:text-base">
                {turn.user.text}
              </p>
            </div>
            <div className="min-w-0">
              <p className="text-xs font-medium uppercase tracking-wide text-muted">Minerva</p>
              {turn.assistant.blocks.map((block, j) =>
                block.kind === "card" ? (
                  <Result key={j} card={block} />
                ) : (
                  <Prose key={j} block={block} />
                ),
              )}
            </div>
          </li>
        ))}
      </ol>
    </figure>
  );
}
