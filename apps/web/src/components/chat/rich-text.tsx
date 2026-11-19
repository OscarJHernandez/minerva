import { EquationNumber } from "./equation-number";
import { MARK_TEXT } from "@minerva/ui/marks";
import { Math } from "@minerva/ui/math";
import { type ReactNode, useEffect, useRef, useState } from "react";
import type { FormulaStatus, Segment } from "@/lib/chat-types";
import { FormulaButtons } from "./formula-actions";

const STYLE: Record<FormulaStatus, { underline: string; box: string; tone: string }> = {
  from_tool: { underline: "", box: "border-rule bg-raised", tone: "" },
  quoted: {
    underline: "border-b border-dotted border-muted",
    box: "border-rule bg-sunk",
    tone: "text-muted",
  },
  checked: {
    underline: "border-b border-dotted border-ok",
    box: "border-ok/40 bg-ok-soft/40",
    tone: "text-ok",
  },
  conditional: {
    underline: "border-b border-dashed border-warn",
    box: "border-warn/50 bg-warn-soft/40",
    tone: "text-warn",
  },
  contradicted: {
    underline: "border-b border-solid border-bad",
    box: "border-bad/50 bg-bad-soft/50",
    tone: "text-bad",
  },
  unchecked: {
    underline: "border-b border-dotted border-warn",
    box: "border-warn/50 bg-warn-soft/40",
    tone: "text-warn",
  },
};

/** Each mark's words (shared with the landing page, `@minerva/ui/marks`) and its look in the chat. */
export const MARKS = Object.fromEntries(
  (Object.keys(STYLE) as FormulaStatus[]).map((s) => [s, { ...MARK_TEXT[s], ...STYLE[s] }]),
) as Record<
  FormulaStatus,
  { label: string | null; title: string; underline: string; box: string; tone: string }
>;

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

/**
 * `*italic*` or `_italic_` inside one run of text. The marker must touch the words it wraps and must not
 * sit inside a word, so `2 * 3 * 4`, `snake_case_name` and an unmatched `*` stay as typed. It never spans a
 * formula: a run of text between two formulas is the most it can cover.
 */
const EMPHASIS = /((?<![\w*])\*(?=\S)[^*\n]*?\S\*(?![\w*])|(?<!\w)_(?=\S)[^_\n]*?\S_(?!\w))/;

/** Inline markup: `**bold**` (which may enclose formulas), `*italic*` and `code`. Block markup is in `RichText`. */
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
        piece.split(EMPHASIS).forEach((part, ei) => {
          if (part === "") return;
          const ekey = `${key}-${ei}`;
          const italic = ei % 2 === 1; // odd entries are the captured emphasis
          const content = italic ? <em key={ekey}>{part.slice(1, -1)}</em> : part;
          out.push(bold ? <strong key={ekey}>{content}</strong> : content);
        });
      }
    });
  });
  return out;
}

const BULLET = /^\s*[*-]\s+/;

/** A lone symbol (`r`, `\\theta`, `r_0`) names something: nothing to verify, flag or copy. */
const LONE_SYMBOL = /^\\?[A-Za-z]+(?:[_^]\{?[A-Za-z0-9]+\}?)?$/;
const isLoneSymbol = (latex: string) => LONE_SYMBOL.test(latex.trim());

function Formula({ seg }: { seg: Segment }) {
  const [open, setOpen] = useState(false);
  const [barOpen, setBarOpen] = useState(false);
  const [alignEnd, setAlignEnd] = useState(false);
  const host = useRef<HTMLSpanElement>(null);
  const bar = useRef<HTMLSpanElement>(null);
  // A tap outside the formula closes the bar that a tap opened (hover and focus need no state).
  useEffect(() => {
    if (!barOpen) return;
    const away = (event: PointerEvent) => {
      if (!host.current?.contains(event.target as globalThis.Node)) setBarOpen(false);
    };
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [barOpen]);
  // `null`: written by the user or still streaming. Minerva has made no claim about it.
  if (seg.status === null) return <Math latex={seg.value} />;
  const status = seg.status ?? "unchecked";
  const mark = MARKS[status];
  // A lone symbol names something; it asserts nothing, so there is nothing to verify or flag.
  const plain = status === "unchecked" && !isClaim(seg.value);
  const actions = !isLoneSymbol(seg.value);
  // Keep the bar inside the screen: it opens to the right of the formula's start unless that runs off.
  const place = () => {
    const box = host.current?.getBoundingClientRect();
    const width = bar.current?.offsetWidth ?? 0;
    if (box) setAlignEnd(box.left + width > window.innerWidth - 8);
  };
  return (
    <span
      ref={host}
      className="group/formula relative"
      onMouseEnter={actions ? place : undefined}
      onFocus={actions ? place : undefined}
    >
      <span
        title={plain ? undefined : explain(seg)}
        data-status={plain ? "symbol" : status}
        className={plain ? undefined : mark.underline || undefined}
        onClick={
          actions
            ? () => {
                place();
                setBarOpen((o) => !o);
              }
            : undefined
        }
      >
        <Math latex={seg.value} />
        {!plain && mark.label && (
          // A button, so that a keyboard or a finger can read what the mark means (a hover tooltip cannot).
          <sup className="ml-0.5">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                setOpen((o) => !o);
              }}
              aria-expanded={open}
              aria-label={`${status === "unchecked" ? "unchecked" : mark.label}: what this mark means`}
              className={`text-[10px] font-medium ${mark.tone}`}
            >
              {status === "unchecked" ? "?" : mark.label}
            </button>
          </sup>
        )}
      </span>
      {actions && (
        // Out of the line: it takes no room, so the formula and the sentence around it stay together.
        <span
          ref={bar}
          data-testid="formula-actions"
          className={`absolute bottom-full z-10 mb-1 whitespace-nowrap rounded-md border border-rule bg-raised px-2 py-1 text-xs shadow-sm ${
            alignEnd ? "right-0" : "left-0"
          } ${
            barOpen
              ? "opacity-100"
              : "pointer-events-none opacity-0 group-focus-within/formula:pointer-events-auto group-focus-within/formula:opacity-100 group-hover/formula:pointer-events-auto group-hover/formula:opacity-100"
          }`}
        >
          <FormulaButtons latex={seg.value} />
        </span>
      )}
      {open && (
        <span role="note" className="mx-1 text-xs text-muted" data-testid="mark-explanation">
          ({explain(seg)})
        </span>
      )}
    </span>
  );
}

function DisplayFormula({ seg }: { seg: Segment }) {
  const mark = MARKS[seg.status ?? "from_tool"];
  return (
    <div data-status={seg.status} className={`my-2 rounded-md border px-3 py-1 ${mark.box}`}>
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <Math latex={seg.value} display />
        </div>
        {seg.number != null && <EquationNumber n={seg.number} />}
      </div>
      <div className="flex items-start justify-between gap-3 pb-1">
        {mark.label ? (
          <div className={`text-xs ${mark.tone}`} title={mark.title}>
            {seg.status === "quoted"
              ? `Quoted from a tool result, not a verified result${seg.note ? `: ${seg.note}` : ""}`
              : seg.status === "checked"
                ? "Written by the model; checked numerically by Minerva"
                : seg.status === "conditional"
                  ? `Written by the model; ${seg.note ?? "true only under extra assumptions"}`
                  : seg.status === "contradicted"
                    ? `Written by the model; ${seg.note ?? "Minerva's check says this does not hold"}`
                    : `Not verified by Minerva (written by the model)${seg.note ? `: ${seg.note}` : ""}`}
          </div>
        ) : (
          <span />
        )}
        <FormulaButtons latex={seg.value} />
      </div>
    </div>
  );
}

const HEADING = /^\s{0,3}(#{1,6})\s+/;
const NUMBERED = /^\s*(\d{1,3})[.)]\s+/;
const SEPARATOR = /^\|?(?:\s*:?-{2,}:?\s*\|)+\s*(?::?-{2,}:?)?\s*\|?$/;

type Align = "left" | "center" | "right";

const firstText = (line: Line): string | null => {
  if ("display" in line) return null;
  const first = line.nodes[0];
  return first?.kind === "text" ? first.text : null;
};
const isTableRow = (line: Line): boolean => firstText(line)?.trimStart().startsWith("|") ?? false;
const isSeparatorRow = (line: Line): boolean =>
  !("display" in line) &&
  line.nodes.every((n) => n.kind === "text") &&
  SEPARATOR.test(
    line.nodes
      .map((n) => (n.kind === "text" ? n.text : ""))
      .join("")
      .trim(),
  );

/** The cells of a row: only text is split on `|`, so a formula such as `$|x|$` never breaks a cell. */
function splitCells(nodes: Node[]): Node[][] {
  const cells: Node[][] = [[]];
  for (const n of nodes) {
    if (n.kind === "math") {
      cells[cells.length - 1]!.push(n);
      continue;
    }
    n.text.split("|").forEach((part, i) => {
      if (i > 0) cells.push([]);
      if (part !== "") cells[cells.length - 1]!.push({ kind: "text", text: part });
    });
  }
  const blank = (cell: Node[]) => cell.every((n) => n.kind === "text" && n.text.trim() === "");
  if (cells.length > 0 && blank(cells[0]!)) cells.shift(); // a row written | a | b | starts and ends with a bar
  if (cells.length > 0 && blank(cells[cells.length - 1]!)) cells.pop();
  return cells.map(trimCell);
}

/** The spaces around a cell's text belong to the bars, not to the cell. */
function trimCell(cell: Node[]): Node[] {
  const out = [...cell];
  const first = out[0];
  if (first?.kind === "text") out[0] = { ...first, text: first.text.trimStart() };
  const last = out[out.length - 1];
  if (last?.kind === "text") out[out.length - 1] = { ...last, text: last.text.trimEnd() };
  return out.filter((n) => n.kind === "math" || n.text !== "");
}

function alignments(line: Line): Align[] {
  if ("display" in line) return [];
  const text = line.nodes.map((n) => (n.kind === "text" ? n.text : "")).join("");
  return text
    .split("|")
    .map((c) => c.trim())
    .filter((c) => c !== "")
    .map((c) =>
      c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : "left",
    );
}

const ALIGN_CLASS: Record<Align, string> = {
  left: "text-left",
  center: "text-center",
  right: "text-right",
};

function Table({
  head,
  align,
  rows,
  at,
}: {
  head: Line;
  align: Align[];
  rows: Line[];
  at: number;
}) {
  const cells = (line: Line) => ("display" in line ? [] : splitCells(line.nodes));
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-rule">
          <tr>
            {cells(head).map((cell, ci) => (
              <th
                key={ci}
                scope="col"
                className={`px-2 py-1 font-medium ${ALIGN_CLASS[align[ci] ?? "left"]}`}
              >
                {renderNodes(cell, `t${at}h${ci}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-t border-rule">
              {cells(row).map((cell, ci) => (
                <td key={ci} className={`px-2 py-1 align-top ${ALIGN_CLASS[align[ci] ?? "left"]}`}>
                  {renderNodes(cell, `t${at}r${ri}c${ci}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RichText({ segments }: { segments: Segment[] }) {
  const blocks: ReactNode[] = [];
  let items: ReactNode[] = [];
  let listKind: "ul" | "ol" = "ul";
  let listStart = 1;
  const closeList = (at: number) => {
    if (items.length > 0) {
      blocks.push(
        listKind === "ul" ? (
          <ul key={`ul${at}`} className="ml-5 list-disc space-y-1.5 marker:text-muted">
            {items}
          </ul>
        ) : (
          <ol
            key={`ol${at}`}
            start={listStart}
            className="ml-5 list-decimal space-y-1.5 marker:text-muted"
          >
            {items}
          </ol>
        ),
      );
      items = [];
    }
  };
  const withoutMarker = (line: { nodes: Node[] }, marker: RegExp): Node[] => {
    const first = line.nodes[0];
    return first?.kind === "text"
      ? [{ ...first, text: first.text.replace(marker, "") }, ...line.nodes.slice(1)]
      : line.nodes;
  };
  const lines = toLines(segments);
  for (let li = 0; li < lines.length; li++) {
    const line = lines[li]!;
    if ("display" in line) {
      closeList(li);
      blocks.push(<DisplayFormula key={`d${line.key}`} seg={line.display} />);
      continue;
    }
    const next = lines[li + 1];
    if (isTableRow(line) && next !== undefined && isSeparatorRow(next)) {
      closeList(li);
      let end = li + 2;
      while (end < lines.length && isTableRow(lines[end]!) && !isSeparatorRow(lines[end]!)) end++;
      blocks.push(
        <Table
          key={`tb${li}`}
          at={li}
          head={line}
          align={alignments(next)}
          rows={lines.slice(li + 2, end)}
        />,
      );
      li = end - 1;
      continue;
    }
    const text = firstText(line);
    const heading = text === null ? null : HEADING.exec(text);
    if (heading) {
      closeList(li);
      const level = heading[1]!.length;
      const nodes = withoutMarker(line, HEADING);
      blocks.push(
        level <= 3 ? (
          <h3 key={`h${li}`} className="pt-1 font-serif text-lg">
            {renderNodes(nodes, `l${li}`)}
          </h3>
        ) : (
          <h4 key={`h${li}`} className="pt-1 font-medium">
            {renderNodes(nodes, `l${li}`)}
          </h4>
        ),
      );
      continue;
    }
    if (text !== null && BULLET.test(text)) {
      if (listKind !== "ul") closeList(li);
      listKind = "ul";
      items.push(<li key={`li${li}`}>{renderNodes(withoutMarker(line, BULLET), `l${li}`)}</li>);
      continue;
    }
    const numbered = text === null ? null : NUMBERED.exec(text);
    if (numbered) {
      if (listKind !== "ol") closeList(li);
      if (items.length === 0) listStart = Number(numbered[1]);
      listKind = "ol";
      items.push(<li key={`li${li}`}>{renderNodes(withoutMarker(line, NUMBERED), `l${li}`)}</li>);
      continue;
    }
    closeList(li);
    blocks.push(<p key={`p${li}`}>{renderNodes(line.nodes, `l${li}`)}</p>);
  }
  closeList(blocks.length);
  return <div className="space-y-2.5 leading-relaxed">{blocks}</div>;
}
