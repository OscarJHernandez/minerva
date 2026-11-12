import katex from "katex";

/**
 * Display-only aliases for the constants Minerva's parser accepts (`\\exponentialE`, `\\imaginaryI`; MathLive
 * emits `\\differentialD`). Output from the engine already uses `\\mathrm{e}` and `\\mathrm{i}`.
 */
const MACROS = {
  "\\exponentialE": "\\mathrm{e}",
  "\\imaginaryI": "\\mathrm{i}",
  "\\differentialD": "\\mathrm{d}",
};

function render(latex: string, display: boolean): { html: string } | { error: string } {
  try {
    return {
      html: katex.renderToString(latex, {
        displayMode: display,
        throwOnError: true,
        trust: false,
        strict: "ignore",
        output: "htmlAndMathml",
        maxExpand: 1000,
        maxSize: 50,
        macros: { ...MACROS },
      }),
    };
  } catch (e) {
    return { error: e instanceof Error ? e.message : "unknown error" };
  }
}

/**
 * Renders LaTeX with KaTeX. `trust` is off, so KaTeX output contains no scripts, links or
 * raw HTML from the input; this is why `dangerouslySetInnerHTML` is acceptable here.
 * Works on the server and in the browser (no effects), so static pages ship pre-rendered math.
 * LaTeX KaTeX cannot render is shown as its source, marked `data-math-error`, never as a formula.
 */
export function Math({
  latex,
  display = false,
  className,
  label,
}: {
  latex: string;
  display?: boolean;
  className?: string;
  label?: string;
}) {
  const Tag = display ? "div" : "span";
  const out = render(latex, display);
  if ("error" in out) {
    return (
      <Tag className={className}>
        <code
          data-math-error=""
          title={`This LaTeX cannot be displayed: ${out.error}`}
          className="text-bad underline decoration-dotted"
        >
          {latex}
        </code>
      </Tag>
    );
  }
  return (
    <Tag
      className={className}
      aria-label={label}
      role={label ? "math" : undefined}
      dangerouslySetInnerHTML={{ __html: out.html }}
    />
  );
}
