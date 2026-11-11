import katex from "katex";

/**
 * Renders LaTeX with KaTeX. `trust` is off, so KaTeX output contains no scripts, links or
 * raw HTML from the input; this is why `dangerouslySetInnerHTML` is acceptable here.
 * Works on the server and in the browser (no effects), so static pages ship pre-rendered math.
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
  const html = katex.renderToString(latex, {
    displayMode: display,
    throwOnError: false,
    trust: false,
    strict: "ignore",
    output: "htmlAndMathml",
    maxExpand: 1000,
    maxSize: 50,
  });
  const Tag = display ? "div" : "span";
  return (
    <Tag
      className={className}
      aria-label={label}
      role={label ? "math" : undefined}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
