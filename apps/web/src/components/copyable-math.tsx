"use client";

import { Math } from "@minerva/ui/math";
import { FormulaButtons } from "@/components/chat/formula-actions";

/**
 * `Math` for a result the user may want to reuse: a displayed formula gets "Copy LaTeX" (and "Edit
 * in chat" inside a conversation). The text copied is exactly the LaTeX that was rendered.
 * Inline formulas are left alone: they sit in sentences and tables where buttons would be noise.
 */
export function CopyMath({
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
  if (!display) return <Math latex={latex} className={className} label={label} />;
  return (
    <div className="group/formula">
      <Math latex={latex} display className={className} label={label} />
      <div className="flex justify-end">
        <FormulaButtons latex={latex} subtle />
      </div>
    </div>
  );
}
