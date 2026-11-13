import Link from "next/link";
import type { Capability } from "@/lib/capabilities";

/** A round mark: a check for what exists, an exclamation mark for what is only planned. */
function Mark({ planned }: { planned: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`inline-flex size-5 shrink-0 items-center justify-center rounded-full ${
        planned ? "bg-warn-soft text-warn" : "bg-ok-soft text-ok"
      }`}
    >
      <svg viewBox="0 0 16 16" className="size-3" fill="none" stroke="currentColor" strokeWidth="2">
        {planned ? (
          <path d="M8 3.5v5M8 12v.01" strokeLinecap="round" />
        ) : (
          <path d="M3.5 8.5l3 3 6-6.5" strokeLinecap="round" strokeLinejoin="round" />
        )}
      </svg>
    </span>
  );
}

/** One capability: the mark and the title, with the description folded away until asked for. */
export function CapabilityRow({ cap }: { cap: Capability }) {
  const planned = cap.status === "planned";
  return (
    <details className="group border-b border-rule">
      <summary className="flex cursor-pointer list-none items-center gap-3 py-3 [&::-webkit-details-marker]:hidden">
        <Mark planned={planned} />
        <span className="flex-1 font-medium">{cap.title}</span>
        <span className="sr-only">{planned ? "(planned)" : "(available)"}</span>
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          className="size-4 shrink-0 text-muted transition-transform group-open:rotate-90"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M6 3.5l4.5 4.5L6 12.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </summary>
      <div className="pb-4 pl-8 text-sm text-muted">
        <p>{cap.summary}</p>
        {cap.needs_model && <p className="mt-2">Self-hosted; needs a model you connect.</p>}
        {!planned && (
          <p className="mt-2">
            <Link href={`/docs#${cap.id}`} className="underline hover:text-ink">
              Examples and limits
            </Link>
          </p>
        )}
      </div>
    </details>
  );
}
