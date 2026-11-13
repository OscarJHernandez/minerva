import type { Verified } from "@/lib/types";

const LABELS: Record<Verified, { text: string; title: string; tone: string }> = {
  symbolic: {
    text: "Symbolically verified",
    title: "SymPy reduced the difference of both sides to zero, and sampling agreed.",
    tone: "bg-ok-soft text-ok",
  },
  numeric: {
    text: "Numerically verified",
    title: "Checked by numerical evaluation at sample points; no symbolic proof was found.",
    tone: "bg-ok-soft text-ok",
  },
  structural: {
    text: "Structurally verified",
    title: "Checked with the tensor canonicalizer.",
    tone: "bg-ok-soft text-ok",
  },
  dimensional: {
    text: "Dimensionally checked",
    title: "Both sides have the same physical dimensions. This does not prove the equation.",
    tone: "bg-warn-soft text-warn",
  },
  distributional: {
    text: "Verified as distributions",
    title:
      "Equal as distributions: exact conditions were established and a test-function pairing agreed. This is not a pointwise check.",
    tone: "bg-ok-soft text-ok",
  },
  stated: {
    text: "Stated, not machine-checked",
    title: "Recorded from the literature. Minerva has not verified it and never applies it.",
    tone: "bg-sunk text-muted",
  },
};

export function VerificationBadge({ level }: { level: Verified }) {
  const l = LABELS[level];
  return (
    <span
      title={l.title}
      className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${l.tone}`}
    >
      {l.text}
    </span>
  );
}
