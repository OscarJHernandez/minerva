"use client";

/** A status line for something just deleted: what happened, and one click to take it back. */
export function UndoToast({
  message,
  onUndo,
  onDismiss,
}: {
  message: string;
  onUndo: () => void;
  onDismiss: () => void;
}) {
  return (
    <p
      role="status"
      className="flex flex-wrap items-center gap-3 rounded border border-rule bg-accent-soft px-3 py-2 text-sm"
    >
      <span>{message}</span>
      <button type="button" onClick={onUndo} className="font-medium text-accent hover:underline">
        Undo
      </button>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss"
        className="ml-auto text-muted hover:text-ink"
      >
        ✕
      </button>
    </p>
  );
}
