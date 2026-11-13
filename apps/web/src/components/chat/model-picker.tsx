"use client";

import type { ChatStatus } from "@/lib/chat-types";
import { useModelSettings } from "@/lib/model-settings";

/** The compact chip in the composer: which model will answer. Opens the settings panel. */
export function ModelPicker({
  status,
  open,
  onToggle,
}: {
  status: ChatStatus | null;
  open: boolean;
  onToggle: () => void;
}) {
  const s = useModelSettings();
  const defaultCloud = status?.providers?.default === "openrouter";
  const cloud = s.touched ? s.provider === "openrouter" : defaultCloud;
  const label = cloud
    ? `OpenRouter · ${s.touched && s.model ? s.model : (status?.providers?.openrouter.default_model ?? "")}`
    : `Local · ${status?.providers?.ollama.model || status?.model || "model"}`;
  const bad = !cloud && status?.providers?.ollama.reachable === false;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={`Model: ${label}. Change model`}
      title="Change model"
      className="flex h-9 min-w-0 max-w-[11rem] shrink items-center gap-1.5 rounded-lg px-2.5 text-sm text-muted hover:bg-sunk hover:text-ink sm:max-w-[16rem]"
    >
      <span
        aria-hidden="true"
        className={`h-2 w-2 shrink-0 rounded-full ${bad ? "bg-bad" : cloud ? "bg-warn" : "bg-ok"}`}
      />
      <span className="truncate">{label}</span>
    </button>
  );
}
