"use client";

import { useSyncExternalStore } from "react";
import type { ProviderChoice } from "./chat-types";

/**
 * Which model answers, kept in memory for this tab only.
 *
 * The OpenRouter key and the cloud consent are never written to localStorage, sessionStorage or a
 * cookie: they are gone when the page is closed or reloaded, and cleared on sign-out. Only the chosen
 * model id (not secret) is remembered, as a convenience.
 */
export interface ModelSettings {
  provider: "ollama" | "openrouter";
  model: string;
  apiKey: string;
  consent: boolean;
  /** False until the user picks a provider: then the server's default applies. */
  touched: boolean;
}

const MODEL_KEY = "minerva.openrouter.model";

function savedModel(): string {
  try {
    return window.localStorage.getItem(MODEL_KEY) ?? "";
  } catch {
    return "";
  }
}

const initial: ModelSettings = {
  provider: "ollama",
  model: "",
  apiKey: "",
  consent: false,
  touched: false,
};
let state: ModelSettings = initial;
const listeners = new Set<() => void>();

function set(next: Partial<ModelSettings>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

export const modelSettings = {
  get: () => state,
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  useCloud(model: string) {
    set({ provider: "openrouter", model, touched: true });
    try {
      if (model) window.localStorage.setItem(MODEL_KEY, model);
    } catch {
      /* not remembered */
    }
  },
  useLocal: () => set({ provider: "ollama", touched: true }),
  setModel(model: string) {
    set({ model });
    try {
      if (model) window.localStorage.setItem(MODEL_KEY, model);
    } catch {
      /* not remembered */
    }
  },
  setApiKey: (apiKey: string) => set({ apiKey }),
  setConsent: (consent: boolean) => set({ consent }),
  savedModel,
  /** Forget the key and consent (sign-out). Back to the local model. */
  clear() {
    state = initial;
    listeners.forEach((l) => l());
  },
};

export function useModelSettings(): ModelSettings {
  return useSyncExternalStore(modelSettings.subscribe, modelSettings.get, () => initial);
}

/** The provider choice to send with a message, or undefined to use the server's default. */
export function providerChoice(s: ModelSettings): ProviderChoice | undefined {
  if (!s.touched) return undefined;
  if (s.provider === "ollama") return { kind: "ollama" };
  return {
    kind: "openrouter",
    model: s.model || undefined,
    api_key: s.apiKey || undefined,
    cloud_consent: s.consent,
  };
}

const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,99}$/;
export const isValidModelId = (id: string) => MODEL_ID.test(id);
