"use client";

import { useEffect, useState } from "react";

import { chatApi } from "./chat-api";
import type { ChatStatus, ModelInfo } from "./chat-types";

/** What the local model `name` is: the status's identity when it is the default, else asked of the server. */
export function useLocalModelInfo(status: ChatStatus | null, name: string): ModelInfo | null {
  const [fetched, setFetched] = useState<{ name: string; info: ModelInfo | null } | null>(null);
  const known = status?.identity && status.identity.alias === name ? status.identity : null;
  useEffect(() => {
    // Only a server that reports identities has the endpoint; an older one gets the label without it.
    if (known || !name || !status?.enabled || !status.identity) return;
    let live = true;
    chatApi
      .modelInfo(name)
      .then((info) => live && setFetched({ name, info }))
      .catch(() => live && setFetched({ name, info: null }));
    return () => {
      live = false;
    };
  }, [known, name, status?.enabled, status?.identity]);
  if (known) return known;
  return fetched && fetched.name === name ? fetched.info : null;
}
