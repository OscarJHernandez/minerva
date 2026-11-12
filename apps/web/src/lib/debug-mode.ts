"use client";

import { useCallback, useEffect, useState } from "react";

const KEY = "minerva.debug";

function read(): boolean {
  try {
    if (new URLSearchParams(window.location.search).get("debug") === "1") {
      window.localStorage.setItem(KEY, "1");
      return true;
    }
    return window.localStorage.getItem(KEY) === "1";
  } catch {
    return false; // storage blocked: debug mode is simply off
  }
}

/** Debug mode: per-browser, off by default, switched on with the toggle or `?debug=1`. */
export function useDebugMode(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(false);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reading browser storage after mount
    setOn(read());
  }, []);
  const set = useCallback((value: boolean) => {
    setOn(value);
    try {
      if (value) window.localStorage.setItem(KEY, "1");
      else window.localStorage.removeItem(KEY);
    } catch {
      /* not persisted; still applies to this page */
    }
  }, []);
  return [on, set];
}
