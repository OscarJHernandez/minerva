"use client";

import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

export function useFormDraft<T>(
  form: T,
  setForm: Dispatch<SetStateAction<T>>,
  key?: string,
  restoring = false,
) {
  const [loaded, setLoaded] = useState<string>();
  useEffect(() => {
    if (!key) return;
    const timer = setTimeout(() => {
      try {
        const raw = sessionStorage.getItem(key);
        if (raw && !restoring) setForm(JSON.parse(raw) as T);
      } catch {
        // A missing or corrupt draft leaves the form usable.
      }
      setLoaded(key);
    }, 0);
    return () => clearTimeout(timer);
  }, [key, restoring, setForm]);

  useEffect(() => {
    if (!key || loaded !== key) return;
    try {
      sessionStorage.setItem(key, JSON.stringify(form));
    } catch {
      // Storage is optional.
    }
  }, [form, key, loaded]);
}
