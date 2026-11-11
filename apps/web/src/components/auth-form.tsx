"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ApiError, api } from "@/lib/api";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const isRegister = mode === "register";

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (isRegister ? api.register(email, password) : api.login(email, password));
      router.push("/");
      router.refresh();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto mt-16 max-w-sm">
      <h1 className="font-serif text-3xl">{isRegister ? "Create an account" : "Log in"}</h1>
      <p className="mt-2 text-sm text-muted">
        {isRegister
          ? "Your saved calculations are private to your account."
          : "Welcome back to your workspace."}
      </p>
      <form onSubmit={submit} className="mt-8 space-y-4" noValidate>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Email</span>
          <input
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded border border-rule bg-raised px-3 py-2"
          />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Password</span>
          <input
            type="password"
            autoComplete={isRegister ? "new-password" : "current-password"}
            required
            minLength={10}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded border border-rule bg-raised px-3 py-2"
          />
          {isRegister && (
            <span className="mt-1 block text-xs text-muted">At least 10 characters.</span>
          )}
        </label>
        {error && (
          <p role="alert" className="rounded bg-bad-soft px-3 py-2 text-sm text-bad">
            {error}
          </p>
        )}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-60"
        >
          {busy ? "Please wait…" : isRegister ? "Create account" : "Log in"}
        </button>
      </form>
      <p className="mt-6 text-sm text-muted">
        {isRegister ? (
          <>
            Already registered?{" "}
            <Link href="/login" className="text-accent hover:underline">
              Log in
            </Link>
          </>
        ) : (
          <>
            New here?{" "}
            <Link href="/register" className="text-accent hover:underline">
              Create an account
            </Link>
          </>
        )}
      </p>
    </div>
  );
}
