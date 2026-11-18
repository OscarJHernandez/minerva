import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Minerva — Explore mathematics. Own the process.",
  description:
    "Minerva is an open-source mathematical assistant you can run yourself to explore ideas, try different approaches, and verify results on your terms.",
  keywords: [
    "mathematical assistant",
    "math exploration",
    "open source",
    "self-hosted",
    "computer algebra",
    "symbolic computation",
    "SymPy",
    "mathematical identities",
    "tensor algebra",
    "Ollama",
    "local LLM",
  ],
  openGraph: {
    title: "Minerva — Explore mathematics. Own the process.",
    description:
      "Explore mathematics on your own terms. Self-host Minerva, choose what to investigate, and verify the results using open-source tools.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-paper text-ink">{children}</body>
    </html>
  );
}
