import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Minerva — an open-source, self-hostable mathematical workbench",
  description:
    "Open-source mathematical workbench for verified symbolic computation, identity search with explicit assumptions, and tensor symmetries. Self-host it and connect your own local model.",
  keywords: [
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
    title: "Minerva — an open-source, self-hostable mathematical workbench",
    description:
      "Verified symbolic computation and a citable handbook of mathematics and physics. Run it on your own infrastructure and keep your work under your control.",
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
