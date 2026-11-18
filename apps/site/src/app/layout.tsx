import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Minerva — Explore mathematics on your own terms",
  description:
    "An open-source mathematical assistant you control. Explore ideas, verify computations, and choose your own methods. Self-host Minerva and bring your own AI model.",
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
    title: "Minerva — Explore mathematics on your own terms",
    description:
      "An open-source mathematical assistant you run on your own infrastructure. You direct the explorations, choose the model, and verify the mathematics.",
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
