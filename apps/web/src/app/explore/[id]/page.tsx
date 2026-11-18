import { notFound } from "next/navigation";
import { findExploration } from "@minerva/ui/explorations";
import { ChatApp } from "@/components/chat/chat-app";

/**
 * "Explore in Minerva": a verified example from the landing page, continued in the chat. The page opens a new
 * conversation with the example's prompt (metric, coordinates and assumptions included) in the message box and
 * sends nothing: the user decides to run it, and every result then comes from the engine like any other message.
 */
export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const exploration = findExploration(id);
  if (!exploration) notFound();
  return <ChatApp initialDraft={exploration.prompt} />;
}
