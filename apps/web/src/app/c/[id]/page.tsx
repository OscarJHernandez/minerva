import { ChatApp } from "@/components/chat/chat-app";
import { isConversationId } from "@/lib/next-path";

/** A conversation at its own address: reloading, or logging in again, comes back to it. */
export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ChatApp initialId={isConversationId(id) ? id : undefined} />;
}
