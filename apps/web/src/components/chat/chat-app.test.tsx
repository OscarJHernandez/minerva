import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatApp } from "./chat-app";

const sendMessage = vi.fn();
const get = vi.fn();
const router = { push: vi.fn() };
vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("@/lib/api", async (original) => ({
  ...(await original<typeof import("@/lib/api")>()),
  api: { me: async () => ({ id: "user-1", email: "u@example.test" }) },
}));
vi.mock("@/lib/chat-api", () => ({
  sendMessage: (...args: unknown[]) => sendMessage(...args),
  chatApi: {
    status: async () => ({ enabled: true, images: false }),
    list: async () => [
      { id: "a", title: "First", updated_at: "2026-01-01" },
      { id: "b", title: "Second", updated_at: "2026-01-01" },
    ],
    get: (...args: unknown[]) => get(...args),
  },
}));
vi.mock("./composer", () => ({
  Composer: ({ onSend, disabled }: { onSend: (text: string) => void; disabled: boolean }) => (
    <button disabled={disabled} onClick={() => onSend("Question")}>
      Send mock
    </button>
  ),
}));
vi.mock("./message-view", () => ({
  MessageView: ({ message }: { message: { content: { text?: string } } }) => (
    <p>{message.content.text}</p>
  ),
}));

describe("ChatApp conversation isolation", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    sendMessage.mockReset();
    get.mockReset();
  });
  it("keeps the newest conversation when an earlier load resolves last", async () => {
    let finishOld!: () => void;
    const old = new Promise<void>((resolve) => {
      finishOld = resolve;
    });
    get.mockImplementation(async (id: string) => {
      if (id === "a") await old;
      return { id, messages: [{ id, role: "user", content: { text: id + " saved" } }] };
    });
    render(<ChatApp />);
    fireEvent.click((await screen.findByText("First")).closest("button")!);
    expect(screen.getByText("Send mock")).toBeDisabled();
    fireEvent.click(screen.getByText("Second").closest("button")!);
    expect(await screen.findByText("b saved")).toBeInTheDocument();
    await act(async () => {
      finishOld();
      await old;
    });
    expect(screen.queryByText("a saved")).not.toBeInTheDocument();
    expect(screen.getByText("Send mock")).toBeEnabled();
  });

  it("ignores a late streamed message after switching conversations", async () => {
    let onEvent: ((event: unknown) => void) | undefined;
    let finish: (() => void) | undefined;
    sendMessage.mockImplementation((_id, _text, callback) => {
      onEvent = callback;
      return new Promise<void>((resolve) => {
        finish = resolve;
      });
    });
    get.mockImplementation(async (id: string) => ({
      id,
      title: id,
      messages: [
        {
          id: id + "-message",
          role: "user",
          content: { text: id + " saved" },
          created_at: "2026-01-01",
        },
      ],
    }));
    render(<ChatApp />);
    fireEvent.click((await screen.findByText("First")).closest("button")!);
    expect(await screen.findByText("a saved")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Send mock"));
    await waitFor(() => expect(sendMessage).toHaveBeenCalled());
    fireEvent.click(screen.getByText("Second").closest("button")!);
    expect(await screen.findByText("b saved")).toBeInTheDocument();
    onEvent?.({ event: "message", data: { text: "late reply", blocks: [], trace: [] } });
    finish?.();
    expect(screen.queryByText("late reply")).not.toBeInTheDocument();
    expect(screen.getByText("b saved")).toBeInTheDocument();
  });
});
