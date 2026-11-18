import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatApp } from "./chat-app";

const sendMessage = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("@/lib/api", async (original) => ({
  ...(await original<typeof import("@/lib/api")>()),
  api: { me: async () => ({ id: "user-1", email: "u@example.test" }) },
}));
vi.mock("@/lib/chat-api", () => ({
  sendMessage: (...args: unknown[]) => sendMessage(...args),
  chatApi: {
    status: async () => ({ enabled: true, images: false }),
    list: async () => [],
    create: async () => ({ id: "new-conversation" }),
    get: async () => ({ id: "x", messages: [] }),
  },
}));
vi.mock("./composer", () => ({
  Composer: ({
    onSend,
    disabled,
    replaceRequest,
  }: {
    onSend: (text: string) => void;
    disabled: boolean;
    replaceRequest?: { id: number; text: string } | null;
  }) => (
    <div>
      <output data-testid="box">{replaceRequest?.text ?? ""}</output>
      <button disabled={disabled} onClick={() => onSend(replaceRequest?.text ?? "")}>
        Send mock
      </button>
    </div>
  ),
}));

const PROMPT = "Using the Schwarzschild metric ... compute the Ricci tensor.";

describe("ChatApp with a prefilled prompt (Explore in Minerva)", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    sendMessage.mockReset();
  });

  it("puts the prompt in the message box and does not send it", async () => {
    render(<ChatApp initialDraft={PROMPT} />);
    await waitFor(() => expect(screen.getByTestId("box")).toHaveTextContent(PROMPT));
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("sends the prompt only when the user sends it, and does not put it back afterwards", async () => {
    sendMessage.mockResolvedValue(undefined);
    render(<ChatApp initialDraft={PROMPT} />);
    await waitFor(() => expect(screen.getByTestId("box")).toHaveTextContent(PROMPT));
    await waitFor(() => expect(screen.getByText("Send mock")).toBeEnabled());
    fireEvent.click(screen.getByText("Send mock"));
    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    expect(sendMessage.mock.calls[0]?.[1]).toBe(PROMPT);
    await waitFor(() => expect(screen.getByTestId("box")).toBeEmptyDOMElement());
  });

  it("leaves the box alone without a draft", async () => {
    render(<ChatApp />);
    await waitFor(() => expect(screen.getByText("Send mock")).toBeEnabled());
    expect(screen.getByTestId("box")).toBeEmptyDOMElement();
  });
});
