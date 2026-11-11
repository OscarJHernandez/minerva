import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses events split across arbitrary chunk boundaries", () => {
    const parser = new SseParser();
    const raw =
      'event: text\ndata: {"delta":"Hel"}\n\nevent: text\ndata: {"delta":"lo"}\n\nevent: done\ndata: {}\n\n';
    const events = [];
    for (let i = 0; i < raw.length; i += 7) events.push(...parser.push(raw.slice(i, i + 7)));
    expect(events.map((e) => e.event)).toEqual(["text", "text", "done"]);
    expect(events[0]).toEqual({ event: "text", data: { delta: "Hel" } });
  });

  it("handles CRLF and ignores malformed or empty blocks", () => {
    const parser = new SseParser();
    const events = parser.push(
      'event: error\r\ndata: {"message":"x"}\r\n\r\nevent: text\ndata: not json\n\n: comment only\n\n',
    );
    expect(events).toEqual([{ event: "error", data: { message: "x" } }]);
  });
});
