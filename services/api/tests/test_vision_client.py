import asyncio
import base64
import json
from typing import Any

import httpx
import pytest
from minerva_api.agent.llm import LLMError
from minerva_api.agent.vision import MAX_FORMULAS, OllamaFormulaReader, clean_candidates


def reader_for(content: Any, status: int = 200) -> OllamaFormulaReader:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = content if isinstance(content, bytes) else json.dumps(content).encode()
        return httpx.Response(status, content=body)

    r = OllamaFormulaReader(
        "http://model.test:11434/", "vis", transport=httpx.MockTransport(handler)
    )
    r.seen = seen  # type: ignore[attr-defined]
    return r


def reply(formulas: Any) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": json.dumps({"formulas": formulas})}}


def test_request_carries_the_vision_model_the_image_and_no_streaming() -> None:
    r = reader_for(reply(["x+1"]))
    assert asyncio.run(r.read(b"\x89PNGdata")) == ["x+1"]
    req = r.seen[0]  # type: ignore[attr-defined]
    sent = json.loads(req.content)
    assert str(req.url) == "http://model.test:11434/api/chat"
    assert sent["model"] == "vis" and sent["stream"] is False and sent["think"] is False
    assert sent["messages"][0]["images"] == [base64.b64encode(b"\x89PNGdata").decode()]
    assert "do not solve" in sent["messages"][0]["content"].lower()
    assert sent["format"]["properties"]["formulas"]["type"] == "array"


@pytest.mark.parametrize(
    "bad",
    [b"not json", {"message": {"content": "plain words"}}, {"message": {}}, reply("x+1")],
)
def test_unreadable_replies_raise_a_safe_error(bad: Any) -> None:
    with pytest.raises(LLMError) as err:
        asyncio.run(reader_for(bad).read(b"img"))
    assert "unreadable" in str(err.value)


def test_non_200_and_unreachable_hosts_raise() -> None:
    with pytest.raises(LLMError, match="status 500"):
        asyncio.run(reader_for(b"", status=500).read(b"img"))

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    dead = OllamaFormulaReader("http://x.test", "vis", transport=httpx.MockTransport(boom))
    with pytest.raises(LLMError, match="not reachable"):
        asyncio.run(dead.read(b"img"))


def test_clean_candidates_strips_delimiters_and_drops_junk() -> None:
    cleaned = clean_candidates(
        ["$x^2$", "$$ y = 1 $$", r"\[ a+b \]", r"\( c \)", "x^2", "", "   ", 5, None, "z" * 50],
        max_chars=20,
    )
    assert cleaned == ["x^2", "y = 1", "a+b", "c"]  # duplicate, empty, non-string, too long gone


def test_clean_candidates_caps_the_count() -> None:
    assert len(clean_candidates([f"x_{i}" for i in range(50)], 100)) == MAX_FORMULAS
