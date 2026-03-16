import json
from collections.abc import AsyncIterator
from typing import Any

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "Perform a calculation",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to evaluate",
            }
        },
        "required": ["expression"],
    },
}

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}


class FakeStreamResponse:
    def __init__(self, chunks: list[str], status_code: int = 200):
        self._chunks = chunks
        self.status_code = status_code

    async def __aenter__(self) -> "FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_text(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class FakeAsyncClient:
    def __init__(self, chunks: list[str], status_code: int = 200, **_: Any):
        self._chunks = chunks
        self._status_code = status_code

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def stream(self, *_: Any, **__: Any) -> FakeStreamResponse:
        return FakeStreamResponse(self._chunks, self._status_code)


def fake_client_factory(chunks: list[str]):
    def factory(*args: Any, **kwargs: Any) -> FakeAsyncClient:
        return FakeAsyncClient(chunks, **kwargs)
    return factory


async def collect_events(
    generator: AsyncIterator[str],
) -> list[tuple[str, dict[str, Any]]]:
    from anthropic_bridge.protocol import iter_sse_events
    return [event async for event in iter_sse_events(generator)]


def extract_text_from_events(events: list[dict[str, Any]]) -> str:
    text = ""
    for e in events:
        if e["event"] == "content_block_delta":
            delta = e["data"].get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
    return text


def extract_tool_calls_from_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools: dict[int, dict[str, Any]] = {}
    for e in events:
        if e["event"] == "content_block_start":
            block = e["data"].get("content_block", {})
            if block.get("type") == "tool_use":
                idx = e["data"].get("index", 0)
                tools[idx] = {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": "",
                }
        elif e["event"] == "content_block_delta":
            delta = e["data"].get("delta", {})
            if delta.get("type") == "input_json_delta":
                idx = e["data"].get("index", 0)
                if idx in tools:
                    tools[idx]["input"] += delta.get("partial_json", "")

    result = []
    for t in tools.values():
        try:
            t["input"] = json.loads(t["input"]) if t["input"] else {}
        except json.JSONDecodeError:
            t["input"] = {}
        result.append(t)
    return result


async def parse_sse_stream(response) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines.pop()

        current_event = ""
        for line in lines:
            line = line.strip()
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        events.append({"event": current_event, "data": data})
                    except json.JSONDecodeError:
                        pass
    return events
