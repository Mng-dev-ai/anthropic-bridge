import json
import os
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from anthropic_bridge.providers.openai.auth import auth_file_exists
from anthropic_bridge.server import create_app

skip_openai = pytest.mark.skipif(
    not auth_file_exists(),
    reason="OpenAI auth file not found (~/.codex/auth.json)",
)

GEMINI_MODEL = "openrouter/google/gemini-3-pro-preview"
OPENAI_MODEL = "openai/gpt-5.2"

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}

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


@pytest.fixture
def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture
def app(api_key: str):
    return create_app(openrouter_api_key=api_key)


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


class TestMultiRoundStreaming:
    @pytest.mark.asyncio
    async def test_multi_round_conversation_gemini(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            messages: list[dict[str, Any]] = []

            # round 1
            messages.append({"role": "user", "content": "What is 2 + 2?"})
            response = await client.post(
                "/v1/messages",
                json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text1 = extract_text_from_events(events)
            assert len(text1) > 0
            assert "4" in text1
            messages.append({"role": "assistant", "content": text1})

            # round 2
            messages.append({"role": "user", "content": "Multiply that result by 3"})
            response = await client.post(
                "/v1/messages",
                json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text2 = extract_text_from_events(events)
            assert len(text2) > 0
            assert "12" in text2
            messages.append({"role": "assistant", "content": text2})

            # round 3 - verify context retention
            messages.append({"role": "user", "content": "What was my first question?"})
            response = await client.post(
                "/v1/messages",
                json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text3 = extract_text_from_events(events)
            assert len(text3) > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_multi_round_conversation_openai(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            messages: list[dict[str, Any]] = []

            # round 1
            messages.append({"role": "user", "content": "What is 5 + 5?"})
            response = await client.post(
                "/v1/messages",
                json={"model": OPENAI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text1 = extract_text_from_events(events)
            assert len(text1) > 0
            assert "10" in text1
            messages.append({"role": "assistant", "content": text1})

            # round 2
            messages.append({"role": "user", "content": "Add 5 more to that"})
            response = await client.post(
                "/v1/messages",
                json={"model": OPENAI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text2 = extract_text_from_events(events)
            assert len(text2) > 0
            assert "15" in text2


class TestMultiRoundToolCalls:
    @pytest.mark.asyncio
    async def test_multi_round_tool_calls_gemini(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            messages: list[dict[str, Any]] = []

            # round 1 - trigger tool call
            messages.append(
                {
                    "role": "user",
                    "content": "What's the weather in Tokyo? Use the get_weather tool.",
                }
            )
            response = await client.post(
                "/v1/messages",
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "tools": [WEATHER_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            tool_calls = extract_tool_calls_from_events(events)

            assert len(tool_calls) >= 1
            tc = tool_calls[0]
            assert tc["name"] == "get_weather"
            assert "tokyo" in tc["input"].get("location", "").lower()

            text_before_tool = extract_text_from_events(events)
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text_before_tool}
                        if text_before_tool
                        else None,
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"],
                        },
                    ],
                }
            )
            messages[-1]["content"] = [c for c in messages[-1]["content"] if c]

            # round 2 - provide tool result
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": "Sunny, 25°C",
                        }
                    ],
                }
            )
            response = await client.post(
                "/v1/messages",
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "tools": [WEATHER_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text2 = extract_text_from_events(events)
            assert "25" in text2 or "sunny" in text2.lower()
            messages.append({"role": "assistant", "content": text2})

            # round 3 - check that another request works
            messages.append(
                {"role": "user", "content": "Thanks! What city did I ask about?"}
            )
            response = await client.post(
                "/v1/messages",
                json={
                    "model": GEMINI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "tools": [WEATHER_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text3 = extract_text_from_events(events)
            assert len(text3) > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_multi_round_tool_calls_openai(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            messages: list[dict[str, Any]] = []

            # round 1 - trigger tool call
            messages.append(
                {
                    "role": "user",
                    "content": "Calculate 15 * 7 using the calculate tool.",
                }
            )
            response = await client.post(
                "/v1/messages",
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "tools": [CALCULATOR_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            tool_calls = extract_tool_calls_from_events(events)

            assert len(tool_calls) >= 1
            tc = tool_calls[0]
            assert tc["name"] == "calculate"

            text_before_tool = extract_text_from_events(events)
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text_before_tool}
                        if text_before_tool
                        else None,
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["input"],
                        },
                    ],
                }
            )
            messages[-1]["content"] = [c for c in messages[-1]["content"] if c]

            # round 2 - provide tool result
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": "105",
                        }
                    ],
                }
            )
            response = await client.post(
                "/v1/messages",
                json={
                    "model": OPENAI_MODEL,
                    "messages": messages,
                    "max_tokens": 1000,
                    "tools": [CALCULATOR_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text2 = extract_text_from_events(events)
            assert "105" in text2


class TestModelSwitching:
    @skip_openai
    @pytest.mark.asyncio
    async def test_switch_between_models(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            messages: list[dict[str, Any]] = []

            # round 1 - use Gemini
            messages.append({"role": "user", "content": "What is 7 + 3?"})
            response = await client.post(
                "/v1/messages",
                json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text1 = extract_text_from_events(events)
            assert "10" in text1
            messages.append({"role": "assistant", "content": text1})

            # round 2 - switch to OpenAI
            messages.append({"role": "user", "content": "Double that number"})
            response = await client.post(
                "/v1/messages",
                json={"model": OPENAI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text2 = extract_text_from_events(events)
            assert "20" in text2
            messages.append({"role": "assistant", "content": text2})

            # round 3 - back to Gemini
            messages.append({"role": "user", "content": "Add 5 to that"})
            response = await client.post(
                "/v1/messages",
                json={"model": GEMINI_MODEL, "messages": messages, "max_tokens": 500},
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)
            text3 = extract_text_from_events(events)
            assert "25" in text3


class TestSSEStreamStructure:
    @pytest.mark.asyncio
    async def test_sse_event_sequence(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": GEMINI_MODEL,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 100,
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)

            event_types = [e["event"] for e in events]

            assert event_types[0] == "message_start"
            assert "ping" in event_types
            assert "message_delta" in event_types
            assert event_types[-1] == "message_stop"

            # verify message_start structure
            msg_start = events[0]["data"]
            assert msg_start["type"] == "message_start"
            assert "message" in msg_start
            assert msg_start["message"]["role"] == "assistant"

    @skip_openai
    @pytest.mark.asyncio
    async def test_tool_call_sse_structure(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/messages",
                json={
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": "Get weather for NYC"}],
                    "max_tokens": 1000,
                    "tools": [WEATHER_TOOL],
                },
                timeout=60.0,
            )
            assert response.status_code == 200
            events = await parse_sse_stream(response)

            tool_starts = [
                e
                for e in events
                if e["event"] == "content_block_start"
                and e["data"].get("content_block", {}).get("type") == "tool_use"
            ]
            tool_deltas = [
                e
                for e in events
                if e["event"] == "content_block_delta"
                and e["data"].get("delta", {}).get("type") == "input_json_delta"
            ]

            assert len(tool_starts) >= 1
            assert len(tool_deltas) >= 1

            # verify tool_use block structure
            tool_block = tool_starts[0]["data"]["content_block"]
            assert "id" in tool_block
            assert "name" in tool_block
            assert tool_block["type"] == "tool_use"
