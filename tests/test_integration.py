import os
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from anthropic_bridge.providers.openai.auth import auth_file_exists
from anthropic_bridge.server import create_app

from .conftest import (
    CALCULATOR_TOOL,
    WEATHER_TOOL,
    extract_text_from_events,
    extract_tool_calls_from_events,
    parse_sse_stream,
)

pytestmark = pytest.mark.live

skip_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
skip_openai = pytest.mark.skipif(
    not auth_file_exists(),
    reason="OpenAI auth file not found (~/.codex/auth.json)",
)
skip_copilot = pytest.mark.skipif(
    not os.environ.get("GITHUB_COPILOT_TOKEN"),
    reason="GITHUB_COPILOT_TOKEN not set",
)

GEMINI_MODEL = "openrouter/google/gemini-3-pro-preview"
OPENAI_MODEL = "openai/gpt-5.2"
COPILOT_MODEL = "copilot/gpt-5.3-codex"


@pytest.fixture(autouse=True, scope="module")
def require_live_tests() -> None:
    if os.environ.get("ANTHROPIC_BRIDGE_LIVE_TESTS") != "1":
        pytest.skip("Live provider tests require ANTHROPIC_BRIDGE_LIVE_TESTS=1")


@pytest.fixture
def app():
    return create_app(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        copilot_token=os.environ.get("GITHUB_COPILOT_TOKEN"),
    )


@pytest.fixture
async def live_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def streaming_request(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    payload.update(extra)
    return payload


async def post_events(client: AsyncClient, payload: dict[str, Any]) -> list[dict[str, Any]]:
    response = await client.post(
        "/v1/messages",
        json=payload,
        timeout=60.0,
    )
    assert response.status_code == 200
    return await parse_sse_stream(response)


def skip_on_openrouter_issue(events: list[dict[str, Any]]) -> None:
    for event in events:
        if event["event"] != "error":
            continue
        message = event["data"].get("error", {}).get("message", "")
        lower = message.lower()
        if (
            "rate-limit" in lower
            or "rate limited" in lower
            or "retry shortly" in lower
            or "no content received from openrouter api" in lower
            or "may not be available" in lower
        ):
            pytest.skip(message)
        pytest.fail(message)


class TestMultiRoundStreaming:
    @skip_openrouter
    @pytest.mark.asyncio
    async def test_multi_round_conversation_gemini(self, live_client):
        client = live_client
        messages: list[dict[str, Any]] = []

        messages.append({"role": "user", "content": "What is 2 + 2?"})
        events = await post_events(client, streaming_request(GEMINI_MODEL, messages, 500))
        skip_on_openrouter_issue(events)
        text1 = extract_text_from_events(events)
        assert len(text1) > 0
        assert "4" in text1
        messages.append({"role": "assistant", "content": text1})

        messages.append({"role": "user", "content": "Multiply that result by 3"})
        events = await post_events(client, streaming_request(GEMINI_MODEL, messages, 500))
        skip_on_openrouter_issue(events)
        text2 = extract_text_from_events(events)
        assert len(text2) > 0
        assert "12" in text2
        messages.append({"role": "assistant", "content": text2})

        messages.append({"role": "user", "content": "What was my first question?"})
        events = await post_events(client, streaming_request(GEMINI_MODEL, messages, 500))
        skip_on_openrouter_issue(events)
        text3 = extract_text_from_events(events)
        assert len(text3) > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_multi_round_conversation_openai(self, live_client):
        client = live_client
        messages: list[dict[str, Any]] = []

        messages.append({"role": "user", "content": "What is 5 + 5?"})
        events = await post_events(client, streaming_request(OPENAI_MODEL, messages, 500))
        text1 = extract_text_from_events(events)
        assert len(text1) > 0
        assert "10" in text1
        messages.append({"role": "assistant", "content": text1})

        messages.append({"role": "user", "content": "Add 5 more to that"})
        events = await post_events(client, streaming_request(OPENAI_MODEL, messages, 500))
        text2 = extract_text_from_events(events)
        assert len(text2) > 0
        assert "15" in text2


class TestMultiRoundToolCalls:
    @skip_openrouter
    @pytest.mark.asyncio
    async def test_multi_round_tool_calls_gemini(self, live_client):
        client = live_client
        messages: list[dict[str, Any]] = []

        messages.append(
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Use the get_weather tool.",
            }
        )
        events = await post_events(
            client,
            streaming_request(
                GEMINI_MODEL,
                messages,
                1000,
                tools=[WEATHER_TOOL],
            ),
        )
        skip_on_openrouter_issue(events)
        tool_calls = extract_tool_calls_from_events(events)

        if not tool_calls:
            pytest.skip("OpenRouter Gemini did not emit a tool call for this live request")
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
        events = await post_events(
            client,
            streaming_request(
                GEMINI_MODEL,
                messages,
                1000,
                tools=[WEATHER_TOOL],
            ),
        )
        skip_on_openrouter_issue(events)
        text2 = extract_text_from_events(events)
        assert "25" in text2 or "sunny" in text2.lower()
        messages.append({"role": "assistant", "content": text2})

        messages.append(
            {"role": "user", "content": "Thanks! What city did I ask about?"}
        )
        events = await post_events(
            client,
            streaming_request(
                GEMINI_MODEL,
                messages,
                1000,
                tools=[WEATHER_TOOL],
            ),
        )
        skip_on_openrouter_issue(events)
        text3 = extract_text_from_events(events)
        assert len(text3) > 0

    @skip_openai
    @pytest.mark.asyncio
    async def test_multi_round_tool_calls_openai(self, live_client):
        client = live_client
        messages: list[dict[str, Any]] = []

        messages.append(
            {
                "role": "user",
                "content": "Calculate 15 * 7 using the calculate tool.",
            }
        )
        events = await post_events(
            client,
            streaming_request(
                OPENAI_MODEL,
                messages,
                1000,
                tools=[CALCULATOR_TOOL],
                tool_choice={"type": "tool", "name": CALCULATOR_TOOL["name"]},
            ),
        )
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
        events = await post_events(
            client,
            streaming_request(
                OPENAI_MODEL,
                messages,
                1000,
                tools=[CALCULATOR_TOOL],
            ),
        )
        text2 = extract_text_from_events(events)
        assert "105" in text2


class TestModelSwitching:
    @skip_openrouter
    @skip_openai
    @pytest.mark.asyncio
    async def test_switch_between_models(self, live_client):
        client = live_client
        messages: list[dict[str, Any]] = []

        messages.append({"role": "user", "content": "What is 7 + 3?"})
        events = await post_events(client, streaming_request(GEMINI_MODEL, messages, 500))
        skip_on_openrouter_issue(events)
        text1 = extract_text_from_events(events)
        assert "10" in text1
        messages.append({"role": "assistant", "content": text1})

        messages.append({"role": "user", "content": "Double that number"})
        events = await post_events(client, streaming_request(OPENAI_MODEL, messages, 500))
        text2 = extract_text_from_events(events)
        assert "20" in text2
        messages.append({"role": "assistant", "content": text2})

        messages.append({"role": "user", "content": "Add 5 to that"})
        events = await post_events(client, streaming_request(GEMINI_MODEL, messages, 500))
        skip_on_openrouter_issue(events)
        text3 = extract_text_from_events(events)
        assert "25" in text3


class TestSSEStreamStructure:
    @skip_openrouter
    @pytest.mark.asyncio
    async def test_sse_event_sequence(self, live_client):
        client = live_client
        events = await post_events(
            client,
            streaming_request(
                GEMINI_MODEL,
                [{"role": "user", "content": "Say hello"}],
                100,
            ),
        )

        event_types = [e["event"] for e in events]

        assert event_types[0] == "message_start"
        assert "ping" in event_types
        assert "message_delta" in event_types
        assert event_types[-1] == "message_stop"

        msg_start = events[0]["data"]
        assert msg_start["type"] == "message_start"
        assert "message" in msg_start
        assert msg_start["message"]["role"] == "assistant"

    @skip_openai
    @pytest.mark.asyncio
    async def test_tool_call_sse_structure(self, live_client):
        client = live_client
        events = await post_events(
            client,
            streaming_request(
                OPENAI_MODEL,
                [{"role": "user", "content": "Get weather for NYC"}],
                1000,
                tools=[WEATHER_TOOL],
            ),
        )

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

        tool_block = tool_starts[0]["data"]["content_block"]
        assert "id" in tool_block
        assert "name" in tool_block
        assert tool_block["type"] == "tool_use"


class TestCopilotSmoke:
    @skip_copilot
    @pytest.mark.asyncio
    async def test_basic_response_copilot(self, live_client):
        client = live_client
        events = await post_events(
            client,
            streaming_request(
                COPILOT_MODEL,
                [{"role": "user", "content": "Reply with exactly READY"}],
                64,
            ),
        )
        text = extract_text_from_events(events)
        assert "ready" in text.lower()
