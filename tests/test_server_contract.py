import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

import anthropic_bridge.server as server_module
from anthropic_bridge.protocol import estimate_anthropic_input_tokens, iter_sse_events
from anthropic_bridge.server import AnthropicBridge, ProxyConfig

from .conftest import CALCULATOR_TOOL

FAKE_MODEL = "fake-model"
DEFAULT_USAGE = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
}


class FakeProvider:
    def __init__(self, events: list[tuple[str, dict[str, Any]]]):
        self.events = events

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        for event, data in self.events:
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"


def message_start_event(
    *, usage: dict[str, int] | None = None, model: str = FAKE_MODEL
) -> tuple[str, dict[str, Any]]:
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {**DEFAULT_USAGE, **(usage or {})},
            },
        },
    )


def bridge_with_events(
    monkeypatch: pytest.MonkeyPatch, *events: tuple[str, dict[str, Any]]
) -> AnthropicBridge:
    bridge = AnthropicBridge(ProxyConfig())
    monkeypatch.setattr(bridge, "_get_provider", lambda model: FakeProvider(list(events)))
    return bridge


async def post_message(app: Any, payload: dict[str, Any]):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.post("/v1/messages", json=payload)


@pytest.mark.asyncio
async def test_non_stream_messages_return_json(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = bridge_with_events(
        monkeypatch,
        message_start_event(usage={"input_tokens": 11}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Planning"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "calculate",
                    "input": {},
                },
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"expression":"2+2"}',
                },
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "text_delta", "text": "Use the tool."},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 2}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {**DEFAULT_USAGE, "input_tokens": 11, "output_tokens": 7},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )

    response = await post_message(
        bridge.app,
        {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["id"] == "msg_test"
    assert body["stop_reason"] == "tool_use"
    assert body["usage"]["output_tokens"] == 7
    assert body["content"] == [
        {"type": "thinking", "thinking": "Planning"},
        {
            "type": "tool_use",
            "id": "tool_1",
            "name": "calculate",
            "input": {"expression": "2+2"},
        },
        {"type": "text", "text": "Use the tool."},
    ]


@pytest.mark.asyncio
async def test_non_stream_errors_return_json_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = bridge_with_events(
        monkeypatch,
        message_start_event(),
        (
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": "provider failed"},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )

    response = await post_message(
        bridge.app,
        {"model": FAKE_MODEL, "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "provider failed"


@pytest.mark.asyncio
async def test_stream_messages_keep_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = bridge_with_events(
        monkeypatch,
        message_start_event(),
        ("message_stop", {"type": "message_stop"}),
    )

    response = await post_message(
        bridge.app,
        {
            "model": FAKE_MODEL,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text


def test_explicit_prefixes_do_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "auth_file_exists", lambda: False)
    bridge = AnthropicBridge(ProxyConfig(openrouter_api_key="openrouter-key"))

    assert bridge._get_provider("openai/gpt-5.2") is None
    assert bridge._get_provider("copilot/gpt-5.3-codex") is None
    assert bridge._get_provider("gpt-5.2") is not None


@pytest.mark.asyncio
async def test_strict_prefix_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_module, "auth_file_exists", lambda: False)
    bridge = AnthropicBridge(ProxyConfig(openrouter_api_key="openrouter-key"))

    async with AsyncClient(
        transport=ASGITransport(app=bridge.app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/v1/messages",
            json={"model": "openai/gpt-5.2", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert response.status_code == 401
    assert "codex login" in response.json()["error"]["message"]


def test_count_tokens_skips_binary_payloads() -> None:
    small = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "abc",
                        },
                    }
                ],
            }
        ]
    }
    large = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "a" * 10000,
                        },
                    }
                ],
            }
        ]
    }

    assert estimate_anthropic_input_tokens(small) == estimate_anthropic_input_tokens(
        large
    )


def test_count_tokens_include_tools_and_tool_history() -> None:
    plain = {"messages": [{"role": "user", "content": "Hello"}]}
    with_tools = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [CALCULATOR_TOOL],
    }
    with_history = {
        "messages": [
            {"role": "user", "content": "Use the calculator"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "calculate",
                        "input": {"expression": "2+2"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "4",
                    }
                ],
            },
        ],
        "tools": [CALCULATOR_TOOL],
    }

    assert estimate_anthropic_input_tokens(with_tools) > estimate_anthropic_input_tokens(
        plain
    )
    assert estimate_anthropic_input_tokens(
        with_history
    ) > estimate_anthropic_input_tokens(with_tools)


@pytest.mark.asyncio
async def test_count_tokens_endpoint_uses_structured_estimate() -> None:
    bridge = AnthropicBridge(ProxyConfig())
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tools": [CALCULATOR_TOOL],
    }

    async with AsyncClient(
        transport=ASGITransport(app=bridge.app),
        base_url="http://test",
    ) as client:
        response = await client.post("/v1/messages/count_tokens", json=payload)

    assert response.status_code == 200
    assert response.json()["input_tokens"] == estimate_anthropic_input_tokens(payload)


@pytest.mark.asyncio
async def test_sse_parser_round_trips_events() -> None:
    async def chunks() -> AsyncIterator[str]:
        yield 'event: ping\ndata: {"type":"ping"}\n\n'
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    events = [event async for event in iter_sse_events(chunks())]
    assert events == [
        ("ping", {"type": "ping"}),
        ("message_stop", {"type": "message_stop"}),
    ]
