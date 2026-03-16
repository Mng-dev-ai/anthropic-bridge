from collections.abc import AsyncIterator
from typing import Any

import pytest

from anthropic_bridge.providers.copilot.client import CopilotProvider
from anthropic_bridge.providers.openai.client import OpenAIProvider
from anthropic_bridge.providers.openrouter.client import OpenRouterProvider
from anthropic_bridge.providers.responses_api import stream_responses_api

from .conftest import collect_events, fake_client_factory


@pytest.mark.asyncio
async def test_openrouter_parser_skips_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        ": OPENROUTER PROCESSING\n\n",
        'data: {"choices":[]}\n\n',
        'data: {"usage":{"prompt_tokens":3}}\n\n',
        (
            'data: {"choices":[{"index":0,"delta":{"content":"Hi","role":"assistant"},'
            '"finish_reason":null}]}\n\n'
        ),
        (
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
        ),
        "data: [DONE]\n\n",
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    events = await collect_events(
        provider._stream_openrouter(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
        )
    )

    assert ("message_stop", {"type": "message_stop"}) in events
    assert not any(event == "error" for event, _ in events)
    assert any(
        event == "content_block_delta"
        and data["delta"].get("type") == "text_delta"
        and data["delta"].get("text") == "Hi"
        for event, data in events
    )


@pytest.mark.asyncio
async def test_copilot_parser_skips_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: message\n",
        'data: {"choices":[]}\n',
        'data: {"usage":{"prompt_tokens":2}}\n',
        (
            'data: {"choices":[{"index":0,"delta":{"content":"Hi","role":"assistant"},'
            '"finish_reason":null}]}\n'
        ),
        (
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n'
        ),
        "data: [DONE]\n",
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.copilot.client.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    provider = CopilotProvider("copilot/claude-opus-4.6", token="token")
    events = await collect_events(
        provider._stream_chat(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
            "token",
        )
    )

    assert ("message_stop", {"type": "message_stop"}) in events
    assert not any(event == "error" for event, _ in events)
    assert any(
        event == "content_block_delta"
        and data["delta"].get("type") == "text_delta"
        and data["delta"].get("text") == "Hi"
        for event, data in events
    )


@pytest.mark.asyncio
async def test_openai_provider_omits_instructions_without_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_get_auth(*args: Any) -> tuple[str, None, float]:
        return "token", None, 0

    async def fake_stream(
        endpoint: str,
        headers: dict[str, str],
        request_body: dict[str, Any],
        target_model: str,
    ) -> AsyncIterator[str]:
        captured["endpoint"] = endpoint
        captured["headers"] = headers
        captured["body"] = request_body
        captured["target_model"] = target_model
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    monkeypatch.setattr(
        "anthropic_bridge.providers.openai.client.get_auth",
        fake_get_auth,
    )
    monkeypatch.setattr(
        "anthropic_bridge.providers.openai.client.stream_responses_api",
        fake_stream,
    )

    provider = OpenAIProvider("openai/gpt-5.2")
    events = await collect_events(
        provider.handle({"messages": [{"role": "user", "content": "Hi"}]})
    )

    assert events == [("message_stop", {"type": "message_stop"})]
    assert "instructions" not in captured["body"]
    assert captured["body"]["input"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_openrouter_empty_stream_yields_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.httpx.AsyncClient",
        fake_client_factory([": OPENROUTER PROCESSING\n\n", "data: [DONE]\n\n"]),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    events = await collect_events(
        provider._stream_openrouter(
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 10},
        )
    )

    assert any(event == "error" for event, _ in events)


@pytest.mark.asyncio
async def test_responses_api_ignores_duplicate_reasoning_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: response.reasoning_summary_text.delta",
        'data: {"delta":"Plan"}',
        "event: response.reasoning.delta",
        'data: {"delta":"Plan"}',
        "event: response.output_text.delta",
        'data: {"delta":"Answer"}',
        "event: response.completed",
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}',
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.responses_api.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    events = await collect_events(
        stream_responses_api(
            "https://example.test/responses",
            {},
            {
                "input": [{"role": "user", "content": "Hi"}],
                "reasoning": {"effort": "low", "summary": "auto"},
            },
            "gpt-5.2",
        )
    )

    thinking_deltas = [
        data["delta"]["thinking"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"].get("type") == "thinking_delta"
    ]

    assert thinking_deltas == ["Plan"]


@pytest.mark.asyncio
async def test_responses_api_does_not_repeat_tool_args_on_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        "event: response.output_item.added",
        'data: {"item":{"type":"function_call","call_id":"call_1","name":"calc"}}',
        "event: response.function_call_arguments.delta",
        'data: {"call_id":"call_1","delta":"{\\"x\\":1}"}',
        "event: response.output_item.done",
        'data: {"item":{"type":"function_call","call_id":"call_1","arguments":"{\\"x\\":1}"}}',
        "event: response.completed",
        'data: {"response":{"usage":{"input_tokens":1,"output_tokens":1}}}',
    ]
    monkeypatch.setattr(
        "anthropic_bridge.providers.responses_api.httpx.AsyncClient",
        fake_client_factory(chunks),
    )

    events = await collect_events(
        stream_responses_api(
            "https://example.test/responses",
            {},
            {"input": [{"role": "user", "content": "Hi"}]},
            "gpt-5.2",
        )
    )

    arg_deltas = [
        data["delta"]["partial_json"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"].get("type") == "input_json_delta"
    ]

    assert arg_deltas == ['{"x":1}']


def test_openrouter_inject_gemini_reasoning_deduplicates_cached_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCache:
        def get(self, tool_call_id: str) -> list[dict[str, Any]] | None:
            mapping = {
                "tool_a": [{"id": "r1", "type": "reasoning"}],
                "tool_b": [{"id": "r1", "type": "reasoning"}],
            }
            return mapping.get(tool_call_id)

    monkeypatch.setattr(
        "anthropic_bridge.providers.openrouter.client.get_reasoning_cache",
        lambda: FakeCache(),
    )

    provider = OpenRouterProvider("openrouter/google/gemini-3-pro-preview", "token")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "tool_a", "function": {"name": "a", "arguments": "{}"}},
                {"id": "tool_b", "function": {"name": "b", "arguments": "{}"}},
            ],
        }
    ]

    provider._inject_gemini_reasoning(messages)

    assert messages[0]["reasoning_details"] == [{"id": "r1", "type": "reasoning"}]
