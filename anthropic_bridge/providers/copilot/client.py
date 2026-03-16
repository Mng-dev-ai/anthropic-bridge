import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...transform import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_anthropic_tools_to_openai,
)
from ..responses_api import (
    build_responses_input,
    convert_tool_choice_for_responses,
    convert_tools_for_responses,
    stream_responses_api,
)
from ..utils import (
    AnthropicSSEEmitter,
    estimate_input_tokens,
    first_choice,
    map_reasoning_effort,
    sse,
    yield_error_events,
)
from .auth import get_copilot_token

COPILOT_CHAT_API_URL = "https://api.githubcopilot.com/chat/completions"
COPILOT_RESPONSES_API_URL = "https://api.githubcopilot.com/responses"


class CopilotProvider:
    def __init__(self, target_model: str, token: str | None = None):
        self.target_model = target_model.removeprefix("copilot/")
        self._token = token

    def _get_token(self) -> str | None:
        return self._token or get_copilot_token()

    def _should_use_responses_api(self) -> bool:
        model = self.target_model.lower()
        return model.startswith("gpt-5") and "mini" not in model

    def _supports_reasoning(self) -> bool:
        model = self.target_model.lower()
        if "grok" in model:
            return False
        return True

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        token = self._get_token()
        if not token:
            async for event in yield_error_events(
                "GitHub Copilot token not found. Set GITHUB_COPILOT_TOKEN.",
                self.target_model,
            ):
                yield event
            return

        try:
            if self._should_use_responses_api():
                async for event in self._handle_responses(payload, token):
                    yield event
                return

            async for event in self._handle_chat(payload, token):
                yield event
        except Exception as e:
            async for event in yield_error_events(str(e), self.target_model):
                yield event

    async def _handle_responses(
        self, payload: dict[str, Any], token: str
    ) -> AsyncIterator[str]:
        instructions, input_messages = build_responses_input(payload)
        tools = convert_tools_for_responses(payload.get("tools"))

        request_body: dict[str, Any] = {
            "model": self.target_model,
            "input": input_messages,
            "stream": True,
        }

        if instructions:
            request_body["instructions"] = instructions

        max_tokens = payload.get("max_tokens")
        if max_tokens:
            request_body["max_output_tokens"] = max_tokens

        if payload.get("temperature") is not None:
            request_body["temperature"] = payload["temperature"]

        if tools:
            request_body["tools"] = tools
            tool_choice = convert_tool_choice_for_responses(payload.get("tool_choice"))
            if tool_choice:
                request_body["tool_choice"] = tool_choice

        if payload.get("thinking"):
            effort = map_reasoning_effort(
                payload["thinking"].get("budget_tokens"), self.target_model
            )
            if effort:
                request_body["reasoning"] = {"effort": effort, "summary": "auto"}

        headers = self._build_headers(token)

        async for event in stream_responses_api(
            COPILOT_RESPONSES_API_URL,
            headers,
            request_body,
            self.target_model,
        ):
            yield event

    async def _handle_chat(
        self, payload: dict[str, Any], token: str
    ) -> AsyncIterator[str]:
        messages = self._convert_messages(payload)
        tools = convert_anthropic_tools_to_openai(payload.get("tools"))

        copilot_payload: dict[str, Any] = {
            "model": self.target_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        max_tokens = payload.get("max_tokens")
        if max_tokens:
            copilot_payload["max_tokens"] = max_tokens

        if payload.get("temperature") is not None:
            copilot_payload["temperature"] = payload["temperature"]

        if tools:
            copilot_payload["tools"] = tools
            tool_choice = convert_anthropic_tool_choice_to_openai(
                payload.get("tool_choice")
            )
            if tool_choice:
                copilot_payload["tool_choice"] = tool_choice

        if payload.get("thinking"):
            budget = payload["thinking"].get("budget_tokens", 0)
            if "claude" in self.target_model.lower():
                copilot_payload["thinking_budget"] = budget or 4000
            elif self._supports_reasoning():
                copilot_payload["include_reasoning"] = True
                effort = map_reasoning_effort(budget, self.target_model) or "medium"
                copilot_payload["reasoning_effort"] = effort
                copilot_payload["reasoning_summary"] = "auto"
                copilot_payload["include"] = ["reasoning.encrypted_content"]

        async for event in self._stream_chat(copilot_payload, token):
            yield event

    def _convert_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = convert_anthropic_messages_to_openai(
            payload.get("messages", []), payload.get("system")
        )
        self._inject_reasoning_fields(payload.get("messages", []), messages)
        return messages

    def _inject_reasoning_fields(
        self,
        anthropic_messages: list[dict[str, Any]],
        openai_messages: list[dict[str, Any]],
    ) -> None:
        assistant_idx = 0
        for msg in anthropic_messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                assistant_idx += 1
                continue

            reasoning_text = ""
            reasoning_opaque = ""
            for block in content:
                if block.get("type") == "thinking":
                    reasoning_text += block.get("thinking", "")
                    sig = block.get("signature", "")
                    if sig:
                        reasoning_opaque = sig

            if not reasoning_text and not reasoning_opaque:
                assistant_idx += 1
                continue

            count = 0
            for oai_msg in openai_messages:
                if oai_msg.get("role") == "assistant":
                    if count == assistant_idx:
                        if reasoning_opaque:
                            oai_msg["reasoning_text"] = reasoning_text
                            oai_msg["reasoning_opaque"] = reasoning_opaque
                        break
                    count += 1
            assistant_idx += 1

    def _build_headers(self, token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Openai-Intent": "conversation-edits",
            "Editor-Version": "vscode/1.100.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.0",
            "Copilot-Integration-Id": "vscode-chat",
            "x-initiator": "user",
            "User-Agent": "anthropic-bridge/0.1",
        }

    async def _stream_chat(
        self, payload: dict[str, Any], token: str
    ) -> AsyncIterator[str]:
        estimated_input = await asyncio.to_thread(
            estimate_input_tokens,
            payload.get("messages", []),
            payload.get("tools"),
        )

        emitter = AnthropicSSEEmitter(self.target_model, estimated_input)
        for _e in emitter.message_start():
            yield _e

        usage: dict[str, Any] | None = None
        reasoning_opaque: str | None = None

        try:
            async with (
                httpx.AsyncClient(timeout=300.0) as client,
                client.stream(
                    "POST",
                    COPILOT_CHAT_API_URL,
                    headers=self._build_headers(token),
                    json=payload,
                ) as response,
            ):
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise RuntimeError(
                        f"Copilot API returned {response.status_code}: "
                        f"{error_text.decode(errors='replace')}"
                    )

                buffer = ""
                first_data_seen = False
                async for chunk in response.aiter_text():
                    buffer += chunk
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue

                        if not first_data_seen and not line.startswith(
                            ("data: ", "event: ", "id: ", ":")
                        ):
                            try:
                                error_data = json.loads(line)
                                error_msg = (
                                    error_data.get("error", {}).get("message")
                                    or error_data.get("message")
                                    or line
                                )
                            except (json.JSONDecodeError, AttributeError):
                                error_msg = line
                            raise RuntimeError(
                                f"Non-SSE response from Copilot API: {error_msg}"
                            )

                        if not line.startswith("data: "):
                            if line.startswith("event: ") or line.startswith(":"):
                                first_data_seen = True
                            continue

                        first_data_seen = True
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            continue

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if data.get("usage"):
                            usage = data["usage"]

                        choice = first_choice(data)
                        if choice is None:
                            continue

                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            delta = {}

                        if delta.get("reasoning_opaque"):
                            reasoning_opaque = delta["reasoning_opaque"]

                        reasoning = delta.get("reasoning_text") or ""
                        content = delta.get("content") or ""

                        if reasoning:
                            for _e in emitter.thinking_delta(reasoning):
                                yield _e

                        if content:
                            for _e in emitter.close_thinking(reasoning_opaque or ""):
                                yield _e
                            for _e in emitter.text_delta(content):
                                yield _e

                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            if emitter.get_tool(idx) is None:
                                tool_id = tc.get("id") or f"tool_{idx}"
                                for _e in emitter.register_tool(idx, tool_id):
                                    yield _e

                            fn = tc.get("function", {})
                            if fn.get("name"):
                                for _e in emitter.start_tool(idx, fn["name"]):
                                    yield _e

                            tool_entry = emitter.get_tool(idx)
                            if fn.get("arguments") and tool_entry and tool_entry["started"]:
                                for _e in emitter.tool_delta(idx, fn["arguments"]):
                                    yield _e

                        finish = choice.get("finish_reason")
                        if finish == "tool_calls":
                            for key in emitter.tool_keys:
                                for _e in emitter.close_tool(key):
                                    yield _e

        except Exception as e:
            for _e in emitter.error_and_finish(str(e)):
                yield _e
            return

        if not emitter.had_content and not emitter.has_tools:
            yield sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": (
                            f"No content received from Copilot API for model "
                            f"'{self.target_model}'. The model may not be available "
                            f"or the response format may be unsupported."
                        ),
                    },
                },
            )

        for _e in emitter.finish(
            {
                "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) if usage else 0,
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) if usage else 0,
                "output_tokens": usage.get("completion_tokens", 0) if usage else 0,
            },
            signature=reasoning_opaque or "",
        ):
            yield _e
