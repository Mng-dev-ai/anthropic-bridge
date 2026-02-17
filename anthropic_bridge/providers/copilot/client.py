import json
import random
import string
import time
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
    convert_tools_for_responses,
    stream_responses_api,
)
from ..utils import map_reasoning_effort, yield_error_events
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
            "instructions": instructions,
            "input": input_messages,
            "stream": True,
        }

        if payload.get("temperature") is not None:
            request_body["temperature"] = payload["temperature"]

        if tools:
            request_body["tools"] = tools

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
        system = payload.get("system")
        if isinstance(system, list):
            system = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )
        messages = convert_anthropic_messages_to_openai(
            payload.get("messages", []), system
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
            "x-initiator": "user",
            "User-Agent": "anthropic-bridge/0.1",
        }

    async def _stream_chat(
        self, payload: dict[str, Any], token: str
    ) -> AsyncIterator[str]:
        msg_id = f"msg_{int(time.time())}_{self._random_id()}"

        yield self._sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.target_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        yield self._sse("ping", {"type": "ping"})

        text_started = False
        text_idx = -1
        thinking_started = False
        thinking_idx = -1
        cur_idx = 0
        tools: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        had_content = False
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

                        delta = data.get("choices", [{}])[0].get("delta", {})

                        if delta.get("reasoning_opaque"):
                            reasoning_opaque = delta["reasoning_opaque"]

                        reasoning = delta.get("reasoning_text") or ""
                        content = delta.get("content") or ""

                        if reasoning:
                            had_content = True
                            if not thinking_started:
                                thinking_idx = cur_idx
                                cur_idx += 1
                                yield self._sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": thinking_idx,
                                        "content_block": {
                                            "type": "thinking",
                                            "thinking": "",
                                            "signature": "",
                                        },
                                    },
                                )
                                thinking_started = True

                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {
                                        "type": "thinking_delta",
                                        "thinking": reasoning,
                                    },
                                },
                            )

                        if content:
                            had_content = True
                            if thinking_started:
                                yield self._sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": thinking_idx,
                                        "delta": {
                                            "type": "signature_delta",
                                            "signature": reasoning_opaque or "",
                                        },
                                    },
                                )
                                yield self._sse(
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": thinking_idx,
                                    },
                                )
                                thinking_started = False

                            if not text_started:
                                text_idx = cur_idx
                                cur_idx += 1
                                yield self._sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_idx,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                                text_started = True

                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": content},
                                },
                            )

                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            had_content = True
                            idx = tc.get("index", 0)
                            if idx not in tools:
                                if text_started:
                                    yield self._sse(
                                        "content_block_stop",
                                        {
                                            "type": "content_block_stop",
                                            "index": text_idx,
                                        },
                                    )
                                    text_started = False

                                tools[idx] = {
                                    "id": tc.get("id") or f"tool_{int(time.time())}_{idx}",
                                    "name": tc.get("function", {}).get("name", ""),
                                    "block_idx": cur_idx,
                                    "started": False,
                                    "closed": False,
                                }
                                cur_idx += 1

                            t = tools[idx]
                            fn = tc.get("function", {})

                            if fn.get("name") and not t["started"]:
                                t["name"] = fn["name"]
                                yield self._sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": t["block_idx"],
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": t["id"],
                                            "name": t["name"],
                                            "input": {},
                                        },
                                    },
                                )
                                t["started"] = True

                            if fn.get("arguments") and t["started"]:
                                yield self._sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": t["block_idx"],
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": fn["arguments"],
                                        },
                                    },
                                )

                        finish = data.get("choices", [{}])[0].get("finish_reason")
                        if finish == "tool_calls":
                            for t in tools.values():
                                if t["started"] and not t["closed"]:
                                    yield self._sse(
                                        "content_block_stop",
                                        {
                                            "type": "content_block_stop",
                                            "index": t["block_idx"],
                                        },
                                    )
                                    t["closed"] = True

        except Exception as e:
            yield self._sse(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )
            yield self._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                    },
                    "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0},
                },
            )
            yield self._sse("message_stop", {"type": "message_stop"})
            return

        if not had_content and not tools:
            yield self._sse(
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

        if thinking_started:
            yield self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": thinking_idx,
                    "delta": {
                        "type": "signature_delta",
                        "signature": reasoning_opaque or "",
                    },
                },
            )
            yield self._sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": thinking_idx},
            )

        if text_started:
            yield self._sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": text_idx},
            )

        for t in tools.values():
            if t["started"] and not t["closed"]:
                yield self._sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": t["block_idx"]},
                )

        stop_reason = "tool_use" if tools else "end_turn"
        yield self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) if usage else 0,
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) if usage else 0,
                    "output_tokens": (
                        usage.get("completion_tokens", 0) if usage else 0
                    ),
                },
            },
        )
        yield self._sse("message_stop", {"type": "message_stop"})

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
