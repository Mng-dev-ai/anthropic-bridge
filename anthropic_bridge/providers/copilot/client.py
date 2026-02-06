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
from .auth import get_copilot_token

COPILOT_API_URL = "https://api.githubcopilot.com/chat/completions"


class CopilotProvider:
    def __init__(self, target_model: str, token: str | None = None):
        self.target_model = target_model.removeprefix("copilot/")
        self._token = token

    def _get_token(self) -> str | None:
        return self._token or get_copilot_token()

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        token = self._get_token()
        if not token:
            yield self._sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "authentication_error",
                        "message": "GitHub Copilot token not found. Run 'anthropic-bridge login' or set GITHUB_COPILOT_TOKEN.",
                    },
                },
            )
            yield self._sse("message_stop", {"type": "message_stop"})
            yield "data: [DONE]\n\n"
            return

        messages = self._convert_messages(payload)
        tools = convert_anthropic_tools_to_openai(payload.get("tools"))

        copilot_payload: dict[str, Any] = {
            "model": self.target_model,
            "messages": messages,
            "temperature": payload.get("temperature", 1),
            "stream": True,
            "max_tokens": payload.get("max_tokens", 16000),
            "stream_options": {"include_usage": True},
        }

        if tools:
            copilot_payload["tools"] = tools
            tool_choice = convert_anthropic_tool_choice_to_openai(
                payload.get("tool_choice")
            )
            if tool_choice:
                copilot_payload["tool_choice"] = tool_choice

        if payload.get("thinking"):
            copilot_payload["include_reasoning"] = True

        async for event in self._stream_copilot(copilot_payload, token):
            yield event

    def _convert_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        system = payload.get("system")
        if isinstance(system, list):
            system = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )
        return convert_anthropic_messages_to_openai(payload.get("messages", []), system)

    def _build_headers(self, token: str) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "user",
            "User-Agent": "anthropic-bridge/0.1",
        }

    async def _stream_copilot(
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
                    "usage": {"input_tokens": 0, "output_tokens": 0},
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

        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream(
                "POST",
                COPILOT_API_URL,
                headers=self._build_headers(token),
                json=payload,
            ) as response,
        ):
            if response.status_code != 200:
                error_text = await response.aread()
                yield self._sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": error_text.decode(errors="replace"),
                        },
                    },
                )
                yield self._sse("message_stop", {"type": "message_stop"})
                yield "data: [DONE]\n\n"
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                for line in lines:
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

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

                    reasoning = delta.get("reasoning") or ""
                    content = delta.get("content") or ""

                    if reasoning:
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
                        if thinking_started:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {
                                        "type": "signature_delta",
                                        "signature": "",
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

        if thinking_started:
            yield self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": thinking_idx,
                    "delta": {"type": "signature_delta", "signature": ""},
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

        yield self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "output_tokens": (
                        usage.get("completion_tokens", 0) if usage else 0
                    ),
                },
            },
        )
        yield self._sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
