import json
import logging
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from ...transform import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_anthropic_tools_to_openai,
)
from ..utils import map_reasoning_effort
from .auth import get_api_key

logger = logging.getLogger(__name__)


class OpenAIProvider:
    def __init__(self, target_model: str):
        self.target_model = target_model.removeprefix("openai/")
        self._api_key: str | None = None
        self._api_key_expires_at: float = 0

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        self._api_key, self._api_key_expires_at = await get_api_key(
            self._api_key, self._api_key_expires_at
        )
        client = AsyncOpenAI(api_key=self._api_key)

        messages = self._convert_messages(payload)
        tools = convert_anthropic_tools_to_openai(payload.get("tools"))

        kwargs: dict[str, Any] = {
            "model": self.target_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if payload.get("max_tokens"):
            kwargs["max_completion_tokens"] = payload["max_tokens"]

        if payload.get("temperature") is not None:
            kwargs["temperature"] = payload["temperature"]

        if tools:
            kwargs["tools"] = tools
            tool_choice = convert_anthropic_tool_choice_to_openai(
                payload.get("tool_choice")
            )
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        if payload.get("thinking"):
            effort = map_reasoning_effort(
                payload["thinking"].get("budget_tokens"), self.target_model
            )
            if effort:
                kwargs["reasoning_effort"] = effort

        async for event in self._stream_response(client, kwargs):
            yield event

    def _convert_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        system = payload.get("system")
        if isinstance(system, list):
            system = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )

        return convert_anthropic_messages_to_openai(payload.get("messages", []), system)

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
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
        usage: dict[str, int] | None = None

        try:
            stream = await client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens or 0,
                        "completion_tokens": chunk.usage.completion_tokens or 0,
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
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
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        },
                    )

                if delta.content:
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
                            "delta": {"type": "text_delta", "text": delta.content},
                        },
                    )

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index or 0
                        if idx not in tools:
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

                            if text_started:
                                yield self._sse(
                                    "content_block_stop",
                                    {"type": "content_block_stop", "index": text_idx},
                                )
                                text_started = False

                            tools[idx] = {
                                "id": tc.id or f"tool_{int(time.time())}_{idx}",
                                "name": "",
                                "block_idx": cur_idx,
                                "started": False,
                                "closed": False,
                            }
                            cur_idx += 1

                        t = tools[idx]

                        if tc.id:
                            t["id"] = tc.id

                        if tc.function and tc.function.name and not t["started"]:
                            t["name"] = tc.function.name
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

                        if tc.function and tc.function.arguments and t["started"]:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": t["block_idx"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": tc.function.arguments,
                                    },
                                },
                            )

                finish = chunk.choices[0].finish_reason
                if finish == "tool_calls":
                    for t in tools.values():
                        if t["started"] and not t["closed"]:
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": t["block_idx"]},
                            )
                            t["closed"] = True

        except Exception as e:
            logger.error("Error calling OpenAI API: %s", e)
            yield self._sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(e)},
                },
            )
            yield self._sse("message_stop", {"type": "message_stop"})
            yield "data: [DONE]\n\n"
            return

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
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "output_tokens": usage.get("completion_tokens", 0) if usage else 0,
                },
            },
        )
        yield self._sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
